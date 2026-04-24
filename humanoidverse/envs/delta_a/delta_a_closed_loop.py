import torch # 导入 PyTorch 深度学习库
import numpy as np # 导入 NumPy 数学计算库
from pathlib import Path # 导入 Path 用于处理文件路径
import os # 导入 os 模块与操作系统交互
# 从 isaac_utils 导入四元数旋转相关的数学计算函数（用于处理机器人姿态）
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_rotate_inverse
)
# from isaacgym import gymtorch, gymapi, gymutil # （被注释掉的）Isaac Gym 相关库导入
from humanoidverse.envs.env_utils.visualization import Point # 导入环境可视化工具

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree # 导入骨骼树结构（用于解析人体或机器人运动学）

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot # 导入机器人运动库

from termcolor import colored # 导入带颜色的终端输出打印工具
from loguru import logger # 导入高级日志记录工具

from scipy.spatial.transform import Rotation as sRot # 导入 SciPy 的旋转矩阵转换工具
import joblib # 导入 joblib 用于序列化或并行计算

# 导入底层运动跟踪基类
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
# 导入 PyTorch 相关的辅助函数
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float

class DeltaA_ClosedLoop(LeggedRobotMotionTracking):
    # 【类作用说明】
    # 这是一个继承自 LeggedRobotMotionTracking 的环境类，代表嵌入了 Delta-Action (残差动作) 的闭环强化学习环境。
    # 它的核心目的是修改默认的物理引擎计算逻辑，在向 PD 控制器发送目标关节角度前，把策略输出的动作 (Action) 和 
    # 残差动作 (Delta-A) 结合起来，从而在仿真中对齐真实世界的物理动力学鸿沟 (Sim-to-Real Gap)。
    
    def __init__(self, config, device):
        # 【函数说明：初始化函数】
        # 作用：初始化 Delta-A 闭环环境，并额外分配一个存储闭环动作的张量。
        # 被谁调用：在强化学习算法 (如 PPO) 启动时，由 Hydra 配置系统通过实例化环境层 (env) 时调用。
        # import ipdb; ipdb.set_trace() # （被注释掉的）调试断点
        super().__init__(config, device) # 调用父类的初始化方法，搭建基本物理仿真环境

        
        # 初始化闭环动作的缓存张量，维度为 [环境数量, 动作维度]，不计算梯度
        # 这个通常用于存储名义上预训练好的策略动作（Base Policy Action）
        self.closed_loop_actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)

    def _init_buffers(self):
        # 【函数说明：初始化数据缓存区】
        # 作用：除了父类的数据缓冲外，额外初始化用于域随机化（Domain Randomization, DR）的掩码和缩放系数。
        # 这些掩码决定了在训练过程中是否应用 Delta-A，或者对 Delta-A 进行随机缩放，以增强策略鲁棒性。
        # 被谁调用：在基类的 __init__ 方法中，创建物理引擎后被自动调用。
        super()._init_buffers() # 调用父类初始化基本的状态、奖励等 buffer
        if self.config.domain_rand.cotrain_with_without_delta_a: # 如果配置了混合训练（有时用 Delta-A，有时不用）
            without_delta_a_ratio = self.config.domain_rand.without_delta_a_ratio # 获取不使用 Delta-A 的概率比例
            # 生成一个布尔掩码，决定哪些环境实例采用 Delta-A 补偿
            self.with_delta_a_or_not = torch.rand(self.num_envs, device=self.device) > without_delta_a_ratio
            self.with_delta_a_or_not = self.with_delta_a_or_not.unsqueeze(1) # 增加一个维度以匹配动作维度
        
        # 初始化 Delta-A 的缩放系数，默认全为 1
        self.delta_a_scale = torch.ones(self.num_envs, device=self.device, requires_grad=False)
        if self.config.domain_rand.rescale_delta_a: # 如果配置了对 Delta-A 进行随机缩放
            # 在指定范围内为每个自由度 (DoF) 随机生成一个缩放系数
            self.delta_a_scale = torch_rand_float(self.config.domain_rand.delta_a_scale_range[0], self.config.domain_rand.delta_a_scale_range[1], (self.num_envs, self.num_dofs), device=self.device)

    def _episodic_domain_randomization(self, env_ids):
        # 【函数说明：回合制域随机化】
        # 作用：当某些环境实例 (env_ids) 结束并重置时，重新为它们采样 Delta-A 的缩放系数。
        # 被谁调用：当环境回合结束 (done/terminated) 触发 reset 时，由基类的重置逻辑调用。
        super()._episodic_domain_randomization(env_ids) # 调用父类的域随机化（可能包含质量、摩擦力等）
        
        if self.config.domain_rand.rescale_delta_a: # 如果开启了 Delta-A 的随机缩放
            # 仅为那些重置的环境 (env_ids) 重新采样缩放系数
            self.delta_a_scale[env_ids] = torch_rand_float(self.config.domain_rand.delta_a_scale_range[0], self.config.domain_rand.delta_a_scale_range[1], (len(env_ids), self.num_dofs), device=self.device)

    def _compute_torques(self, actions):
        # 【函数说明：计算电机控制扭矩】
        # 作用：这是 ASAP 论文的核心物理对齐实现处。它将策略输出的 action 和 delta-a 结合，
        # 还实现了论文中提到的用来解等式 (2) 的两种数值优化方法：定点迭代 (Fixed-Point Iteration) 和梯度下降 (Gradient-Based)。
        # 然后将修正后的目标位置送入 PD 控制器计算出最终的力矩 (Torques)。
        # 被谁调用：在环境的物理推进步 (`_pre_physics_step`) 中，每一物理帧被调用一次。
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """

        # 将策略输出的网络动作乘以动作缩放系数 (Action Scale)，得到真实物理范围的目标值
        actions_scaled = actions * self.config.robot.control.action_scale
        control_type = self.config.robot.control.control_type # 获取控制类型（通常是 "P" 代表 PD位置控制）
        if self.config['add_extra_action']: # 如果配置中要求附加额外动作（即 Delta-A）
            # 获取当前时间步的名义动作/闭环动作
            motion_action = self.get_closed_loop_action_at_current_timestep()
        else:
            motion_action = torch.zeros_like(actions_scaled) # 否则附加动作为 0
        

        if self.config.domain_rand.cotrain_with_without_delta_a:
            # 应用混合训练掩码（部分机器人在这一步不加上 Delta-a）
            motion_action = motion_action * self.with_delta_a_or_not
        if self.config.domain_rand.rescale_delta_a:
            # 应用随机缩放系数干扰 Delta-a
            motion_action = motion_action * self.delta_a_scale

        # import ipdb; ipdb.set_trace() # （被注释的断点）
        if hasattr(self.config.domain_rand, 'action_noise'): # domain_rand.action_noise_percentage=0.05 # 检查是否有动作噪声配置
            if self.config.domain_rand.action_noise: # 如果开启了动作级噪声
                # import ipdb; ipdb.set_trace() # （被注释的断点）
                print("adding action noise") # 打印日志：正在添加动作噪声
                print("noise percentage", self.config.domain_rand.action_noise_percentage) # 打印噪声百分比
                # 生成范围在 [-1, 1] 的均匀分布噪声，并乘以配置的百分比强度
                action_noise = (torch.rand_like(actions_scaled)*2.-1.) * 1.0 * self.config.domain_rand.action_noise_percentage
                actions_scaled += action_noise # 将噪声叠加到策略动作上
        
        
        # 【重要：对应 ASAP 论文公式 (4) 的梯度优化法】
        if hasattr(self.config, 'delta_a_gradient_search'):
            if self.config.delta_a_gradient_search: # 如果开启了梯度搜索求解策略
                # 确保已经加载了残差模型 (extra_policy) 以及观测输入
                if hasattr(self, 'loaded_extra_policy') and self.current_closed_loop_actor_obs is not None:
                    iter_for_gradient_descent = 2000 # 设置梯度下降的迭代次数
                    torch.set_grad_enabled(True) # 临时开启 PyTorch 梯度计算，因为需要反向传播

                    # import ipdb; ipdb.set_trace() # （被注释的断点）

                    current_closed_loop_actor_obs = self.current_closed_loop_actor_obs # 获取当前状态的观测值

                    # 截取状态输入的前半部分（剔除动作部分），作为 Delta 模型的不变输入
                    fixed_part_obs_for_deltaA = current_closed_loop_actor_obs[:, :-23].detach().clone()

                    # 提取名义策略输出的目标动作
                    norminal_policy_action = current_closed_loop_actor_obs[:, -23:].detach().clone()
                    # 初始化新解 (y) 为名义动作，并要求计算梯度 (requires_grad_(True))
                    new_best_action = current_closed_loop_actor_obs[:, -23:].detach().clone().requires_grad_(True)

                    for i in range(iter_for_gradient_descent): # 开始多步梯度下降
                        # 拼接状态和待优化的动作，作为 Delta 模型的输入
                        new_input_for_deltaA = torch.cat([fixed_part_obs_for_deltaA, new_best_action], dim=1)
                        # 论文公式(4): L(y) = || y + \pi^\Delta(s, y) - \hat{\pi}(s) ||^2
                        loss_fn = new_best_action + self.loaded_extra_policy.eval_policy(new_input_for_deltaA) - norminal_policy_action

                        loss_fn = loss_fn.norm(dim=1) # 计算向量的 L2 范数（模长）作为损失
                        loss_fn.backward() # 反向传播求梯度
                        with torch.no_grad(): # 在不计算梯度的上下文中更新参数
                            new_best_action -= 0.0002 * new_best_action.grad # 步长为 0.0002 的梯度下降更新
                        new_best_action.grad.zero_() # 清空梯度
                        print("iter", i) # 打印当前迭代次数
                        print("loss_fn", loss_fn) # 打印当前损失
                        print("new_best_action", new_best_action) # 打印更新后的动作
                        print("-----------------------------------------")
                        

                    torch.set_grad_enabled(False) # 优化结束，关闭梯度计算以节省内存
                
                    # replace action_scaled with new_best_action *= self.config.robot.control.action_scale
                    # 将求解出的最优动作赋值给缩放后的动作，准备发送给底层电机
                    actions_scaled = new_best_action * self.config.robot.control.action_scale

        
        # 【重要：对应 ASAP 论文公式 (3) 的定点迭代法】
        if hasattr(self.config, 'delta_a_fixed_point_iteration'):
            if self.config.delta_a_fixed_point_iteration: # 如果开启了定点迭代求解
                # 确保模型和观测都已就绪
                if hasattr(self, 'loaded_extra_policy') and self.current_closed_loop_actor_obs is not None:
                    iter_for_gradient_descent = 10 # 定点迭代不需要求导，通常迭代次数很少就能收敛 (此处设为 10)



                    current_closed_loop_actor_obs = self.current_closed_loop_actor_obs

                    # 截取不变的状态部分
                    fixed_part_obs_for_deltaA = current_closed_loop_actor_obs[:, :-23].detach().clone()

                    # y_0 = \hat{\pi}(s) 提取初始的名义策略动作
                    norminal_policy_action = current_closed_loop_actor_obs[:, -23:].detach().clone()
                    new_best_action = current_closed_loop_actor_obs[:, -23:].detach().clone().requires_grad_(True)

                    for i in range(iter_for_gradient_descent):
                        new_input_for_deltaA = torch.cat([fixed_part_obs_for_deltaA, new_best_action], dim=1)
                        # 论文公式(3): y_{k+1} = \hat{\pi}(s) - \pi^\Delta(s, y_k)
                        new_best_action =  norminal_policy_action - self.loaded_extra_policy.eval_policy(new_input_for_deltaA)

                        print("iter", i) # 打印迭代次数
                        print("new_best_action", new_best_action) # 打印本轮迭代的结果
                        print("-----------------------------------------")
                        


                
                    # replace action_scaled with new_best_action *= self.config.robot.control.action_scale
                    # 迭代结束后，将收敛的解作为最终动作
                    actions_scaled = new_best_action * self.config.robot.control.action_scale



        # 【重要：对应 ASAP 论文中真实世界评估的 4-DoF 踝关节模型设定】
        if hasattr(self.config, 'anklePR'):
            if self.config.anklePR: # 如果只针对脚踝的 Pitch (俯仰) 和 Roll (横滚) 开启补偿
                # 将 4, 5, 10, 11 以外的其他 19 个自由度的附加补偿动作 (Delta) 强制置 0
                motion_action[:, [i for i in range(actions_scaled.shape[1]) if i not in [4,5,10,11]]] *= 0
                print("zeroing out non-anklePR actions") # 打印日志告知：仅保留脚踝补偿



        if control_type=="P": # 如果是基于位置的 PD 控制 (最常见的情况)
            # torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
            # 核心公式：目标位置 = 策略动作 (actions_scaled) + 残差补偿动作 (motion_action) + 默认站立姿态
            # 扭矩 = Kp * (目标位置 - 当前位置) - Kd * 当前速度
            torques = self._kp_scale * self.p_gains*(actions_scaled + motion_action + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V": # 如果是纯速度控制
            # 扭矩 = Kp * (目标速度 - 当前速度) - Kd * (加速度阻尼)
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_dt
        elif control_type=="T": # 如果策略直接输出扭矩
            torques = actions_scaled # 直接透传扭矩
        else:
            raise NameError(f"Unknown controller type: {control_type}") # 不支持的控制器报错
        
        if self.config.domain_rand.randomize_torque_rfi: # 如果开启了电机扭矩的高频随机干扰 (RFI)
            # 在最终计算出的扭矩上叠加随机扰动力矩
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        if self.config.robot.control.clip_torques: # 如果启用了扭矩物理裁剪
            # 限制扭矩不超出硬件安全范围
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        else:
            return torques # 返回无裁剪的扭矩
        
    def get_closed_loop_action_at_current_timestep(self):
        # 【函数说明：获取闭环动作】
        # 作用：获取当前时间步保存的闭环预训练策略的动作。
        # 被谁调用：在上方 `_compute_torques` 函数中用来提取名义附加动作。
        return self.actions_closed_loop.clone()

    def _get_obs_actions_closed_loop(self):
        # 【函数说明：获取用于观测的闭环动作】
        # 作用：向强化学习网络的观测输入提供闭环动作的数据。
        # 被谁调用：在基类构建 observation 字典时通过反射机制调用。
        return self.actions_closed_loop.clone()
    
    def _get_obs_actions_sim2real_policy(self):
        # 【函数说明：获取 sim2real 策略动作】
        # 作用：返回当前实际用于 sim-to-real 对齐的策略动作。
        # 被谁调用：基类观测状态组装阶段。
        return self.actions

    def _get_obs_dof_pos_ankle_pitch_roll(self):
        # 【函数说明：获取踝关节特定位置】
        # 作用：仅提取 G1 机器人左右脚踝的 Pitch 和 Roll 关节 (索引4,5,10,11) 的角度位置。
        # 被谁调用：特定于 4-DoF Delta 模型训练时的状态组装。
        # import ipdb; ipdb.set_trace()
        return self.simulator.dof_pos[:, [4,5,10,11]]
    
    def _get_obs_dof_vel_ankle_pitch_roll(self):
        # 【函数说明：获取踝关节特定速度】
        # 作用：仅提取踝关节 (索引4,5,10,11) 的角速度。
        # 被谁调用：同上。
        return self.simulator.dof_vel[:, [4,5,10,11]]

    def _get_obs_actions_closed_loop_ankle_pitch_roll(self):
        # 【函数说明：获取踝关节特定闭环动作】
        # 作用：仅提取名义策略输出中对应踝关节的控制指令。
        # 被谁调用：同上。
        return self.actions_closed_loop.clone()[:, [4,5,10,11]]
    
    def _get_obs_actions_sim2real_policy_ankle_pitch_roll(self):
        # 【函数说明：获取踝关节的 sim2real 动作】
        # 作用：提取最终决定的踝关节对齐补偿动作。
        # 被谁调用：同上。
        return self.actions[:, [4,5,10,11]]
    
    def step(self, actor_state):
        # 【函数说明：物理环境执行单步交互】
        # 作用：接收网络推理出的状态字典 (actor_state)，提取出动作并加以裁剪，
        # 然后推进物理引擎的模拟仿真，最后返回新的观测和奖励。
        # 被谁调用：在 PPO 等强化学习算法的主循环或评测循环中每帧被调用。
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actor_state["actions"] # 提取 Delta 策略输出的动作
        self.actions_closed_loop = actor_state["actions_closed_loop"] # 提取闭环名义策略的动作
        clip_action_limit = self.config.robot.control.action_clip_value # 获取软件层面的动作幅度限制阈值
        # 裁剪动作，防止由于网络崩溃输出超大值导致物理引擎爆炸
        self.actions_closed_loop = torch.clip(self.actions_closed_loop, -clip_action_limit, clip_action_limit).to(self.device)


        # --------------------------- UNCOMMENT THIS FOR CURRENT_CLOSED_LOOP_ACTOR_OBS ---------------------------
        # 【废弃的注释区：之前用于外部手动传入或挂载闭环策略观测数据的逻辑】
        # if hasattr(self, 'current_closed_loop_actor_obs') and 'current_closed_loop_actor_obs' in actor_state.keys():
        #     self.current_closed_loop_actor_obs = actor_state['current_closed_loop_actor_obs']
        #     print("current_closed_loop_actor_obs found in self, set it to actor_state['current_closed_loop_actor_obs']")
        # else:
        #     if 'current_closed_loop_actor_obs' in actor_state.keys():
        #         setattr(self, 'current_closed_loop_actor_obs', actor_state['current_closed_loop_actor_obs'])
        #         print("current_closed_loop_actor_obs not found in self, set it to actor_state['current_closed_loop_actor_obs']")
        #     else:
        #         setattr(self, 'current_closed_loop_actor_obs', None)
        #         print("current_closed_loop_actor_obs not found in actor_state")
        # --------------------------------------------------------------------------------------------------------


        
        # import ipdb; ipdb.set_trace() # （被注释的断点）
        # print('actions in env.step: ', actions) # （被注释的打印：调试输出动作分布）
        # print('closed_loop_actions in env.step: ', actor_state['actions_closed_loop'])
        
        # 按顺序执行物理引擎的三部曲：
        self._pre_physics_step(actions) # 1. 物理计算前置处理 (将动作转为扭矩送到仿真器，即调用上面的 _compute_torques)
        self._physics_step()            # 2. 仿真器前向推演一步时间戳
        self._post_physics_step()       # 3. 物理计算后处理 (计算奖励、检查跌倒、更新内部观测变量)


        # 返回给强化学习算法的四个核心数据：下一步观测、获得的奖励、是否重置信号、额外信息字典
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def step_for_reset_all(self, actor_state):
        # 【函数说明：物理环境执行单步交互】
        # 作用：接收网络推理出的状态字典 (actor_state)，提取出动作并加以裁剪，
        # 然后推进物理引擎的模拟仿真，最后返回新的观测和奖励。
        # 被谁调用：在 PPO 等强化学习算法的主循环或评测循环中每帧被调用。
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actor_state["actions"] # 提取 Delta 策略输出的动作
        self.actions_closed_loop = actor_state["actions_closed_loop"] # 提取闭环名义策略的动作
        clip_action_limit = self.config.robot.control.action_clip_value # 获取软件层面的动作幅度限制阈值
        # 裁剪动作，防止由于网络崩溃输出超大值导致物理引擎爆炸
        self.actions_closed_loop = torch.clip(self.actions_closed_loop, -clip_action_limit, clip_action_limit).to(self.device)


        # --------------------------- UNCOMMENT THIS FOR CURRENT_CLOSED_LOOP_ACTOR_OBS ---------------------------
        # 【废弃的注释区：之前用于外部手动传入或挂载闭环策略观测数据的逻辑】
        # if hasattr(self, 'current_closed_loop_actor_obs') and 'current_closed_loop_actor_obs' in actor_state.keys():
        #     self.current_closed_loop_actor_obs = actor_state['current_closed_loop_actor_obs']
        #     print("current_closed_loop_actor_obs found in self, set it to actor_state['current_closed_loop_actor_obs']")
        # else:
        #     if 'current_closed_loop_actor_obs' in actor_state.keys():
        #         setattr(self, 'current_closed_loop_actor_obs', actor_state['current_closed_loop_actor_obs'])
        #         print("current_closed_loop_actor_obs not found in self, set it to actor_state['current_closed_loop_actor_obs']")
        #     else:
        #         setattr(self, 'current_closed_loop_actor_obs', None)
        #         print("current_closed_loop_actor_obs not found in actor_state")
        # --------------------------------------------------------------------------------------------------------


        
        # import ipdb; ipdb.set_trace() # （被注释的断点）
        # print('actions in env.step: ', actions) # （被注释的打印：调试输出动作分布）
        # print('closed_loop_actions in env.step: ', actor_state['actions_closed_loop'])
        
        # 按顺序执行物理引擎的三部曲：
        self._pre_physics_step(actions) # 1. 物理计算前置处理 (将动作转为扭矩送到仿真器，即调用上面的 _compute_torques)
        #self._physics_step()            # 2. 仿真器前向推演一步时间戳
        self._post_physics_step_for_reset_all()       # 3. 物理计算后处理 (计算奖励、检查跌倒、更新内部观测变量)


        # 返回给强化学习算法的四个核心数据：下一步观测、获得的奖励、是否重置信号、额外信息字典
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras


    def reset_all(self):
        # 【函数说明：硬重置所有环境】
        # 作用：强制把所有并行的机器人都拉回到初始设定的姿态和状态，清空所有动作历史。
        # 被谁调用：通常在训练最开始的 Epoch 或者整个算法手动请求重置时调用。
        """ Reset all robots"""
        # 生成全部环境的索引列表，并通知基类清空它们的记录
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        # 强制底层仿真器将所有机器人的根节点（基座）状态拉回初始配置
        self.simulator.set_actor_root_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.all_root_states)
        # 强制底层仿真器将所有机器人的关节自由度状态拉回初始配置
        self.simulator.set_dof_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.dof_state)
        # self._refresh_env_idx_tensors(torch.arange(self.num_envs, device=self.device))
        
        # 把两种控制动作全部归零
        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actions_closed_loop = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False) 
        
        # 封装进字典准备进行空转 step
        actor_state = {}
        actor_state["actions"] = actions
        actor_state["actions_closed_loop"] = actions_closed_loop
        # 用全 0 动作前向推演一帧，以刷新拿到干净的、重置后的第一帧观测数据 obs_dict
        obs_dict, _, _, _ = self.step(actor_state)
        return obs_dict # 返回初始观测给算法