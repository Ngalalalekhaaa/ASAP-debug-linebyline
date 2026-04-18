import torch # PyTorch：深度学习和张量计算的核心库
import numpy as np # 数值计算库
from pathlib import Path # 路径操作库
import os # 系统操作库
from isaac_utils.rotations import ( # 底层运动学四元数工具
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_rotate_inverse
)
# from isaacgym import gymtorch, gymapi, gymutil
from humanoidverse.envs.env_utils.visualization import Point # 可视化工具

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree # 骨骼树解析

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot # 动作库类

from termcolor import colored # 终端彩色打印
from loguru import logger # 日志记录器

from scipy.spatial.transform import Rotation as sRot # 高精度旋转变换库
import joblib # pkl 文件读取工具

from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking # 导入基础的动作追踪（模仿学习）环境类

# ============================================================================
# 核心类：DeltaA_OpenLoop
# 继承自 LeggedRobotMotionTracking (基础模仿学习环境)
# 【作用】：专门用于训练 ASAP 论文中提到的“开环残差控制（Delta-A）”策略。
# ============================================================================
class DeltaA_OpenLoop(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        """
        【函数作用】：初始化 Delta-A 环境，并设置用于监控和分析网络输出（微调幅度）的统计变量（热力图）。
        【被谁调用】：由强化学习框架在环境实例化时调用。
        """
        # import ipdb; ipdb.set_trace()
        super().__init__(config, device) # 初始化基础动作追踪环境的物理引擎、动作库等
        
        # 初始化两个张量，用于记录和统计网络输出的 delta-a 动作的绝对值（生成热力图分析）
        self.delta_action_dof_heatmaps = torch.zeros((self.simulator.num_envs, self.num_dofs)).to(device)
        # 记录 delta-a 占原始动作的百分比，评估网络是不是在“用力过猛”
        self.detla_action_percentage_heatmaps = torch.zeros((self.simulator.num_envs, self.num_dofs)).to(device)
        self.delta_action_cnt = 1 # 统计步数计数器，用于计算移动平均

    def _compute_torques(self, actions):
        """ 
        【函数作用】：将网络输出的 action 转化为底层 PD 控制器需要的力矩 (Torques)。
        【ASAP 论文背景】：这是 Delta-A 策略的灵魂所在！在这个函数里，网络的输出 `actions` 变成了 $\Delta a$。
        最终发给 PD 控制器的目标位置是： 原始动画预设动作 (motion_action) + 网络微调动作 (actions_scaled)！
        
        Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions (网络输出的原始无量纲的预测值)

        Returns:
            [torch.Tensor]: Torques sent to the simulation (发送给物理引擎的关节力矩)
        【被谁调用】：物理引擎每推演一步之前，被基类的 `step()` 函数底层调用以获取控制信号。
        """
        
        # 如果配置文件里要求强制把 delta-a 设为 0 (纯开环测试，不带网络微调)
        if hasattr(self.config, 'zero_delta_a'):
            if self.config.zero_delta_a:                
                actions *= 0. # 乘以 0 抹除网络的输出
                print("actions", actions)
                pass
                
        # 将网络输出的无量纲数值（通常在 -1 到 1 之间）乘上比例因子，转化为物理世界的弧度微调值
        actions_scaled = actions * self.config.robot.control.action_scale
        control_type = self.config.robot.control.control_type # 获取控制类型（通常是 P: 位置控制）
        
        # 如果配置允许叠加基础动作 (也就是标准的 Delta-A 模式)
        if self.config['add_extra_action']:
            # 去动作库里查出当前这一帧，原本该下发什么动作 (Open Loop Action)
            motion_action = self.get_open_loop_action_at_current_timestep()
            # motion_action *= 0
            # print("motion_action", motion_action)
            # print("self.get_open_loop_action_at_current_timestep()", self.get_open_loop_action_at_current_timestep())
            # 把基础动作也乘上比例因子，转为物理弧度
            motion_action *= self.config.robot.control.action_scale
        else:
            # 如果不叠加，那就退化成了普通的强化学习，让网络从零学起全权接管
            print("zeroing out motion_action") 
            motion_action = torch.zeros_like(actions_scaled)

        # 脚踝专属实验控制逻辑 (Ankle Pitch/Roll 隔离测试)
        if hasattr(self.config, 'anklePR'):
            if self.config.anklePR:
                # 除了 4, 5, 10, 11 (这四个通常是双腿踝关节的俯仰和侧倾电机)，其余部位的网络微调全部强行归零。
                # 意思是：上半身和腿只听话照做，只有脚踝允许通过网络动态微调来维持平衡。
                actions_scaled[:, [i for i in range(actions_scaled.shape[1]) if i not in [4,5,10,11]]] *= 0
                
                # print("zeroing out non-anklePR actions")

        
        # add action_to_delta_a_heatmap
        # 在线计算移动平均，更新 delta-a 热力图，方便分析网络对哪个关节微调得最狠
        self.delta_action_dof_heatmaps = self.delta_action_dof_heatmaps * self.delta_action_cnt / (1+self.delta_action_cnt) + 1/(1+self.delta_action_cnt) * torch.abs(actions_scaled)
        self.detla_action_percentage_heatmaps = self.detla_action_percentage_heatmaps * self.delta_action_cnt / (1+self.delta_action_cnt) + 1/(1+self.delta_action_cnt) * torch.abs(actions_scaled / motion_action)
        self.delta_action_cnt += 1


        # print("self.delta_action_dof_heatmaps", self.delta_action_dof_heatmaps)
        # print("self.detla_action_percentage_heatmaps", self.detla_action_percentage_heatmaps)
        # import ipdb; ipdb.set_trace()
        # print("motion_action", motion_action)
        # print("motion_action", motion_action)
        # print("motion_action", motion_action.shape)
        # print('actions', actions)
        # print('motion_action', motion_action)
        # import ipdb; ipdb.set_trace(

        # handcrafted perfect delta_a for 65Kp (硬编码的测试代码注释)
        # perfect_delta_a_for_65Kp = (motion_action + self.default_dof_pos - self.simulator.dof_pos) * -0.35
        # actions_scaled = perfect_delta_a_for_65Kp

        # =====================================================================
        # 核心公式实现： PD 控制器的目标位置
        # =====================================================================
        if control_type=="P":
            # 基础 RL (被注释): torques = Kp * (网络动作 + 默认静止姿态 - 实际当前姿态) - Kd * 实际速度
            # torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
            
            # ★ Delta-A 公式: torques = Kp * (微调动作(Δa) + 开环预设动作(a_ref) + 默认静止姿态 - 实际当前姿态) - Kd * 实际速度
            torques = self._kp_scale * self.p_gains*(actions_scaled + motion_action + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V":
            # 纯速度控制
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_dt
        elif control_type=="T":
            # 纯力矩控制
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        # 领域随机化 (Domain Randomization)：给算出来的力矩加一点高斯噪声，防止网络过拟合到纯净的物理引擎里
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        # 为了保护电机，截断过高的力矩指令
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        else:
            return torques
        
    # 以下三个函数是用来计算“正则化惩罚项”的。
    # 【ASAP 论文背景】：既然让网络输出残差(Delta-A)，我们就希望这个微调越小越好！
    # 如果微调太大，机器人就不像是在“跟练”了，而是在跳一种面目全非的舞。
    def _reward_minimal_action_norm(self):
        """【函数作用】：对动作张量(L2范数)进行指数衰减惩罚。动作幅度越大，该项得分越低（越接近0）。"""
        # exp(-norm(actions))
        return torch.exp(-torch.norm(self.actions, dim=-1))


    def _reward_normalized_penalty_minimal_action_norm(self):
        """【函数作用】：归一化版本的惩罚。网络完全不动时得分最高(0分)，动作越大分数越负。"""
        # exp(-norm(actions))
        return torch.exp(-torch.norm(self.actions, dim=-1)) - 1

    def _reward_penalty_minimal_action_norm(self):
        """
        【函数作用】：基于指数增长的非线性强惩罚。
        （作者注：这里用正指数，意味着动作越大，返回的正向数值越大，但因为在配置文件里会给这项配一个负的权重系数(-0.1)，所以这最终依然是一个惩罚项！）
        【被谁调用】：由强化学习 PPO 算法中收集各维度 Reward 字典的调度器自动调用。
        """
        # print("self.actions", self.actions)
        # print("torch.norm(self.actions, dim=-1)", torch.norm(self.actions, dim=-1))
        # # import ipdb; ipdb.set_trace()
        # # print max abs values of rewards
        # print("max abs values of rewards", torch.max(torch.abs(torch.exp(torch.norm(self.actions, dim=-1))-1)))
        # # throw out the reward if the action norm is too large, and tell me which action caused it
        # if torch.max(torch.abs(torch.exp(torch.norm(self.actions, dim=-1))-1)) > 100000:
        #     print("action causing the reward to be thrown out", self.actions[torch.argmax(torch.abs(torch.exp(torch.norm(self.actions, dim=-1))-1))])
        #     import ipdb; ipdb.set_trace()
        #     return torch.exp(torch.norm(self.actions, dim=-1))-1
        #     # return torch.tensor(0.0)
        # else:
        #     return torch.exp(torch.norm(self.actions, dim=-1))-1
        
        # clip the reward from -1000 to 1000
        # 返回指数增长的动作范数惩罚项
        return torch.exp(torch.norm(self.actions, dim=-1))-1
        # return torch.clip(torch.exp(torch.norm(self.actions, dim=-1))-1, -1000, 1000)
        
    def get_open_loop_action_at_current_timestep(self):
        """
        【函数作用】：获取当前时间戳下的标准开环动作（也就是我们在 pkl 文件里存入的原始 Action）。
        【被谁调用】：在 `_compute_torques` 中提供预设轨迹基准，以及给下面的 Observation(观测值) 提取函数供数。
        """
        # 预测下一帧的时间 (当前帧经过时长 + dt + 各自片段的随机起始时长)
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        # 从底层的动作库(MotionLib)中取出指定环境在指定时间该下发的电机动作。保存的电机动作是否经过乘以比例因子转化为弧度了？
        motion_action = self._motion_lib.get_motion_actions(self.motion_ids, motion_times)

        return motion_action
    
    # NOTE: this is the perfect delta_a for 0.65Kp scenario
    def _get_perfect_delta_a(self):
        """
        【函数作用】：提供一个硬编码计算出来的“完美残差值”。通常用于在 Tensorboard / WebLogger 中画对比图观察网络学得好不好，实际不参与环境推演。
        """
        return (self.get_open_loop_action_at_current_timestep() * self.config.robot.control.action_scale + self.default_dof_pos - self.simulator.dof_pos) * -0.35

    # 以下四个函数均以 `_get_obs_` 开头，这是在向神经网络（Actor & Critic）提供“观测值(Observation/State)”的组件。
    def _get_obs_actions_open_loop(self):
        """
        【函数作用】：将当前的“开环动作指令”作为观测值喂给神经网络。
        【RL 背景】：告诉网络“下一步原本该干嘛”，有助于网络学习在哪种动作下容易摔倒，从而提前给出补偿。
        """
        return self.get_open_loop_action_at_current_timestep()

    def _get_obs_dof_pos_ankle_pitch_roll(self):
        """
        【函数作用】：只提取 4 个踝关节（俯仰和侧倾）的实际位置作为观测值。通常用于消融实验或分层控制。
        """
        # import ipdb; ipdb.set_trace()
        return self.simulator.dof_pos[:, [4,5,10,11]]
    
    def _get_obs_dof_vel_ankle_pitch_roll(self):
        """【函数作用】：只提取 4 个踝关节的实际转速作为观测值。"""
        return self.simulator.dof_vel[:, [4,5,10,11]]

    def _get_obs_actions_ankle_pitch_roll(self):
        """【函数作用】：提取网络在上一帧下发给 4 个踝关节的动作指令历史。"""
        return self.actions[:, [4,5,10,11]]
    
    def _get_obs_actions_open_loop_ankle_pitch_roll(self):
        """【函数作用】：提取动画库中记录的 4 个踝关节的标准开环动作。"""
        return self.get_open_loop_action_at_current_timestep()[:, [4,5,10,11]]