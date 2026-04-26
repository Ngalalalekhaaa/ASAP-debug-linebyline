from time import time
import numpy as np
import matplotlib.pyplot as plt
import os

# 导入自定义的 PyTorch 工具函数（比如张量转换、随机数生成等）
from humanoidverse.utils.torch_utils import *
# 以下是原本用于 Isaac Gym 的库，目前被注释掉了，可能当前框架已经做了一层封装或者准备迁移到 Isaac Sim/IsaacLab
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

# 导入一些环境辅助工具，比如类转字典、四元数/欧拉角转换工具
from humanoidverse.envs.env_utils.general import class_to_dict
from isaac_utils.rotations import get_euler_xyz_in_tensor
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.base_task.base_task import BaseTask

# 导入历史观测处理器（用于堆叠 T-1, T-2 步的历史数据，这对解决机器人的部分可观测马尔可夫决策过程 POMDP 至关重要）
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from termcolor import colored
from humanoidverse.utils.helpers import parse_observation
from humanoidverse.envs.env_utils.visualization import Point

from loguru import logger
import copy

# 腿足机器人强化学习任务的基础类，继承自管理物理引擎底层的 BaseTask
class LeggedRobotBase(BaseTask):
    """
    【函数作用】：环境类的构造函数，初始化强化学习所需的各种特化模块（域随机化、奖励、历史记录器等）。
    【RL知识补充】：这是构建强化学习环境的第一步，它将读取你写好的 yaml 配置文件并搭建训练场。
    【被谁调用】：由外部的训练主脚本（如 train.py 或者 PPO 算法的 Runner 类）在程序启动时调用一次。
    """
    def __init__(self, config, device):
        self.init_done = False # 初始化状态标记
        super().__init__(config, device) # 调用父类初始化物理仿真器、创建成百上千个并行机器人环境
        self._domain_rand_config() # 解析 Sim2Real 域随机化（Domain Randomization）相关的配置文件
        self._prepare_reward_function() # 动态绑定并在内存中准备好所有的奖励函数（Reward Functions）
        # 初始化历史记录器，分配显存用于记录过去的观测序列（比如过去10步的关节位置）
        self.history_handler = HistoryHandler(self.num_envs, config.obs.obs_auxiliary, config.obs.obs_dims, device)
        self.is_evaluating = False # 标记当前是否处于部署/测试评估状态
        self.init_done = True # 初始化完成标志

    """
    【函数作用】：在 GPU 显存中预先分配所有用于存储机器人状态和计算过程的 PyTorch 张量（Buffers）。
    【RL知识补充】：为了让训练极速运行，必须避免在每一步循环中重新分配内存。所有的状态数据（Observation、Reward、Done）都在这里提前挖好“坑”。
    【被谁调用】：被父类 BaseTask 的 __init__ 函数调用。
    """
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers() # 调用父类初始化基础的观测、奖励、重置标志张量

        self.base_quat = self.simulator.base_quat # 机器人的基座四元数（姿态）
        self.rpy = get_euler_xyz_in_tensor(self.base_quat) # 将四元数转为欧拉角 (Roll, Pitch, Yaw)

        # initialize some data used later on
        self._init_counters() # 初始化各类步数计数器
        self.extras = {} # 用于存放发送给强化学习算法的额外日志信息（Info dict）
        # 构建一个指向 Z 轴负方向的全局重力向量，并复制 num_envs 份
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # 构建一个指向 X 轴正方向的全局前向向量
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        
        # 预分配存放各个关节最终输出力矩的张量
        self.torques = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 预分配 PD 控制器的 P 增益（刚度）和 D 增益（阻尼）张量
        self.p_gains = torch.zeros(self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 存放神经网络输出的原始 Action
        self.actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 存放经过模拟延迟处理后真正发给电机的 Action
        self.actions_after_delay = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 记录上一步的 Action（常用于计算 Action 的抖动惩罚）
        self.last_actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros_like(self.simulator.dof_pos) # 上一步的关节位置
        self.last_dof_vel = torch.zeros_like(self.simulator.dof_vel) # 上一步的关节速度
        self.last_root_vel = torch.zeros_like(self.simulator.robot_root_states[:, 7:13]) # 上一步的基座线速度和角速度
        
        # 记录每只脚当前的腾空时间
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        # 记录上一帧哪些脚踩在地上了
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        # 将基座在世界坐标系下的速度转换到机器人自己的局部坐标系下
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        # 重力向量在机器人局部坐标系下的投影，用来判断机器人身体倾斜了多少
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # joint positions offsets and PD gains
        # 解析配置文件，提取并设置每个关节的初始角度（零点位置）、P 增益和 D 增益
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.config.robot.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            # 遍历 yaml 配置中的刚度字典进行匹配
            for dof_name in self.config.robot.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.config.robot.control.stiffness[dof_name]
                    self.d_gains[i] = self.config.robot.control.damping[dof_name]
                    found = True
                    logger.debug(f"PD gain of joint {name} were defined, setting them to {self.p_gains[i]} and {self.d_gains[i]}")
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.config.robot.control.control_type in ["P", "V"]: # 如果使用位置控制或速度控制却没有找到 PD 参数，就报错
                    logger.warning(f"PD gain of joint {name} were not defined, setting them to zero")
                    raise ValueError(f"PD gain of joint {name} were not defined. Should be defined in the yaml file.")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        # 初始化与域随机化（Sim2Real）相关的张量
        self._init_domain_rand_buffers()

        # for reward penalty curriculum (用于惩罚项课程学习的参数)
        # 平均每个回合的长度。若机器人在环境里站得越久，说明能力越强，就可以增加训练难度
        self.average_episode_length = 0. # num_compute_average_epl last termination episode length
        self.last_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.num_compute_average_epl = self.config.rewards.num_compute_average_epl # 算平均值所用的滑动窗口大小

        # 标记哪些环境在当前这一步被破坏（比如摔倒）并需要完全重置物理状态
        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        # 观测噪声课程学习：控制加在网络输入上的随机噪声随训练进度的变大或变小
        self.add_noise_currculum = self.config.obs.add_noise_currculum
        self.current_noise_curriculum_value = self.config.obs.noise_initial_value

    """
    【函数作用】：根据配置设定每个机器人多久被随机外力“推”一次。
    【被谁调用】：__init__
    """
    def _domain_rand_config(self):
        if self.config.domain_rand.push_robots:
            # 在一个区间内（比如每 5-15 秒）随机抽取下次被推的时间间隔
            self.push_interval_s = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (self.num_envs,), device=self.device)

    """
    【函数作用】：初始化各类步数计数器。
    【被谁调用】：_init_buffers
    """
    def _init_counters(self):
        self.common_step_counter = 0 # 仿真总步数
        self.push_robot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False) # 距离上次被推经过了多少步
        self.push_robot_plot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False) # 渲染界面画力向量用的计数器
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False) # 距离上次下发新指令经过了多少步

    """
    【函数作用】：在每次物理引擎前进一步后，更新各类步数计数器。
    【被谁调用】：_post_physics_step
    """
    def _update_counters_each_step(self):
        self.common_step_counter  +=1
        self.push_robot_counter[:] += 1
        self.push_robot_plot_counter[:] += 1
        self.command_counter[:] += 1

    """
    【函数作用】：分配给各种 Sim2Real 随机化技术所需的底层显存。
    【被谁调用】：_init_buffers
    """
    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        # 动作延迟队列：模拟实际主板算完动作发给底层电机的通信延迟
        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(self.num_envs, self.config.domain_rand.ctrl_delay_step_range[1]+1, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay_idx = torch.randint(self.config.domain_rand.ctrl_delay_step_range[0], 
                                                self.config.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)

        # self._link_mass_scale = torch.ones(self.num_envs, len(self.config.robot.randomize_link_body_names), dtype=torch.float, device=self.device, requires_grad=False)
        # 电机属性随机缩放系数
        self._kp_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._kd_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._rfi_lim_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 用于记录施加给机器人的瞬间线速度脉冲（推力）
        self.push_robot_vel_buf = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.record_push_robot_vel_buf = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        # 过滤后的历史接触点记录（解决仿真器中刚体在地面抖动造成的连续快速触地/离地误判问题）
        self.last_contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        # 记录某只脚在这个腾空周期内达到的最大高度
        self.feet_air_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

    """
    【函数作用】：读取配置，利用反射机制把所有的奖励计算逻辑挂载好，并初始化它们的课程学习进度。
    【RL知识补充】：这是 Reward Shaping 的关键。为了在不同控制频率下奖励总和大致不变，通常要把原始系数乘上时间步长 (dt)。
    【被谁调用】：__init__
    """
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        logger.info(colored(f"{self.config.rewards.set_reward} set reward on {self.config.rewards.set_reward_date}", "green"))
        
        self.reward_scales = self.config.rewards.reward_scales
        # remove zero scales + multiply non-zero ones by dt
        # 移除权重为0的项，并且对于非零权重，乘以动作间隔时间(dt)
        for key in list(self.reward_scales.keys()):
            logger.info(f"Scale: {key} = {self.reward_scales[key]}")
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt

        # 如果开启了惩罚项课程（初期少惩罚，后期机器人稳了就加大惩罚力度）
        self.use_reward_penalty_curriculum = self.config.rewards.reward_penalty_curriculum
        if self.use_reward_penalty_curriculum:
            self.reward_penalty_scale = self.config.rewards.reward_initial_penalty_scale

        logger.info(colored(f"Use Reward Penalty: {self.use_reward_penalty_curriculum}", "green"))
        if self.use_reward_penalty_curriculum:
            logger.info(f"Penalty Reward Names: {self.config.rewards.reward_penalty_reward_names}")
            logger.info(f"Penalty Reward Initial Scale: {self.config.rewards.reward_initial_penalty_scale}")
        
        # 各种软限制（比如限制关节只能在非常安全的范围内运动）的课程设置
        self.use_reward_limits_dof_pos_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum
        self.use_reward_limits_dof_vel_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum
        self.use_reward_limits_torque_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum

        # 初始化这些软限制当前的阈值（初始值可能较宽容，训练后期逐渐收紧）
        if self.use_reward_limits_dof_pos_curriculum:
            logger.info(f"Use Reward Limits DOF Curriculum: {self.use_reward_limits_dof_pos_curriculum}")
            logger.info(f"Reward Limits DOF Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit}")
            logger.info(f"Reward Limits DOF Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit}")
            logger.info(f"Reward Limits DOF Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit}")
            self.soft_dof_pos_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit
        
        if self.use_reward_limits_dof_vel_curriculum:
            logger.info(f"Use Reward Limits DOF Vel Curriculum: {self.use_reward_limits_dof_vel_curriculum}")
            logger.info(f"Reward Limits DOF Vel Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit}")
            logger.info(f"Reward Limits DOF Vel Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit}")
            logger.info(f"Reward Limits DOF Vel Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit}")
            self.soft_dof_vel_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit
        
        if self.use_reward_limits_torque_curriculum:
            logger.info(f"Use Reward Limits Torque Curriculum: {self.use_reward_limits_torque_curriculum}")
            logger.info(f"Reward Limits Torque Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit}")
            logger.info(f"Reward Limits Torque Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit}")
            logger.info(f"Reward Limits Torque Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit}")
            self.soft_torque_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit

        # prepare list of functions
        # 动态绑定名字如 _reward_termination 对应的具体函数对象，方便后续快速遍历调用
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination": # 终止惩罚很特殊，单独处理，不在循环里算
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name)) # 利用 python getattr 拿到函数指针
            # reward episode sums：给每个奖励项预分配 Tensor 用来记录总得分
            self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                for name in self.reward_scales.keys()}

    """
    【函数作用】：标记进入部署/测试环境。
    【被谁调用】：测试脚本（如 play.py）。
    """
    def set_is_evaluating(self):
        logger.info("Setting Env is evaluating")
        self.is_evaluating = True
    
    """
    【函数作用】：核心强化学习交互 API！执行动作并返回新状态。
    【RL知识补充】：符合 Gym 范式的标准接口。输入 Action，返回 (Obs, Reward, Done, Info)。
    【被谁调用】：被外部的 RL 算法主循环（比如 PPO 采样时）不断调用。
    """
    def step(self, actor_state):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actor_state["actions"]
        # actions *= 0.0 # 调试用代码：强制动作归零以观察机器人自由落体状态
        
        # 1. 发送给物理引擎前的预处理
        self._pre_physics_step(actions)
        # 2. 调用物理引擎步进计算
        self._physics_step()
        # 3. 物理计算后的收尾，包括计算奖励和准备下一帧的观测
        self._post_physics_step()

        # if self.episode_length_buf[0] == 1:
        #     import ipdb; ipdb.set_trace() # 调试用的断点

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def step_for_reset_all(self, actor_state):#没有学会运行4个时间状态下的动作（第一次强制摔倒），所以在reset_all调用时，不使用环境步进，让机器人直接从初始位置开始通过决策输出动作
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actor_state["actions"]
        # actions *= 0.0 # 调试用代码：强制动作归零以观察机器人自由落体状态
        
        # 1. 发送给物理引擎前的预处理
        # self._pre_physics_step(actions)
        # 2. 调用物理引擎步进计算
        #self._physics_step()
        # 3. 物理计算后的收尾，包括计算奖励和准备下一帧的观测
        #self._post_physics_step()
        self._post_physics_step_for_reset_all()


        # if self.episode_length_buf[0] == 1:
        #     import ipdb; ipdb.set_trace() # 调试用的断点

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    """
    【函数作用】：将网络输出的动作进行安全截断，并在需要时存入延迟队列模拟真实通信延迟。
    【被谁调用】：step
    """
    def _pre_physics_step(self, actions):
        clip_action_limit = self.config.robot.control.action_clip_value
        # 裁切网络动作（比如限定在 [-1.0, 1.0] 范围内）防爆
        self.actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

        # 记录有多少动作顶到了极限值（这可以监控网络的学习是否变得病态）
        self.log_dict["action_clip_frac"] = (
                self.actions.abs() == clip_action_limit
            ).sum() / self.actions.numel()#torch.numel()返回张量的总元素个数（2*3*4=24）

        # 处理延迟控制（极其关键的 Sim2Real 技巧）
        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone() # 队列元素整体向后移动一步
            self.action_queue[:, 0] = self.actions.clone() # 新生成的动作放进队头
            # 根据每个环境预先随机好的延迟索引，从队列中抽取“过去发出的”那个动作来作为这步生效的动作
            self.actions_after_delay = self.action_queue[torch.arange(self.num_envs), self.action_delay_idx].clone()
        else:
            self.actions_after_delay = self.actions.clone()

    """
    【函数作用】：驱动底层物理引擎。
    【RL知识补充】：为了稳定性，物理引擎步长（如 0.005s）通常比控制策略的步长（如 0.02s）小。这就是 Control Decimation，这里通过 for 循环来实现控制保持。
    【被谁调用】：step
    """
    def _physics_step(self):
        self.render() # 刷新图形渲染。genesis中什么也不会执行
        for _ in range(self.config.simulator.config.sim.control_decimation):
            self._apply_force_in_physics_step() # 施加真实力矩
            self.simulator.simulate_at_each_physics_step() # 物理世界向前演进一小步

    """
    【函数作用】：将位置指令（角度）通过 PD 控制器公式计算成扭矩（力），发送给底层 API。
    【被谁调用】：_physics_step
    """
    def _apply_force_in_physics_step(self):
        self.torques = self._compute_torques(self.actions_after_delay).view(self.torques.shape)
        self.simulator.apply_torques_at_dof(self.torques)

    """
    【函数作用】：这一大步结束后，统计分数、看看谁摔倒了、给摔倒的复活。
    【被谁调用】：step
    """
    def _post_physics_step(self):
        self._refresh_sim_tensors() # 从仿真器底层拉取最新的所有物理状态张量
        self.episode_length_buf += 1 # 存活步数加一。在一步后加一。
        # update counters
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback() # 在算观测前做一些坐标系转换的准备工作
        self._update_tasks_callback() # 处理特定的环境任务（比如此时是否需要推一下机器人）
        
        # compute observations, rewards, resets, ...
        self._check_termination() # 检查有没有机器人摔倒/超时了 (计算 Done)
        self._compute_reward() # 结算这步的奖励 (计算 Reward)
        
        # check terminations
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten() # 捞出所有被判死刑的环境 ID
        self.reset_envs_idx(env_ids) # 真正去重置它们的参数位置

        # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            # 对于重置的机器，把最新的（被拉回原点的）坐标强制塞进底层物理引擎
            self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
            # self.need_to_refresh_envs[refresh_env_ids] = False
            
        self._refresh_sim_tensors() # 从仿真器底层拉取最新的所有物理状态张量
        self._pre_compute_observations_callback() # 在算观测前做一些坐标系转换的准备工作

        self._compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        # 获取最新的观测向量 (计算 Obs)
        
        self._post_compute_observations_callback()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():#obs_buf_dict是按照字母顺序排列好的观测值字典，2个键是actor_obs和critic_obs,2个值是对应的观测张量列表
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs) # 同样限制观测值的极值防爆

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key]) # 将当前状态hist_obs_dict(加噪且放缩过的这一期的观测值)推入历史序列，供未来使用

        self.extras["to_log"] = self.log_dict
        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
            if self.debug_viz:
                self._draw_debug_vis()
                
        # if len(refresh_env_ids) > 0:
        #     for obs_key, obs_val in self.obs_buf_dict.items():#obs_buf_dict是按照字母顺序排列好的观测值字典，2个键是actor_obs和critic_obs,2个值是对应的观测张量列表
        #         # self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs) # 同样限制观测值的极值防爆
        #         print(obs_key, obs_val)

            # for key in self.history_handler.history.keys():
                # self.history_handler.add(key, self.hist_obs_dict[key]) # 将当前状态hist_obs_dict(加噪且放缩过的这一期的观测值)推入历史序列，供未来使用
                # print(key, self.hist_obs_dict[key])
        self.need_to_refresh_envs[refresh_env_ids] = False

    """
    【函数作用】：这一大步结束后，统计分数、看看谁摔倒了、给摔倒的复活。因为reset_all时不想让机器人先被物理引擎演进一步再算奖励和观测，所以唯一注释了self.episode_length_buf += 1这一行。
    【被谁调用】：step_for_reset_all（这是在 reset_all 时调用的版本，区别在于它不会调用物理引擎步进，所以机器人会直接从初始位置开始通过决策输出动作，而不是先让物理引擎演进一步再算奖励和观测）
    """
    def _post_physics_step_for_reset_all(self):
        # self._refresh_sim_tensors() # 从仿真器底层拉取最新的所有物理状态张量
        #self.episode_length_buf += 1 # 存活步数加一。在一步后加一。
        # update counters
        # self._update_counters_each_step()
        # self.last_episode_length_buf = self.episode_length_buf.clone()

        
        # print(f"_pre_compute_observations_callback前的self.simulator.all_root_states: {self.simulator.all_root_states}")
        # self._pre_compute_observations_callback() # 在算观测前做一些坐标系转换的准备工作
        # print(f"_pre_compute_observations_callback后的self.simulator.all_root_states: {self.simulator.all_root_states}")
        # self._update_tasks_callback() # 处理特定的环境任务（比如此时是否需要推一下机器人）
        
        # compute observations, rewards, resets, ...
        # self._check_termination() # 检查有没有机器人摔倒/超时了 (计算 Done)
        # # self._compute_reward() # 结算这步的奖励 (计算 Reward)
        
        # # check terminations
        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten() # 捞出所有被判死刑的环境 ID
        # self.reset_envs_idx(env_ids) # 真正去重置它们的参数位置

        # # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()#思路就是给reset_all的重置创造和循环中reset一模一样的条件，这句需要保留是因为最后一行要专门赋值为False，否则进入循环中的紧下面几行会将已经经过一步错当作初始状态重新赋值给仿真器。模型第一步会倒就是因为_post_physics_step_for_reset_all最后一行错误地注释掉了，实际不能注释
        # if len(refresh_env_ids) > 0:
        #     # 对于重置的机器，把最新的（被拉回原点的）坐标强制塞进底层物理引擎
        #     self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
        #     self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
        #     # self.need_to_refresh_envs[refresh_env_ids] = False
            
        self._refresh_sim_tensors() # 从仿真器底层拉取最新的所有物理状态张量
        self._pre_compute_observations_callback() # 在算观测前做一些坐标系转换的准备工作

        self._compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        # 获取最新的观测向量 (计算 Obs)
        
        self._post_compute_observations_callback()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():#obs_buf_dict是按照字母顺序排列好的观测值字典，2个键是actor_obs和critic_obs,2个值是对应的观测张量列表
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs) # 同样限制观测值的极值防爆

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key]) # 将当前状态hist_obs_dict(加噪且放缩过的这一期的观测值)推入历史序列，供未来使用

        self.extras["to_log"] = self.log_dict
        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
            if self.debug_viz:
                self._draw_debug_vis()
                
        # if len(refresh_env_ids) > 0:
        #     for obs_key, obs_val in self.obs_buf_dict.items():#obs_buf_dict是按照字母顺序排列好的观测值字典，2个键是actor_obs和critic_obs,2个值是对应的观测张量列表
        #         # self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs) # 同样限制观测值的极值防爆
        #         print(obs_key, obs_val)

            # for key in self.history_handler.history.keys():
                # self.history_handler.add(key, self.hist_obs_dict[key]) # 将当前状态hist_obs_dict(加噪且放缩过的这一期的观测值)推入历史序列，供未来使用
                # print(key, self.hist_obs_dict[key])
        self.need_to_refresh_envs[refresh_env_ids] = False#这句不能注释掉，否则进入循环中的紧下面几行会将已经经过一步错当作初始状态重新赋值给仿真器。模型第一步会倒就是因为这句错误地注释掉了，实际不能注释
        self.reset_buf[:]=0#需要。否则打印信息中的奖励会丢失第1帧（0值），且无法save动作数据



    """
    【函数作用】：从_post_physics_step修改而来，获取用于 MPJPE 评估时delta-a模型的观测数据
    【被谁调用】：evaluate_policy_for_mpjpe
    """
    def get_observation_for_delta_a_openloop_for_mpjpe(self):
        self._refresh_sim_tensors() # 从仿真器底层拉取最新的所有物理状态张量
        # self.episode_length_buf += 1 # 存活步数加一
        # # update counters
        # self._update_counters_each_step()
        # self.last_episode_length_buf = self.episode_length_buf.clone()

        # self._pre_compute_observations_callback() # 在算观测前做一些坐标系转换的准备工作
        # self._update_tasks_callback() # 处理特定的环境任务（比如此时是否需要推一下机器人）
        
        # # compute observations, rewards, resets, ...
        # self._check_termination() # 检查有没有机器人摔倒/超时了 (计算 Done)
        # self._compute_reward() # 结算这步的奖励 (计算 Reward)
        
        # # check terminations
        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten() # 捞出所有被判死刑的环境 ID
        # self.reset_envs_idx(env_ids) # 真正去重置它们的参数位置

        # set envs
        # refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        # if len(refresh_env_ids) > 0:
        #     # 对于重置的机器，把最新的（被拉回原点的）坐标强制塞进底层物理引擎
        #     self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
        #     self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
        #     self.need_to_refresh_envs[refresh_env_ids] = False

        self._compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        # 获取最新的观测向量 (计算 Obs)
        
        self._post_compute_observations_callback()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():#obs_buf_dict是按照字母顺序排列好的观测值字典，2个键是actor_obs和critic_obs,2个值是对应的观测张量列表
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs) # 同样限制观测值的极值防爆

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key]) # 将当前状态hist_obs_dict(加噪且放缩过的这一期的观测值)推入历史序列，供未来使用

        # self.extras["to_log"] = self.log_dict
        # if self.viewer:
        #     self._setup_simulator_control()
        #     self._setup_simulator_next_task()
        #     if self.debug_viz:
        #         self._draw_debug_vis()
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    
    
    # 占位符回调函数，子类可以重写它们
    def _setup_simulator_next_task(self):
        pass

    def _setup_simulator_control(self):
        pass

    """
    【函数作用】：把需要使用的全局物理量转换成机器人本体局部坐标系的数据（非常重要，因为机器人眼睛长在自己身上）。
    【被谁调用】：_post_physics_step
    """
    def _pre_compute_observations_callback(self):
        # prepare quantities
        self.base_quat[:] = self.simulator.base_quat[:]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        # 基于基座的四元数逆向旋转，计算局部线速度
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        # print("self.base_lin_vel", self.base_lin_vel)
        # 计算局部角速度
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        # print(f"旋转后的self.base_ang_vel: {self.base_ang_vel}")
        # print("self.base_ang_vel", self.base_ang_vel)
        # 投影重力向量（完美反映了机器人目前的俯仰和侧倾角度）
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    """
    【函数作用】：根据配置设定的倒计时，到时间了就推一下机器人。
    【被谁调用】：_post_physics_step
    """
    def _update_tasks_callback(self):
        if self.config.domain_rand.push_robots:
            # 找到哪些机器人的计数器已经走到了它专属的惩罚时间间隔
            push_robot_env_ids = (self.push_robot_counter == (self.push_interval_s / self.dt).int()).nonzero(as_tuple=False).flatten()
            self.push_robot_counter[push_robot_env_ids] = 0 # 清零计数器
            self.push_robot_plot_counter[push_robot_env_ids] = 0
            # 随机分配它的下一次惩罚间隔
            self.push_interval_s[push_robot_env_ids] = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (len(push_robot_env_ids),), device=self.device, requires_grad=False)
            self._push_robots(push_robot_env_ids) # 真的动手推

    """
    【函数作用】：当前步走完后，更新一批带 "last" 前缀的变量。
    【被谁调用】：_post_physics_step
    """
    def _post_compute_observations_callback(self):
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.simulator.dof_pos[:]
        self.last_dof_vel[:] = self.simulator.dof_vel[:]
        self.last_root_vel[:] = self.simulator.robot_root_states[:, 7:13]


    """
    【函数作用】：检查回合终止条件。
    【被谁调用】：_post_physics_step
    """
    def _check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = 0
        # self.time_out_buf = 0
        # Note: DO NOT USE FOLLOWING TWO LINES STYLE
        # 原作者提示：不可以直接等于0，这会改变张量的类型，必须用 [:] 进行原位清零
        self.reset_buf[:] = 0#reset_buf在每轮_check_termination开始时被清零，只有满足特定条件的环境会被标记为1（需要重置）
        self.time_out_buf[:] = 0

        self._update_reset_buf()
        self._update_timeout_buf()

        # 如果是被打死了，或者是寿终正寝（超时）了，在这一回合里都被记作“结束”（Done）
        self.reset_buf |= self.time_out_buf
        # print(self.episode_sums["mpjpe"])
        # print(self.episode_sums["teleop_body_position_extend"][self.reset_buf.bool()])

    """
    【函数作用】：判断机器人有没有因为摔倒或其他违规行为“死亡”（需要重置）。
    【被谁调用】：_check_termination
    """
    def _update_reset_buf(self):
        # 1. 不该碰地的部位（比如身体、大腿）碰地了，直接判死
        if self.config.termination.terminate_by_contact:
            skip_motion_far_for_saving = getattr(self, "save_motion", False)
            if not skip_motion_far_for_saving:
                self.reset_buf |= torch.any(torch.norm(self.simulator.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        # 2. 严重倾斜侧翻（通过投影重力判定），判死
        if self.config.termination.terminate_by_gravity:
            # print(self.projected_gravity)
            skip_motion_far_for_saving = getattr(self, "save_motion", False)
            if not skip_motion_far_for_saving:
                self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 0:1]) > self.config.termination_scales.termination_gravity_x, dim=1)
                self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 1:2]) > self.config.termination_scales.termination_gravity_y, dim=1)
        
        # 3. 基座高度太低（比如深蹲起不来了或者直接趴下），判死
        if self.config.termination.terminate_by_low_height:
            # import ipdb; ipdb.set_trace()
            skip_motion_far_for_saving = getattr(self, "save_motion", False)
            if not skip_motion_far_for_saving:
                self.reset_buf |= torch.any(self.simulator.robot_root_states[:, 2:3] < self.config.termination_scales.termination_min_base_height, dim=1)

        # 4. 触碰关节软限制（为了保护真机的硬件极限而设置的）
        if self.config.termination.terminate_when_close_to_dof_pos_limit:
            out_of_dof_pos_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.) # lower limit
            out_of_dof_pos_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.)
            
            out_of_dof_pos_limits = torch.sum(out_of_dof_pos_limits, dim=1)
            # get random number between 0 and 1, if it is smaller than self.config.termination_probality.terminate_when_close_to_dof_pos_limit, apply the termination
            # 有一定概率被判死，不用绝对一刀切，给网络一定的探索空间
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_pos_limit:
                skip_motion_far_for_saving = getattr(self, "save_motion", False)
                if not skip_motion_far_for_saving:
                    self.reset_buf |= out_of_dof_pos_limits > 0.
        
        # 5. 速度超限判死
        if self.config.termination.terminate_when_close_to_dof_vel_limit:
            out_of_dof_vel_limits = torch.sum((torch.abs(self.simulator.dof_vel) - self.dof_vel_limits * self.config.termination_scales.termination_close_to_dof_vel_limit).clip(min=0., max=1.), dim=1)
            
            

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_vel_limit:
                skip_motion_far_for_saving = getattr(self, "save_motion", False)
                if not skip_motion_far_for_saving:
                    self.reset_buf |= out_of_dof_vel_limits > 0.
        
        # 6. 力矩超限判死
        if self.config.termination.terminate_when_close_to_torque_limit:
            out_of_torque_limits = torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.termination_scales.termination_close_to_torque_limit).clip(min=0., max=1.), dim=1)
            
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_torque_limit:
                skip_motion_far_for_saving = getattr(self, "save_motion", False)
                if not skip_motion_far_for_saving:
                    self.reset_buf |= out_of_torque_limits > 0.


    """
    【函数作用】：更新超时判定（寿终正寝）。
    【被谁调用】：_check_termination
    """
    def _update_timeout_buf(self):
        # 存活步数大于设定的最大步数
        self.time_out_buf |= self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

    """
    【函数作用】：对指定 ID 的环境执行重置流程。
    【被谁调用】：_post_physics_step 发现有人死亡时，或者外部评估时强制复位调用。
    """
    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
            target_states (dict): Dictionary containing lists of target states for the robot
        """
        if len(env_ids) == 0:
            return
        
        for i in range(len(self.reward_functions)):
                    name = self.reward_names[i]#reward_names[i]和reward_functions[i]是一一对应的，都是按照配置中reward_scales里非零的奖励项顺序排列的。reward_names[i]是字符串名字，reward_functions[i]是函数指针。
                    print(f"各项累计奖励：{name}:{self.episode_sums[name]}")
        
        self.need_to_refresh_envs[env_ids] = True#在CPU修改了状态，需要同步刷新到GPU的环境
        self._reset_buffers_callback(env_ids, target_buf) # 清空旧数据：重力投影，参考相位等不会清空（等下一轮计算obs自动赋值，但这会影响接下来一步动作的输出）。等actions，上一步动作，延迟后动作，上一步速度，足端腾空时间等等。
        self._reset_tasks_callback(env_ids)        #包含动作起始点的选择（这个函数在motion_tracking.py的_reset_tasks_callback中包含，直接点进去是其父类），域随机化，惩罚项系数，关节物理限制系数，观测噪声系数。 if target_states is not None, reset to target states # 随机出新的物理难度(与奖励的计算标准有关)。域随机化，惩罚项系数，关节物理限制系数，观测噪声系数。
        self._reset_robot_states_callback(env_ids, target_states) # 从配置读取各关节和根状态的值，并给关节位置和根节点速度加一点随机化。把他丢回出生点.

        # fill extras：用于 tensorboard 打印出这个结束的 episode 各项奖励的均值
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0. # 记录完毕清零
        self.extras["time_outs"] = self.time_out_buf
        # self._refresh_sim_tensors()


    """
    【函数作用】：让机器人的身体和关节物理状态复位。
    【被谁调用】：reset_envs_idx
    """
    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # if target_states is not None, reset to target states
        if target_states is not None:
            self._reset_dofs(env_ids, target_states["dof_states"])
            self._reset_root_states(env_ids, target_states["root_states"])
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

    """
    【函数作用】：回合重置时，执行各种课程学习的升级/降级和参数随机化。
    【被谁调用】：reset_envs_idx
    """
    def _reset_tasks_callback(self, env_ids):
        self._episodic_domain_randomization(env_ids) # 重新掷骰子摇随机参数。配置中没有开启域随机化。
        if self.use_reward_penalty_curriculum:
            self._update_reward_penalty_curriculum()
        if self.use_reward_limits_dof_pos_curriculum or self.use_reward_limits_dof_vel_curriculum or self.use_reward_limits_torque_curriculum:
            self._update_reward_limits_curriculum()
        if self.add_noise_currculum:
            self._update_obs_noise_curriculum()
    
    """
    【函数作用】：根据大家存活的时间（表现）来调整加给神经网络的噪音难度。
    【被谁调用】：_reset_tasks_callback
    """
    def _update_obs_noise_curriculum(self):
        # 如果大家活的时间还很短，就降低噪音让它好学一点
        if self.average_episode_length < self.config.obs.soft_dof_pos_curriculum_level_down_threshold:
            self.current_noise_curriculum_value *= (1 - self.config.obs.soft_dof_pos_curriculum_degree)
        # 如果大家活得足够长（稳如老狗），就大幅加大噪音难度
        elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
            self.current_noise_curriculum_value *= (1 + self.config.obs.soft_dof_pos_curriculum_degree)

        # 钳制数值范围
        self.current_noise_curriculum_value = np.clip(self.current_noise_curriculum_value, 
                                                      self.config.obs.noise_value_min, 
                                                      self.config.obs.noise_value_max)

    """
    【函数作用】：重置时清空各相关的追踪数组。
    【被谁调用】：reset_envs_idx
    """
    def _reset_buffers_callback(self, env_ids, target_buf=None):
        if target_buf is not None:
            self.simulator.dof_pos[env_ids] = target_buf["dof_pos"].to(self.simulator.dof_pos.dtype)
            self.simulator.dof_vel[env_ids] = target_buf["dof_vel"].to(self.simulator.dof_vel.dtype)
            self.base_quat[env_ids] = target_buf["base_quat"].to(self.base_quat.dtype)
            self.base_lin_vel[env_ids] = target_buf["base_lin_vel"].to(self.base_lin_vel.dtype)
            self.base_ang_vel[env_ids] = target_buf["base_ang_vel"].to(self.base_ang_vel.dtype)
            self.projected_gravity[env_ids] = target_buf["projected_gravity"].to(self.projected_gravity.dtype)
            self.torques[env_ids] = target_buf["torques"].to(self.torques.dtype)
            self.actions[env_ids] = target_buf["actions"].to(self.actions.dtype)
            self.last_actions[env_ids] = target_buf["last_actions"].to(self.last_actions.dtype)
            self.last_dof_pos[env_ids] = target_buf["last_dof_pos"].to(self.last_dof_pos.dtype)
            self.last_dof_vel[env_ids] = target_buf["last_dof_vel"].to(self.last_dof_vel.dtype)
            self.episode_length_buf[env_ids] = target_buf["episode_length_buf"].to(self.episode_length_buf.dtype)
            self.reset_buf[env_ids] = target_buf["reset_buf"].to(self.reset_buf.dtype)
            self.time_out_buf[env_ids] = target_buf["time_out_buf"].to(self.time_out_buf.dtype)
            self.feet_air_time[env_ids] = target_buf["feet_air_time"].to(self.feet_air_time.dtype)
            self.last_contacts[env_ids] = target_buf["last_contacts"].to(self.last_contacts.dtype)
            self.last_contacts_filt[env_ids] = target_buf["last_contacts_filt"].to(self.last_contacts_filt.dtype)
            self.feet_air_max_height[env_ids] = target_buf["feet_air_max_height"].to(self.feet_air_max_height.dtype)
        else:
            self.actions[env_ids] = 0.
            self.last_actions[env_ids] = 0.
            self.actions_after_delay[env_ids] = 0.
            self.last_dof_pos[env_ids] = 0.
            self.last_dof_vel[env_ids] = 0.
            self.feet_air_time[env_ids] = 0.
            self.episode_length_buf[env_ids] = 0
            # self.reset_buf[env_ids] = 0
            # self.time_out_buf[env_ids] = 0
            self.reset_buf[env_ids] = 1 # 告知下一个流程：“我已经重置好了”
            self._update_average_episode_length(env_ids)

            self.history_handler.reset(env_ids) # 抹除历史观察记录（重生了，不带入前世记忆）

    """
    【函数作用】：获取用于 MPPI (基于模型的轨迹预测控制算法) 等传统控制算法所需的缓冲区深拷贝快照。
    【被谁调用】：外部特定算法（非 PPO 等模型无关算法）调用。
    """
    def get_mppi_buffers(self, env_ids):
        """ Get buffers for MPPI
        MPPI algo need to store the buffers to replicate environments
        """
        return {
            "dof_pos": copy.deepcopy(self.simulator.dof_pos[env_ids]),
            "dof_vel": copy.deepcopy(self.simulator.dof_vel[env_ids]),
            "base_quat": copy.deepcopy(self.base_quat[env_ids]),
            "base_lin_vel": copy.deepcopy(self.base_lin_vel[env_ids]),
            "base_ang_vel": copy.deepcopy(self.base_ang_vel[env_ids]),
            "projected_gravity": copy.deepcopy(self.projected_gravity[env_ids]),
            "torques": copy.deepcopy(self.torques[env_ids]),
            "actions": copy.deepcopy(self.actions[env_ids]),
            "last_actions": copy.deepcopy(self.last_actions[env_ids]),
            "last_dof_pos": copy.deepcopy(self.last_dof_pos[env_ids]),
            "last_dof_vel": copy.deepcopy(self.last_dof_vel[env_ids]),
            "episode_length_buf": copy.deepcopy(self.episode_length_buf[env_ids]),
            "reset_buf": copy.deepcopy(self.reset_buf[env_ids]),
            "time_out_buf": copy.deepcopy(self.time_out_buf[env_ids]),
            "feet_air_time": copy.deepcopy(self.feet_air_time[env_ids]),
            "last_contacts": copy.deepcopy(self.last_contacts[env_ids]),
            "last_contacts_filt": copy.deepcopy(self.last_contacts_filt[env_ids]),
            "feet_air_max_height": copy.deepcopy(self.feet_air_max_height[env_ids]),
        }

    """
    【函数作用】：结算当前时间步所有并行机器人获取的奖励/惩罚。
    【RL知识补充】：非常核心的部分。这里通过循环执行之前动态绑定的各个 _reward_xxx 业务函数，算出总得分。
    【被谁调用】：_post_physics_step
    """
    def _compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]#reward_names[i]和reward_functions[i]是一一对应的，都是按照配置中reward_scales里非零的奖励项顺序排列的。reward_names[i]是字符串名字，reward_functions[i]是函数指针。
            # 计算特定的单项奖励，并乘以你在配置文件里设定的该项权重（Scale）
            rew = self.reward_functions[i]() * self.reward_scales[name]#名字，函数，奖励权重这3个都是一一对应的（基于reward_scales的顺序构建的）
            try:
                assert rew.shape[0] == self.num_envs
            except:
                import ipdb; ipdb.set_trace()
            # penalty curriculum：如果是惩罚项，它可能还会被课程系数打折扣
            if name in self.config.rewards.reward_penalty_reward_names:
                if self.config.rewards.reward_penalty_curriculum:
                    rew *= self.reward_penalty_scale
            self.rew_buf += rew # 累加到最终的总分上
            self.episode_sums[name] += rew # 累加到统计字典里，只为看 Log。#episode_sums是一个字典，键是奖励项的名字，值是一个长度为环境数量的张量，记录了每个环境在这一回合里这个奖励项的累计得分。它会在 reset 的时候被清零。
            
        # 根据配置，截断负数惩罚，很多论文认为负奖励会让机器人更倾向于消极避世（立刻摔倒来停止由于乱动带来的扣分）
        if self.config.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            
        # add termination reward after clipping：如果是致死惩罚，则在上面截断0之后再减去大额负分
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        # 把当前课程学习的各项系数写进 Log，方便看曲线有没有在变化
        if self.use_reward_penalty_curriculum:
            self.log_dict["penalty_scale"] = torch.tensor(self.reward_penalty_scale, dtype=torch.float)
            self.log_dict["average_episode_length"] = self.average_episode_length
        
        if self.use_reward_limits_dof_pos_curriculum:
            self.log_dict["soft_dof_pos_curriculum_value"] = torch.tensor(self.soft_dof_pos_curriculum_value, dtype=torch.float)
        if self.use_reward_limits_dof_vel_curriculum:
            self.log_dict["soft_dof_vel_curriculum_value"] = torch.tensor(self.soft_dof_vel_curriculum_value, dtype=torch.float)
        if self.use_reward_limits_torque_curriculum:
            self.log_dict["soft_torque_curriculum_value"] = torch.tensor(self.soft_torque_curriculum_value, dtype=torch.float)
        
        if self.add_noise_currculum:
            self.log_dict["current_noise_curriculum_value"] = torch.tensor(self.current_noise_curriculum_value, dtype=torch.float)
            
    """
    【函数作用】：收集物理引擎反馈的数据，将其包装为神经网络所需的特征形式（张量）。
    【RL知识补充】：所有的特征在送入网络前通常会被解析、加入随机高斯噪声（提高鲁棒性）、并经过固定缩放。
    【被谁调用】：_post_physics_step
    """
    def _compute_observations(self):
        """ Computes observations
        """
        # super().compute_observations()
        self.obs_buf_dict_raw = {}
        self.hist_obs_dict = {}
        
        if self.add_noise_currculum:
            noise_extra_scale = self.current_noise_curriculum_value
        else:
            noise_extra_scale = 1.
        # print("noise_extra_scale", noise_extra_scale)
        # compute Algo observations
        for obs_key, obs_config in self.config.obs.obs_dict.items():#观测值总有2个：动作网络和价值网络。如果有闭环微调，还有闭环微调的观测值。
            self.obs_buf_dict_raw[obs_key] = dict()
            # 这是一个外部帮助函数，它会依据配置找到下面的 `_get_obs_xxx` 方法，调用后自动施加噪声。
            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], self.config.obs.obs_scales, self.config.obs.noise_scales, noise_extra_scale)
        
        # Compute history observations
        # 处理并抽取存好的历史状态队列
        history_obs_list = self.history_handler.history.keys()#history_handler是一个HistoryHandler对象，.history是一个字典，每个键是各个观测维度，值以3维张量存储所有环境所有历史期数所有观测值维度的历史观测值。？？为什么是字典？
        parse_observation(self, history_obs_list, self.hist_obs_dict, self.config.obs.obs_scales, self.config.obs.noise_scales, noise_extra_scale)
        #上一行得到的hist_obs_dict只是当前这一期的观测值，加过噪声且放缩过后的
        self._post_config_observation_callback()#把self.obs_buf_dict_raw按字母顺序拼接成self.obs_buf_dict字典，键为网络名称，值为该网络各个观测值按照字母排序的值构成的一维张量供全连接网络使用。

    """
    【函数作用】：把分散的特征按照字母顺序（保证统一）拼接成一个巨大的一维张量，供全连接网络使用。
    【被谁调用】：_compute_observations
    """
    def _post_config_observation_callback(self):
        self.obs_buf_dict = dict()
        
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            obs_keys = sorted(obs_config) # 强制按照字母顺序排列 key，防止不同时候拼出来的特征顺序乱掉导致网络瞎掉
            # print("obs_keys", obs_keys)            
            self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1) # 沿着最后一维（特征维）拼接
            
            # if self.episode_length_buf.max() < 1:
    #             self.obs_buf_dict["actor_obs"]=torch.tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  5.2807e-03,  5.6546e-03,
    #      -1.9112e-02, -4.0459e-03,  1.4637e-02,  2.7219e-01, -2.5757e-01,
    #       2.0141e-01,  0.0000e+00, -1.3613e-01, -1.3153e-01, -4.8082e-02,
    #      -2.0730e-01,  2.0428e-01,  0.0000e+00,  2.0814e-01, -6.7075e-02,
    #      -2.7382e-01,  2.5760e-01,  3.9115e-01, -4.2474e-01,  9.9456e-01,
    #       2.2172e-01, -3.0844e-01,  2.6735e-01,  9.8759e-01, -1.3740e-03,
    #      -2.2959e-03,  1.1763e-03, -1.4489e-03, -9.4786e-05,  0.0000e+00,
    #      -4.1740e-04,  3.3123e-04,  3.4656e-05,  1.5207e-04, -1.1137e-05,
    #       0.0000e+00, -2.4071e-03, -1.0776e-03, -1.1750e-03, -4.8322e-03,
    #      -6.0613e-03, -1.9516e-03, -1.8067e-03, -2.5364e-03,  5.3511e-03,
    #       8.7900e-04, -3.7086e-04,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00, -0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00, -0.0000e+00, -0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00,
    #       0.0000e+00, -0.0000e+00, -0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00,
    #      -0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00, -0.0000e+00, -0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  2.3538e-01, -5.3127e-02, -9.7045e-01,  5.0847e-03]],
    #    device='cuda:0')
                # print(f"self.obs_buf_dict[obs_key]是：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：{obs_key}: {self.obs_buf_dict[obs_key]}")
            
            # refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
            # if len(refresh_env_ids) > 0:
            #     for key in obs_keys:
            #         print(f"{key}: {self.obs_buf_dict_raw[obs_key][key]}")

    
    """
    【函数作用】：PD 伺服控制器方程的底层实现，将期望值转换为实际驱动力矩。
    【RL知识补充】：RL 网络输出的很可能是一个范围如 [-1, 1] 的“期望偏移量”，这里通过 KP 和 KD 增益把它变成牛顿米。
    【被谁调用】：_apply_force_in_physics_step
    """
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # 将归一化的 action 乘以我们定义的一个动作放大系数（Action Scale）
        actions_scaled = actions * self.config.robot.control.action_scale
        control_type = self.config.robot.control.control_type
        if control_type=="P":
            # 位置控制模式（最常用）：扭矩 = KP * (目标角度 - 真实角度) - KD * 真实角速度
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V":
            # 速度控制模式
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_dt
        elif control_type=="T":
            # 纯扭矩模式（网络直接暴力输出力矩大小）
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        # 为力矩施加随机干扰（RFI），模拟真实电机发热和齿轮旷量导致的不精准
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        # 根据机器人的物理极限夹断扭矩
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        else:
            return torques

    def _create_terrain(self):
        super()._create_terrain()

    """
    【函数作用】：在 GUI 界面绘制用于辅助 Debug 的调试信息。
    【被谁调用】：_post_physics_step（条件开启）
    """
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        self.gym.clear_lines(self.viewer)
        self._refresh_sim_tensors()

        # 对刚被外部“推”了一下的机器人，在画面里画出一根表示受力方向和大小的彩色线条
        draw_env_ids = (self.push_robot_plot_counter < 10).nonzero(as_tuple=False).flatten()
        not_draw_env_ids = (self.push_robot_plot_counter >= 10).nonzero(as_tuple=False).flatten()
        self.record_push_robot_vel_buf[not_draw_env_ids] *=0
        self.push_robot_plot_counter[not_draw_env_ids] = 0
        
        for env_id in draw_env_ids:
            push_vel = self.record_push_robot_vel_buf[env_id]
            push_vel = torch.cat([push_vel, torch.zeros(1, device=self.device)])
            push_pos = self.simulator.robot_root_states[env_id, :3]
            push_vel_list = [push_vel]
            push_pos_list = [push_pos]
            push_mag_list = [1]
            push_color_schems = [(0.851, 0.144, 0.07)]
            push_line_widths = [0.03]
            for push_vel, push_pos, push_mag, push_color, push_line_width in zip(push_vel_list, push_pos_list, push_mag_list, push_color_schems, push_line_widths):
                for _ in range(200):
                    gymutil.draw_line(Point(push_pos +torch.rand(3, device=self.device) * push_line_width),
                                        Point(push_pos + push_vel * push_mag),
                                        Point(push_color),
                                        self.gym, self.viewer, self.envs[env_id])


    ################ Curriculum (所有的课程学习滑动均值更新机制) #################

    def _update_average_episode_length(self, env_ids):
        # 使用基于比例的滑动平均法，平滑记录大家能够活多久
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.last_episode_length_buf[env_ids], dtype=torch.float)
        
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)

        
    def _update_reward_penalty_curriculum(self):
        """
        Update the penalty curriculum based on the average episode length.

        If the average episode length is below the penalty level down threshold,
        decrease the penalty scale by a certain level degree.
        If the average episode length is above the penalty level up threshold,
        increase the penalty scale by a certain level degree.
        Clip the penalty scale within the specified range.

        Returns:
            None
        """
        # 如果死得快（平均长度小），就降低惩罚倍率，鼓励网络敢于乱动探索；如果活得久，就加大惩罚倍率，要求它的动作不仅有效还得漂亮。
        if self.average_episode_length < self.config.rewards.reward_penalty_level_down_threshold:
            self.reward_penalty_scale *= (1 - self.config.rewards.reward_penalty_degree)
        elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
            self.reward_penalty_scale *= (1 + self.config.rewards.reward_penalty_degree)

        self.reward_penalty_scale = np.clip(self.reward_penalty_scale, self.config.rewards.reward_min_penalty_scale, self.config.rewards.reward_max_penalty_scale)

    def _update_reward_limits_curriculum(self):
        """
        Update the reward limits curriculum based on the average episode length.
        """
        # 与上面类似，存活越久，就把关节可以活动的安全范围收得越紧，强迫网络收敛到非常保守的运动学核心安全区中。
        if self.use_reward_limits_dof_pos_curriculum:
            if self.average_episode_length < self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_down_threshold:
                self.soft_dof_pos_curriculum_value *= (1 + self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree)
            elif self.average_episode_length > self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_up_threshold:
                self.soft_dof_pos_curriculum_value *= (1 - self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree)
            self.soft_dof_pos_curriculum_value = np.clip(self.soft_dof_pos_curriculum_value, 
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit, 
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit)
        
        if self.use_reward_limits_dof_vel_curriculum:
            if self.average_episode_length < self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_down_threshold:
                self.soft_dof_vel_curriculum_value *= (1 + self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree)
            elif self.average_episode_length > self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_up_threshold:
                self.soft_dof_vel_curriculum_value *= (1 - self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree)
            self.soft_dof_vel_curriculum_value = np.clip(self.soft_dof_vel_curriculum_value, 
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit, 
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit)
        
        if self.use_reward_limits_torque_curriculum:
            if self.average_episode_length < self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_level_down_threshold:
                self.soft_torque_curriculum_value *= (1 + self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_degree)
            elif self.average_episode_length > self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_level_up_threshold:
                self.soft_torque_curriculum_value *= (1 - self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_degree)
            self.soft_torque_curriculum_value = np.clip(self.soft_torque_curriculum_value, 
                                                        self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit, 
                                                        self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit)

    #------------ reward functions (具体的各种打分函数) ----------------
    ########################### PENALTY REWARDS ###########################

    def _reward_termination(self):
        # Terminal reward / penalty
        # 因为非正常操作（摔倒碰地）重置的环境扣大分。因超时顺利活下来的不扣分（~self.time_out_buf 排除）。
        return self.reset_buf * ~self.time_out_buf

    def _reward_penalty_torques(self):
        # Penalize torques
        # 能量效率惩罚：力矩平方的累加和。力出的越大扣分越多，鼓励省电和利用自身重力。
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_penalty_dof_vel(self):
        # Penalize dof velocities
        # 肢体挥动惩罚：关节速度平方的累加和。不鼓励无用的快速空挥。
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)
    
    def _reward_penalty_dof_acc(self):
        # Penalize dof accelerations
        # 加速度惩罚（平滑性惩罚）：惩罚速度的快速变号突变，保护实体减速机不被打齿。
        return torch.sum(torch.square((self.last_dof_vel - self.simulator.dof_vel) / self.dt), dim=1)
    
    def _reward_penalty_action_rate(self):
        # Penalize changes in actions
        # Action抖动惩罚：惩罚本步骤输出的动作和上一步相差太大，防止机器人得帕金森抽搐。
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_penalty_orientation(self):
        # Penalize non flat base orientation
        # 姿态惩罚：重力向量在局部XY平面上的投影越大，说明身子越歪（如果是人型或四足，应当保持躯干与地面基本平行）。
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    ######################## LIMITS REWARDS #########################

    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit
        # 如果关节极其接近物理或软件限定的死角极限，进行扣分。
        if self.use_reward_limits_dof_pos_curriculum:
            m = (self.simulator.hard_dof_pos_limits[:, 0] + self.simulator.hard_dof_pos_limits[:, 1]) / 2
            r = self.simulator.hard_dof_pos_limits[:, 1] - self.simulator.hard_dof_pos_limits[:, 0]
            lower_soft_limit = m - 0.5 * r * self.soft_dof_pos_curriculum_value
            upper_soft_limit = m + 0.5 * r * self.soft_dof_pos_curriculum_value
        else:
            lower_soft_limit = self.simulator.dof_pos_limits[:, 0]
            upper_soft_limit = self.simulator.dof_pos_limits[:, 1]
        out_of_limits = -(self.simulator.dof_pos - lower_soft_limit).clip(max=0.) # lower limit
        out_of_limits += (self.simulator.dof_pos - upper_soft_limit).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_limits_dof_vel(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        if self.use_reward_limits_dof_vel_curriculum:
            return torch.sum((torch.abs(self.simulator.dof_vel) - self.dof_vel_limits * self.soft_dof_vel_curriculum_value).clip(min=0., max=1.), dim=1)
        else:
            return torch.sum((torch.abs(self.simulator.dof_vel) - self.dof_vel_limits * self.config.rewards.reward_limit.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_limits_torque(self):
        # penalize torques too close to the limit
        if self.use_reward_limits_torque_curriculum:
            return torch.sum((torch.abs(self.torques) - self.torque_limits * self.soft_torque_curriculum_value).clip(min=0., max=1.), dim=1)
        else:
            return torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.rewards.reward_limit.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_penalty_slippage(self):
        # 打滑惩罚（重要）：当脚部接触地面的力大于1时（意味着已经踩实），如果该脚还拥有水平线速度（foot_vel），说明它在贴地滑行，必须重罚。
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)

    def _reward_feet_max_height_for_this_air(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # Jiawei: Key ingredient.
        # 鼓励高抬腿：在脚底刚发生触地的一瞬间（也就是这一步踏实了结算的时候），根据这整个腾空周期中达到的最高高度给奖励。
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) # 滤波防抖
        from_air_to_contact = torch.logical_and(contact_filt, ~self.last_contacts_filt) # 找到刚触地的那一帧
        self.last_contacts = contact
        self.last_contacts_filt = contact_filt
        self.feet_air_max_height = torch.max(self.feet_air_max_height, self.simulator._rigid_body_pos[:, self.feet_indices, 2]) # 不断更新最大高度

        # 只在落地那一帧（from_air_to_contact）计算奖励值
        rew_feet_max_height = torch.sum((torch.clamp_min(self.config.rewards.desired_feet_max_height_for_this_air - self.feet_air_max_height, 0)) * from_air_to_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_max_height *= ~contact_filt # 一旦落地，高度记录清零，为下一次腾空做准备
        return rew_feet_max_height
    
    def _reward_feet_heading_alignment(self):
        # 脚向对齐惩罚：惩罚机器人八字步。希望脚趾头的指向（前向向量）永远和基座的前方保持绝对一致。
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]

        forward_left_feet = quat_apply(left_quat, self.forward_vec)
        heading_left_feet = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
        forward_right_feet = quat_apply(right_quat, self.forward_vec)
        heading_right_feet = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])


        root_forward = quat_apply(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(wrap_to_pi(heading_left_feet - heading_root))
        heading_diff_right = torch.abs(wrap_to_pi(heading_right_feet - heading_root))
        
        return heading_diff_left + heading_diff_right
    
    def _reward_penalty_feet_ori(self):
        # 垫脚尖惩罚：将全局重力向量用脚部的局部四元数反求，如果脚底板是平铺的，重力投影在xy上应该是0。以此来惩罚用脚尖或脚跟走路的奇怪姿态。
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 
    
    """
    【函数作用】：在每一个回合开启时分配物理世界的随机数。
    【被谁调用】：_reset_tasks_callback
    """
    def _episodic_domain_randomization(self, env_ids):
        """ Update scale of Kp, Kd, rfi lim"""
        if len(env_ids) == 0:
            return
        
        # 这里的代码已经修好了你之前的缩进问题，完美对齐：
        if self.config.domain_rand.randomize_pd_gain:
            self._kp_scale[env_ids] = torch_rand_float(self.config.domain_rand.kp_range[0], self.config.domain_rand.kp_range[1], (len(env_ids), self.num_dofs), device=self.device)
            self._kd_scale[env_ids] = torch_rand_float(self.config.domain_rand.kd_range[0], self.config.domain_rand.kd_range[1], (len(env_ids), self.num_dofs), device=self.device)
    
        if self.config.domain_rand.randomize_rfi_lim:
            self._rfi_lim_scale[env_ids] = torch_rand_float(self.config.domain_rand.rfi_lim_range[0], self.config.domain_rand.rfi_lim_range[1], (len(env_ids), self.num_dofs), device=self.device)

        if self.config.domain_rand.randomize_ctrl_delay:            
            # self.action_queue[env_ids] = 0.delay: # (原代码中被注掉的疑似错误代码保留)
            self.action_queue[env_ids] *= 0. # 重新投胎重置延迟队列
            # self.action_queue[env_ids] = 0.
            self.action_delay_idx[env_ids] = torch.randint(self.config.domain_rand.ctrl_delay_step_range[0], 
                                            self.config.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)

    """
    【函数作用】：对指定的机器人施加一个极短的外部速度干预，用来训练抗干扰能力。
    【被谁调用】：_update_tasks_callback
    """
    def _push_robots(self, env_ids):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel = self.config.domain_rand.max_push_vel_xy
        # 在XY平面上随机产生一个速度向量
        self.push_robot_vel_buf[env_ids] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2), device=str(self.device))  # lin vel x/y
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone() # 记录下来用于画图
        # 直接暴力改写物理引擎池子里的速度参数
        self.simulator.robot_root_states[env_ids, 7:9] = self.push_robot_vel_buf[env_ids]
        # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.simulator.all_root_states))


    ############ TERRAIN AND COMMANDS

    ################ ENV CALLBACKS #################

    """
    【函数作用】：重置被选中环境机器人的各个驱动关节。
    【被谁调用】：_reset_robot_states_callback
    """
    def _reset_dofs(self, env_ids, target_state=None):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        If target_state is not None, reset to target_state

        Args:
            env_ids (List[int]): Environemnt ids
            target_state (Tensor): Target state
        """
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
        else:
            # 在 0.5 到 1.5 倍默认零位之间进行随机初始化位置（防过拟合死记硬背）。self.default_dof_pos从配置文件中读取
            self.simulator.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device))
            # self.simulator.dof_pos[env_ids] = self.default_dof_pos
            # import ipdb; ipdb.set_trace()
            
            # 初始速度一定重置为0
            self.simulator.dof_vel[env_ids] = 0.

    """
    【函数作用】：重置被选中环境机器人的基座躯干坐标。
    【被谁调用】：_reset_robot_states_callback
    """
    def _reset_root_states(self, env_ids, target_root_states=None):
        """ Resets ROOT states position and velocities of selected environmments
            if target_root_states is not None, reset to target_root_states
        Args:
            env_ids (List[int]): Environemnt ids
            target_root_states (Tensor): Target root states
        """
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]

        else:
            # base position
            # custom_origins 通常指复杂的地形（阶梯、粗糙地面等）
            if self.custom_origins:
                self.simulator.robot_root_states[env_ids] = self.base_init_state
                self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
                # 在指定出生点周围1米范围内随机刷新位置
                self.simulator.robot_root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=str(self.device)) # xy position within 1m of the center
            else:
                self.simulator.robot_root_states[env_ids] = self.base_init_state#从配置里读出来的，直接同步到仿真中的robot_root_states里
                self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
            # base velocities
            
            # 出生时给一个介于 -0.5 到 0.5 的微小速度微扰
            self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=str(self.device)) # [7:10]: lin vel, [10:13]: ang vel


    def _plot_domain_rand_params(self):
        raise NotImplementedError

    ######################### Observations #########################
    # 以下函数均为提供给网络构建 Observation 特征的接口方法。
    
    def _get_obs_base_pos_z(self,):
        # 躯干 Z 轴绝对高度
        return self.simulator.robot_root_states[:, 2:3]
    
    def _get_obs_feet_contact_force(self,):
        # 脚底力传感器数值（接触力）
        return self.simulator.contact_forces[:, self.feet_indices, :].view(self.num_envs, -1)
          
    
    def _get_obs_base_lin_vel(self,):
        # 躯干局部线速度
        return self.base_lin_vel
    
    def _get_obs_base_ang_vel(self,):
        # 躯干局部角速度
        return self.base_ang_vel
    
    def _get_obs_projected_gravity(self,):
        # 投影重力（身体倾斜姿态角）
        return self.projected_gravity
    
    def _get_obs_dof_pos(self,):
        # 当前关节角度减去默认零点偏置
        return self.simulator.dof_pos - self.default_dof_pos
    
    def _get_obs_dof_vel(self,):
        # 关节速度
        return self.simulator.dof_vel
    
    def _get_obs_history(self,):
        # 历史特征记录堆叠
        assert "history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_short_history(self,):
        # 短期历史特征记录堆叠（逻辑同上）
        assert "short_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['short_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_long_history(self,):
        # 长期历史特征记录堆叠（用于长时距记忆感知）
        assert "long_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['long_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_actions(self,):
        # 获取上一帧网络自身的输出，让它“知道自己刚做了什么决策”
        return self.actions