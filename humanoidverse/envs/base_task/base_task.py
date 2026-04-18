import sys
import os
import numpy as np
import torch
from rich.progress import Progress
# from humanoidverse.envs.env_utils.terrain import Terrain

import logging
from loguru import logger
from humanoidverse.utils.logging import capture_stdout_to_loguru
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
from hydra.utils import instantiate, get_class

from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
from termcolor import colored

# Base class for RL tasks
# RL任务的基础类，所有的具体机器人任务（如行走、奔跑、上下楼梯）都会继承这个类。
class BaseTask():
    """
    【函数作用】：类的构造函数，负责初始化整个强化学习训练环境。
    【RL知识补充】：在强化学习中，环境（Environment）负责接收动作（Action），推进物理仿真，
                  并返回新的状态/观测（Observation）、奖励（Reward）和完成信号（Done/Termination）。
                  这个初始化函数就是在搭建这样一个用于批量处理的强化学习环境。
    【被谁调用】：通常被训练脚本（如 train.py 或者 PPO Runner 初始化阶段）调用。
    """
    def __init__(self, config, device):
        # 保存传入的配置字典（通常由Hydra等配置管理工具生成，包含各种超参数）
        self.config = config
        
        # optimization flags for pytorch JIT
        # 关闭 PyTorch JIT 的性能分析模式，通常在强化学习部署模型时可以稍微提升运行速度
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # self.simulator = instantiate(config=self.config.simulator, device=device)
        # 根据配置文件动态获取仿真器（Simulator）的类，并实例化。仿真器负责底层的物理计算。
        SimulatorClass = get_class(self.config.simulator._target_)
        self.simulator: BaseSimulator = SimulatorClass(config=self.config, device=device)
        
        # 判断是否为无头模式（headless）。无头模式指不渲染3D图形界面，通常在服务器上训练时开启，能极大提升训练速度。
        self.headless = config.headless
        self.simulator.set_headless(self.headless)
        # 建立仿真器基础设置
        self.simulator.setup()
        # 获取仿真器运行的硬件设备（如 'cuda:0'），RL的大规模并行极度依赖GPU
        self.device = self.simulator.sim_device
        # 获取物理引擎仿真的时间步长（例如 0.005 秒，即物理引擎每秒计算200次）
        self.sim_dt = self.simulator.sim_dt
        # 设定世界坐标系的向上轴，2 代表 Z 轴向上（0:X, 1:Y, 2:Z）
        self.up_axis_idx = 2 # Jiawei: HARD CODE FOR NOW
        
        # 【重要RL概念：Control Decimation 控制降采样】
        # 物理引擎为了稳定，需要很小的步长（如 200Hz），但神经网络不需要这么快做决策（通常 50Hz 就够了）。
        # control_decimation 就是这个比率（如 200/50 = 4）。
        # 所以 RL 策略的时间步 dt = 物理时间步 * decimation。
        self.dt = self.config.simulator.config.sim.control_decimation * self.sim_dt
        
        # 强化学习每个 Episode（回合）的最大时长（秒）
        self.max_episode_length_s = self.config.max_episode_length_s
        # 转换成 RL 步数：最大时间 / RL时间步长
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        # 【重要RL概念：并行环境】
        # num_envs 是并行环境的数量。在 GPU 上我们可能同时跑 4096 个机器人来快速收集经验数据。
        self.num_envs = self.config.num_envs
        # Actor 网络（策略网络）的观测（输入）维度大小
        self.dim_obs = self.config.robot.policy_obs_dim
        # Critic 网络（价值网络）的观测维度大小（非对称Actor-Critic架构中，Critic通常可以看到特权信息）
        self.dim_critic_obs = self.config.robot.critic_obs_dim
        # 动作空间维度（通常等于机器人的驱动关节数，即自由度 DOF）
        self.dim_actions = self.config.robot.actions_dim

        # 初始化地形（如平地、粗糙地面、楼梯等），这对于机器人的 Locomotion（运动控制）至关重要
        terrain_mesh_type = self.config.terrain.mesh_type
        self.simulator.setup_terrain(terrain_mesh_type)
        # 初始化一些用于可视化的实体（比如显示力的箭头、目标点等）
        self.setup_visualize_entities()

        # create envs, sim and viewer
        # 加载机器人模型资产（URDF/MJCF 文件）
        self._load_assets()
        # 计算每个并行机器人在三维空间中的出生点坐标
        self._get_env_origins()
        # 正式在仿真器中创建这些机器人实例
        self._create_envs()
        
        # 获取机器人的物理限制：关节位置限制、关节速度限制、关节最大扭矩限制
        # RL 动作输出时经常需要这些限制进行截断（Clipping）或缩放（Scaling）
        self.dof_pos_limits, self.dof_vel_limits, self.torque_limits = self.simulator.get_dof_limits_properties()
        
        # 建立机器人各个身体部件的名字和索引的映射（比如哪个序号是膝盖，哪个是脚底，用于写奖励函数）
        self._setup_robot_body_indices()#主要是将刚体和关节分组，每个组的索引需要存下来。完整的刚体和关节是直接从仿真里赋值的，和仿真完全一样
        
        # self._create_sim()
        # 准备开始仿真，锁定内存等操作
        # while True:
            # self.simulator.prepare_sim()
        self.simulator.prepare_sim()
        
        # if running with a viewer, set up keyboard shortcuts and camera
        # 如果不是无头模式，则需要配置 3D 渲染窗口（Viewer）
        self.viewer = None
        if self.headless == False:
            self.debug_viz = False
            self.simulator.setup_viewer()
            ###########################################################################
            # Jiawei: Should be removed
            ###########################################################################
            self.viewer = self.simulator.viewer
            
        # 【关键】初始化所有用于存储强化学习状态和反馈的 PyTorch Tensor 缓冲区
        self._init_buffers()

        ###########################################################################
        #### Jiawei: Should be removed
        ###########################################################################
        # self.gym = self.simulator.gym
        # self.sim = self.simulator.sim
        if self.headless == False:
            self.viewer = self.simulator.viewer


    """
    【函数作用】：初始化各类 PyTorch 张量（Buffer）以存储仿真数据。
    【RL知识补充】：为了最大化 GPU 效率，Isaac Gym 等框架会预先分配好连续的 GPU 内存（张量）。
                  RL 循环中的 Observation（观测）、Reward（奖励）、Done（终止标志）都会在这些预先分配好的张量上直接原地修改（In-place update），避免内存反复分配的开销。
    【被谁调用】：由自身的 `__init__` 函数调用。
    """
    def _init_buffers(self):
        # 字典，用于存放各种观测张量（可能包括本体感受 proprioception、视觉、特权信息等）
        self.obs_buf_dict = {}
        # 奖励张量，维度为 (num_envs,)，记录当前步每个环境获得的奖励得分
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # 重置张量，1 表示该环境需要重置（比如机器人摔倒了，或者时间到了），0 表示继续
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        # 记录每个环境当前已经运行了多少个 RL 时间步（用于判断是否超时 Time out）
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # 记录每个环境是否是因为“超时”而结束的（处理截断 Truncation 时需要，详见 PPO 算法中对超时的 value bootstrapping）
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # 存放一些额外信息的字典（常用于 logging，比如记录本次 episode 中的平均速度、总违规惩罚等）
        self.extras = {}
        # 用于向 Tensorboard/Wandb 记录日志数据的字典
        self.log_dict = {}

    """
    【函数作用】：从物理引擎中刷新并获取最新的状态张量。
    【RL知识补充】：每次物理引擎 `step` 前进之后，底层的刚体位置、速度等数据发生了改变。
                  调用此函数，将底层的最新物理数据同步到 PyTorch 能够访问的张量中，供我们在计算 Reward 和 Observation 时使用。
    【被谁调用】：通常被子类（如具体的行走任务类）的 `step()` 函数，在完成物理动作施加后调用。
    """
    def _refresh_sim_tensors(self):
        self.simulator.refresh_sim_tensors()
        return

    """
    【函数作用】：强制重置所有的环境（将所有机器人拉回初始状态）。
    【RL知识补充】：通常在开始训练（Episode 0）或者在评估/测试（Testing）阶段之前调用。
                  需要将机器人的根节点（位置、姿态）以及所有关节（角度、角速度）恢复到默认值。
    【被谁调用】：通常被外部的训练循环（Training Loop）在初始化或者评估时调用。
    """
    def reset_all(self):
        """ Reset all robots"""
        # 生成一个包含所有环境 ID 的张量 [0, 1, 2, ..., num_envs-1]
        # 并调用底层的方法重置它们的状态（位置、速度等）。初始状态都是从配置文件(并非，模仿学习中会新增或重写相关逻辑，改成从动作库抽取)中读取的，没有域随机化
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))#这句对所有4096个环境进行重置
        
        # 将机器人的基座（Root）状态写入仿真器
        self.simulator.set_actor_root_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.all_root_states)
        # 将机器人的各个关节（DOF）状态写入仿真器
        self.simulator.set_dof_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.dof_state)
        # self._refresh_env_idx_tensors(torch.arange(self.num_envs, device=self.device))
        
        # 构造一个全0的动作输入（因为刚重置，还没有网络输出动作）
        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actor_state = {}
        actor_state["actions"] = actions
        
        # 驱动仿真器走一步，从而获得刚重置后的第一个环境观测值（Observation）
        # 返回的是初始状态的观测，提供给策略网络进行第一次推理
        #for i in range(1000):
        obs_dict, _, _, _ = self.step_for_reset_all(actor_state)
        return obs_dict
        
        # self._compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        # # 获取最新的观测向量 (计算 Obs)
        
        # self._post_compute_observations_callback()

        # # return clipped obs, clipped states (None), rewards, dones and infos
        # clip_obs = self.config.normalization.clip_observations
        # for obs_key, obs_val in self.obs_buf_dict.items():#obs_buf_dict是按照字母顺序排列好的观测值字典，2个键是actor_obs和critic_obs,2个值是对应的观测张量列表
        #     self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs) # 同样限制观测值的极值防爆

        # for key in self.history_handler.history.keys():
        #     self.history_handler.add(key, self.hist_obs_dict[key]) # 将当前状态hist_obs_dict推入历史序列，供未来使用

        # self.extras["to_log"] = self.log_dict
        # if self.viewer:
        #     self._setup_simulator_control()
        #     self._setup_simulator_next_task()
        #     if self.debug_viz:
        #         self._draw_debug_vis()

        # return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    
    # （被注释掉的旧代码，说明了底层是通过 gym 的 API 更新状态的）
    # def _refresh_env_idx_tensors(self, env_ids):
    #     env_ids_int32 = env_ids.to(dtype=torch.int32)
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                 gymtorch.unwrap_tensor(self.all_root_states),
    #                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    #     self.gym.set_dof_state_tensor_indexed(self.sim,
    #                                             gymtorch.unwrap_tensor(self.dof_state),
    #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    """
    【函数作用】：渲染一帧图像到屏幕上。
    【RL知识补充】：RL 训练完全可以在后台无头（headless）全速运行。但在 Debug 或者看训练效果展示（Sim-to-Real 的可视化评估）时，需要把物理世界渲染出来。
    【被谁调用】：通常在评估（Evaluate/Play）脚本中被 `while True` 的控制循环每步调用。
    """
    def render(self, sync_frame_time=True):
        if self.viewer:
            self.simulator.render(sync_frame_time)

    ###########################################################################
    #### Helper functions
    ###########################################################################
    
    """
    【函数作用】：计算并设置这成百上千个机器人在虚拟世界里的“出生点”（初始三维坐标）。
    【RL知识补充】：
        - 如果是平地环境，机器人会排成一个整齐的网格阵列（Grid）。
        - 如果是复杂地形（Terrain），这里使用了【课程学习 Curriculum Learning】的思想。
          地形网格按照难度递增排列，机器人的出生点取决于它的表现水平。
    【被谁调用】：由自身的 `__init__` 函数调用。
    """
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        # 如果启用了复杂地形（基于高度图或三角网格）
        if self.config.terrain.mesh_type in ["heightfield", "trimesh"]:
            # import ipdb; ipdb.set_trace()
            self.custom_origins = True
            # 初始化一个 (num_envs, 3) 的张量存放 xyz 坐标
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            
            # 最大初始地形难度级别
            max_init_level = self.config.terrain.max_init_terrain_level
            # 如果没有开启课程学习（动态升级），则允许出生在任何地形难度级别
            if not self.config.terrain.curriculum: max_init_level = self.config.terrain.num_rows - 1
            
            # 随机给每个机器人分配一个出生难度级别 (terrain_levels) 和 地形种类 (terrain_types，比如有的是楼梯，有的是斜坡)
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.config.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.config.terrain.num_rows
            
            # 获取由仿真器生成的地形网格原点数据
            if isinstance(self.simulator.terrain.env_origins, np.ndarray):
                self.terrain_origins = torch.from_numpy(self.simulator.terrain.env_origins).to(self.device).to(torch.float)
            else:
                self.terrain_origins = self.simulator.terrain.env_origins.to(self.device).to(torch.float)   
                
            # 根据分配的难度和种类，到预生成的地形网格中查表得到具体的 XYZ 坐标
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            # import ipdb; ipdb.set_trace()
            # print(self.terrain_origins.shape)
            
        else:
            # 如果不需要复杂地形，就在一片平地上创建一个正方形阵列（Grid）
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            
            # 计算每行每列需要多少个机器人（开平方根）
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            # 生成网格的横纵坐标索引
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            
            # 机器人之间的间距
            spacing = self.config.env_spacing
            
            # 计算具体的 X 和 Y 坐标，Z坐标（高度）设为 0
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    """
    【函数作用】：加载机器人 3D 物理资产文件（通常是 URDF 格式）。
    【RL知识补充】：策略网络（Policy）输出的动作 $a_t$ 通常对应机器人的各个驱动关节。这里需要统计机器人一共有多少个关节（DOFs）、多少个刚体部件（Bodies），并进行维度的对齐检查。同时也构建机器人的初始姿态（位置、旋转、速度）。
    【被谁调用】：由自身的 `__init__` 函数调用。
    """
    def _load_assets(self):
        # 通知底层的物理引擎去加载 URDF 文件解析刚体和关节
        self.simulator.load_assets()
        # 将底层解析出来的 自由度数量、刚体数量、自由度名称列表、刚体名称列表 提取到当前类中
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = self.simulator.num_dof, self.simulator.num_bodies, self.simulator.dof_names, self.simulator.body_names
        
        # check dimensions
        # 安全断言：神经网络输出的动作维度，必须等于机器人的自由度（关节数量）。如果不对齐，后面的驱动控制就会崩溃。
        assert self.num_dof == self.dim_actions, "Number of DOFs must be equal to number of actions"

        # other properties
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        
        # 组装机器人的初始状态向量。包含：
        # pos(位置x,y,z) + rot(旋转四元数/欧拉角) + lin_vel(线速度) + ang_vel(角速度)
        base_init_state_list = self.config.robot.init_state.pos + self.config.robot.init_state.rot + self.config.robot.init_state.lin_vel + self.config.robot.init_state.ang_vel
        # 将这个列表转换为 PyTorch 张量并存入 GPU
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

    """
    【函数作用】：通过调用仿真器，正式在物理引擎中批量创建强化学习的并行环境（实例化机器人）。
    【RL知识补充】：这是 Sim2Real 非常核心的一步。Isaac Gym 之所以快，就是因为它可以一次性向物理系统提交 `num_envs` 个创建请求，共享相同的碰撞体和材质属性。
    【被谁调用】：由自身的 `__init__` 函数调用。
    """
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # env_config = self.config
        # 传入环境数量、出生点张量、机器人初始状态张量，创建环境
        self.simulator.create_envs(self.num_envs, 
                                    self.env_origins, 
                                    self.base_init_state)

    """
    【函数作用】：根据配置文件的规则，找到机器人身体各个关键部件在物理引擎内部的编号（Index）。
    【RL知识补充】：在设计 Reward Function（奖励函数）和 Termination Condition（终止条件）时极其重要。
                  例如：
                  - 我们需要奖励“脚（Foot）”接触地面，那我们必须知道“脚”这个刚体的 index，才能去获取它的接触力。
                  - 如果“膝盖（Knee）”或者“躯干（Torso）”碰到了地面，说明机器人摔倒了（Termination），应该给一个大的惩罚并重置环境。
    【被谁调用】：由自身的 `__init__` 函数调用。
    """
    def _setup_robot_body_indices(self):
        # 利用列表推导式，从刚体名称列表 body_names 中筛选出匹配配置文件所定义的脚部名字的字符串
        feet_names = [s for s in self.body_names if self.config.robot.foot_name in s]#self.body_names和self.dof_names是仿真器解析URDF后生成的列表赋值的，和仿真器中完全一样，顺序也一样
        # 筛选膝盖名字
        knee_names = [s for s in self.body_names if self.config.robot.knee_name in s]
        
        # 筛选那些“如果发生碰撞（比如碰撞到其他腿或身体），就给予扣分（Penalize）”的刚体部件
        penalized_contact_names = []
        for name in self.config.robot.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
            
        # 筛选那些“一旦发生碰撞（如躯干着地），就直接结束回合（Terminate）”的刚体部件
        termination_contact_names = []
        for name in self.config.robot.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        # 初始化用于存放脚部索引的张量，去仿真器查询对应刚体名字的数字索引并存入
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.simulator.find_rigid_body_indice(feet_names[i])
        
        # 查膝盖的索引
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.simulator.find_rigid_body_indice(knee_names[i])

        # 查需要被惩罚的碰撞体索引
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.simulator.find_rigid_body_indice(penalized_contact_names[i])

        # 查导致终止的碰撞体索引
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.simulator.find_rigid_body_indice(termination_contact_names[i])
            
        # 如果是具有上半身关节的机器人（比如人形机器人 Humanoid）
        # 需要区分上肢和下肢的关节。在有些控制策略中，下肢用来保持平衡行走，上肢可能执行其他手臂抓取任务
        if self.config.robot.has_upper_body_dof:
            # maintain upper/lower dof idxs
            self.upper_dof_names = self.config.robot.upper_dof_names
            self.lower_dof_names = self.config.robot.lower_dof_names
            # 在所有关节名字列表中查出对应的数字索引
            self.upper_dof_indices = [self.dof_names.index(dof) for dof in self.upper_dof_names]
            self.lower_dof_indices = [self.dof_names.index(dof) for dof in self.lower_dof_names]
            # 腰部关节索引
            self.waist_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.waist_dof_names]

        # 如果机器人有明确的躯干定义（用于计算机器人的质心高度、倾斜角度或者基础线速度奖励）
        if self.config.robot.has_torso:
            self.torso_name = self.config.robot.torso_name
            self.torso_index = self.simulator.find_rigid_body_indice(self.torso_name)

    """
    【函数作用】：一个用于设置可视化实体的占位函数（Placeholder）。
    【RL知识补充】：在实际编写代码时，开发者可能希望继承这个基类后重写此方法，以在渲染界面上画出期望的速度向量箭头、机器人的目标点（Goal position）等信息。
    【被谁调用】：由自身的 `__init__` 函数调用。
    """
    def setup_visualize_entities(self):
        pass