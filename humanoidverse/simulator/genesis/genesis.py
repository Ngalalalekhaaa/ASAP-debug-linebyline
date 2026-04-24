import sys
import os
from loguru import logger # 导入强大的日志打印库
# from isaacgym import gymtorch, gymapi, gymutil # 原始基于 IsaacGym 的导入，由于现在换了 Genesis，被注释掉了
import torch # 导入 PyTorch，强化学习的核心库，仿真器里的所有状态数据都是张量 (Tensor) 格式，直接在 GPU 上和神经网络交互
import genesis as gs # 导入 Genesis 物理引擎核心库
gs.init(backend=gs.gpu) # 初始化 Genesis 后端，强制使用 GPU。GPU 并行化是目前机器人 RL 能在几小时内训出复杂动作的关键
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver # 导入刚体求解器（用于计算机器人的碰撞、受力、运动学）
from humanoidverse.simulator.genesis.tmp_gs_utils import * # 导入 HumanoidVerse 框架写的一些 Genesis 工具函数（比如四元数转换等）
from humanoidverse.simulator.genesis.genesis_viewer import Viewer # 导入可视化渲染器（用于在屏幕上看到机器人）
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float # 导入一些 PyTorch 辅助工具
import numpy as np # 导入 NumPy，用于 CPU 端的数组计算
from termcolor import colored # 导入终端颜色打印
from rich.progress import Progress # 导入进度条展示库
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator # 导入基类。HumanoidVerse 支持多种仿真器（IsaacGym, IsaacSim, Genesis, MuJoCo），它们都继承自这个基类以保证 API 统一
import copy # 导入深拷贝库，防止修改配置字典时影响原数据

class Genesis(BaseSimulator):
    """
    【RL 知识背景】：
    这个类是 RL 环境（Environment）和底层物理引擎（Physics Engine）之间的桥梁。
    RL 环境的 `step()` 和 `reset()` 函数底层都会调用这个类里的方法。
    它负责把 RL 输出的抽象 Action 转化为真实的电机力矩，并把物理引擎算出来的真实坐标转化为 RL 需要的 Observation。
    """
    def __init__(self, config, device):
        """
        【函数作用】：初始化 Genesis 仿真器实例，读取物理参数、机器人参数，设定运行设备。
        【被谁调用】：通常被上层环境类（如 `legged_robot_base.py` 或 `motion_tracking.py`）在实例化环境时调用。
        
        Args:
            config (dict): 包含所有 RL 实验设置的字典（比如重力、摩擦力、重置条件等）。
            device (str): 运行设备 ('cpu' 或 'cuda:0')。RL 通常在 'cuda' 上运行以避免 CPU-GPU 通讯延迟。
        """
        self.cfg = config # 保存全局配置
        self.sim_cfg = config.simulator.config # 专门保存仿真器相关的配置（如时间步长 dt）
        self.robot_cfg = config.robot # 保存机器人本体的配置（如 URDF 路径，初始位姿）
        self.device = device # 记录 PyTorch 张量应该存放在哪个设备上
        self.sim_device = device # 仿真器计算设备
        self.headless = False # 默认不开启无头模式（即默认会弹出可视化窗口）
        # gs.init(backend=gs.gpu if 'cuda' in self.device else gs.cpu) # 动态初始化后端（已被移到文件开头）
        # from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


    # ----- Configuration Setup Methods -----

    def set_headless(self, headless):
        """
        【函数作用】：设置是否在无头模式（Headless）下运行。开启无头模式意味着关闭图形渲染界面。
        【RL 知识背景】：在服务器集群上训练模型时，由于没有显示器，且渲染极其消耗显卡算力，我们必须开启无头模式，让显卡 100% 专注于物理计算和神经网络反向传播。只有在本地 Debug 或验证（Eval）时才设为 False。
        【被谁调用】：上层环境初始化时，根据命令行参数（如 `headless=True`）调用。
        """
        self.headless = headless # 赋值给类的内部变量

    def setup(self):
        """
        【函数作用】：初始化物理仿真场景 (Scene)，定义世界的基本物理法则（如时间步长、控制频率、碰撞检测）。
        【RL 知识背景】：RL 是离散步进的。`fps`（物理刷新率）通常是 200Hz 或更高的内部积分频率，而 `control_decimation`（控制降采样）决定了神经网络多久输出一次动作。例如物理算 4 步，网络才出 1 步，这叫 Action Repeat。
        【被谁调用】：在 `load_assets` 之前，由上层环境类调用以搭建物理世界的舞台。
        """
        # 计算物理引擎每次计算的时间间隔 dt (例如 fps=200，则 dt=0.005 秒)
        self.sim_dt = 1 / self.sim_cfg.sim.fps 
        self.sim_substeps = self.sim_cfg.sim.substeps # 每个 dt 内部再细分的积分步数，用于提高物理计算精度（防止穿模）
        
        # create scene (创建一个 Genesis 物理场景)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions( # 仿真核心选项
                dt=self.sim_dt,
                substeps=self.sim_substeps,
            ),
            viewer_options=gs.options.ViewerOptions( # 渲染窗口选项
                max_FPS=int(1 / self.sim_dt * self.sim_cfg.sim.control_decimation), # 限制屏幕刷新率，不浪费算力
                camera_pos=(2.0, 0.0, 2.5), # 默认摄像机位置
                camera_lookat=(0.0, 0.0, 0.5), # 默认摄像机视角中心
                camera_fov=40, # 视野广度
            ),
            rigid_options=gs.options.RigidOptions( # 刚体动力学选项
                enable_self_collision=True, # 开启自碰撞（防止机器人的手穿过自己的身体，对于全身动作/摔倒模拟非常重要）
            ),
            # vis_options=gs.options.VisOptions(
            #     n_rendered_envs=1, # 原来用来控制只渲染一个环境（节省算力），被注释掉了
            # ),
            show_viewer=not self.headless, # 如果不是无头模式，就展示窗口
            show_FPS=False, # 不在窗口上显示帧率
        )

        # 遍历场景中的求解器，找出并保存刚体求解器 RigidSolver，后续用来读取和修改刚体状态
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

    # ----- Terrain Setup Methods -----

    def setup_terrain(self, mesh_type):
        """
        【函数作用】：配置地形。
        【RL 知识背景】：为了让机器人学会适应复杂环境，RL 会采用“地形课程学习（Terrain Curriculum）”。从简单的平地开始，慢慢加入上坡、下坡、台阶。本函数根据配置在物理引擎中生成对应的地形网格（Mesh）。
        【被谁调用】：由上层环境（`terrain.py` 相关逻辑）在生成场景时调用。
        """
        if mesh_type == 'plane': # 如果指定地形是平地
            # this is somehow deprecated (这一段是用旧方法加载 URDF 平地，已弃用)
            # self.scene.add_entity(
            #     gs.morphs.URDF(file='urdf/plane/plane.urdf', scale=20.0, fixed=True),
            # )
            plane = self.scene.add_entity(gs.morphs.Plane()) # 使用 Genesis 原生接口添加一个无限大的刚体平面
        elif mesh_type == 'trimesh': # 如果是复杂的不规则三角网格地形（如随机生成的坑洼）
            # 当前未在 Genesis 子类中实现，抛出异常
            raise NotImplementedError(f"Mesh type {mesh_type} hasn't been implemented in genesis subclass.")

    # ----- Robot Asset Setup Methods -----

    def load_assets(self):
        """
        【函数作用】：把机器人的 3D 模型（URDF 文件）加载到物理引擎中，并建立极为重要的“关节/刚体索引映射（Index Mapping）”。
        【RL 知识背景】：非常关键的一步！不同的物理引擎（如 IsaacGym, Genesis, MuJoCo）在解析同一个 URDF 文件时，给关节排的编号顺序可能是不同的！但 RL 神经网络输出的 Action 向量必须对应固定的电机（比如第0个数字必须控制左胯）。因此，我们需要提取引擎中的顺序，并与我们期待的顺序进行映射。
        【被谁调用】：由上层环境类在 `setup_terrain` 之后调用。
        """
        # 读取配置中的机器人初始旋转状态（四元数）
        init_quat_xyzw = self.robot_cfg.init_state.rot 
        # 将配置的 (x, y, z, w) 格式转换为 Genesis 引擎默认要求的 (w, x, y, z) 格式
        init_quat_wxyz = init_quat_xyzw[-1:] + init_quat_xyzw[:3] 
        
        # 将初始位置转化为 GPU 张量
        self.base_init_pos = torch.tensor(
            self.robot_cfg.init_state.pos, device=self.device
        )
        # self.base_init_pos[2] += 1.5 # 可以在这里硬编码抬高机器人初始高度，让它掉下来，这有助于数据增强
        
        # 将初始姿态转化为 GPU 张量
        self.base_init_quat = torch.tensor(
            init_quat_wxyz, device=self.device
        )

        # 拼接 URDF 文件的完整路径
        asset_root = self.robot_cfg.asset.asset_root # 比如 humanoidverse/data/robots/
        asset_file = self.robot_cfg.asset.urdf_file  # 比如 g1/g1_29dof_anneal_23dof.urdf
        asset_path = os.path.join(asset_root, asset_file)

        # 使用 Genesis 接口真正将机器人实体（Entity）添加到物理场景中
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=asset_path,
                merge_fixed_links=True, # 优化性能：把用 Fixed 关节连接的零件合并成一个刚体，减少求解器负担
                links_to_keep=self.robot_cfg.body_names, # 必须保留的刚体名称（比如手和脚，因为我们需要计算它们的坐标作为奖励）
                pos=self.base_init_pos.cpu().numpy(), # 设置初始位置
                quat=self.base_init_quat.cpu().numpy(), # 设置初始姿态
            ),
            visualize_contact=False, # 不在画面上画出碰撞点（会卡顿）
        )
        # （这是你上传代码里附带的结构图说明，非常形象，原封不动保留）
# ┌─────────────────────────────────────────┐
# │ Genesis Scene (gs.Scene)                │
# │ ├─ Plane (地面)                         │
# │ └─ Entity: self.robot                   │
# │    ├─ Links (刚体)                      │
# │    │  ├─ link_0 (躯干)                  │
# │    │  ├─ link_1 (左腿)                  │
# │    │  └─ ...                            │
# │    └─ Joints (关节)                     │
# │       ├─ joint_0 (DOF[6-7])            │
# │       ├─ joint_1 (DOF[8-9])            │
# │       └─ ...                            │
# └─────────────────────────────────────────┘
        # 深拷贝预定义的关节名称列表（也就是我们期待的、RL 网络对应的标准顺序）
        dof_names_list = copy.deepcopy(self.robot_cfg.dof_names)

        # 获取 Genesis 解析出来的刚体名称列表
        self.genesis_link_names = [link.name for link in self.robot.links]
        # 获取 HumanoidVerse 框架中定义的刚体名称列表（标准顺序）
        self.humanoidverse_link_names = self.robot_cfg.body_names
        
        # 【核心逻辑】：构建映射数组。计算出“框架期待的刚体”在“Genesis实际刚体列表”中的索引位置。
        # 后面在提取状态（如刚体线速度）时，通过这个数组切片，就能把打乱的顺序重新理顺回标准顺序。
        self.link_mapping_genesis_to_humanoidverse_idx = [self.genesis_link_names.index(name) for name in self.humanoidverse_link_names]
        
        # 将关节名称转换为在 Genesis 中对应的内部索引 ID
        self.dof_ids = [
            self.robot.get_joint(name).dof_idx_local
            for name in dof_names_list
        ]

        # 保存一些基本统计信息供外部环境调用
        self.body_names = self.robot_cfg.body_names
        self.num_bodies = len(self.body_names)                # 刚体总数
        self.dof_names = dof_names_list
        self.num_dof = len(dof_names_list)                    # 自由度（关节）总数，也就是 Action 的维度

    # ----- Environment Creation Methods -----

    def find_rigid_body_indice(self, body_name):
        """
        【函数作用】：根据刚体名称（如 "left_hand"），查找它在物理引擎中对应的索引编号。
        【RL 知识背景】：在写 Reward Function（奖励函数）时经常用到。比如我们要奖励机器人“手的位置接近参考轨迹的手的位置”，就需要用这个索引去底层的 state tensor 里把手的坐标提取出来。
        【被谁调用】：由上层环境类在初始化奖励函数配置时调用。

        Args:
            body_name (str): 刚体的名字
        Returns:
            int: 刚体在当前机器人实体中的局部索引
        """
        for link in self.robot.links:
            flag = False # 无用变量，大概是之前写代码留下的
            if body_name in link.name:
                return link.idx - self.robot.link_start # 返回相对于机器人自身的局部索引（因为全局索引还包含了环境里的其他物体如地面）

    def create_envs(self, num_envs, env_origins, base_init_state):
        """
        【函数作用】：在 GPU 上“克隆”出成千上万个平行的环境，这是 GPU 物理加速的灵魂步骤。
        【RL 知识背景】：PPO 等 RL 算法极其吃数据。用传统的 CPU 引擎一次只能跑 1 个机器人，而在 Genesis 中，调用 `build` 后，会在一个巨大的扁平内存张量里同时模拟 4096 个互不干扰的机器人。
        【被谁调用】：由上层环境类调用，传入需要生成的环境数量。

        Args:
            num_envs (int): 要并行的环境数量 (如 4096)。
            env_origins (list): 每个环境在 3D 空间中的中心原点坐标（排成网格状，防止机器人互相撞到）。
            base_init_state (array): 基座的初始状态。
        """
        # build
        self.num_envs = num_envs # 记录环境数量
        self._body_list = [link.name for link in self.robot.links] # 记录链接名称列表
        
        # （一段废弃代码注释）
        # build a body list that are compatible with the robot_config
        # self._body_list = []
        # for body_name in self.robot_cfg.body_names:
        #     self._body_list.append(self._genesis_body_list[body_name])
        
        # 核心：调用 Genesis 的 build 方法，传入 n_envs，物理引擎在显存中开辟空间，真正开始就绪！
        self.scene.build(n_envs=num_envs)
        
        self.env_origins = env_origins # 记录各环境原点
        self.base_init_state = base_init_state # 记录初始状态

        return None, None # 返回两个 None（可能是为了兼容旧的 IsaacGym 接口需要返回 actor_handles 和 env_ptrs）

    # ----- Property Retrieval Methods -----

    def get_dof_limits_properties(self):
        """
        【函数作用】：获取机器人的物理限制属性（关节角度上限/下限、最大速度、最大输出力矩）。
        【RL 知识背景】：
            1. 截断 (Clipping)：如果 RL 网络输出的目标角度超出了物理极限，会引发真实电机保护甚至损坏，所以底层控制前要卡住范围。
            2. 惩罚 (Penalty)：如果机器人运动过猛逼近极限，我们在 Reward 中可以加一个 Penalty (惩罚项) 来抑制它，这叫 Soft Limits。
            3. 缩放 (Scaling)：神经网络通常输出 [-1, 1] 的数值，我们需要知道 `torque_limits` 才能把它按比例放大成真实的牛米 (Nm)。
        【被谁调用】：由环境类在初始化结束前调用，提取并保存到 `env` 实例中。
        """
        # 初始化一堆形状为 (num_dof, ) 或 (num_dof, 2) 的张量，放在 GPU 上
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        
        # 遍历每个关节，从配置中读取极限值并填入 Tensor 中
        for i in range(self.num_dof):
            # 记录硬极限（机械死区）
            self.hard_dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            
            # 准备记录软极限
            self.dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            
            # 最大速度和力矩
            self.dof_vel_limits[i] = self.robot_cfg.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_cfg.dof_effort_limit_list[i]
            
            # 计算软极限 (Soft Limits)：在原本的极限区间内再收缩一点比例（比如 0.9），用于在 Reward 里惩罚靠近极限的动作
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2 # 找到区间中点
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0] # 计算区间跨度
            # 上下限按比例收缩
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.reward_limit.soft_dof_pos_limit
            
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits # 返回这三个张量

    # ----- Simulation Preparation and Refresh Methods -----

    def prepare_sim(self):
        """
        【函数作用】：在 RL 循环真正开始之前，试运行物理引擎（进行一次步进），以确保所有物理张量被正确填充。
        【RL 知识背景】：如果在物理引擎还没“热身”出有效数据时就去取 Observation，全都是 0 的张量会导致网络第一步产生错误输出。
        【被谁调用】：在整个框架 `setup` 的最后一步调用。
        """
        # while True: # 这原本可能是为了让机器人自己掉到地上稳住写的预热代码，被注释了
        #     self.scene.step()
        self.scene.step() # 让 Genesis 物理引擎前向计算一个步长 (dt)

        # 获取机器人的基座绝对位置 [num_envs, 3]
        self.base_pos = self.robot.get_pos()
        # 获取基座姿态 [num_envs, 4]，Genesis 返回的是 (w,x,y,z)
        base_quat = self.robot.get_quat()
        # 将 (w,x,y,z) 切片重组为 (x,y,z,w)，这是绝大多数 RL 代码库（包括 HumanoidVerse）采用的标准格式
        self.base_quat = base_quat[..., [1, 2, 3, 0,]]

        # （一段被注释的代码，原本试图利用四元数把全局速度转换成基座局部坐标系下的速度）
        # inv_base_quat = gs_inv_quat(base_quat)
        # self.base_lin_vel = gs_transform_by_quat(self.robot.get_vel(), inv_base_quat)
        # self.base_ang_vel = gs_transform_by_quat(self.robot.get_ang(), inv_base_quat)
        
        # 直接获取全局线速度和角速度
        self.base_lin_vel = self.robot.get_vel()
        self.base_ang_vel = self.robot.get_ang()

        # 将位置、姿态、线速度、角速度拼接成一个长向量：Root State (根状态)，维度为 3+4+3+3=13
        self.all_root_states = torch.cat(
            [
                self.base_pos,
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
            ], dim=-1
        )
        self.robot_root_states = torch.cat(
            [
                self.base_pos,
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
            ], dim=-1
        )

        # 根据前面记录的 dof_ids 顺序，从物理引擎拉取关节实际角度 (q) 和速度 (dq)
        self.dof_pos = self.robot.get_dofs_position(self.dof_ids)
        self.dof_vel = self.robot.get_dofs_velocity(self.dof_ids)

        # 获取所有刚体受到的接触力总和（用于判断是否摔倒，或者计算脚踩地的接触惩罚）
        self.contact_forces = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )



    def refresh_sim_tensors(self):
        """
        【函数作用】：核心的数据抓取函数！每执行完一次动作后，从仿真引擎内存中“刷新”并读取机器人最新的一帧状态。
        【RL 知识背景】：这是构建 RL 中 $s_{t+1}$（下一刻观测值 Observation）的数据源。PPO 等算法的 `compute_observations()` 方法会严重依赖这里更新的张量。如果在这一步没拿到正确的数据，智能体就“瞎”了。
        【被谁调用】：通常被仿真器的 `simulate_at_each_physics_step` 之后立刻调用，或直接在环境的 `step()` 逻辑中调用。
        """
        # 更新基座全局位置
        self.base_pos = self.robot.get_pos()
        # 更新基座全局四元数姿态，并转换为 xyzw 格式
        base_quat = self.robot.get_quat()
        self.base_quat = base_quat[..., [1, 2, 3, 0,]]

        # 更新全局线速度和角速度
        self.base_lin_vel = self.robot.get_vel()
        self.base_ang_vel = self.robot.get_ang()

        print(f"genesis.py中在refresh_sim_tensor中使用api读取的根节点位置base_pos{self.base_pos}")
        print(f"genesis.py中在refresh_sim_tensor中使用api读取的根节点角速度base_ang_vel{self.base_ang_vel}")



        # 拼接更新 13 维的 Root State 张量
        self.all_root_states = torch.cat(
            [
                self.base_pos,
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
            ], dim=-1
        )
        self.robot_root_states = self.all_root_states

        # 更新所有电机的实际位置和速度
        self.dof_pos = self.robot.get_dofs_position(self.dof_ids)
        self.dof_vel = self.robot.get_dofs_velocity(self.dof_ids)

        # 更新刚体受到的环境接触力
        self.contact_forces = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )
        
        # 【重要映射操作】：获取所有刚体（末端、大腿等）的位姿和速度。
        # 这里用到了在 `load_assets` 里算出来的 link_mapping_genesis_to_humanoidverse_idx！
        # 物理引擎返回的列表是打乱的，我们利用 [:, mapping_idx] 的高级切片语法，在 GPU 显存级别
        # 将刚体数据瞬间重组为 HumanoidVerse 期待的顺序，保证神经网络读入的末端特征（如左手、右脚）永远在固定的索引上。
        self._rigid_body_pos = self.robot.get_links_pos()[:, self.link_mapping_genesis_to_humanoidverse_idx]
        
        self._rigid_body_rot = self.robot.get_links_quat()[:, self.link_mapping_genesis_to_humanoidverse_idx]  
        # 再次处理姿态四元数：从 wxyz 变为 xyzw 格式，保持全库一致
        self._rigid_body_rot = self._rigid_body_rot[..., [1, 2, 3, 0]]
        
        self._rigid_body_vel = self.robot.get_links_vel()[:, self.link_mapping_genesis_to_humanoidverse_idx]
        self._rigid_body_ang_vel = self.robot.get_links_ang()[:, self.link_mapping_genesis_to_humanoidverse_idx]

    # ----- Control Application Methods -----

    def apply_torques_at_dof(self, torques):
        """
        【函数作用】：把计算好的目标力矩施加到虚拟电机的关节上。
        【RL 知识背景】：这对应了马尔可夫决策过程（MDP）中的“执行动作 (Action)”。在基于力矩控制（Torque Control）或计算完 PD 控制的前馈力之后，RL 环境会将这个力的张量传给仿真引擎执行。
        【被谁调用】：由环境类 `LeggedRobotBase.step()` 在执行动作指令时调用。

        Args:
            torques (tensor): 要施加的关节力矩张量，形状为 (num_envs, num_dof)。
        """
        # 调用 Genesis 原生接口，为特定的由 self.dof_ids 指定的关节下发控制力
        self.robot.control_dofs_force(torques, self.dof_ids)

    
    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        """
        【函数作用】：强行“瞬间移动”机器人基座到指定的状态（位置、姿态、速度）。
        【RL 知识背景】：强化学习是一个“不断摔倒又不断重来”的过程。当一个回合（Episode）结束时，或者我们要使用动作捕捉数据进行“状态参考初始化 (RSI)”时，我们需要不经过物理计算，强行重置基座状态，这就是 Reset 的核心。
        【被谁调用】：在环境的 `reset_idx()` 或者重采样轨迹时调用。

        Args:
            set_env_ids (tensor): 哪些环境（比如第 5, 17, 39 号并行环境）需要被重置。
            root_states (tensor): 并没有被实际用上的参数（看下面代码，被自己覆盖了，这可能是原作者留下的小 Bug，但靠读取自身状态进行了修复）。
        """
        # 这里用自身状态重新组装了一下，然后取出了对应 set_env_ids 的状态进行覆盖。
        # 理论上应该是外部传来的 root_states，但由于这里的逻辑覆盖，它表现为将特定的环境强行设定为其缓存的目标状态。
        root_states = torch.cat(
            [
                self.base_pos,
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
            ], dim=-1
        )
        root_states = self.robot_root_states[set_env_ids]

        # 拆解出新的位姿和速度参数。注意四元数重新转回了 Genesis 需要的 wxyz 格式 ([6, 3, 4, 5])。
        base_pos = root_states[..., :3]
        base_quat = root_states[..., [6, 3, 4, 5]]
        base_lin_vel = root_states[..., 7:10]
        base_ang_vel = root_states[..., 10:13]

        print(f"genesis.py中的根节点位置base_pos{base_pos}，马上用api赋给真正仿真器")
        print(f"genesis.py中的根节点角速度base_ang_vel{base_ang_vel}，马上用api赋给真正仿真器")

        # reset root states - position (调用底层方法重置)
        # zero_velocity=False 代表不仅设位置，连速度也要设过去（这在初始化抛掷机器人时很重要）
        self.robot.set_pos(
            base_pos, zero_velocity=False, envs_idx=set_env_ids
        )
        self.robot.set_quat(
            base_quat, zero_velocity=False, envs_idx=set_env_ids
        )
        self.robot.set_dofs_velocity(
            base_lin_vel, dofs_idx_local=[0, 1, 2],  envs_idx=set_env_ids
        )
        self.robot.set_dofs_velocity(
            base_ang_vel, dofs_idx_local=[3, 4, 5],  envs_idx=set_env_ids
        )
    
    def set_dof_state_tensor(self, set_env_ids, dof_states):
        """
        【函数作用】：强行“瞬间掰动”机器人的所有关节到指定的角度和角速度。
        【RL 知识背景】：和重置基座同理。让机器人强行摆出特定的姿势准备开启下一个回合（Episode Reset）。
        【被谁调用】：由环境的 `reset_idx()` 配合 `set_actor_root_state_tensor` 一起调用。

        Args:
            set_env_ids (tensor): 需要被重置的环境索引。
            dof_states (tensor): 包含新关节角度和速度的张量。
        """
        # 将扁平的张量重新塑形 (Reshape) 为 [num_envs, num_dof, 2]，提取目标位置和目标速度
        dof_pos = dof_states.view(self.num_envs, -1, 2)[set_env_ids, :, 0]
        dof_vel = dof_states.view(self.num_envs, -1, 2)[set_env_ids, :, 1]

        # reset dofs (重置关节物理状态)
        self.robot.set_dofs_position(
            position=dof_pos,
            dofs_idx_local=self.dof_ids,
            envs_idx=set_env_ids,
        )
        self.robot.set_dofs_velocity(
            velocity=dof_vel,
            dofs_idx_local=self.dof_ids,
            envs_idx=set_env_ids,
        )

    def simulate_at_each_physics_step(self):
        """
        【函数作用】：指挥物理引擎往前积分模拟一个步长 (dt)。
        【RL 知识背景】：这是 RL 推进时间的最核心代码！当 Action 转换成力矩被设置好之后，调用这个函数，物理引擎就会计算 $F=ma$，算碰撞，解算位置，推进虚拟世界的时间。
        【被谁调用】：由环境的 `step()` 方法调用。一次网络推理可能对应调用此函数 `sim_substeps` 次。
        """
        self.scene.step()
        # self.refresh_sim_tensors() # （原注释掉了刷新张量，因为目前架构是推演完所有控制周期后，统一再外面调一次 refresh）

    # ----- Viewer Setup and Rendering Methods -----

    def setup_viewer(self):
        """
        【函数作用】：初始化可视化组件（如果你没有开启 headless 模式）。
        【RL 知识背景】：人类检查 RL 训练是否有效的唯一直观方式就是看画面。Viewer 就是渲染这些画面的工具。
        【被谁调用】：在整个框架 `setup` 期间，判断需要渲染时调用。
        """
        self.viewer = Viewer() # 实例化自定义或原生的 Viewer 类

    def render(self, sync_frame_time=True):
        """
        【函数作用】：渲染最新一帧的画面到屏幕上。
        【RL 知识背景】：如果在推理 (Eval) 时不加 `sync_frame_time` 限制频率，仿真会以几万帧每秒运行，人类肉眼根本看不清机器人动作。
        【被谁调用】：由环境主循环 `env.step()` 或评估器 `algo.evaluate_policy()` 每隔几个 timestep 调用一次。

        Args:
            sync_frame_time (bool): 是否同步真实时间（控制渲染速度等于真实物理流逝速度）。
        """
        # 由于 Genesis 在调用 self.scene.step() 且 show_viewer=True 时会在内部自动处理渲染更新，
        # 所以这里的原生 render 方法为空 (Pass)。
        return

    @property
    def dof_state(self):
        """
        【函数作用】：Python 的 Property 属性，用于快速返回当前所有电机的角度和角速度组合张量。
        【RL 知识背景】：方便上层提取并构造 Observation 数组。
        """
        # This will always use the latest dof_pos and dof_vel (保证使用最新抓取的数据)
        # 将角度和速度增加一个维度并拼接在一起返回
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)
    
    def add_visualize_entities(self, num_visualize_markers):
        """
        【函数作用】：在场景中加入一定数量的小球（Marker），作为调试用的画笔。
        【RL 知识背景】：极其有用的 Debug 功能！比如你要练 C罗的动作，你可以通过调用这个函数把人类示范动作中手和脚的“虚拟目标位置”用发光的球画在空间中，看机器人的真手有没有去追随它。
        【被谁调用】：通常被专门写好的可视化 Callback 回调函数初始化时调用。
        """
        # self.scene.add_entity(gs.morphs.Sphere())
        self.visualize_entities = [] # 初始化存放小球的列表
        for i in range(num_visualize_markers):
            # 添加半径为 4cm 的球，visualization=True 表示这只是一团光影，collision=False 表示它没有体积，机器人可以穿过它而不被绊倒
            self.visualize_entities.append(self.scene.add_entity(gs.morphs.Sphere(radius=0.04, visualization=True, collision=False)))
    
     # debug visualization
    def clear_lines(self):
        """
        【函数作用】：清除画面上的辅助线/辅助球。
        【被谁调用】：每渲染新的一帧之前调用，清除上一帧的残影。
        """
        # self.scene.clear_debug_objects()
        pass # 由于用的是持续存在的实体小球，所以无需特殊的 clear 函数（每次只是移动球的位置）

    def draw_sphere(self, pos, radius, color, env_id, pos_id=0):
        """
        【函数作用】：移动并显示 Debug 小球到指定位置。
        【被谁调用】：在 `render` 期间或 `step` 期间由追踪模块调用。
        """
        # self.scene.draw_debug_sphere(pos, radius, color)
        # 找到对应编号的小球，更新它的全局坐标
        # 多环境时必须指定 envs_idx，否则 Genesis 会按 scene.n_envs 校验第一维，导致 shape 不匹配
        self.visualize_entities[pos_id].set_pos(pos.reshape(1, 3), envs_idx=[env_id])

    def draw_line(self, start_point, end_point, color, env_id):
        """
        【函数作用】：画线。通常用于可视化力矢量（比如地面对脚反作用力的方向）。
        """
        pass # 当前子类尚未实现