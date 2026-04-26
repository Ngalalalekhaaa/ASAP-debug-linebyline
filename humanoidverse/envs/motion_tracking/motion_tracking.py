import torch # 强化学习张量计算核心库
import numpy as np # CPU数值计算库
from pathlib import Path # 路径处理
import os # 系统文件操作
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase # 导入步行机器人的基类（处理基本的物理引擎交互、奖励框架）
from isaac_utils.rotations import ( # 导入一堆四元数（Quaternion）计算工具，机器人的姿态全靠四元数算
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_conjugate,
    quat_to_angle_axis,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw
)
# from isaacgym import gymtorch, gymapi, gymutil # (旧的IsaacGym接口，已废弃)
from humanoidverse.envs.env_utils.visualization import Point # 可视化点位工具

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree # 骨骼树，用于解析机器人连杆的层级关系

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot # 核心类：动作库！负责加载、解析、插值 .pkl 动捕数据

from termcolor import colored # 终端彩色打印
from loguru import logger # 高级日志打印库

from scipy.spatial.transform import Rotation as sRot # 用于四元数和旋转向量/欧拉角之间的转换
import joblib # 用于读取或保存 .pkl 数据集

# ==============================================================================================
# 类定义：腿部机器人动作追踪环境
# 继承自 LeggedRobotBase，专为“模仿学习/动作追踪”任务设计。
# ==============================================================================================
class LeggedRobotMotionTracking(LeggedRobotBase):
    def __init__(self, config, device):
        """
        【函数作用】：环境的初始化入口。
        【RL 背景】：搭建整个追踪训练的舞台。加载参考动作、设置虚拟追踪点、设定终止条件。
        【被谁调用】：通常被 eval_agent.py 或 train_agent.py 通过 hydra.utils.instantiate() 动态实例化时调用。
        """
        self.init_done = False # 标记初始化未完成
        self.debug_viz = True # 开启 Debug 可视化（画面里会画出追踪点的小球）
        
        super().__init__(config, device) # 调用父类初始化（这里会创建物理引擎、几千个并行环境等）
        
        self._init_motion_lib() # 初始化动作库（把 .pkl 数据加载进显存）
        self._init_motion_extend() # 初始化扩展点（比如在手掌心虚拟出一个点来精确追踪）
        self._init_tracking_config() # 初始化追踪配置（定义哪些点算上半身，哪些算下半身）

        self.init_done = True # 标记初始化完成
        self.debug_viz = True

        self._init_save_motion() # 如果配置了要保存动作，初始化相关的显存缓冲区

        # 如果开启了遥操作（Teleoperation，比如用苹果 Vision Pro 真人控制机器人）
        if self.config.use_teleop_control:
            # 预分配存放头显和手柄 3 个空间坐标点的张量
            self.teleop_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
            import rclpy # 导入 ROS2 Python 接口
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
            # 启动一个 ROS2 节点，订阅 Vision Pro 发来的坐标数据，实时赋给机器人当目标
            self.node = Node("motion_tracking")
            self.teleop_sub = self.node.create_subscription(Float64MultiArray, "vision_pro_data", self.teleop_callback, 1)

        # 【课程学习 (Curriculum Learning) 终止条件设置】
        # 如果机器人动作太离谱（和参考动作差太远），就直接终止回合 (Terminate)，不要浪费时间乱跑。
        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            # 初始容忍度：一开始让它随便动，只要不差得十万八千里就不重置
            self.terminate_when_motion_far_threshold = self.config.termination_curriculum.terminate_when_motion_far_initial_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

        else:
            # 固定容忍度
            self.terminate_when_motion_far_threshold = self.config.termination_scales.termination_motion_far_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

    def teleop_callback(self, msg):
        """
        【函数作用】：ROS2 的回调函数，接收人穿戴 VR 时的姿态数据。
        【被谁调用】：由 ROS2 底层事件循环自动调用。
        """
        self.teleop_marker_coords = torch.tensor(msg.data, device=self.device)

    def _init_save_motion(self):
        """
        【函数作用】：初始化用来“录像存盘”的数据结构。
        【RL 背景】：用于生成前置的模仿学习数据集，或者用来分析训练出来的策略是否逼真。
        【被谁调用】：自己类的 __init__。
        """
        if "save_motion" in self.config:
            self.save_motion = self.config.save_motion
            if self.save_motion:
                os.makedirs(Path(self.config.ckpt_dir) / "motions", exist_ok = True)
                
                # 决定保存的文件名
                if hasattr(self.config, 'dump_motion_name'):
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / (str(self.config.eval_timestamp) + "_" + self.config.dump_motion_name)
                else:
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / f"{self.config.save_note}_{self.config.eval_timestamp}"
                self.save_motion = True
                self.num_augment_joint = len(self.config.robot.motion.extend_config)
                # 初始化一个大字典，用来存盘（注意这里存的东西和我们之前跑 Sim2Sim 数据集的很像）
                self.motions_for_saving = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'actor_obs':[], 'action':[], 'terminate':[],
                                            'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[]}
                self.motion_times_buf = []
                self.start_save = False
        else:
            self.save_motion = False

    def _init_motion_lib(self):
        """
        【函数作用】：把 .pkl 数据集加载进系统的硬盘或显存中，作为“标准答案库”。
        【被谁调用】：自己类的 __init__。
        """
        self.config.robot.motion.step_dt = self.dt
        self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        
        # 评估(Eval)时不打乱动作，训练(Train)时随机打乱各个环境播放的动作
        if self.is_evaluating:
            self._motion_lib.load_motions(random_sample=False)
        else:
            self._motion_lib.load_motions(random_sample=True)
            
        # 让所有机器人立刻在参考动作的随机时间点“重生”（RSI 初始化）
        res = self._resample_motion_times(torch.arange(self.num_envs))
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions

    def _init_tracking_config(self):
        """
        【函数作用】：读取配置，把字符串名称（比如 "left_hand"）转成物理引擎里对应的索引数字。
        【被谁调用】：自己类的 __init__。
        """
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]
        if self.config.resample_motion_when_training:
            # 算出过多少个 step 需要强制重置换下一个动作
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)
        
    def _init_motion_extend(self):
        """
        【函数作用】：构建“虚拟标记点（Markers）”。
        【RL 背景】：有时候单纯追踪关节角度不够，比如拿剑时剑尖的位置。我们可以通过相对父连杆的位移 (pos) 和旋转 (rot)，凭空定义一些虚拟追踪点。
        【被谁调用】：自己类的 __init__。
        """
        if "extend_config" in self.config.robot.motion:
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.robot.motion.extend_config:
                extend_parent_ids.append(self.simulator._body_list.index(extend_config["parent_name"]))
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
                self.simulator._body_list.append(extend_config["joint_name"])

            # 保存这些虚拟点的父连杆 ID 和相对位姿张量到 GPU
            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.num_extend_bodies = len(extend_parent_ids)

            # 预分配张量用来存放这些扩展点在物理世界里的实际坐标和参考坐标
            self.marker_coords = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.dif_global_body_pos = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def start_compute_metrics(self):
        self.compute_metrics = True
        self.start_idx = 0
    
    def forward_motion_samples(self):
        pass
    
    def _init_buffers(self):
        """
        【函数作用】：初始化仿真过程中用来记录各种状态时间的 Tensor 缓冲区。
        【被谁调用】：在父类的 __init__ 过程中被触发。
        """
        super()._init_buffers()
        self.vr_3point_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.realtime_vr_keypoints_pos = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.realtime_vr_keypoints_vel = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) 
        
        # motion_ids: 记录这 4096 个环境分别正在学哪一个动作（如 0 号动作是跳，1 号是跑）
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        # 记录每个环境当前这个动作开始播放的时间截（比如从动画的第 0.5 秒开始截断播放）
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        # 记录分配给该环境的这个动作的总长
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        
    def _init_domain_rand_buffers(self):
        super()._init_domain_rand_buffers()
        self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def _reset_tasks_callback(self, env_ids):
        """
        【函数作用】：当有机器人摔倒或者超时，触发环境重置时的回调。
        【被谁调用】：由父类的 env.step() 内部判断哪些 env_ids 挂了，然后调用这里重置它们。
        """
        if len(env_ids) == 0:
            return
        super()._reset_tasks_callback(env_ids)
        # 给这些摔倒重生的机器人重新随机抽取一个“动作库”里的时间起点（这就是传说中的 RSI）
        self._resample_motion_times(env_ids) 
        # 如果容忍度随着训练动态变严（课程学习），在这里更新它
        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self._update_terminate_when_motion_far_curriculum()
    
    def _update_terminate_when_motion_far_curriculum(self):
        """
        【函数作用】：自适应课程学习 (Curriculum Learning)。
        【RL 背景】：如果策略现在普遍能活得很久（average_episode_length很大），就收紧“跟丢”的容忍度，要求它追踪得更准；如果天天摔跤，就放宽容忍度，让它先学会大概怎么动。
        【被谁调用】：每次有机器人重置时被上面调用。
        """
        assert self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum
        if self.average_episode_length < self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            # 活得太短了，放宽阈值
            self.terminate_when_motion_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        elif self.average_episode_length > self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            # 活得挺长，收紧阈值
            self.terminate_when_motion_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        # 卡死在最大和最小限制内
        self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_max)

    def _update_tasks_callback(self):
        """
        【函数作用】：定期强制大洗牌。如果设定了时间间隔，不管机器人摔没摔，强制重新抽取一遍动作目标。
        """
        super()._update_tasks_callback()
        if self.config.resample_motion_when_training:
            if self.common_step_counter % self.resample_time_interval == 0:
                logger.info(f"Resampling motion at step {self.common_step_counter}")
                self.resample_motion()

    def set_is_evaluating(self):
        super().set_is_evaluating()

    def _check_termination(self):
        """
        【函数作用】：判断本回合是否玩砸了该立刻终结（Done = True）。
        【RL 背景】：除了倒地算失败，在模仿学习中，如果机器人的姿势 (dif_global_body_pos) 和参考动画差得太离谱（超过了 threshold），就算模仿失败，直接拉闸重练。
        【被谁调用】：由父类的步进循环调用。
        """
        super()._check_termination()
        if self.config.termination.terminate_when_motion_far:
            # 计算所有追踪点的位置差值的 L2 范数（欧氏距离），只要有一个点偏差大于阈值，就标记为重置
            reset_buf_motion_far = torch.any(torch.norm(self.dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)

            # 诊断结论：save_motion 评估时会在 step0 被 motion_far 立即清空，导致首生命无任何可保存帧。
            # 因此在“评估+存动作”模式下，跳过 motion_far 终止，仅保留其它终止条件（超时/接触/姿态等）。
            skip_motion_far_for_saving = getattr(self, "save_motion", False)
            if not skip_motion_far_for_saving:
                self.reset_buf |= reset_buf_motion_far # 管道合并：把动作跟丢标记加入总重置标志中
            
            
        if self.reset_buf.sum() > 0:
            # print(self.reset_buf)
            print("全部self.episode_sums['mpjpe']取值:", self.episode_sums["mpjpe"])
            print("self.episode_sums['mpjpe']的平均值:", self.episode_sums["mpjpe"].mean())

            # 把当前的容忍度记录下来画图表
            if self.config.termination_curriculum.terminate_when_motion_far_curriculum:
                self.log_dict["terminate_when_motion_far_threshold"] = torch.tensor(self.terminate_when_motion_far_threshold, dtype=torch.float)

    def _update_timeout_buf(self):
        """
        【函数作用】：判断动作片段是否播完了。播完了算 Timeout（自然结束），不算摔倒（Penalty），处理逻辑和摔倒是不一样的。
        """
        super()._update_timeout_buf()
        if self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.time_out_buf |= current_time > self.motion_len

    def next_task(self):
        """
        【函数作用】：在评估模式 (Eval) 下，按顺序播放并测试下一个动捕片段。
        【被谁调用】：我们之前看过的 `on_press` 函数（按下键盘 'n' 时）。
        """
        self.motion_start_idx += self.num_envs
        if self.motion_start_idx >= self.num_motions:
            self.motion_start_idx = 0
        self._motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
        self.reset_all()

    def _resample_motion_times(self, env_ids):
        """
        【函数作用】：RSI (参考状态初始化) 的核心抽样函数。为刚重置的机器人从动画库中随机抽取一个起始时间戳。
        """
        if len(env_ids) == 0:
            return
        # 拿到这批动作的总长度
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        # print(f"重置动作的长度self.motion_len[env_ids]={self.motion_len[env_ids]}")
        # 如果是评估测试，一律从头 (0.0 秒) 开始播
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
            # print(f"动作起始时间self.motion_start_times[env_ids]={self.motion_start_times[env_ids]}")
        else:
            # 如果是训练，随机抽一个时间点 (比如动画总长 3 秒，随机抽在 1.5 秒出生)
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])

    def resample_motion(self):
        """重新随机分配整个批次的动作任务"""
        self._motion_lib.load_motions(random_sample=True)
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))


    def _pre_compute_observations_callback(self):
            """
            【函数作用】：强化学习每一步的心脏！它在物理引擎演算完后、提取神经网络 Observation 前执行。
            它的主要工作是计算参考动画在此刻“应该是什么姿势”，并与机器人的“实际姿势”作差，算出误差 (dif)。
            【被谁调用】：由基类的 env.step() 底层调用。
            """
            # 1. 基础物理状态更新
            super()._pre_compute_observations_callback()
            
            offset = self.env_origins # 获取环境矩阵网格的物理偏移系
            B = self.motion_ids.shape[0] # 环境总数
            
            # 【前瞻性 (Future Window)】
            # episode_length_buf + 1：我们提取的是“下一帧”参考动作。告诉神经网络“你下一步该往哪走”，而不是“你这步该在哪”。
            motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
            
            # 从 C罗 的动捕数据库里，插值计算出此时刻机器人的理想完美状态
            motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

    #         motion_res={'root_pos': torch.tensor([[ 0.0035, -0.2307,  0.9065]], device='cuda:0'), 'root_rot': torch.tensor([[-0.1065, -0.0585,  0.7433, -0.6578]], device='cuda:0'), 'dof_pos': torch.tensor([[-0.1040,  0.0146,  0.2722,  0.0424,  0.0014,  0.0000, -0.2361, -0.1315,
    #      -0.0481,  0.0927,  0.0043,  0.0000,  0.2081, -0.0671, -0.2738,  0.2576,
    #       0.3911, -0.4247,  0.9946,  0.2217, -0.3084,  0.2673,  0.9876]],
    #    device='cuda:0'), 'root_vel': torch.tensor([[-0.0055,  0.0255,  0.0135]], device='cuda:0'), 'root_ang_vel': torch.tensor([[ 0.0263, -0.0059, -0.0780]], device='cuda:0'), 'dof_vel': torch.tensor([[-0.0275, -0.0459,  0.0235, -0.0290, -0.0019,  0.0000, -0.0083,  0.0066,
    #       0.0007,  0.0030, -0.0002,  0.0000, -0.0481, -0.0216, -0.0235, -0.0966,
    #      -0.1212, -0.0390, -0.0361, -0.0507,  0.1070,  0.0176, -0.0074]],
    #    device='cuda:0'), 'motion_aa': torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    #    device='cuda:0'), 'motion_bodies': torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    #    device='cuda:0'), 'rg_pos': torch.tensor([[[ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 7.5702e-02, -2.1562e-01,  8.1025e-01],
    #      [ 1.2931e-01, -2.1844e-01,  7.8286e-01],
    #      [ 1.3375e-01, -2.4932e-01,  6.6016e-01],
    #      [ 1.3582e-01, -1.8039e-01,  4.7898e-01],
    #      [ 1.4475e-01, -1.3092e-01,  1.8321e-01],
    #      [ 1.4527e-01, -1.2799e-01,  1.6591e-01],
    #      [-5.1962e-02, -1.9915e-01,  8.0340e-01],
    #      [-1.0185e-01, -1.9266e-01,  7.7022e-01],
    #      [-1.1767e-01, -2.3682e-01,  6.5261e-01],
    #      [-1.2329e-01, -1.8890e-01,  4.6486e-01],
    #      [-1.3775e-01, -1.5824e-01,  1.6677e-01],
    #      [-1.3858e-01, -1.5637e-01,  1.4933e-01],
    #      [ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 2.8940e-04, -2.3482e-01,  9.4133e-01],
    #      [ 1.0468e-05, -2.3903e-01,  9.5985e-01],
    #      [ 9.0632e-02, -2.2231e-01,  1.2009e+00],
    #      [ 1.3045e-01, -2.1532e-01,  1.1998e+00],
    #      [ 1.7575e-01, -1.8997e-01,  1.1104e+00],
    #      [ 2.0312e-01, -1.8493e-01,  1.0332e+00],
    #      [-1.0887e-01, -2.4026e-01,  1.1938e+00],
    #      [-1.4909e-01, -2.4097e-01,  1.1898e+00],
    #      [-1.8375e-01, -2.2634e-01,  1.0934e+00],
    #      [-2.0160e-01, -2.2924e-01,  1.0134e+00]]], device='cuda:0'), 'rb_rot': torch.tensor([[[-0.1065, -0.0585,  0.7433, -0.6578],
    #      [-0.0677, -0.0243,  0.7479, -0.6600],
    #      [-0.0070,  0.0390,  0.7506, -0.6595],
    #      [ 0.0016, -0.0396, -0.6542,  0.7553],
    #      [ 0.0726,  0.0426, -0.6502,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [-0.0182,  0.0194,  0.7507, -0.6601],
    #      [ 0.0904,  0.0275,  0.7529, -0.6514],
    #      [ 0.0897,  0.0297,  0.7683, -0.6331],
    #      [-0.0136, -0.0550,  0.7734, -0.6314],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [ 0.1120,  0.0472, -0.6710,  0.7315],
    #      [ 0.0874,  0.0696, -0.6690,  0.7348],
    #      [-0.0047, -0.0313, -0.6747,  0.7374],
    #      [ 0.1825, -0.0300, -0.6457,  0.7408],
    #      [ 0.2236, -0.0661, -0.6431,  0.7295],
    #      [-0.2325,  0.0175,  0.7824, -0.5775],
    #      [ 0.5776,  0.2602, -0.5767,  0.5159],
    #      [-0.0322,  0.1433, -0.6801,  0.7182],
    #      [-0.0427,  0.1532, -0.6780,  0.7177],
    #      [-0.0219,  0.1575, -0.5763,  0.8016],
    #      [ 0.2538,  0.5186, -0.5178,  0.6312]]], device='cuda:0'), 'body_vel': torch.tensor([[[-0.0055,  0.0255,  0.0135],
    #      [-0.0032,  0.0214,  0.0146],
    #      [-0.0032,  0.0166,  0.0150],
    #      [-0.0097,  0.0174,  0.0146],
    #      [-0.0107,  0.0188,  0.0151],
    #      [-0.0149,  0.0096,  0.0135],
    #      [-0.0151,  0.0090,  0.0134],
    #      [-0.0016,  0.0331,  0.0137],
    #      [-0.0007,  0.0379,  0.0134],
    #      [-0.0029,  0.0401,  0.0128],
    #      [ 0.0040,  0.0419,  0.0131],
    #      [ 0.0113,  0.0468,  0.0133],
    #      [ 0.0118,  0.0471,  0.0133],
    #      [-0.0055,  0.0255,  0.0135],
    #      [-0.0061,  0.0249,  0.0134],
    #      [-0.0065,  0.0245,  0.0133],
    #      [-0.0041,  0.0158,  0.0130],
    #      [-0.0030,  0.0093,  0.0126],
    #      [-0.0093, -0.0076,  0.0048],
    #      [-0.0176, -0.0202,  0.0012],
    #      [-0.0065,  0.0425,  0.0134],
    #      [-0.0065,  0.0474,  0.0133],
    #      [ 0.0041,  0.0480,  0.0096],
    #      [ 0.0109,  0.0470,  0.0082]]], device='cuda:0'), 'body_ang_vel': torch.tensor([[[ 2.6257e-02, -5.9151e-03, -7.7960e-02],
    #      [ 8.6228e-03, -3.2692e-03, -7.9365e-02],
    #      [ 1.2471e-02,  2.9898e-02, -8.1405e-02],
    #      [ 1.1354e-02,  3.1757e-02, -5.2361e-02],
    #      [-2.7502e-02,  2.5645e-02, -5.3717e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [ 7.3476e-03, -3.5769e-03, -7.9060e-02],
    #      [ 6.4085e-03, -1.1238e-02, -7.8942e-02],
    #      [ 6.6325e-03, -9.6965e-03, -7.6678e-02],
    #      [ 1.2110e-02, -1.0904e-02, -7.6448e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 3.0077e-02,  3.9392e-03, -1.2007e-01],
    #      [ 3.0118e-02,  3.7528e-03, -1.2054e-01],
    #      [-1.7129e-02, -1.0014e-04, -1.2165e-01],
    #      [-9.8071e-02, -8.3816e-03, -1.4892e-01],
    #      [-1.1705e-01,  1.0668e-01, -1.2682e-01],
    #      [-1.0299e-01,  1.1524e-01, -1.5953e-01],
    #      [ 4.6978e-02, -4.2387e-02,  1.0965e-01],
    #      [-4.3933e-02, -2.2689e-03, -1.1495e-01],
    #      [-4.1613e-02, -8.3487e-02, -1.2895e-01],
    #      [-4.1718e-02, -8.3550e-02, -1.2975e-01],
    #      [-1.1604e-02, -7.3208e-02, -1.3703e-01]]], device='cuda:0'), 'rg_pos_t': torch.tensor([[[ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 7.5702e-02, -2.1562e-01,  8.1025e-01],
    #      [ 1.2931e-01, -2.1844e-01,  7.8286e-01],
    #      [ 1.3375e-01, -2.4932e-01,  6.6016e-01],
    #      [ 1.3582e-01, -1.8039e-01,  4.7898e-01],
    #      [ 1.4475e-01, -1.3092e-01,  1.8321e-01],
    #      [ 1.4527e-01, -1.2799e-01,  1.6591e-01],
    #      [-5.1962e-02, -1.9915e-01,  8.0340e-01],
    #      [-1.0185e-01, -1.9266e-01,  7.7022e-01],
    #      [-1.1767e-01, -2.3682e-01,  6.5261e-01],
    #      [-1.2329e-01, -1.8890e-01,  4.6486e-01],
    #      [-1.3775e-01, -1.5824e-01,  1.6677e-01],
    #      [-1.3858e-01, -1.5637e-01,  1.4933e-01],
    #      [ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 2.8940e-04, -2.3482e-01,  9.4133e-01],
    #      [ 1.0468e-05, -2.3903e-01,  9.5985e-01],
    #      [ 9.0632e-02, -2.2231e-01,  1.2009e+00],
    #      [ 1.3045e-01, -2.1532e-01,  1.1998e+00],
    #      [ 1.7575e-01, -1.8997e-01,  1.1104e+00],
    #      [ 2.0312e-01, -1.8493e-01,  1.0332e+00],
    #      [-1.0887e-01, -2.4026e-01,  1.1938e+00],
    #      [-1.4909e-01, -2.4097e-01,  1.1898e+00],
    #      [-1.8375e-01, -2.2634e-01,  1.0934e+00],
    #      [-2.0160e-01, -2.2924e-01,  1.0134e+00],
    #      [ 2.5299e-01, -2.5856e-01,  7.9952e-01],
    #      [-2.2016e-01, -3.2685e-01,  7.8401e-01],
    #      [-1.6733e-02, -2.1839e-01,  1.3790e+00]]], device='cuda:0'), 'rg_rot_t': torch.tensor([[[-0.1065, -0.0585,  0.7433, -0.6578],
    #      [-0.0677, -0.0243,  0.7479, -0.6600],
    #      [-0.0070,  0.0390,  0.7506, -0.6595],
    #      [ 0.0016, -0.0396, -0.6542,  0.7553],
    #      [ 0.0726,  0.0426, -0.6502,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [-0.0182,  0.0194,  0.7507, -0.6601],
    #      [ 0.0904,  0.0275,  0.7529, -0.6514],
    #      [ 0.0897,  0.0297,  0.7683, -0.6331],
    #      [-0.0136, -0.0550,  0.7734, -0.6314],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [ 0.1120,  0.0472, -0.6710,  0.7315],
    #      [ 0.0874,  0.0696, -0.6690,  0.7348],
    #      [-0.0047, -0.0313, -0.6747,  0.7374],
    #      [ 0.1825, -0.0300, -0.6457,  0.7408],
    #      [ 0.2236, -0.0661, -0.6431,  0.7295],
    #      [-0.2325,  0.0175,  0.7824, -0.5775],
    #      [ 0.5776,  0.2602, -0.5767,  0.5159],
    #      [-0.0322,  0.1433, -0.6801,  0.7182],
    #      [-0.0427,  0.1532, -0.6780,  0.7177],
    #      [-0.0219,  0.1575, -0.5763,  0.8016],
    #      [ 0.2538,  0.5186, -0.5178,  0.6312],
    #      [ 0.5776,  0.2602, -0.5767,  0.5159],
    #      [ 0.2538,  0.5186, -0.5178,  0.6312],
    #      [-0.0047, -0.0313, -0.6747,  0.7374]]], device='cuda:0'), 'body_vel_t': torch.tensor([[[-0.0055,  0.0255,  0.0135],
    #      [-0.0032,  0.0214,  0.0146],
    #      [-0.0032,  0.0166,  0.0150],
    #      [-0.0097,  0.0174,  0.0146],
    #      [-0.0107,  0.0188,  0.0151],
    #      [-0.0149,  0.0096,  0.0135],
    #      [-0.0151,  0.0090,  0.0134],
    #      [-0.0016,  0.0331,  0.0137],
    #      [-0.0007,  0.0379,  0.0134],
    #      [-0.0029,  0.0401,  0.0128],
    #      [ 0.0040,  0.0419,  0.0131],
    #      [ 0.0113,  0.0468,  0.0133],
    #      [ 0.0118,  0.0471,  0.0133],
    #      [-0.0055,  0.0255,  0.0135],
    #      [-0.0061,  0.0249,  0.0134],
    #      [-0.0065,  0.0245,  0.0133],
    #      [-0.0041,  0.0158,  0.0130],
    #      [-0.0030,  0.0093,  0.0126],
    #      [-0.0093, -0.0076,  0.0048],
    #      [-0.0176, -0.0202,  0.0012],
    #      [-0.0065,  0.0425,  0.0134],
    #      [-0.0065,  0.0474,  0.0133],
    #      [ 0.0041,  0.0480,  0.0096],
    #      [ 0.0109,  0.0470,  0.0082],
    #      [-0.0593, -0.0560,  0.0039],
    #      [ 0.0159,  0.0447,  0.0088],
    #      [-0.0034,  0.0328,  0.0130]]], device='cuda:0'), 'body_ang_vel_t': torch.tensor([[[ 2.6257e-02, -5.9151e-03, -7.7960e-02],
    #      [ 8.6228e-03, -3.2692e-03, -7.9365e-02],
    #      [ 1.2471e-02,  2.9898e-02, -8.1405e-02],
    #      [ 1.1354e-02,  3.1757e-02, -5.2361e-02],
    #      [-2.7502e-02,  2.5645e-02, -5.3717e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [ 7.3476e-03, -3.5769e-03, -7.9060e-02],
    #      [ 6.4085e-03, -1.1238e-02, -7.8942e-02],
    #      [ 6.6325e-03, -9.6965e-03, -7.6678e-02],
    #      [ 1.2110e-02, -1.0904e-02, -7.6448e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 3.0077e-02,  3.9392e-03, -1.2007e-01],
    #      [ 3.0118e-02,  3.7528e-03, -1.2054e-01],
    #      [-1.7129e-02, -1.0014e-04, -1.2165e-01],
    #      [-9.8071e-02, -8.3816e-03, -1.4892e-01],
    #      [-1.1705e-01,  1.0668e-01, -1.2682e-01],
    #      [-1.0299e-01,  1.1524e-01, -1.5953e-01],
    #      [ 4.6978e-02, -4.2387e-02,  1.0965e-01],
    #      [-4.3933e-02, -2.2689e-03, -1.1495e-01],
    #      [-4.1613e-02, -8.3487e-02, -1.2895e-01],
    #      [-4.1718e-02, -8.3550e-02, -1.2975e-01],
    #      [-1.1604e-02, -7.3208e-02, -1.3703e-01],
    #      [ 4.6978e-02, -4.2387e-02,  1.0965e-01],
    #      [-1.1604e-02, -7.3208e-02, -1.3703e-01],
    #      [-1.7129e-02, -1.0014e-04, -1.2165e-01]]], device='cuda:0')}
            
            # 拆包：理想位置、速度、姿态。主要读的是刚体和关节参考状态
            ref_body_pos_extend = motion_res["rg_pos_t"]            
            self.ref_body_pos_extend[:] = ref_body_pos_extend       
            ref_body_vel_extend = motion_res["body_vel_t"]          
            self.ref_body_rot_extend = ref_body_rot_extend = motion_res["rg_rot_t"] 
            ref_body_ang_vel_extend = motion_res["body_ang_vel_t"]  
            ref_joint_pos = motion_res["dof_pos"]                   
            ref_joint_vel = motion_res["dof_vel"]                   


            # -----------------------------------------------------------------
            # 【正向运动学计算 Forward Kinematics (FK)】
            # 算真实的机器人那几个额外虚拟挂载点（比如剑尖、掌心）现在在空间什么位置。
            # -----------------------------------------------------------------
            # 1. 把虚拟点相对于手腕的偏置坐标，用手腕现在的实际空间姿态(四元数)旋转出来
            rotated_pos_in_parent = my_quat_rotate(#虚拟点相对于父连杆的位置，在世界坐标系下的坐标
                self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                self.extend_body_pos_in_parent.reshape(-1, 3)
            )
            # 2. 加上手腕本身的绝对坐标，得出虚拟点的世界绝对坐标
            extend_curr_pos = my_quat_rotate(
                self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                rotated_pos_in_parent
            ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
            
            # 3. 把原有的身体节点和新增的虚拟节点拼起来
            self._rigid_body_pos_extend = torch.cat([self.simulator._rigid_body_pos, extend_curr_pos], dim=1)

            # (同理：计算这些虚拟点的旋转、角速度、线速度)
            extend_curr_rot = quat_mul(self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                                        self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                                        w_last=True).view(self.num_envs, -1, 4)
            self._rigid_body_rot_extend = torch.cat([self.simulator._rigid_body_rot, extend_curr_rot], dim=1)
            
            self._rigid_body_ang_vel_extend = torch.cat([self.simulator._rigid_body_ang_vel, self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]], dim=1)#
        
            self._rigid_body_ang_vel_global = self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]
            # 虚拟线速度的计算：手腕线速度 + 手腕旋转带来的线速度贡献 (w x r 叉乘)
            angular_velocity_contribution = torch.cross(self._rigid_body_ang_vel_global, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2)
            extend_curr_vel = self.simulator._rigid_body_vel[:, self.extend_body_parent_ids] + angular_velocity_contribution.view(self.num_envs, -1, 3)
            self._rigid_body_vel_extend = torch.cat([self.simulator._rigid_body_vel, extend_curr_vel], dim=1)

            # -----------------------------------------------------------------
            # 【核心：计算误差 (Tracking Errors)】
            # 目标就是让这些误差趋近于 0。
            # -----------------------------------------------------------------
            self.dif_global_body_pos = ref_body_pos_extend - self._rigid_body_pos_extend # 空间坐标误差
            self.dif_global_body_rot = quat_mul(ref_body_rot_extend, quat_conjugate(self._rigid_body_rot_extend, w_last=True), w_last=True) # 旋转误差（四元数差就是乘共轭）
            self.dif_global_body_vel = ref_body_vel_extend - self._rigid_body_vel_extend # 线速度误差
            self.dif_global_body_ang_vel = ref_body_ang_vel_extend - self._rigid_body_ang_vel_extend # 刚体角速度误差
            self.dif_joint_angles = ref_joint_pos - self.simulator.dof_pos # 底层电机关节误差
            self.dif_joint_velocities = ref_joint_vel - self.simulator.dof_vel # 关节速度误差

            # -----------------------------------------------------------------
            # 【非常重要：观测值局部化 (Heading Invariance Transform)】
            # 如果不把目标转换到机器人自身的前方(Local)，机器人在面朝北学会的动作，在面朝南时就完全不知道怎么做了。
            # -----------------------------------------------------------------
            self.marker_coords[:] = ref_body_pos_extend.reshape(B, -1, 3) # 留给 UI 画参考点用

            env_batch_size = self.simulator._rigid_body_pos.shape[0]#环境数
            num_rigid_bodies = self.simulator._rigid_body_pos.shape[1]#刚体数（不包括扩展点的原因？？？：扩展点是虚拟的，不参与物理仿真，所以它们的坐标不受基座朝向影响）

            # 提取基座朝向(Heading)的逆旋转矩阵
            heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
            heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)#所有刚体（包括扩展点）的朝向逆旋转四元数，准备批量应用于坐标变换

            # 提取正向 Heading (用于其他可能需要的坐标变换)
            heading_rot = calc_heading_quat(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
            heading_rot_expand = heading_rot.unsqueeze(1).expand(-1, num_rigid_bodies, -1).reshape(-1, 4)

            # 1. 局部刚体位置误差观测 (让网络知道：我的手离目标的手，在我自己的坐标系下偏左还是偏右)。将世界坐标系下与下一步目标位置坐标差向量，统一转为机器人当前坐标系下的误差向量。无论目标动作到哪里，这个计算结果是各个刚体在机器人当前位置下相对于自身需要努力的方向。
            dif_global_body_pos_for_obs_compute = ref_body_pos_extend.view(env_batch_size, -1, 3) - self._rigid_body_pos_extend.view(env_batch_size, -1, 3)
            dif_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), dif_global_body_pos_for_obs_compute.view(-1, 3))
            self._obs_dif_local_rigid_body_pos = dif_local_body_pos_flat.view(env_batch_size, -1) 

            # 2. 局部参考动作绝对位置 (让网络知道：下一步的目标定在哪)。下一步目标各个刚体的位置，在机器人自身坐标系下相对于机器人当前根节点的向量。
            global_ref_rigid_body_pos = ref_body_pos_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)  
            local_ref_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_pos.view(-1, 3))
            self._obs_local_ref_rigid_body_pos = local_ref_rigid_body_pos_flat.view(env_batch_size, -1) 

            # 3. 局部参考动作速度 
            global_ref_body_vel = ref_body_vel_extend.view(env_batch_size, -1, 3)#如果ref_body_vel_extend是世界坐标系下的绝对速度，那么转化后也是当前机器人朝向下的世界坐标系速度。如果ref_body_vel_extend是相对于机器人根节点的速度，那么转化后就是当前机器人朝向下的相对速度。
            local_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_vel.view(-1, 3))
            self._obs_local_ref_rigid_body_vel = local_ref_rigid_body_vel_flat.view(env_batch_size, -1) 

            # 如果接了真实世界的头盔追踪设备，要把真实的头盔坐标转化为局部的观测给网络看
            if not self.config.use_teleop_control:
                ref_vr_3point_pos = ref_body_pos_extend.view(env_batch_size, -1, 3)[:, self.motion_tracking_id, :]
            else:
                ref_vr_3point_pos = self.teleop_marker_coords
                
            vr_2root_pos = (ref_vr_3point_pos - self.simulator.robot_root_states[:, 0:3].view(env_batch_size, 1, 3))
            heading_inv_rot_vr = heading_inv_rot.repeat(3,1)
            self._obs_vr_3point_pos = my_quat_rotate(heading_inv_rot_vr.view(-1, 4), vr_2root_pos.view(-1, 3)).view(env_batch_size, -1)
            
            # 【相位 (Phase)】非常重要！告诉网络：现在你在放这段武术的百分之几了（0表示刚起手，1表示收招完）。有助于策略周期性记忆。
            self._ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
            self._ref_motion_phase = motion_times / self._ref_motion_length 
            
            # print(f"{self._ref_motion_phase} = {motion_times} / {self._ref_motion_length}")
            
            if not (torch.all(self._ref_motion_phase >= 0) and torch.all(self._ref_motion_phase <= 1.05)): 
                max_phase = self._ref_motion_phase.max()
                
            self._ref_motion_phase = self._ref_motion_phase.unsqueeze(1)
            
            self._log_motion_tracking_info() # 把误差平均值发给 tensorboard 去画曲线
            
    def _compute_reward(self):
        """挂载计算奖励"""
        super()._compute_reward()
        self.extras["ref_body_pos_extend"] = self.ref_body_pos_extend.clone()
        self.extras["ref_body_rot_extend"] = self.ref_body_rot_extend.clone()

    def _log_motion_tracking_info(self):
        """计算上下半身 L2 误差均值并扔给日志。用来直观地看机器人是不是学得越来越像了"""
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        joint_pos_diff = self.dif_joint_angles

        upper_body_diff_norm = upper_body_diff.norm(dim=-1).mean()#.mean()默认对所有元素求平均，结果是标量
        lower_body_diff_norm = lower_body_diff.norm(dim=-1).mean()
        vr_3point_diff_norm = vr_3point_diff.norm(dim=-1).mean()
        joint_pos_diff_norm = joint_pos_diff.norm(dim=-1).mean()#各个环境中所有电机实际位置与参考位置的差值的 L2 范数的平均值。即：所有环境平均电机位置误差

        self.log_dict["upper_body_diff_norm"] = upper_body_diff_norm
        self.log_dict["lower_body_diff_norm"] = lower_body_diff_norm
        self.log_dict["vr_3point_diff_norm"] = vr_3point_diff_norm
        self.log_dict["joint_pos_diff_norm"] = joint_pos_diff_norm
        

    def _draw_debug_vis(self):
        """画出代表参考动作的悬浮小球，方便开发者盯着屏幕看机器人手有没有碰到小球"""
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(self.num_envs):
            if not self.config.use_teleop_control:
                for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): 
                    if self.config.robot.motion.visualization.customize_color:
                        color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
                    else:
                        color_inner = (0.3, 0.3, 0.3)
                    color_inner = tuple(color_inner)

                    self.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id, pos_id)
            else:
                for pos_id, pos_joint in enumerate(self.teleop_marker_coords[env_id]):
                    self.simulator.draw_sphere(pos_joint, 0.04, (0.851, 0.144, 0.07), env_id, pos_id)

    def _reset_root_states(self, env_ids):
        """
        【函数作用】：Reference State Initialization (RSI) 的关键一步：瞬间移动基座！
        【RL 背景】：机器人摔倒后，不回老家，而是瞬间瞬移到动画某处继续学。
        【被谁调用】：环境重启时，被父类的 reset 机制调用。
        """
        if self.custom_origins: # trimesh
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times 
            offset = self.env_origins
            # 查询此时应该在什么绝对坐标
            motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
            # 暴力覆盖仿真器内存，瞬移！
            self.simulator.robot_root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(motion_res['root_rot'][env_ids])
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
                raise NotImplementedError
            self.simulator.robot_root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids]
            self.simulator.robot_root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
            
        else: # (下面的逻辑一样，只是加了一些随机噪声让策略泛化性更强)
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times 
            offset = self.env_origins
            motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

    #         motion_res={'root_pos': torch.tensor([[ 0.0035, -0.2307,  0.9065]], device='cuda:0'), 'root_rot': torch.tensor([[-0.1065, -0.0585,  0.7433, -0.6578]], device='cuda:0'), 'dof_pos': torch.tensor([[-0.1040,  0.0146,  0.2722,  0.0424,  0.0014,  0.0000, -0.2361, -0.1315,
    #      -0.0481,  0.0927,  0.0043,  0.0000,  0.2081, -0.0671, -0.2738,  0.2576,
    #       0.3911, -0.4247,  0.9946,  0.2217, -0.3084,  0.2673,  0.9876]],
    #    device='cuda:0'), 'root_vel': torch.tensor([[-0.0055,  0.0255,  0.0135]], device='cuda:0'), 'root_ang_vel': torch.tensor([[ 0.0263, -0.0059, -0.0780]], device='cuda:0'), 'dof_vel': torch.tensor([[-0.0275, -0.0459,  0.0235, -0.0290, -0.0019,  0.0000, -0.0083,  0.0066,
    #       0.0007,  0.0030, -0.0002,  0.0000, -0.0481, -0.0216, -0.0235, -0.0966,
    #      -0.1212, -0.0390, -0.0361, -0.0507,  0.1070,  0.0176, -0.0074]],
    #    device='cuda:0'), 'motion_aa': torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    #    device='cuda:0'), 'motion_bodies': torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    #    device='cuda:0'), 'rg_pos': torch.tensor([[[ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 7.5702e-02, -2.1562e-01,  8.1025e-01],
    #      [ 1.2931e-01, -2.1844e-01,  7.8286e-01],
    #      [ 1.3375e-01, -2.4932e-01,  6.6016e-01],
    #      [ 1.3582e-01, -1.8039e-01,  4.7898e-01],
    #      [ 1.4475e-01, -1.3092e-01,  1.8321e-01],
    #      [ 1.4527e-01, -1.2799e-01,  1.6591e-01],
    #      [-5.1962e-02, -1.9915e-01,  8.0340e-01],
    #      [-1.0185e-01, -1.9266e-01,  7.7022e-01],
    #      [-1.1767e-01, -2.3682e-01,  6.5261e-01],
    #      [-1.2329e-01, -1.8890e-01,  4.6486e-01],
    #      [-1.3775e-01, -1.5824e-01,  1.6677e-01],
    #      [-1.3858e-01, -1.5637e-01,  1.4933e-01],
    #      [ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 2.8940e-04, -2.3482e-01,  9.4133e-01],
    #      [ 1.0468e-05, -2.3903e-01,  9.5985e-01],
    #      [ 9.0632e-02, -2.2231e-01,  1.2009e+00],
    #      [ 1.3045e-01, -2.1532e-01,  1.1998e+00],
    #      [ 1.7575e-01, -1.8997e-01,  1.1104e+00],
    #      [ 2.0312e-01, -1.8493e-01,  1.0332e+00],
    #      [-1.0887e-01, -2.4026e-01,  1.1938e+00],
    #      [-1.4909e-01, -2.4097e-01,  1.1898e+00],
    #      [-1.8375e-01, -2.2634e-01,  1.0934e+00],
    #      [-2.0160e-01, -2.2924e-01,  1.0134e+00]]], device='cuda:0'), 'rb_rot': torch.tensor([[[-0.1065, -0.0585,  0.7433, -0.6578],
    #      [-0.0677, -0.0243,  0.7479, -0.6600],
    #      [-0.0070,  0.0390,  0.7506, -0.6595],
    #      [ 0.0016, -0.0396, -0.6542,  0.7553],
    #      [ 0.0726,  0.0426, -0.6502,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [-0.0182,  0.0194,  0.7507, -0.6601],
    #      [ 0.0904,  0.0275,  0.7529, -0.6514],
    #      [ 0.0897,  0.0297,  0.7683, -0.6331],
    #      [-0.0136, -0.0550,  0.7734, -0.6314],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [ 0.1120,  0.0472, -0.6710,  0.7315],
    #      [ 0.0874,  0.0696, -0.6690,  0.7348],
    #      [-0.0047, -0.0313, -0.6747,  0.7374],
    #      [ 0.1825, -0.0300, -0.6457,  0.7408],
    #      [ 0.2236, -0.0661, -0.6431,  0.7295],
    #      [-0.2325,  0.0175,  0.7824, -0.5775],
    #      [ 0.5776,  0.2602, -0.5767,  0.5159],
    #      [-0.0322,  0.1433, -0.6801,  0.7182],
    #      [-0.0427,  0.1532, -0.6780,  0.7177],
    #      [-0.0219,  0.1575, -0.5763,  0.8016],
    #      [ 0.2538,  0.5186, -0.5178,  0.6312]]], device='cuda:0'), 'body_vel': torch.tensor([[[-0.0055,  0.0255,  0.0135],
    #      [-0.0032,  0.0214,  0.0146],
    #      [-0.0032,  0.0166,  0.0150],
    #      [-0.0097,  0.0174,  0.0146],
    #      [-0.0107,  0.0188,  0.0151],
    #      [-0.0149,  0.0096,  0.0135],
    #      [-0.0151,  0.0090,  0.0134],
    #      [-0.0016,  0.0331,  0.0137],
    #      [-0.0007,  0.0379,  0.0134],
    #      [-0.0029,  0.0401,  0.0128],
    #      [ 0.0040,  0.0419,  0.0131],
    #      [ 0.0113,  0.0468,  0.0133],
    #      [ 0.0118,  0.0471,  0.0133],
    #      [-0.0055,  0.0255,  0.0135],
    #      [-0.0061,  0.0249,  0.0134],
    #      [-0.0065,  0.0245,  0.0133],
    #      [-0.0041,  0.0158,  0.0130],
    #      [-0.0030,  0.0093,  0.0126],
    #      [-0.0093, -0.0076,  0.0048],
    #      [-0.0176, -0.0202,  0.0012],
    #      [-0.0065,  0.0425,  0.0134],
    #      [-0.0065,  0.0474,  0.0133],
    #      [ 0.0041,  0.0480,  0.0096],
    #      [ 0.0109,  0.0470,  0.0082]]], device='cuda:0'), 'body_ang_vel': torch.tensor([[[ 2.6257e-02, -5.9151e-03, -7.7960e-02],
    #      [ 8.6228e-03, -3.2692e-03, -7.9365e-02],
    #      [ 1.2471e-02,  2.9898e-02, -8.1405e-02],
    #      [ 1.1354e-02,  3.1757e-02, -5.2361e-02],
    #      [-2.7502e-02,  2.5645e-02, -5.3717e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [ 7.3476e-03, -3.5769e-03, -7.9060e-02],
    #      [ 6.4085e-03, -1.1238e-02, -7.8942e-02],
    #      [ 6.6325e-03, -9.6965e-03, -7.6678e-02],
    #      [ 1.2110e-02, -1.0904e-02, -7.6448e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 3.0077e-02,  3.9392e-03, -1.2007e-01],
    #      [ 3.0118e-02,  3.7528e-03, -1.2054e-01],
    #      [-1.7129e-02, -1.0014e-04, -1.2165e-01],
    #      [-9.8071e-02, -8.3816e-03, -1.4892e-01],
    #      [-1.1705e-01,  1.0668e-01, -1.2682e-01],
    #      [-1.0299e-01,  1.1524e-01, -1.5953e-01],
    #      [ 4.6978e-02, -4.2387e-02,  1.0965e-01],
    #      [-4.3933e-02, -2.2689e-03, -1.1495e-01],
    #      [-4.1613e-02, -8.3487e-02, -1.2895e-01],
    #      [-4.1718e-02, -8.3550e-02, -1.2975e-01],
    #      [-1.1604e-02, -7.3208e-02, -1.3703e-01]]], device='cuda:0'), 'rg_pos_t': torch.tensor([[[ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 7.5702e-02, -2.1562e-01,  8.1025e-01],
    #      [ 1.2931e-01, -2.1844e-01,  7.8286e-01],
    #      [ 1.3375e-01, -2.4932e-01,  6.6016e-01],
    #      [ 1.3582e-01, -1.8039e-01,  4.7898e-01],
    #      [ 1.4475e-01, -1.3092e-01,  1.8321e-01],
    #      [ 1.4527e-01, -1.2799e-01,  1.6591e-01],
    #      [-5.1962e-02, -1.9915e-01,  8.0340e-01],
    #      [-1.0185e-01, -1.9266e-01,  7.7022e-01],
    #      [-1.1767e-01, -2.3682e-01,  6.5261e-01],
    #      [-1.2329e-01, -1.8890e-01,  4.6486e-01],
    #      [-1.3775e-01, -1.5824e-01,  1.6677e-01],
    #      [-1.3858e-01, -1.5637e-01,  1.4933e-01],
    #      [ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 2.8940e-04, -2.3482e-01,  9.4133e-01],
    #      [ 1.0468e-05, -2.3903e-01,  9.5985e-01],
    #      [ 9.0632e-02, -2.2231e-01,  1.2009e+00],
    #      [ 1.3045e-01, -2.1532e-01,  1.1998e+00],
    #      [ 1.7575e-01, -1.8997e-01,  1.1104e+00],
    #      [ 2.0312e-01, -1.8493e-01,  1.0332e+00],
    #      [-1.0887e-01, -2.4026e-01,  1.1938e+00],
    #      [-1.4909e-01, -2.4097e-01,  1.1898e+00],
    #      [-1.8375e-01, -2.2634e-01,  1.0934e+00],
    #      [-2.0160e-01, -2.2924e-01,  1.0134e+00],
    #      [ 2.5299e-01, -2.5856e-01,  7.9952e-01],
    #      [-2.2016e-01, -3.2685e-01,  7.8401e-01],
    #      [-1.6733e-02, -2.1839e-01,  1.3790e+00]]], device='cuda:0'), 'rg_rot_t': torch.tensor([[[-0.1065, -0.0585,  0.7433, -0.6578],
    #      [-0.0677, -0.0243,  0.7479, -0.6600],
    #      [-0.0070,  0.0390,  0.7506, -0.6595],
    #      [ 0.0016, -0.0396, -0.6542,  0.7553],
    #      [ 0.0726,  0.0426, -0.6502,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [-0.0182,  0.0194,  0.7507, -0.6601],
    #      [ 0.0904,  0.0275,  0.7529, -0.6514],
    #      [ 0.0897,  0.0297,  0.7683, -0.6331],
    #      [-0.0136, -0.0550,  0.7734, -0.6314],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [ 0.1120,  0.0472, -0.6710,  0.7315],
    #      [ 0.0874,  0.0696, -0.6690,  0.7348],
    #      [-0.0047, -0.0313, -0.6747,  0.7374],
    #      [ 0.1825, -0.0300, -0.6457,  0.7408],
    #      [ 0.2236, -0.0661, -0.6431,  0.7295],
    #      [-0.2325,  0.0175,  0.7824, -0.5775],
    #      [ 0.5776,  0.2602, -0.5767,  0.5159],
    #      [-0.0322,  0.1433, -0.6801,  0.7182],
    #      [-0.0427,  0.1532, -0.6780,  0.7177],
    #      [-0.0219,  0.1575, -0.5763,  0.8016],
    #      [ 0.2538,  0.5186, -0.5178,  0.6312],
    #      [ 0.5776,  0.2602, -0.5767,  0.5159],
    #      [ 0.2538,  0.5186, -0.5178,  0.6312],
    #      [-0.0047, -0.0313, -0.6747,  0.7374]]], device='cuda:0'), 'body_vel_t': torch.tensor([[[-0.0055,  0.0255,  0.0135],
    #      [-0.0032,  0.0214,  0.0146],
    #      [-0.0032,  0.0166,  0.0150],
    #      [-0.0097,  0.0174,  0.0146],
    #      [-0.0107,  0.0188,  0.0151],
    #      [-0.0149,  0.0096,  0.0135],
    #      [-0.0151,  0.0090,  0.0134],
    #      [-0.0016,  0.0331,  0.0137],
    #      [-0.0007,  0.0379,  0.0134],
    #      [-0.0029,  0.0401,  0.0128],
    #      [ 0.0040,  0.0419,  0.0131],
    #      [ 0.0113,  0.0468,  0.0133],
    #      [ 0.0118,  0.0471,  0.0133],
    #      [-0.0055,  0.0255,  0.0135],
    #      [-0.0061,  0.0249,  0.0134],
    #      [-0.0065,  0.0245,  0.0133],
    #      [-0.0041,  0.0158,  0.0130],
    #      [-0.0030,  0.0093,  0.0126],
    #      [-0.0093, -0.0076,  0.0048],
    #      [-0.0176, -0.0202,  0.0012],
    #      [-0.0065,  0.0425,  0.0134],
    #      [-0.0065,  0.0474,  0.0133],
    #      [ 0.0041,  0.0480,  0.0096],
    #      [ 0.0109,  0.0470,  0.0082],
    #      [-0.0593, -0.0560,  0.0039],
    #      [ 0.0159,  0.0447,  0.0088],
    #      [-0.0034,  0.0328,  0.0130]]], device='cuda:0'), 'body_ang_vel_t': torch.tensor([[[ 2.6257e-02, -5.9151e-03, -7.7960e-02],
    #      [ 8.6228e-03, -3.2692e-03, -7.9365e-02],
    #      [ 1.2471e-02,  2.9898e-02, -8.1405e-02],
    #      [ 1.1354e-02,  3.1757e-02, -5.2361e-02],
    #      [-2.7502e-02,  2.5645e-02, -5.3717e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [ 7.3476e-03, -3.5769e-03, -7.9060e-02],
    #      [ 6.4085e-03, -1.1238e-02, -7.8942e-02],
    #      [ 6.6325e-03, -9.6965e-03, -7.6678e-02],
    #      [ 1.2110e-02, -1.0904e-02, -7.6448e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 3.0077e-02,  3.9392e-03, -1.2007e-01],
    #      [ 3.0118e-02,  3.7528e-03, -1.2054e-01],
    #      [-1.7129e-02, -1.0014e-04, -1.2165e-01],
    #      [-9.8071e-02, -8.3816e-03, -1.4892e-01],
    #      [-1.1705e-01,  1.0668e-01, -1.2682e-01],
    #      [-1.0299e-01,  1.1524e-01, -1.5953e-01],
    #      [ 4.6978e-02, -4.2387e-02,  1.0965e-01],
    #      [-4.3933e-02, -2.2689e-03, -1.1495e-01],
    #      [-4.1613e-02, -8.3487e-02, -1.2895e-01],
    #      [-4.1718e-02, -8.3550e-02, -1.2975e-01],
    #      [-1.1604e-02, -7.3208e-02, -1.3703e-01],
    #      [ 4.6978e-02, -4.2387e-02,  1.0965e-01],
    #      [-1.1604e-02, -7.3208e-02, -1.3703e-01],
    #      [-1.7129e-02, -1.0014e-04, -1.2165e-01]]], device='cuda:0')}
            
            # 获取噪声比例
            # root_pos_noise = self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
            # root_rot_noise = self.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.noise_to_initial_level
            # root_vel_noise = self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
            # root_ang_vel_noise = self.config.init_noise_scale.root_ang_vel * self.config.noise_to_initial_level
            root_pos_noise = 0
            root_rot_noise = 0            
            root_vel_noise = 0
            root_ang_vel_noise = 0

            
            

            root_pos = motion_res['root_pos'][env_ids]
            root_rot = motion_res['root_rot'][env_ids]
            root_vel = motion_res['root_vel'][env_ids]
            root_ang_vel = motion_res['root_ang_vel'][env_ids]

            # 瞬移基座，并加一点随机扰动，防过拟合
            self.simulator.robot_root_states[env_ids, :3] = root_pos + torch.randn_like(root_pos) * root_pos_noise
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True))
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'mujoco':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            else:
                raise NotImplementedError
            self.simulator.robot_root_states[env_ids, 7:10] = root_vel + torch.randn_like(root_vel) * root_vel_noise
            self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise
            
            # print(f"_reset_root_states中读取仿真器的self.simulator.robot_root_states[env_ids, :3]: {self.simulator.robot_root_states[env_ids, :3]}")
            # print(f"_reset_root_states中读取仿真器的self.simulator.robot_root_states[env_ids, 3:7]: {self.simulator.robot_root_states[env_ids, 3:7]}")
            # print(f"_reset_root_states中读取仿真器的self.simulator.robot_root_states[env_ids, 7:10]: {self.simulator.robot_root_states[env_ids, 7:10]}")
            # print(f"_reset_root_states中读取仿真器的self.simulator.robot_root_states[env_ids, 10:13]: {self.simulator.robot_root_states[env_ids, 10:13]}")

    def small_random_quaternions(self, n, max_angle):
            """生成小角度的随机偏转四元数（用于加噪声）"""
            axis = torch.randn((n, 3), device=self.device)
            axis = axis / torch.norm(axis, dim=1, keepdim=True)  
            angles = max_angle * torch.rand((n, 1), device=self.device)
            
            sin_half_angle = torch.sin(angles / 2)
            cos_half_angle = torch.cos(angles / 2)
            
            q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)  
            return q

    def _reset_dofs(self, env_ids):
        """
        【函数作用】：瞬间将机器人的关节也掰到动画中的那个姿势。
        【被谁调用】：环境重启时，随同 _reset_root_states 一起被调用。
        """
        motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times 
        offset = self.env_origins
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

        # print(f"读取动作库中状态{motion_res} = self._motion_lib.get_motion_state(动作编号，动作时刻，原点偏移）{self.motion_ids}, {motion_times}, {offset})")
        # print(f"动作库初始动作motion_res：{motion_res}")
    #     motion_res={'root_pos': torch.tensor([[ 0.0035, -0.2307,  0.9065]], device='cuda:0'), 'root_rot': torch.tensor([[-0.1065, -0.0585,  0.7433, -0.6578]], device='cuda:0'), 'dof_pos': torch.tensor([[-0.1040,  0.0146,  0.2722,  0.0424,  0.0014,  0.0000, -0.2361, -0.1315,
    #      -0.0481,  0.0927,  0.0043,  0.0000,  0.2081, -0.0671, -0.2738,  0.2576,
    #       0.3911, -0.4247,  0.9946,  0.2217, -0.3084,  0.2673,  0.9876]],
    #    device='cuda:0'), 'root_vel': torch.tensor([[-0.0055,  0.0255,  0.0135]], device='cuda:0'), 'root_ang_vel': torch.tensor([[ 0.0263, -0.0059, -0.0780]], device='cuda:0'), 'dof_vel': torch.tensor([[-0.0275, -0.0459,  0.0235, -0.0290, -0.0019,  0.0000, -0.0083,  0.0066,
    #       0.0007,  0.0030, -0.0002,  0.0000, -0.0481, -0.0216, -0.0235, -0.0966,
    #      -0.1212, -0.0390, -0.0361, -0.0507,  0.1070,  0.0176, -0.0074]],
    #    device='cuda:0'), 'motion_aa': torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    #    device='cuda:0'), 'motion_bodies': torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    #    device='cuda:0'), 'rg_pos': torch.tensor([[[ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 7.5702e-02, -2.1562e-01,  8.1025e-01],
    #      [ 1.2931e-01, -2.1844e-01,  7.8286e-01],
    #      [ 1.3375e-01, -2.4932e-01,  6.6016e-01],
    #      [ 1.3582e-01, -1.8039e-01,  4.7898e-01],
    #      [ 1.4475e-01, -1.3092e-01,  1.8321e-01],
    #      [ 1.4527e-01, -1.2799e-01,  1.6591e-01],
    #      [-5.1962e-02, -1.9915e-01,  8.0340e-01],
    #      [-1.0185e-01, -1.9266e-01,  7.7022e-01],
    #      [-1.1767e-01, -2.3682e-01,  6.5261e-01],
    #      [-1.2329e-01, -1.8890e-01,  4.6486e-01],
    #      [-1.3775e-01, -1.5824e-01,  1.6677e-01],
    #      [-1.3858e-01, -1.5637e-01,  1.4933e-01],
    #      [ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 2.8940e-04, -2.3482e-01,  9.4133e-01],
    #      [ 1.0468e-05, -2.3903e-01,  9.5985e-01],
    #      [ 9.0632e-02, -2.2231e-01,  1.2009e+00],
    #      [ 1.3045e-01, -2.1532e-01,  1.1998e+00],
    #      [ 1.7575e-01, -1.8997e-01,  1.1104e+00],
    #      [ 2.0312e-01, -1.8493e-01,  1.0332e+00],
    #      [-1.0887e-01, -2.4026e-01,  1.1938e+00],
    #      [-1.4909e-01, -2.4097e-01,  1.1898e+00],
    #      [-1.8375e-01, -2.2634e-01,  1.0934e+00],
    #      [-2.0160e-01, -2.2924e-01,  1.0134e+00]]], device='cuda:0'), 'rb_rot': torch.tensor([[[-0.1065, -0.0585,  0.7433, -0.6578],
    #      [-0.0677, -0.0243,  0.7479, -0.6600],
    #      [-0.0070,  0.0390,  0.7506, -0.6595],
    #      [ 0.0016, -0.0396, -0.6542,  0.7553],
    #      [ 0.0726,  0.0426, -0.6502,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [-0.0182,  0.0194,  0.7507, -0.6601],
    #      [ 0.0904,  0.0275,  0.7529, -0.6514],
    #      [ 0.0897,  0.0297,  0.7683, -0.6331],
    #      [-0.0136, -0.0550,  0.7734, -0.6314],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [ 0.1120,  0.0472, -0.6710,  0.7315],
    #      [ 0.0874,  0.0696, -0.6690,  0.7348],
    #      [-0.0047, -0.0313, -0.6747,  0.7374],
    #      [ 0.1825, -0.0300, -0.6457,  0.7408],
    #      [ 0.2236, -0.0661, -0.6431,  0.7295],
    #      [-0.2325,  0.0175,  0.7824, -0.5775],
    #      [ 0.5776,  0.2602, -0.5767,  0.5159],
    #      [-0.0322,  0.1433, -0.6801,  0.7182],
    #      [-0.0427,  0.1532, -0.6780,  0.7177],
    #      [-0.0219,  0.1575, -0.5763,  0.8016],
    #      [ 0.2538,  0.5186, -0.5178,  0.6312]]], device='cuda:0'), 'body_vel': torch.tensor([[[-0.0055,  0.0255,  0.0135],
    #      [-0.0032,  0.0214,  0.0146],
    #      [-0.0032,  0.0166,  0.0150],
    #      [-0.0097,  0.0174,  0.0146],
    #      [-0.0107,  0.0188,  0.0151],
    #      [-0.0149,  0.0096,  0.0135],
    #      [-0.0151,  0.0090,  0.0134],
    #      [-0.0016,  0.0331,  0.0137],
    #      [-0.0007,  0.0379,  0.0134],
    #      [-0.0029,  0.0401,  0.0128],
    #      [ 0.0040,  0.0419,  0.0131],
    #      [ 0.0113,  0.0468,  0.0133],
    #      [ 0.0118,  0.0471,  0.0133],
    #      [-0.0055,  0.0255,  0.0135],
    #      [-0.0061,  0.0249,  0.0134],
    #      [-0.0065,  0.0245,  0.0133],
    #      [-0.0041,  0.0158,  0.0130],
    #      [-0.0030,  0.0093,  0.0126],
    #      [-0.0093, -0.0076,  0.0048],
    #      [-0.0176, -0.0202,  0.0012],
    #      [-0.0065,  0.0425,  0.0134],
    #      [-0.0065,  0.0474,  0.0133],
    #      [ 0.0041,  0.0480,  0.0096],
    #      [ 0.0109,  0.0470,  0.0082]]], device='cuda:0'), 'body_ang_vel': torch.tensor([[[ 2.6257e-02, -5.9151e-03, -7.7960e-02],
    #      [ 8.6228e-03, -3.2692e-03, -7.9365e-02],
    #      [ 1.2471e-02,  2.9898e-02, -8.1405e-02],
    #      [ 1.1354e-02,  3.1757e-02, -5.2361e-02],
    #      [-2.7502e-02,  2.5645e-02, -5.3717e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [ 7.3476e-03, -3.5769e-03, -7.9060e-02],
    #      [ 6.4085e-03, -1.1238e-02, -7.8942e-02],
    #      [ 6.6325e-03, -9.6965e-03, -7.6678e-02],
    #      [ 1.2110e-02, -1.0904e-02, -7.6448e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 3.0077e-02,  3.9392e-03, -1.2007e-01],
    #      [ 3.0118e-02,  3.7528e-03, -1.2054e-01],
    #      [-1.7129e-02, -1.0014e-04, -1.2165e-01],
    #      [-9.8071e-02, -8.3816e-03, -1.4892e-01],
    #      [-1.1705e-01,  1.0668e-01, -1.2682e-01],
    #      [-1.0299e-01,  1.1524e-01, -1.5953e-01],
    #      [ 4.6978e-02, -4.2387e-02,  1.0965e-01],
    #      [-4.3933e-02, -2.2689e-03, -1.1495e-01],
    #      [-4.1613e-02, -8.3487e-02, -1.2895e-01],
    #      [-4.1718e-02, -8.3550e-02, -1.2975e-01],
    #      [-1.1604e-02, -7.3208e-02, -1.3703e-01]]], device='cuda:0'), 'rg_pos_t': torch.tensor([[[ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 7.5702e-02, -2.1562e-01,  8.1025e-01],
    #      [ 1.2931e-01, -2.1844e-01,  7.8286e-01],
    #      [ 1.3375e-01, -2.4932e-01,  6.6016e-01],
    #      [ 1.3582e-01, -1.8039e-01,  4.7898e-01],
    #      [ 1.4475e-01, -1.3092e-01,  1.8321e-01],
    #      [ 1.4527e-01, -1.2799e-01,  1.6591e-01],
    #      [-5.1962e-02, -1.9915e-01,  8.0340e-01],
    #      [-1.0185e-01, -1.9266e-01,  7.7022e-01],
    #      [-1.1767e-01, -2.3682e-01,  6.5261e-01],
    #      [-1.2329e-01, -1.8890e-01,  4.6486e-01],
    #      [-1.3775e-01, -1.5824e-01,  1.6677e-01],
    #      [-1.3858e-01, -1.5637e-01,  1.4933e-01],
    #      [ 3.5142e-03, -2.3072e-01,  9.0649e-01],
    #      [ 2.8940e-04, -2.3482e-01,  9.4133e-01],
    #      [ 1.0468e-05, -2.3903e-01,  9.5985e-01],
    #      [ 9.0632e-02, -2.2231e-01,  1.2009e+00],
    #      [ 1.3045e-01, -2.1532e-01,  1.1998e+00],
    #      [ 1.7575e-01, -1.8997e-01,  1.1104e+00],
    #      [ 2.0312e-01, -1.8493e-01,  1.0332e+00],
    #      [-1.0887e-01, -2.4026e-01,  1.1938e+00],
    #      [-1.4909e-01, -2.4097e-01,  1.1898e+00],
    #      [-1.8375e-01, -2.2634e-01,  1.0934e+00],
    #      [-2.0160e-01, -2.2924e-01,  1.0134e+00],
    #      [ 2.5299e-01, -2.5856e-01,  7.9952e-01],
    #      [-2.2016e-01, -3.2685e-01,  7.8401e-01],
    #      [-1.6733e-02, -2.1839e-01,  1.3790e+00]]], device='cuda:0'), 'rg_rot_t': torch.tensor([[[-0.1065, -0.0585,  0.7433, -0.6578],
    #      [-0.0677, -0.0243,  0.7479, -0.6600],
    #      [-0.0070,  0.0390,  0.7506, -0.6595],
    #      [ 0.0016, -0.0396, -0.6542,  0.7553],
    #      [ 0.0726,  0.0426, -0.6502,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [ 0.0730,  0.0431, -0.6501,  0.7551],
    #      [-0.0182,  0.0194,  0.7507, -0.6601],
    #      [ 0.0904,  0.0275,  0.7529, -0.6514],
    #      [ 0.0897,  0.0297,  0.7683, -0.6331],
    #      [-0.0136, -0.0550,  0.7734, -0.6314],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [-0.0152, -0.0564,  0.7734, -0.6313],
    #      [ 0.1120,  0.0472, -0.6710,  0.7315],
    #      [ 0.0874,  0.0696, -0.6690,  0.7348],
    #      [-0.0047, -0.0313, -0.6747,  0.7374],
    #      [ 0.1825, -0.0300, -0.6457,  0.7408],
    #      [ 0.2236, -0.0661, -0.6431,  0.7295],
    #      [-0.2325,  0.0175,  0.7824, -0.5775],
    #      [ 0.5776,  0.2602, -0.5767,  0.5159],
    #      [-0.0322,  0.1433, -0.6801,  0.7182],
    #      [-0.0427,  0.1532, -0.6780,  0.7177],
    #      [-0.0219,  0.1575, -0.5763,  0.8016],
    #      [ 0.2538,  0.5186, -0.5178,  0.6312],
    #      [ 0.5776,  0.2602, -0.5767,  0.5159],
    #      [ 0.2538,  0.5186, -0.5178,  0.6312],
    #      [-0.0047, -0.0313, -0.6747,  0.7374]]], device='cuda:0'), 'body_vel_t': torch.tensor([[[-0.0055,  0.0255,  0.0135],
    #      [-0.0032,  0.0214,  0.0146],
    #      [-0.0032,  0.0166,  0.0150],
    #      [-0.0097,  0.0174,  0.0146],
    #      [-0.0107,  0.0188,  0.0151],
    #      [-0.0149,  0.0096,  0.0135],
    #      [-0.0151,  0.0090,  0.0134],
    #      [-0.0016,  0.0331,  0.0137],
    #      [-0.0007,  0.0379,  0.0134],
    #      [-0.0029,  0.0401,  0.0128],
    #      [ 0.0040,  0.0419,  0.0131],
    #      [ 0.0113,  0.0468,  0.0133],
    #      [ 0.0118,  0.0471,  0.0133],
    #      [-0.0055,  0.0255,  0.0135],
    #      [-0.0061,  0.0249,  0.0134],
    #      [-0.0065,  0.0245,  0.0133],
    #      [-0.0041,  0.0158,  0.0130],
    #      [-0.0030,  0.0093,  0.0126],
    #      [-0.0093, -0.0076,  0.0048],
    #      [-0.0176, -0.0202,  0.0012],
    #      [-0.0065,  0.0425,  0.0134],
    #      [-0.0065,  0.0474,  0.0133],
    #      [ 0.0041,  0.0480,  0.0096],
    #      [ 0.0109,  0.0470,  0.0082],
    #      [-0.0593, -0.0560,  0.0039],
    #      [ 0.0159,  0.0447,  0.0088],
    #      [-0.0034,  0.0328,  0.0130]]], device='cuda:0'), 'body_ang_vel_t': torch.tensor([[[ 2.6257e-02, -5.9151e-03, -7.7960e-02],
    #      [ 8.6228e-03, -3.2692e-03, -7.9365e-02],
    #      [ 1.2471e-02,  2.9898e-02, -8.1405e-02],
    #      [ 1.1354e-02,  3.1757e-02, -5.2361e-02],
    #      [-2.7502e-02,  2.5645e-02, -5.3717e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [-3.0063e-02,  2.5143e-02, -5.3587e-02],
    #      [ 7.3476e-03, -3.5769e-03, -7.9060e-02],
    #      [ 6.4085e-03, -1.1238e-02, -7.8942e-02],
    #      [ 6.6325e-03, -9.6965e-03, -7.6678e-02],
    #      [ 1.2110e-02, -1.0904e-02, -7.6448e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 1.1735e-02, -1.1172e-02, -7.7030e-02],
    #      [ 3.0077e-02,  3.9392e-03, -1.2007e-01],
    #      [ 3.0118e-02,  3.7528e-03, -1.2054e-01],
    #      [-1.7129e-02, -1.0014e-04, -1.2165e-01],
    #      [-9.8071e-02, -8.3816e-03, -1.4892e-01],
    #      [-1.1705e-01,  1.0668e-01, -1.2682e-01],
    #      [-1.0299e-01,  1.1524e-01, -1.5953e-01],
    #      [ 4.6978e-02, -4.2387e-02,  1.0965e-01],
    #      [-4.3933e-02, -2.2689e-03, -1.1495e-01],
    #      [-4.1613e-02, -8.3487e-02, -1.2895e-01],
    #      [-4.1718e-02, -8.3550e-02, -1.2975e-01],
    #      [-1.1604e-02, -7.3208e-02, -1.3703e-01],
    #      [ 4.6978e-02, -4.2387e-02,  1.0965e-01],
    #      [-1.1604e-02, -7.3208e-02, -1.3703e-01],
    #      [-1.7129e-02, -1.0014e-04, -1.2165e-01]]], device='cuda:0')}
        
        dof_pos_noise = self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
        dof_vel_noise = self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
        dof_pos = motion_res['dof_pos'][env_ids]
        dof_vel = motion_res['dof_vel'][env_ids]
        
        # print(f"动作库中读取的该动作当前时刻的关节速度: motion_res['dof_vel'][env_ids]={dof_vel}")
        
        # 强制复写电机角度和速度，完成重生
        # self.simulator.dof_pos[env_ids] = dof_pos + torch.randn_like(dof_pos) * dof_pos_noise
        # self.simulator.dof_vel[env_ids] = dof_vel + torch.randn_like(dof_vel) * dof_vel_noise
        self.simulator.dof_pos[env_ids] = dof_pos 
        self.simulator.dof_vel[env_ids] = dof_vel 
        # print(f"reset_dof中读取仿真器中self.simulator.dof_pos[env_ids]: {self.simulator.dof_pos[env_ids]}")
        # print(f"reset_dof中读取仿真器中self.simulator.dof_vel[env_ids]: {self.simulator.dof_vel[env_ids]}")



    def _post_physics_step__old_disabled(self):
        """
        【原函数保留（已禁用）】
        【函数作用】：如果开启了 save_motion，把刚才产生的这一步真真实实的物理交互数据全部追加到内存里。够长了就存成 .pkl 文件。
        【被谁调用】：每次 env.step 结尾被调用。
        """
        super()._post_physics_step()
        
        if self.save_motion:    
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times

            # 满了设定步长就全写进硬盘，直接退出程序
            if (len(self.motions_for_saving['dof'])) > self.config.save_total_steps:
                for k, v in self.motions_for_saving.items():
                    self.motions_for_saving[k] = torch.stack(v[3:]).transpose(0,1).numpy()
                
                self.motions_for_saving['motion_times'] = torch.stack(self.motion_times_buf[3:]).transpose(0,1).numpy()
                
                dump_data = {}
                num_motions = self.num_envs 
                keys_to_save = self.motions_for_saving.keys()

                for i in range(num_motions):
                    motion_key = f"motion{i}" 
                    dump_data[motion_key] = {
                        key: self.motions_for_saving[key][i] for key in keys_to_save
                    }
                    dump_data[motion_key]['fps'] = 1 / self.dt
    
                joblib.dump(dump_data, f'{self.save_motion_dir}.pkl')
                
                print(colored(f"Saved motion data to {self.save_motion_dir}.pkl", 'green'))
                import sys
                sys.exit()

            # 没满就抓紧时间从物理内存往这里倒腾（和我们写过的数据收集极其神似）
            root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
            if self.config.simulator.config.name == "isaacgym":
                root_rot = self.simulator.robot_root_states[:, 3:7].cpu() 
            elif self.config.simulator.config.name == "isaacsim":
                root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu()   
            elif self.config.simulator.config.name == "genesis":
                root_rot = self.simulator.robot_root_states[:,  3:7].cpu() 
            else:
                raise NotImplementedError
            
            # 转轴角方便做重定向或者重新导入到动画软件
            root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float()
            dof = self.simulator.dof_pos.cpu()
            
            pose_aa = torch.cat([root_rot_vec[:, None, :], self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None], torch.zeros((self.num_envs, self.num_augment_joint, 3))], axis = 1)
            self.motions_for_saving['root_trans_offset'].append(root_trans)
            self.motions_for_saving['root_rot'].append(root_rot)
            self.motions_for_saving['dof'].append(dof)
            self.motions_for_saving['pose_aa'].append(pose_aa)
            self.motions_for_saving['action'].append(self.actions.cpu())
            self.motions_for_saving['actor_obs'].append(self.obs_buf_dict['actor_obs'].cpu())
            self.motions_for_saving['terminate'].append(self.reset_buf.cpu())
            self.motions_for_saving['dof_vel'].append(self.simulator.dof_vel.cpu())
            self.motions_for_saving['root_lin_vel'].append(self.simulator.robot_root_states[:, 7:10].cpu())
            self.motions_for_saving['root_ang_vel'].append(self.simulator.robot_root_states[:, 10:13].cpu())
            
            self.motion_times_buf.append(motion_times.cpu())

            self.start_save = True

    def _post_physics_step(self):
        """
        【函数作用】：只记录每个环境“第一次生命”（首次 reset 前）的连续数据，避免重置后轨迹突变。
        【被谁调用】：每次 env.step 结尾被调用。
        """
        # 先执行父类后处理：刷新状态、计算奖励、终止判断、重置等基础流程都在父类里完成。
        super()._post_physics_step()

        # ========================= 评估期奖励追踪（新增） =========================
        # 目标：在 eval 时记录“每一步奖励”与“整回合 return（含终止步奖励）”，并直接打印到终端用于比较模型。
        # 说明：
        # 1) self.rew_buf 是本步总奖励（已包含各项奖励与 termination 项）；
        # 2) 在父类 super()._post_physics_step() 内，reset 已经执行完，但 reset_buf/time_out_buf 仍可用于识别本步结束的 episode；
        # 3) 因此在这里统计不会漏掉“最后一步奖励”。
        if self.is_evaluating:
            # 懒初始化：仅在首次进入评估步时创建缓存。
            if not hasattr(self, "_eval_return_tracker_ready"):
                # 当前 episode 的累计回报（按 env 独立累计）。
                self._eval_episode_returns = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                # 当前 episode 的逐步奖励序列（按 env 独立 list）。
                self._eval_episode_step_rewards = [[] for _ in range(self.num_envs)]
                # 每个 env 已完成了多少个 episode（用于编号）。
                self._eval_episode_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                # 仅保留轻量统计用于终端显示（不落盘）。
                self._eval_completed_returns = []
                self._eval_completed_lengths = []

                self._eval_return_tracker_ready = True

            # 逐步累积：将本步奖励计入每个 env 的当前 episode。
            step_rewards_cpu = self.rew_buf.detach().cpu()
            for env_idx in range(self.num_envs):
                rew_i = float(step_rewards_cpu[env_idx].item())
                self._eval_episode_returns[env_idx] += rew_i
                self._eval_episode_step_rewards[env_idx].append(rew_i)

            # 检测本步结束的 episode（timeout 或其它终止）。
            done_ids = torch.nonzero(self.reset_buf.bool(), as_tuple=False).flatten().tolist()
            for env_idx in done_ids:
                # 整回合 return（已包含最后一步奖励）。
                ep_return = float(self._eval_episode_returns[env_idx].item())
                # 回合长度：使用 last_episode_length_buf（避免 reset 后被清零）。
                ep_length = int(self.last_episode_length_buf[env_idx].item()) if hasattr(self, "last_episode_length_buf") else len(self._eval_episode_step_rewards[env_idx])
                # 回合结束类型：优先标注 timeout，其它归为 terminated。
                done_type = "timeout" if bool(self.time_out_buf[env_idx].item()) else "terminated"
                # episode 编号。
                ep_idx = int(self._eval_episode_index[env_idx].item())

                # 汇总到全局已完成回合统计（用于实时均值显示）。
                self._eval_completed_returns.append(ep_return)
                self._eval_completed_lengths.append(ep_length)
                recent_n = 20#最近20回合的均值
                recent_returns = self._eval_completed_returns[-recent_n:]
                recent_lengths = self._eval_completed_lengths[-recent_n:]
                recent_return_mean = float(np.mean(recent_returns)) if len(recent_returns) > 0 else 0.0
                recent_length_mean = float(np.mean(recent_lengths)) if len(recent_lengths) > 0 else 0.0
                total_finished = len(self._eval_completed_returns)

                


                # 控制台打印：每个结束回合输出一行摘要 + 最近20回合均值。
                print(colored(
                    f"[eval-return] env={env_idx} ep={ep_idx} done={done_type} len={ep_length} return={ep_return:.6f} | "
                    f"recent{min(recent_n, total_finished)}_mean_return={recent_return_mean:.6f} "
                    f"recent{min(recent_n, total_finished)}_mean_len={recent_length_mean:.2f} total_finished={total_finished}",
                    "cyan"
                ))

                # 清空该 env 当前 episode 缓存，开始下一回合统计。
                self._eval_episode_returns[env_idx] = 0.0
                self._eval_episode_step_rewards[env_idx] = []
                self._eval_episode_index[env_idx] += 1
        # ======================================================================

        # 若未开启保存动作，直接退出，不做任何额外记录。
        if not self.save_motion:
            # 提前返回，避免无意义的 CPU 拷贝与列表追加。
            return

        # 懒初始化：第一次进入时创建“首生命期”追踪器
        if not hasattr(self, "_first_life_active"):
            # 标记每个 env 是否仍处于“首生命”阶段（True 表示还没经历第一次 reset）。
            self._first_life_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            # 标记是否已经完成并保存过首生命数据（防止重复 dump）。
            self._first_life_saved = False
            # 记录首生命录制已经走了多少步（全局步计数，不是每个 env 单独计数）。
            self._first_life_step_counter = 0
            # 记录每个 env 首次 reset 发生在第几步，-1 表示尚未发生。
            self._first_life_reset_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            # 记录每个 env 首次 reset 的原因字符串（诊断用）。
            self._first_life_reset_reason = ["" for _ in range(self.num_envs)]
            # 为每个 key 建立“按 env 分桶”的列表缓存：每个 env 各自有一段连续轨迹。
            self.motions_for_saving = {
                'root_trans_offset': [[] for _ in range(self.num_envs)],
                'pose_aa': [[] for _ in range(self.num_envs)],
                'dof': [[] for _ in range(self.num_envs)],
                'root_rot': [[] for _ in range(self.num_envs)],
                'actor_obs': [[] for _ in range(self.num_envs)],
                'action': [[] for _ in range(self.num_envs)],
                'terminate': [[] for _ in range(self.num_envs)],
                'root_lin_vel': [[] for _ in range(self.num_envs)],
                'root_ang_vel': [[] for _ in range(self.num_envs)],
                'dof_vel': [[] for _ in range(self.num_envs)],
            }
            # 单独缓存每个 env 的 motion time 序列，便于后续按 env 导出。
            self.motion_times_buf = [[] for _ in range(self.num_envs)]

        # 已经保存过首生命数据，不再重复记录
        if self._first_life_saved:
            # 已完成一次导出后，后续 step 全部直接返回。
            return

        # 计算当前参考动作时间：episode 已走步数 * dt + 该 env 的动作起始偏移。
        motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times

        # 当前时刻快照（CPU），供按 env 切分记录
        # 根平移位置（x,y,z）拷贝到 CPU，后续逐 env append。
        root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
        # print(f"写入.pkl的root_trans:{root_trans}")
        # 按 simulator 后端提取根四元数字段（不同后端四元数布局可能不同）。
        if self.config.simulator.config.name == "isaacgym":
            # IsaacGym 使用 [x,y,z,w] 存在 3:7。
            root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
        elif self.config.simulator.config.name == "isaacsim":
            # IsaacSim 这里按当前代码约定重排到 [x,y,z,w]。
            root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu()
        elif self.config.simulator.config.name == "genesis":
            # Genesis 与 IsaacGym 一致读取 3:7。
            root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
        else:
            # 未支持后端直接抛异常，避免静默写错数据。
            raise NotImplementedError

        # 把根四元数转成旋转向量（axis-angle 的 rotvec 形式），用于构造 pose_aa。
        root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float()
        # 当前关节位置拷贝到 CPU。
        dof = self.simulator.dof_pos.cpu()
        # 拼接 pose_aa：根旋转向量 + 每个 dof 的轴角 + augment 关节补零。
        pose_aa = torch.cat([
            root_rot_vec[:, None, :],
            self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None],
            torch.zeros((self.num_envs, self.num_augment_joint, 3)),
        ], axis=1)

        # 只记录“未发生首次 reset”的环境；本步 reset 的环境不写入（避免突变帧）
        # 先取当前“首生命活跃”掩码副本。
        active_mask = self._first_life_active.clone()#active_mask:当前这一帧需要记录的环境序号
        if hasattr(self, "reset_buf"):
            # 本步已 reset 的 env 置为 False，避免把突变帧写入轨迹。
            active_mask = active_mask & (~self.reset_buf.bool())

        # 把活跃掩码转为 Python 列表索引，便于 for 循环逐 env 追加。
        active_ids = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
        # 对每个仍在首生命且本步未 reset 的 env，逐 key 追加一帧数据。
        for env_idx in active_ids:
            # 根平移一帧。
            self.motions_for_saving['root_trans_offset'][env_idx].append(root_trans[env_idx].clone())
            # 根旋转四元数一帧。
            self.motions_for_saving['root_rot'][env_idx].append(root_rot[env_idx].clone())
            # 关节位置一帧。
            self.motions_for_saving['dof'][env_idx].append(dof[env_idx].clone())
            # 轴角姿态一帧。
            self.motions_for_saving['pose_aa'][env_idx].append(pose_aa[env_idx].clone())
            # actor 动作一帧（detach+cpu，避免图和显存绑定）。
            self.motions_for_saving['action'][env_idx].append(self.actions[env_idx].detach().cpu().clone())
            # actor 观测一帧。
            self.motions_for_saving['actor_obs'][env_idx].append(self.obs_buf_dict['actor_obs'][env_idx].detach().cpu().clone())
            # 终止标记一帧（当步通常为 False，因为 active_mask 已过滤 reset）。
            self.motions_for_saving['terminate'][env_idx].append(self.reset_buf[env_idx].detach().cpu().clone())
            # 关节速度一帧。
            self.motions_for_saving['dof_vel'][env_idx].append(self.simulator.dof_vel[env_idx].detach().cpu().clone())
            # 根线速度一帧。
            self.motions_for_saving['root_lin_vel'][env_idx].append(self.simulator.robot_root_states[env_idx, 7:10].detach().cpu().clone())
            # 根角速度一帧。
            self.motions_for_saving['root_ang_vel'][env_idx].append(self.simulator.robot_root_states[env_idx, 10:13].detach().cpu().clone())
            # 对应参考时间戳一帧。
            self.motion_times_buf[env_idx].append(motion_times[env_idx].detach().cpu().clone())

        # 诊断：记录每个 env 首次“掉出首生命”的步数和原因
        if hasattr(self, "reset_buf"):
            # 找到“这一步首次从活跃变为 reset”的 env（活跃且 reset_buf=True）。
            just_reset_mask = self._first_life_active & self.reset_buf.bool()#前面实际赋值这一帧是将self._first_life_active克隆到了active_mask，所以前者还没有改变
            # 转成 env id 列表。#这一帧just_reset_mask对应的环境在上面没有被赋值
            just_reset_ids = torch.nonzero(just_reset_mask, as_tuple=False).flatten().tolist()

            # 逐个刚 reset 的 env 记录首次 reset 信息。
            for env_idx in just_reset_ids:
                # 若此前已记录过首次 reset，直接跳过。
                if self._first_life_reset_step[env_idx].item() >= 0:
                    # continue 保证“首次”语义不被覆盖。
                    continue

                # 收集本 env 首次 reset 的命中原因（可多项并存）。
                reasons = []

                # timeout 原因
                if hasattr(self, "time_out_buf") and bool(self.time_out_buf[env_idx].item()):
                    # 命中超时（自然结束）。
                    reasons.append("timeout")

                # motion_far 原因（motion tracking 任务特有）
                if self.config.termination.terminate_when_motion_far:
                    # 重新按当前阈值计算该 env 是否 motion_far（用于诊断解释）。
                    motion_far_now = bool(torch.any(torch.norm(self.dif_global_body_pos[env_idx], dim=-1) > self.terminate_when_motion_far_threshold).item())
                    if motion_far_now:
                        # 命中 motion_far。
                        reasons.append("motion_far")

                # base 通用终止条件
                if self.config.termination.terminate_by_contact:
                    # 非允许触碰体发生大接触力则记 contact。
                    by_contact = bool(torch.any(torch.norm(self.simulator.contact_forces[env_idx, self.termination_contact_indices, :], dim=-1) > 1.).item())
                    if by_contact:
                        # 命中 contact。
                        reasons.append("contact")

                if self.config.termination.terminate_by_gravity:
                    # 判断重力投影 x 分量是否越界。
                    by_gravity_x = bool(torch.any(torch.abs(self.projected_gravity[env_idx, 0:1]) > self.config.termination_scales.termination_gravity_x).item())
                    # 判断重力投影 y 分量是否越界。
                    by_gravity_y = bool(torch.any(torch.abs(self.projected_gravity[env_idx, 1:2]) > self.config.termination_scales.termination_gravity_y).item())
                    if by_gravity_x or by_gravity_y:
                        # 命中姿态越界。
                        reasons.append("gravity")

                if self.config.termination.terminate_by_low_height:
                    # 判断基座高度是否低于最小高度阈值。
                    by_low_height = bool(torch.any(self.simulator.robot_root_states[env_idx, 2:3] < self.config.termination_scales.termination_min_base_height).item())
                    if by_low_height:
                        # 命中低高度终止。
                        reasons.append("low_height")

                if len(reasons) == 0:
                    # 若未匹配到显式规则，归类为未知或概率型终止。
                    reasons.append("unknown_or_probabilistic")

                # 记录首次 reset 发生的全局录制步。
                self._first_life_reset_step[env_idx] = self._first_life_step_counter
                # 记录首次 reset 原因（多原因用 + 连接）。
                self._first_life_reset_reason[env_idx] = "+".join(reasons)

                # 打印该 env 首次 reset 的诊断信息，便于追查为何丢轨迹。
                print(colored(
                    f"[first-life-diag] env={env_idx} first_reset_at_step={self._first_life_step_counter} "
                    f"ep_len={int(self.episode_length_buf[env_idx].item())} reason={self._first_life_reset_reason[env_idx]}",
                    "yellow"
                ))

        # 更新“首生命仍存活”掩码
        if hasattr(self, "reset_buf"):
            # 把本步 reset 的 env 从首生命活跃集合中永久移除。
            self._first_life_active = self._first_life_active & (~self.reset_buf.bool())
        # 全局录制步计数 +1。
        self._first_life_step_counter += 1

        # 若所有 env 首生命都结束，则可触发导出。
        all_first_life_finished = not bool(self._first_life_active.any().item())
        # 兜底上限：即使仍有 env 未 reset，超过 1000 步也强制导出。
        reached_max_steps = self._first_life_step_counter >= 1000

        # 满足任一导出条件时，组装并保存数据。
        if all_first_life_finished or reached_max_steps:
            # dump_data 结构：motion0/motion1/... 每个 env 一条轨迹。
            dump_data = {}
            # 统计最终真正写入的 env 数。
            saved_envs = 0
            # 要保存的所有键。
            keys_to_save = list(self.motions_for_saving.keys())
            # 与旧逻辑对齐：丢弃前 3 帧热身。
            warmup_trim = 0

            # 诊断：导出前打印每个 env 的首生命采样长度与保留情况
            # 打印总览标题。
            print(colored("[first-life-diag] per-env summary before dump:", "cyan"))
            # 逐 env 打印 seq_len 与是否会被保留。
            for env_idx in range(self.num_envs):
                # 该 env 首生命已缓存帧数。
                seq_len_i = len(self.motion_times_buf[env_idx])
                # 是否通过 warmup_trim 过滤。
                keep_i = seq_len_i > warmup_trim
                # 读取首次 reset 步（若存在诊断字段）。
                reset_step_i = int(self._first_life_reset_step[env_idx].item()) if hasattr(self, "_first_life_reset_step") else -1
                # 读取首次 reset 原因（若存在诊断字段）。
                reset_reason_i = self._first_life_reset_reason[env_idx] if hasattr(self, "_first_life_reset_reason") else ""
                # 输出该 env 的汇总诊断。
                print(colored(
                    f"  env={env_idx} seq_len={seq_len_i} keep={keep_i} warmup_trim={warmup_trim} "
                    f"first_reset_step={reset_step_i} reason={reset_reason_i if reset_reason_i else 'not_reset_yet'}",
                    "cyan"
                ))

            # 正式按 env 组装 dump_data。
            for env_idx in range(self.num_envs):
                # 当前 env 的轨迹长度。
                seq_len = len(self.motion_times_buf[env_idx])
                # 与旧逻辑保持一致：跳过最前面的 3 帧
                if seq_len <= warmup_trim:
                    # 太短则不写入（避免空数组或无意义片段）。
                    continue

                # motion{env_idx} 作为该 env 的导出键。
                motion_key = f"motion{env_idx}"
                # 对每个字段做 stack 后转 numpy，时间维在前。
                dump_data[motion_key] = {
                    key: torch.stack(self.motions_for_saving[key][env_idx][warmup_trim:], dim=0).numpy()
                    for key in keys_to_save
                }
                # 保存对齐后的 motion_times。
                dump_data[motion_key]['motion_times'] = torch.stack(self.motion_times_buf[env_idx][warmup_trim:], dim=0).numpy()
                # 记录帧率（1/dt）。
                dump_data[motion_key]['fps'] = 1 / self.dt
                # 成功保存 env 计数 +1。
                saved_envs += 1

            if saved_envs > 0:
                # 至少一个 env 有效，写入 pkl。
                joblib.dump(dump_data, f'{self.save_motion_dir}.pkl')
                # 打印成功信息与有效 env 数。
                print(colored(f"Saved first-life motion data to {self.save_motion_dir}.pkl (envs={saved_envs})", 'green'))
            else:
                # 没有任何 env 满足最小长度，提示未保存。
                print(colored("No valid first-life data collected; nothing saved.", 'yellow'))

            # 标记已保存，防止重复进入导出逻辑。
            self._first_life_saved = True
            # 按当前脚本设计，保存后立即退出进程。
            import sys
            # 结束运行，避免继续评估循环。
            sys.exit()
            
    
    def _post_physics_step_for_reset_all(self):
        """
        【函数作用】：只记录每个环境“第一次生命”（首次 reset 前）的连续数据，避免重置后轨迹突变。
        【被谁调用】：每次 env.step 结尾被调用。
        """
        # 先执行父类后处理：刷新状态、计算奖励、终止判断、重置等基础流程都在父类里完成。
        super()._post_physics_step_for_reset_all()

        # ========================= 评估期奖励追踪（新增） =========================
        # 目标：在 eval 时记录“每一步奖励”与“整回合 return（含终止步奖励）”，并直接打印到终端用于比较模型。
        # 说明：
        # 1) self.rew_buf 是本步总奖励（已包含各项奖励与 termination 项）；
        # 2) 在父类 super()._post_physics_step() 内，reset 已经执行完，但 reset_buf/time_out_buf 仍可用于识别本步结束的 episode；
        # 3) 因此在这里统计不会漏掉“最后一步奖励”。
        if self.is_evaluating:
            # 懒初始化：仅在首次进入评估步时创建缓存。
            if not hasattr(self, "_eval_return_tracker_ready"):
                # 当前 episode 的累计回报（按 env 独立累计）。
                self._eval_episode_returns = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                # 当前 episode 的逐步奖励序列（按 env 独立 list）。
                self._eval_episode_step_rewards = [[] for _ in range(self.num_envs)]
                # 每个 env 已完成了多少个 episode（用于编号）。
                self._eval_episode_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                # 仅保留轻量统计用于终端显示（不落盘）。
                self._eval_completed_returns = []
                self._eval_completed_lengths = []

                self._eval_return_tracker_ready = True

            # 逐步累积：将本步奖励计入每个 env 的当前 episode。
            step_rewards_cpu = self.rew_buf.detach().cpu()
            for env_idx in range(self.num_envs):
                rew_i = float(step_rewards_cpu[env_idx].item())
                self._eval_episode_returns[env_idx] += rew_i
                self._eval_episode_step_rewards[env_idx].append(rew_i)

            # 检测本步结束的 episode（timeout 或其它终止）。
            done_ids = torch.nonzero(self.reset_buf.bool(), as_tuple=False).flatten().tolist()
            for env_idx in done_ids:
                # 整回合 return（已包含最后一步奖励）。
                ep_return = float(self._eval_episode_returns[env_idx].item())
                # 回合长度：使用 last_episode_length_buf（避免 reset 后被清零）。
                ep_length = int(self.last_episode_length_buf[env_idx].item()) if hasattr(self, "last_episode_length_buf") else len(self._eval_episode_step_rewards[env_idx])
                # 回合结束类型：优先标注 timeout，其它归为 terminated。
                done_type = "timeout" if bool(self.time_out_buf[env_idx].item()) else "terminated"
                # episode 编号。
                ep_idx = int(self._eval_episode_index[env_idx].item())

                # 汇总到全局已完成回合统计（用于实时均值显示）。
                self._eval_completed_returns.append(ep_return)
                self._eval_completed_lengths.append(ep_length)
                recent_n = 20#最近20回合的均值
                recent_returns = self._eval_completed_returns[-recent_n:]
                recent_lengths = self._eval_completed_lengths[-recent_n:]
                recent_return_mean = float(np.mean(recent_returns)) if len(recent_returns) > 0 else 0.0
                recent_length_mean = float(np.mean(recent_lengths)) if len(recent_lengths) > 0 else 0.0
                total_finished = len(self._eval_completed_returns)

                # 控制台打印：每个结束回合输出一行摘要 + 最近20回合均值。
                print(colored(
                    f"[eval-return] env={env_idx} ep={ep_idx} done={done_type} len={ep_length} return={ep_return:.6f} | "
                    f"recent{min(recent_n, total_finished)}_mean_return={recent_return_mean:.6f} "
                    f"recent{min(recent_n, total_finished)}_mean_len={recent_length_mean:.2f} total_finished={total_finished}",
                    "cyan"
                ))

                # 清空该 env 当前 episode 缓存，开始下一回合统计。
                self._eval_episode_returns[env_idx] = 0.0#reset_all中self.reset_buf是否置0只会，在这个判断条件只会影响打印，以及下一行self._eval_episode_step_rewards[env_idx]记录的每帧奖励，是否包含第一帧为0的这个值（第一帧为0,或从实际第2帧开始）
                self._eval_episode_step_rewards[env_idx] = []
                self._eval_episode_index[env_idx] += 1
        # ======================================================================

        # 若未开启保存动作，直接退出，不做任何额外记录。
        if not self.save_motion:
            # 提前返回，避免无意义的 CPU 拷贝与列表追加。
            return

        # 懒初始化：第一次进入时创建“首生命期”追踪器
        if not hasattr(self, "_first_life_active"):
            # 标记每个 env 是否仍处于“首生命”阶段（True 表示还没经历第一次 reset）。
            self._first_life_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            # 标记是否已经完成并保存过首生命数据（防止重复 dump）。
            self._first_life_saved = False
            # 记录首生命录制已经走了多少步（全局步计数，不是每个 env 单独计数）。
            self._first_life_step_counter = 0
            # 记录每个 env 首次 reset 发生在第几步，-1 表示尚未发生。
            self._first_life_reset_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            # 记录每个 env 首次 reset 的原因字符串（诊断用）。
            self._first_life_reset_reason = ["" for _ in range(self.num_envs)]
            # 为每个 key 建立“按 env 分桶”的列表缓存：每个 env 各自有一段连续轨迹。
            self.motions_for_saving = {
                'root_trans_offset': [[] for _ in range(self.num_envs)],
                'pose_aa': [[] for _ in range(self.num_envs)],
                'dof': [[] for _ in range(self.num_envs)],
                'root_rot': [[] for _ in range(self.num_envs)],
                'actor_obs': [[] for _ in range(self.num_envs)],
                'action': [[] for _ in range(self.num_envs)],
                'terminate': [[] for _ in range(self.num_envs)],
                'root_lin_vel': [[] for _ in range(self.num_envs)],
                'root_ang_vel': [[] for _ in range(self.num_envs)],
                'dof_vel': [[] for _ in range(self.num_envs)],
            }
            # 单独缓存每个 env 的 motion time 序列，便于后续按 env 导出。
            self.motion_times_buf = [[] for _ in range(self.num_envs)]

        # 已经保存过首生命数据，不再重复记录
        if self._first_life_saved:
            # 已完成一次导出后，后续 step 全部直接返回。
            return

        # 计算当前参考动作时间：episode 已走步数 * dt + 该 env 的动作起始偏移。
        motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times

        # 当前时刻快照（CPU），供按 env 切分记录
        # 根平移位置（x,y,z）拷贝到 CPU，后续逐 env append。
        root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
        # print(f"写入.pkl的root_trans:{root_trans}")
        # 按 simulator 后端提取根四元数字段（不同后端四元数布局可能不同）。
        if self.config.simulator.config.name == "isaacgym":
            # IsaacGym 使用 [x,y,z,w] 存在 3:7。
            root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
        elif self.config.simulator.config.name == "isaacsim":
            # IsaacSim 这里按当前代码约定重排到 [x,y,z,w]。
            root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu()
        elif self.config.simulator.config.name == "genesis":
            # Genesis 与 IsaacGym 一致读取 3:7。
            root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
        else:
            # 未支持后端直接抛异常，避免静默写错数据。
            raise NotImplementedError

        # 把根四元数转成旋转向量（axis-angle 的 rotvec 形式），用于构造 pose_aa。
        root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float()
        # 当前关节位置拷贝到 CPU。
        dof = self.simulator.dof_pos.cpu()
        # 拼接 pose_aa：根旋转向量 + 每个 dof 的轴角 + augment 关节补零。
        pose_aa = torch.cat([
            root_rot_vec[:, None, :],
            self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None],
            torch.zeros((self.num_envs, self.num_augment_joint, 3)),
        ], axis=1)

        # 只记录“未发生首次 reset”的环境；本步 reset 的环境不写入（避免突变帧）
        # 先取当前“首生命活跃”掩码副本。
        active_mask = self._first_life_active.clone()#active_mask:当前这一帧需要记录的环境序号
        if hasattr(self, "reset_buf"):
            # 本步已 reset 的 env 置为 False，避免把突变帧写入轨迹。
            active_mask = active_mask & (~self.reset_buf.bool())

        # 把活跃掩码转为 Python 列表索引，便于 for 循环逐 env 追加。
        active_ids = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
        # 对每个仍在首生命且本步未 reset 的 env，逐 key 追加一帧数据。
        for env_idx in active_ids:
            # 根平移一帧。
            self.motions_for_saving['root_trans_offset'][env_idx].append(root_trans[env_idx].clone())
            # 根旋转四元数一帧。
            self.motions_for_saving['root_rot'][env_idx].append(root_rot[env_idx].clone())
            # 关节位置一帧。
            self.motions_for_saving['dof'][env_idx].append(dof[env_idx].clone())
            # 轴角姿态一帧。
            self.motions_for_saving['pose_aa'][env_idx].append(pose_aa[env_idx].clone())
            # actor 动作一帧（detach+cpu，避免图和显存绑定）。
            self.motions_for_saving['action'][env_idx].append(self.actions[env_idx].detach().cpu().clone())
            # actor 观测一帧。
            self.motions_for_saving['actor_obs'][env_idx].append(self.obs_buf_dict['actor_obs'][env_idx].detach().cpu().clone())
            # 终止标记一帧（当步通常为 False，因为 active_mask 已过滤 reset）。
            self.motions_for_saving['terminate'][env_idx].append(self.reset_buf[env_idx].detach().cpu().clone())
            # 关节速度一帧。
            self.motions_for_saving['dof_vel'][env_idx].append(self.simulator.dof_vel[env_idx].detach().cpu().clone())
            # 根线速度一帧。
            self.motions_for_saving['root_lin_vel'][env_idx].append(self.simulator.robot_root_states[env_idx, 7:10].detach().cpu().clone())
            # 根角速度一帧。
            self.motions_for_saving['root_ang_vel'][env_idx].append(self.simulator.robot_root_states[env_idx, 10:13].detach().cpu().clone())
            # 对应参考时间戳一帧。
            self.motion_times_buf[env_idx].append(motion_times[env_idx].detach().cpu().clone())

        # 诊断：记录每个 env 首次“掉出首生命”的步数和原因
        if hasattr(self, "reset_buf"):
            # 找到“这一步首次从活跃变为 reset”的 env（活跃且 reset_buf=True）。
            just_reset_mask = self._first_life_active & self.reset_buf.bool()#前面实际赋值这一帧是将self._first_life_active克隆到了active_mask，所以前者还没有改变
            # 转成 env id 列表。#这一帧just_reset_mask对应的环境在上面没有被赋值
            just_reset_ids = torch.nonzero(just_reset_mask, as_tuple=False).flatten().tolist()

            # 逐个刚 reset 的 env 记录首次 reset 信息。
            for env_idx in just_reset_ids:
                # 若此前已记录过首次 reset，直接跳过。
                if self._first_life_reset_step[env_idx].item() >= 0:
                    # continue 保证“首次”语义不被覆盖。
                    continue

                # 收集本 env 首次 reset 的命中原因（可多项并存）。
                reasons = []

                # timeout 原因
                if hasattr(self, "time_out_buf") and bool(self.time_out_buf[env_idx].item()):
                    # 命中超时（自然结束）。
                    reasons.append("timeout")

                # motion_far 原因（motion tracking 任务特有）
                if self.config.termination.terminate_when_motion_far:
                    # 重新按当前阈值计算该 env 是否 motion_far（用于诊断解释）。
                    motion_far_now = bool(torch.any(torch.norm(self.dif_global_body_pos[env_idx], dim=-1) > self.terminate_when_motion_far_threshold).item())
                    if motion_far_now:
                        # 命中 motion_far。
                        reasons.append("motion_far")

                # base 通用终止条件
                if self.config.termination.terminate_by_contact:
                    # 非允许触碰体发生大接触力则记 contact。
                    by_contact = bool(torch.any(torch.norm(self.simulator.contact_forces[env_idx, self.termination_contact_indices, :], dim=-1) > 1.).item())
                    if by_contact:
                        # 命中 contact。
                        reasons.append("contact")

                if self.config.termination.terminate_by_gravity:
                    # 判断重力投影 x 分量是否越界。
                    by_gravity_x = bool(torch.any(torch.abs(self.projected_gravity[env_idx, 0:1]) > self.config.termination_scales.termination_gravity_x).item())
                    # 判断重力投影 y 分量是否越界。
                    by_gravity_y = bool(torch.any(torch.abs(self.projected_gravity[env_idx, 1:2]) > self.config.termination_scales.termination_gravity_y).item())
                    if by_gravity_x or by_gravity_y:
                        # 命中姿态越界。
                        reasons.append("gravity")

                if self.config.termination.terminate_by_low_height:
                    # 判断基座高度是否低于最小高度阈值。
                    by_low_height = bool(torch.any(self.simulator.robot_root_states[env_idx, 2:3] < self.config.termination_scales.termination_min_base_height).item())
                    if by_low_height:
                        # 命中低高度终止。
                        reasons.append("low_height")

                if len(reasons) == 0:
                    # 若未匹配到显式规则，归类为未知或概率型终止。
                    reasons.append("unknown_or_probabilistic")

                # 记录首次 reset 发生的全局录制步。
                self._first_life_reset_step[env_idx] = self._first_life_step_counter
                # 记录首次 reset 原因（多原因用 + 连接）。
                self._first_life_reset_reason[env_idx] = "+".join(reasons)

                # 打印该 env 首次 reset 的诊断信息，便于追查为何丢轨迹。
                print(colored(
                    f"[first-life-diag] env={env_idx} first_reset_at_step={self._first_life_step_counter} "
                    f"ep_len={int(self.episode_length_buf[env_idx].item())} reason={self._first_life_reset_reason[env_idx]}",
                    "yellow"
                ))

        # 更新“首生命仍存活”掩码
        if hasattr(self, "reset_buf"):
            # 把本步 reset 的 env 从首生命活跃集合中永久移除。
            self._first_life_active = self._first_life_active & (~self.reset_buf.bool())
        # 全局录制步计数 +1。
        self._first_life_step_counter += 1

        # 若所有 env 首生命都结束，则可触发导出。
        all_first_life_finished = not bool(self._first_life_active.any().item())
        # 兜底上限：即使仍有 env 未 reset，超过 1000 步也强制导出。
        reached_max_steps = self._first_life_step_counter >= 1000

        # 满足任一导出条件时，组装并保存数据。
        if all_first_life_finished or reached_max_steps:
            # dump_data 结构：motion0/motion1/... 每个 env 一条轨迹。
            dump_data = {}
            # 统计最终真正写入的 env 数。
            saved_envs = 0
            # 要保存的所有键。
            keys_to_save = list(self.motions_for_saving.keys())
            # 与旧逻辑对齐：丢弃前 3 帧热身。
            warmup_trim = 0

            # 诊断：导出前打印每个 env 的首生命采样长度与保留情况
            # 打印总览标题。
            print(colored("[first-life-diag] per-env summary before dump:", "cyan"))
            # 逐 env 打印 seq_len 与是否会被保留。
            for env_idx in range(self.num_envs):
                # 该 env 首生命已缓存帧数。
                seq_len_i = len(self.motion_times_buf[env_idx])
                # 是否通过 warmup_trim 过滤。
                keep_i = seq_len_i > warmup_trim
                # 读取首次 reset 步（若存在诊断字段）。
                reset_step_i = int(self._first_life_reset_step[env_idx].item()) if hasattr(self, "_first_life_reset_step") else -1
                # 读取首次 reset 原因（若存在诊断字段）。
                reset_reason_i = self._first_life_reset_reason[env_idx] if hasattr(self, "_first_life_reset_reason") else ""
                # 输出该 env 的汇总诊断。
                print(colored(
                    f"  env={env_idx} seq_len={seq_len_i} keep={keep_i} warmup_trim={warmup_trim} "
                    f"first_reset_step={reset_step_i} reason={reset_reason_i if reset_reason_i else 'not_reset_yet'}",
                    "cyan"
                ))

            # 正式按 env 组装 dump_data。
            for env_idx in range(self.num_envs):
                # 当前 env 的轨迹长度。
                seq_len = len(self.motion_times_buf[env_idx])
                # 与旧逻辑保持一致：跳过最前面的 3 帧
                if seq_len <= warmup_trim:
                    # 太短则不写入（避免空数组或无意义片段）。
                    continue

                # motion{env_idx} 作为该 env 的导出键。
                motion_key = f"motion{env_idx}"
                # 对每个字段做 stack 后转 numpy，时间维在前。
                dump_data[motion_key] = {
                    key: torch.stack(self.motions_for_saving[key][env_idx][warmup_trim:], dim=0).numpy()
                    for key in keys_to_save
                }
                # 保存对齐后的 motion_times。
                dump_data[motion_key]['motion_times'] = torch.stack(self.motion_times_buf[env_idx][warmup_trim:], dim=0).numpy()
                # 记录帧率（1/dt）。
                dump_data[motion_key]['fps'] = 1 / self.dt
                # 成功保存 env 计数 +1。
                saved_envs += 1

            if saved_envs > 0:
                # 至少一个 env 有效，写入 pkl。
                joblib.dump(dump_data, f'{self.save_motion_dir}.pkl')
                # 打印成功信息与有效 env 数。
                print(colored(f"Saved first-life motion data to {self.save_motion_dir}.pkl (envs={saved_envs})", 'green'))
            else:
                # 没有任何 env 满足最小长度，提示未保存。
                print(colored("No valid first-life data collected; nothing saved.", 'yellow'))

            # 标记已保存，防止重复进入导出逻辑。
            self._first_life_saved = True
            # 按当前脚本设计，保存后立即退出进程。
            import sys
            # 结束运行，避免继续评估循环。
            sys.exit()

            

    # ############################################################
    # 以下函数群：提供各个零散的观测张量，会被上层收集成一个字典送入神经网络
    def _get_obs_dif_local_rigid_body_pos(self):
        return self._obs_dif_local_rigid_body_pos
    
    def _get_obs_local_ref_rigid_body_pos(self):
        return self._obs_local_ref_rigid_body_pos
    
    def _get_obs_ref_motion_phase(self):
        return self._ref_motion_phase
    
    def _get_obs_vr_3point_pos(self):
        return self._obs_vr_3point_pos

    ######################### Observations #########################
    def _get_obs_history_actor(self,):
        """获取演员网络(Actor)需要的历史信息序列（例如过去 5 帧的动作和状态，对抗物理延迟必带的武器）"""
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # 拍平成 [Batch, T*Dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_critic(self,):
        """获取评论家网络(Critic)需要的历史信息。Critic 用来估计价值，需要更多上帝视角的真值历史"""
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    ###############################################################

    # 以下函数群：强化学习的“奖励塑形 (Reward Shaping)”。
    # 使用标准公式： Reward = exp(-误差的平方 / 容忍度常量sigma)
    # 差得越少，Reward 越接近 1，差得远，Reward 衰减成 0。
    def _reward_teleop_body_position_extend(self):
        """奖励：让真实身体（甚至包括剑尖等扩展点）紧贴动画位置"""
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)#每个环境所有上半身刚体位置差的平方的平均值
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)#0.03
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)#0.1
        # 上下半身分别给权重的目的是：保证腿必须准（不然就摔了），手可以稍微飘一点。#0.5,1。
        r_body_pos = r_body_pos_lower * self.config.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.config.rewards.teleop_body_pos_upperbody_weight
    
        return r_body_pos
    
    def _reward_mpjpe(self):
        """奖励：让真实身体（甚至包括剑尖等扩展点）紧贴动画位置"""
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)#每个环境所有上半身刚体位置差的平方的平均值
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        # r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)#0.03
        # r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)#0.1
        # 上下半身分别给权重的目的是：保证腿必须准（不然就摔了），手可以稍微飘一点。#0.5,1。
        # r_body_pos = r_body_pos_lower * self.config.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.config.rewards.teleop_body_pos_upperbody_weight
        r_body_pos=diff_body_pos_dist_upper+diff_body_pos_dist_lower
    
        return r_body_pos*1000
    
    
    def _reward_teleop_vr_3point(self):
        """奖励：如果是遥控或者用3个核心点追踪，让真实末端贴紧那3个点"""
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        vr_3point_dist = (vr_3point_diff**2).mean(dim=-1).mean(dim=-1)
        r_vr_3point = torch.exp(-vr_3point_dist / self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)
        return r_vr_3point

    def _reward_teleop_body_position_feet(self):
        """重中之重奖励：脚部的精准跟随，防止滑步！"""
        feet_diff = self.dif_global_body_pos[:, self.feet_indices, :]
        feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        r_feet = torch.exp(-feet_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_pos)
        return r_feet
    
    def _reward_teleop_body_rotation_extend(self):
        """奖励：驱干姿态对齐"""
        rotation_diff = quat_to_angle_axis(self.dif_global_body_rot)[0]#轴角表示法，1维弧度和3维旋转轴。将四元数旋转差 self.dif_global_body_rot 转换为旋转角度（angle），结果保存在 rotation_diff 里。本质上，rotation_diff 表示每个刚体的旋转误差有多大（弧度制）。
        diff_body_rot_dist = (rotation_diff**2).mean(dim=-1)
        r_body_rot = torch.exp(-diff_body_rot_dist / self.config.rewards.reward_tracking_sigma.teleop_body_rot)
        return r_body_rot

    def _reward_teleop_body_velocity_extend(self):
        """奖励：线速度对齐"""
        velocity_diff = self.dif_global_body_vel    
        diff_body_vel_dist = (velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_vel = torch.exp(-diff_body_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_vel)
        return r_body_vel
    
    def _reward_teleop_body_ang_velocity_extend(self):
        """奖励：角速度对齐"""
        ang_velocity_diff = self.dif_global_body_ang_vel
        diff_body_ang_vel_dist = (ang_velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_ang_vel = torch.exp(-diff_body_ang_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_ang_vel)
        return r_body_ang_vel

    def _reward_teleop_joint_position(self):
        """底层奖励：让每个电机的实际位置去贴合参考位置。这个其实经常被调低权重，鼓励涌现行为"""
        joint_pos_diff = self.dif_joint_angles
        diff_joint_pos_dist = (joint_pos_diff**2).mean(dim=-1)
        r_joint_pos = torch.exp(-diff_joint_pos_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_pos)
        return r_joint_pos
    
    def _reward_teleop_joint_velocity(self):
        """底层奖励：电机转速对齐"""
        joint_vel_diff = self.dif_joint_velocities
        diff_joint_vel_dist = (joint_vel_diff**2).mean(dim=-1)
        r_joint_vel = torch.exp(-diff_joint_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_vel)
        return r_joint_vel
    
    def setup_visualize_entities(self):
        """
        【函数作用】：挂载那些虚拟的发光小球实体（用于追踪可视化）。
        【被谁调用】：由初始化物理世界环境最后阶段调用。
        """
        if self.debug_viz and self.config.simulator.config.name == "genesis":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        elif self.debug_viz and self.config.simulator.config.name == "mujoco":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        else:
            pass