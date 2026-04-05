import time # 导入时间模块，用于获取时间戳（记录数据帧的时间）和控制循环频率
import sys # 导入系统模块，用于处理系统退出和环境变量
import signal # 导入信号模块，用于捕获 Ctrl+C (SIGINT) 信号，以确保安全退出并保存数据
import numpy as np # 导入 numpy，用于高效的多维数组计算和数据存储（保存为 .npz 格式）
from collections import deque # 导入双端队列，用作固定大小的滑动窗口缓冲区（buffer），防止内存溢出
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize # 导入宇树 SDK 的通道订阅者和工厂初始化模块，用于接收机器人底层数据
# from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_ # （原代码注释掉的内容）用于 Go 系列机器人的底层状态和指令消息类型
from nav_msgs.msg import Odometry # 导入 ROS2 中的里程计消息类型，这里被借用来接收动作捕捉系统 (Mocap) 发送的机器人全局位姿和速度
import rclpy # 导入 ROS2 Python 客户端库，用于节点管理和消息订阅
from rclpy.node import Node # 导入 ROS2 的基础 Node 类，DataLogger 将继承自该类
import threading # 导入多线程模块，用于在后台单独运行键盘监听，不阻塞主数据收集循环
from pynput import keyboard # 导入 pynput 的键盘模块，用于监听用户按键（例如控制录制的开始与结束）
import os # 导入操作系统接口模块，用于处理文件路径和创建目录
import argparse # 导入命令行参数解析模块，用于接收运行脚本时传入的 config 和 exp_name 参数
import yaml # 导入 yaml 模块，用于读取机器人的配置文件（如包含自由度、接口等信息）
sys.path.append('./rl_policy/..') # 将上一级目录添加到系统路径，以便导入其他自定义模块
from utils.robot import Robot # 从自定义的 utils 模块中导入 Robot 类，用于获取机器人的关节数量等元数据

class DataLogger(Node):
    """
    【函数作用】: 核心数据记录类，继承自 ROS2 Node。负责同步接收并记录机器人的本体感觉数据（本体传感器）、策略指令（网络输出）以及外部动捕数据（全局基座状态）。
    【ASAP联系】: 这是构建 Delta-a 训练数据集(D^r = {s_0^r, a_0^r, ..., s_T^r, a_T^r})的引擎。
    【被谁调用】: 被本脚本底部的 main() 函数实例化并运行。
    """
    def __init__(self, save_path='data_log.npz', config=None, log_interval=1.0, buffer_rate=50.0, buffer_size=50*600, num_actions=29):
        """
        Initializes the DataLogger node.

        Args:
            save_path (str): Path to save the .npz log file. (保存 .npz 日志文件的路径，用于后续 Delta-a 训练)
            log_interval (float): Interval in seconds for throttled logging. (控制台打印日志的间隔时间，防止刷屏)
            buffer_rate (float): Frequency in Hz to update the observation buffer. (缓冲区更新频率，ASAP 中需要高频对齐真实状态与动作)
            buffer_size (int): Maximum number of entries to store in the buffer. (缓冲区最大容量，防止内存爆掉，默认存10分钟左右的数据)
        """
        super().__init__('data_logger') # 初始化父类 ROS2 Node，命名节点为 'data_logger'

        self.config = config # 保存传入的配置字典（包含机器人类型、通信接口等）
        # 根据配置文件中的机器人类型，动态导入对应型号的底层状态和底层指令数据结构
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2": # 如果是 h1 或 go2 机器人
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ # 导入 Go 系列的底层指令 (LowCmd: 包含目标 q, dq, kp, kd, tau)
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ # 导入 Go 系列的底层状态 (LowState: 包含实际 q, dq, 估算力矩, IMU)
        elif self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2": # 如果是 G1 (ASAP 论文中主要使用的 29 自由度人形机器人) 或 h1-2
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ # 导入 HG 系列 (G1) 的底层指令
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ # 导入 HG 系列 (G1) 的底层状态
        else: # 如果机器人类型不匹配，抛出未实现异常
            raise NotImplementedError(f"Robot type {self.config['ROBOT_TYPE']} is not supported yet")
        self.robot = Robot(config) # 实例化 Robot 对象，用于获取该型号机器人的具体参数
        self.num_actions = self.robot.NUM_JOINTS # 获取机器人的驱动关节数量（RL 动作空间的维度，例如 23 或 29），对应 a_t 的维度
        
        self.start_recording = False # 标志位：控制当前是否处于录制状态，默认不录制，等待用户按键触发
        self.motion_episode_cnt = 0 # 记录当前录制的是第几个动作片段（Episode），ASAP 中通常需要收集短小精悍的敏捷动作片段
        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True) # 创建后台线程运行键盘监听，daemon=True 保证主程序退出时该线程一并退出
        self.key_listener_thread.start() # 启动键盘监听线程

        # Initialize buffer for high-frequency data (200 Hz) (原注释：初始化高频数据缓冲区)
        self.obs_buffer = deque(maxlen=buffer_size)  # Buffer size is 200*300 = 60000 (使用 deque 作为缓冲区，超过 maxlen 后最旧的数据会被自动挤出。在 ASAP 中用于暂存一条完整的轨迹 D^r)

        # Initialize subscribers for mocap data (初始化外部动作捕捉系统的数据订阅者)
        # 动捕数据极其重要，因为 RL 中机器人的状态 s_t 往往需要全局坐标系的线速度和角速度，这在真机上仅靠 IMU 和编码器很难精确获取
        self.create_subscription(Odometry, "/odometry", self.mocap_callback, 10) # 订阅名为 "/odometry" 的 ROS2 话题，队列长度为 10，收到数据后调用 self.mocap_callback

        # Initialize subscribers for Unitree data (初始化宇树底层通信的订阅者，这里是直接监听局域网中的 DDS 消息)
        self.sub_state = ChannelSubscriber("rt/lowstate", LowState_) # 订阅底层状态话题 "rt/lowstate"，数据类型为 LowState_
        self.sub_state.Init(self.LowStateHandler, 10) # 初始化该订阅者，绑定回调函数 LowStateHandler，队列长度 10
                
        self.sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_) # 订阅底层控制指令话题 "rt/lowcmd"，也就是目前跑在真机上的预训练策略发出的 a_t
        self.sub_cmd.Init(self.LowCmdHandler, 10) # 初始化该订阅者，绑定回调函数 LowCmdHandler，队列长度 10

        # Set up the signal handler (设置信号处理器)
        signal.signal(signal.SIGINT, self.sigINT_handler) # 当用户在终端按下 Ctrl+C 时，触发 sigINT_handler，防止数据没保存就丢失

        # Throttled logging setup (日志限流设置，防止高频打印卡死终端)
        self.log_interval = log_interval  # seconds (设置日志打印的间隔时间)
        self.last_log_time = time.time() # 记录上一次打印日志的时间
        self.log_timer = self.create_timer(0.1, self.throttled_log_callback)  # High-frequency timer for checking log interval (创建一个 10Hz 的 ROS 定时器，周期性检查是否需要打印终端信息)

        # Buffer update timer at 200 Hz (设定核心的数据打包定时器频率，ASAP 的控制频率往往在 50Hz-200Hz)
        self.buffer_rate = buffer_rate  # Hz (从初始化参数读入的合并频率)
        self.buffer_timer = self.create_timer(1.0 / self.buffer_rate, self.buffer_update_callback) # 创建 ROS 定时器，按设定频率严格执行 buffer_update_callback 将零散数据打包进 buffer

        self.get_logger().info("DataLogger node initialized.") # 使用 ROS 的日志系统打印初始化完成的提示

        # Initialize latest data holders (初始化变量，用于存放三个异步数据源最近收到的一帧数据)
        self.latest_low_state = None # 存放最新的本体感觉数据（位置、速度、力矩、IMU）
        self.latest_low_cmd = None # 存放最新的策略输出指令（目标位置、速度、PD增益）
        self.latest_mocap = None # 存放最新的动捕基座姿态数据
        
        # Save path for the .npz file (存放当前实验保存路径)
        self.save_path = save_path # 将外界传入的存储路径赋值给实例变量

    def clear_buffer(self):
        """
        【函数作用】: 清空当前的观测数据缓冲区。
        【ASAP联系】: 在每次采集新的真实运动轨迹（例如一次新的跳跃动作录制开始时）调用，以保证每个 npz 文件对应一段干净的、完整的轨迹。
        【被谁调用】: 被 `start_key_listener` 中的按键分发逻辑调用。
        """
        self.obs_buffer.clear() # 调用 deque 自带的 clear 方法，清空所有元素
        self.get_logger().info("Buffer cleared.") # 控制台打印清空完毕的提示

    def start_key_listener(self):
        """
        【函数作用】: 启动独立线程的键盘监听功能。
        【ASAP联系】: 由于敏捷动作（Agile Skills）往往只有几秒钟，使用键盘触发录制（按';'）和结束录制并保存（按"'"）能精确捕捉有效数据段，提高 Delta-a 模型的学习效率。
        【被谁调用】: 在 __init__ 中作为独立线程的 target 启动。
        """
        def on_press(key): # 定义内部按键回调函数
            try:
                if key.char == ";": # 如果按下的键是分号 ';' (通常是准备让机器人执行某个敏捷动作)
                    self.start_recording = True # 开启录制标志，（不过注意看后面 process 代码，其实缓冲区一直在更新，这里只是清空了之前的无效等待数据）
                    self.clear_buffer() # 立刻清空旧数据，开始截取当前轨迹
                    self.get_logger().info("Start recording for episode {}".format(self.motion_episode_cnt)) # 打印开始第几次录制
                elif key.char == "'": # 如果按下的键是单引号 "'" (通常是机器人动作执行完毕)
                    self.start_recording = False # 停止录制标志
                    self.get_logger().info("Stop recording for episode {}".format(self.motion_episode_cnt)) # 打印停止录制提示
                    self.process_and_save_data() # **关键步骤**：调用函数将刚才收集到 buffer 里的轨迹处理成数组并存为 npz 硬盘文件
                    self.clear_buffer() # 存完后清空 buffer，为下一次录制做准备
                    self.motion_episode_cnt += 1 # 录制片段计数器加1
            except AttributeError: # 捕获特殊按键（如 Shift, Ctrl）没有 .char 属性的异常
                pass  # Handle special keys if needed (忽略特殊按键，不做处理)

        listener = keyboard.Listener(on_press=on_press) # 实例化 pynput 的监听器，绑定刚才定义的按键按下回调
        listener.start() # 启动键盘监听器
        listener.join()  # Keep the thread alive (阻塞当前后台线程，保持监听器持续存活)

    # Handler for Unitree low state messages
    def LowStateHandler(self, msg):
        """
        【函数作用】: 宇树底层状态（LowState）数据的回调处理函数。
        【ASAP联系】: 提取真实机器人状态 $s_t^r$ 的核心函数。获取电机的实时位置 q、速度 dq、估算力矩 tau_est，以及 IMU 的姿态，这些是仿真对齐（Sim-to-Real Alignment）的基准。
        【被谁调用】: 由 Unitree SDK 底层 C++ 通过网络收到机器人的 UDP/DDS 广播后异步回调。
        """
        timestamp = time.time() # 获取收到本帧数据的当前机器时间戳
        self.latest_low_state = { # 构造字典，覆盖更新实例变量 latest_low_state
            'timestamp': timestamp, # 存入时间戳
            # 以下提取前 num_actions 个电机的实际角度 (q)，ASAP 通常不训练末端手指，所以可能只取前 23/29 个腿臂关节
            'motor_positions': np.array([motor.q for motor in msg.motor_state[:self.num_actions]]), 
            # 提取电机的实际角速度 (dq)
            'motor_velocities': np.array([motor.dq for motor in msg.motor_state[:self.num_actions]]),
            # 提取电机当前的估算输出力矩 (tau_est)
            'motor_torques': np.array([motor.tau_est for motor in msg.motor_state[:self.num_actions]]),
            # 提取基座 IMU 的姿态四元数 (w, x, y, z)
            'imu_quaternion': np.array(msg.imu_state.quaternion),
            # 提取基座 IMU 的陀螺仪数据 (角速度)
            'imu_gyroscope': np.array(msg.imu_state.gyroscope),
            # 提取基座 IMU 的加速度计数据 (线加速度)
            'imu_accelerometer': np.array(msg.imu_state.accelerometer)
        }

    # Handler for Unitree low command messages
    def LowCmdHandler(self, msg):
        """
        【函数作用】: 宇树底层指令（LowCmd）数据的回调处理函数。
        【ASAP联系】: 记录预训练强化学习策略发出的原始参考动作 $a_t^r$。后续在仿真中训练 $\pi^\Delta$ 时，仿真器会把仿真策略加上这个原始动作进行前向模拟： $s_{t+1} = f^{sim}(s_t, a_t^r + \Delta a_t)$。
        【被谁调用】: 由 Unitree SDK 监听到本地/远端控制程序向机器人下发的指令时触发回调。
        """
        timestamp = time.time() # 获取收到本帧指令的当前机器时间戳
        self.latest_low_cmd = { # 构造字典，覆盖更新实例变量 latest_low_cmd
            'timestamp': timestamp, # 存入时间戳
            'motor_commands': { # 构建包含控制命令的子字典
                # RL 网络输出的目标关节角度 (Target Joint Position)
                "q": np.array([motor_cmd.q for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                # RL 网络输出的目标关节速度 (Target Joint Velocity，通常设为 0)
                "dq": np.array([motor_cmd.dq for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                # 关节控制的比例增益 (Stiffness / Kp)
                "kp": np.array([motor_cmd.kp for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                # 关节控制的微分增益 (Damping / Kd)
                "kd": np.array([motor_cmd.kd for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                # 前馈目标力矩 (Feedforward Torque)
                "tau": np.array([motor_cmd.tau for motor_cmd in msg.motor_cmd[:self.num_actions]])
            }
        }
        

    # Callback function for mocap data
    def mocap_callback(self, msg):
        """
        【函数作用】: ROS2 里程计（Odometry）话题的回调函数，用于接收动作捕捉（Mocap，如 OptiTrack, Vicon）传来的精确位姿数据。
        【ASAP联系】: 对于敏捷的全身动作（如跳跃、踢球），机器人基座的绝对位置 $p^{base}$ 和全局线速度 $v^{base}$ 极为关键。真机难以自身计算，需要外部 Mocap 辅助记录，以此作为仿真对照的真值（Ground Truth）。
        【被谁调用】: 由 ROS2 机制在节点收到 `/odometry` 话题的消息时异步触发。
        """
        timestamp = time.time() # 获取收到动捕数据的机器时间戳
        self.latest_mocap = { # 构造字典，覆盖更新实例变量 latest_mocap
            'timestamp': timestamp, # 存入时间戳
            'position': [ # 提取全局坐标系下的基座三维坐标 (x, y, z)
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ],
            'orientation': [ # 提取全局坐标系下的基座姿态四元数 (w, x, y, z，需注意特定系统的四元数排列顺序)
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z
            ],
            'linear_velocity': [ # 提取全局坐标系下的基座线速度 (vx, vy, vz)
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ],
            'angular_velocity': [ # 提取全局坐标系下的基座角速度 (wx, wy, wz)
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ]
        }

    # Throttled log callback
    def throttled_log_callback(self):
        """
        【函数作用】: 定时打印当前状态的简要日志。
        【ASAP联系】: 给研究人员提供监控反馈，确认数据采集系统（尤其是缓冲区的积累状况）运作正常，无延时堵塞。
        【被谁调用】: 由 __init__ 中设置的 self.log_timer 按照 10Hz (0.1秒) 的频率触发。
        """
        current_time = time.time() # 获取当前时间
        if (current_time - self.last_log_time) >= self.log_interval: # 如果距离上次打印的时间超过了设定的 log_interval (例如 1.0 秒)
            buffer_length = len(self.obs_buffer) # 计算当前滑动窗口 obs_buffer 里已经存了多少帧对齐好的数据
            self.get_logger().info( # 使用 ROS 日志打印
                f"Running... Buffer Size: {buffer_length}" # 输出 "运行中... 缓冲大小: XXX"
            )
            self.last_log_time = current_time # 更新上一次打印的时间戳

    # Buffer update callback at 200 Hz
    def buffer_update_callback(self):
        """
        【函数作用】: 最核心的数据同步组装函数。强制按照固定频率（如 ASAP 默认的 50-200Hz）抓取目前手头上最新的 State、Cmd 和 Mocap 数据，拼成一帧 Observation 存入队列。
        【ASAP联系】: 这是解决异步传感器频率不同步的关键。仿真中 RL 策略是以固定的 dt 进行步进的，因此我们需要离散的、等间隔的真实轨迹序列以供 Delta Action 学习。
        【被谁调用】: 由 __init__ 中设置的 self.buffer_timer 按照指定的 buffer_rate (如 50Hz) 定时触发。
        """
        # Fetch the latest data from each category if available (如果三大类数据均已收到至少一帧)

        if self.latest_low_state and self.latest_low_cmd and self.latest_mocap: # 检查是否已经从所有回调中获取了初始数据
            obs = { # 构建单帧的综合观测字典
                'timestamp': time.time(), # 统一采用组装时的当前时间作为该帧的时间戳
                'low_state': self.latest_low_state, # 放入最新的本体感觉状态
                'low_cmd': self.latest_low_cmd, # 放入最新的底层指令
                'mocap': self.latest_mocap # 放入最新的动捕信息
            }
            self.obs_buffer.append(obs) # 将这完整的一帧存入滑动队列 deque 的最右侧，如果超过 maxlen 会挤掉最左侧的旧数据
        else:
            # print(self.latest_low_cmd,self.latest_low_state,self.latest_mocap)
            # If any data is missing, log a warning (throttled) (如果任何一个通道的数据没有到位，触发节流警告打印)
            # print which data is missing (排查究竟是哪个硬件/通信链路掉线了)
            if not self.latest_low_state: # 如果缺少机器人状态数据
                self.get_logger().warn("Missing low state data for buffer update.", throttle_duration_sec=1.0) # ROS警告，并限制同类警告每秒只弹一次
            if not self.latest_low_cmd: # 如果缺少机器人指令数据
                self.get_logger().warn("Missing low command data for buffer update.", throttle_duration_sec=1.0) # 同上
            if not self.latest_mocap: # 如果缺少动作捕捉数据（实操中 OptiTrack 容易被遮挡掉帧）
                self.get_logger().warn("Missing mocap data for buffer update.", throttle_duration_sec=1.0) # 同上
            self.get_logger().warn("Incomplete data for buffer update.", throttle_duration_sec=1.0) # 总体提示：数据不完整，无法更新 buffer 帧

    # Signal handler to save data upon interruption
    def sigINT_handler(self, sig, frame):
        """
        【函数作用】: 捕获进程中断信号（如终端中按下 Ctrl+C）。
        【ASAP联系】: 真实硬件实验充满了意外（比如机器人眼看要摔倒，研究员急忙终止程序）。此函数保证了即使紧急关闭，也能正常打印退出信息和已完成的片段数。由于保存逻辑已移交到键盘分号/单引号控制，这里仅做善后打印。
        【被谁调用】: 操作系统的中断信号机制触发，或者在 main() 的 exception 捕获中手动调用。
        """
        # self.get_logger().info("SIGINT received. Saving data...") (原注释内容)
        # print("Saving data to ", self.save_path)
        # print("Buffer samples: ", len(self.obs_buffer))
        
        # # Process and save all collected data
        # self.process_and_save_data()
        
        # self.get_logger().info("Data saved successfully for episode {}".format(self.motion_episode_cnt))
        self.get_logger().info("Exiting..., total episodes: {}".format(self.motion_episode_cnt)) # 打印总共成功采集了几条轨迹数据
        self.get_logger().info("Data saved to {}".format(self.save_path)) # 打印数据存储的根目录
        sys.exit(0) # 干净地终止 Python 解释器并退出

    def process_and_save_data(self):
        """
        【函数作用】: 将缓冲队列中的字典列表拆解、重组，转换为 Numpy 多维数组，并将其打包压缩为一个 .npz 文件存储在硬盘。
        【ASAP联系】: 此函数生成的 .npz 文件直接作为 `train_delta_a.py` 的 Offline Dataset。包含了真实世界的状态、动作基准值，使得 $\pi^\Delta$ 可以通过最小化 MSE 来校准仿真引擎的物理表现。
        【被谁调用】: 被 `start_key_listener` 中的停止按键（"'"）逻辑调用。
        """
        # Initialize lists for processed data (初始化一大批空列表，用于分别存储当前 episode 中每帧抽出的特定数据项)
        buffer_timestamps = [] # 时间戳列表
        buffer_low_state_motor_positions = [] # 电机实际位置列表 (q)
        buffer_low_state_motor_velocities = [] # 电机实际速度列表 (dq)
        buffer_low_state_motor_torques = [] # 电机实际/估算力矩列表 (tau)
        buffer_low_state_imu_quaternion = [] # IMU 姿态四元数列表
        buffer_low_state_imu_gyroscope = [] # IMU 陀螺仪角速度列表
        buffer_low_state_imu_accelerometer = [] # IMU 加速度计列表
        # buffer_low_state_foot_force = [] (ASAP 可能未使用直接的足底力传感器)
        # buffer_low_state_foot_force_est = []

        buffer_low_cmd_q = [] # RL 策略发出的目标关节位置 (q_target)
        buffer_low_cmd_dq = [] # RL 策略发出的目标关节速度 (dq_target)
        buffer_low_cmd_kp = [] # 控制 PD 增益 kp
        buffer_low_cmd_kd = [] # 控制 PD 增益 kd
        buffer_low_cmd_tau = [] # 前馈力矩 tau_ff

        buffer_mocap_positions = [] # Mocap 记录的基座绝对空间位置 (x,y,z)
        buffer_mocap_orientations = [] # Mocap 记录的基座全局四元数
        buffer_mocap_linear_velocities = [] # Mocap 记录的基座绝对线速度
        buffer_mocap_angular_velocities = [] # Mocap 记录的基座绝对角速度

        # Iterate through the buffer and extract data (遍历当前缓冲区中积累的所有帧，拆分并装入对应的列表中)
        for obs in self.obs_buffer:
            buffer_timestamps.append(obs['timestamp']) # 追加时间戳

            # Low State (解包底层状态)
            low_state = obs['low_state']
            buffer_low_state_motor_positions.append(low_state['motor_positions'])
            buffer_low_state_motor_velocities.append(low_state['motor_velocities'])
            buffer_low_state_motor_torques.append(low_state['motor_torques'])
            
            buffer_low_state_imu_quaternion.append(low_state['imu_quaternion'])
            buffer_low_state_imu_gyroscope.append(low_state['imu_gyroscope'])
            buffer_low_state_imu_accelerometer.append(low_state['imu_accelerometer'])
            # buffer_low_state_foot_force.append(low_state['foot_force'])
            # buffer_low_state_foot_force_est.append(low_state['foot_force_est'])

            # Low Command (解包底层控制命令)
            low_cmd = obs['low_cmd']
            buffer_low_cmd_q.append(low_cmd['motor_commands']['q'])
            buffer_low_cmd_dq.append(low_cmd['motor_commands']['dq'])
            buffer_low_cmd_kp.append(low_cmd['motor_commands']['kp'])
            buffer_low_cmd_kd.append(low_cmd['motor_commands']['kd'])
            buffer_low_cmd_tau.append(low_cmd['motor_commands']['tau'])

            # Mocap (解包全局动捕数据)
            mocap = obs['mocap']
            buffer_mocap_positions.append(mocap['position'])
            buffer_mocap_orientations.append(mocap['orientation'])
            buffer_mocap_linear_velocities.append(mocap['linear_velocity'])
            buffer_mocap_angular_velocities.append(mocap['angular_velocity'])

        # Convert lists to NumPy arrays (将所有 Python 列表转换为 Numpy 数组，提升存储和后续在 IsaacGym 中加载的性能)
        buffer_timestamps = np.array(buffer_timestamps) # 变成形状为 (T,) 的数组，T是帧数

        buffer_low_state_motor_positions = np.array(buffer_low_state_motor_positions) # 变成形状为 (T, num_actions) 的数组
        buffer_low_state_motor_velocities = np.array(buffer_low_state_motor_velocities) # 同上
        buffer_low_state_motor_torques = np.array(buffer_low_state_motor_torques) # 同上
        buffer_low_state_imu_quaternion = np.array(buffer_low_state_imu_quaternion) # (T, 4)
        buffer_low_state_imu_gyroscope = np.array(buffer_low_state_imu_gyroscope) # (T, 3)
        buffer_low_state_imu_accelerometer = np.array(buffer_low_state_imu_accelerometer) # (T, 3)
        # buffer_low_state_foot_force = np.array(buffer_low_state_foot_force)
        # buffer_low_state_foot_force_est = np.array(buffer_low_state_foot_force_est)

        buffer_low_cmd_q = np.array(buffer_low_cmd_q) # (T, num_actions)
        buffer_low_cmd_dq = np.array(buffer_low_cmd_dq) # 同上
        buffer_low_cmd_kp = np.array(buffer_low_cmd_kp) # 同上
        buffer_low_cmd_kd = np.array(buffer_low_cmd_kd) # 同上
        buffer_low_cmd_tau = np.array(buffer_low_cmd_tau) # 同上

        buffer_mocap_positions = np.array(buffer_mocap_positions) # (T, 3)
        buffer_mocap_orientations = np.array(buffer_mocap_orientations) # (T, 4)
        buffer_mocap_linear_velocities = np.array(buffer_mocap_linear_velocities) # (T, 3)
        buffer_mocap_angular_velocities = np.array(buffer_mocap_angular_velocities) # (T, 3)
        

        import os # 再次导入 os（实际在开头已导入，这里属于局部重新引入）
        if not os.path.exists(os.path.dirname(self.save_path)): # 检查并确保存储目录存在，若不存在则递归创建
            os.makedirs(os.path.dirname(self.save_path))
            
            
        # Save data into npz file (将整理好的庞大 numpy 矩阵集合打包成一个 .npz 文件)
        npz_file_for_this_episode = os.path.join(self.save_path, f"motion_{self.motion_episode_cnt}.npz") # 为本段采集拼接专属文件名，如 motion_0.npz, motion_1.npz
        
        print("\n\n Saving data to ", npz_file_for_this_episode) # 在终端显式打印保存路径
        np.savez( # 使用 numpy 的 savez 进行字典式键值对存储
            npz_file_for_this_episode, 
            time=buffer_timestamps, # 保存时间轴
            joint_pos=buffer_low_state_motor_positions, # 保存机器人的绝对关节角度(重要状态 s_t)
            joint_vel=buffer_low_state_motor_velocities, # 保存机器人的绝对关节速度(重要状态 s_t)
            tau_est=buffer_low_state_motor_torques, # 估算力矩，可能用于动力学推断或约束
            IMU_quaternion=buffer_low_state_imu_quaternion, # 保存基座相对躯干的姿态
            IMU_gyro=buffer_low_state_imu_gyroscope, # 保存基座本地旋转速度
            IMU_acc=buffer_low_state_imu_accelerometer, # 保存基座本地线性加速度
            # foot_force=buffer_low_state_foot_force,
            # foot_force_est=buffer_low_state_foot_force_est,
            joint_pos_cmd=buffer_low_cmd_q, # 最重要：保存策略原封不动发出的 a_t，Delta-a 的本质是学习在它上面加个残差，让仿真器长得像真实的
            joint_vel_cmd=buffer_low_cmd_dq, # 保存发出的目标速度
            kp=buffer_low_cmd_kp, # 保存 PD 控制系数
            kd=buffer_low_cmd_kd,
            tau_cmd=buffer_low_cmd_tau, # 前馈控制值
            pos=buffer_mocap_positions, # 保存全局真值基座位置 (Sim2Real 对比指标，即 $p_t^{base}$)
            quat=buffer_mocap_orientations, # 保存全局真值基座姿态 ($\alpha_t^{base}$)
            lin_vel=buffer_mocap_linear_velocities, # 保存全局真值线速度 ($v_t^{base}$)
            ang_vel=buffer_mocap_angular_velocities # 保存全局真值角速度 ($\omega_t^{base}$)
        )

        self.get_logger().info(f"Data saved to {self.save_path}.") # 记录 ROS 级别的成功保存日志
        # print first 10 timestamps for all data (打印部分保存数据的概览，用于快速检查是否收集到了非零的合法数值)
        data = np.load(npz_file_for_this_episode, allow_pickle=True) # 使用 np.load 重新读取刚才保存的文件验证
        print("First 10 timestamps for all data:")
        for key in data.files: # 遍历 npz 内所有的数据键值对
            print(key, data[key][:10]) # 仅打印每个矩阵的前 10 行以供查阅验证

def main(args=None):
    """
    【函数作用】: 程序的入口。负责解析命令行参数、加载 YAML 配置、初始化通信通道和 ROS2 系统，最后拉起 DataLogger 节点。
    【ASAP联系】: 这个入口脚本将在实机实验期间持续运行，随时随地待命，接收来自策略脚本的指令，并将珍贵的 Sim-to-Real 数据固化。
    【被谁调用】: Python 解释器启动当前脚本 (`__main__` 块) 时调用。
    """
    parser = argparse.ArgumentParser(description='Robot') # 创建命令行参数解析器
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file') # 定义 --config 参数，默认读取 h1.yaml (存放了机器人构型、DDS通信的网卡名称等)
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name') # 定义 --exp_name 参数，用于区分不同动作或不同天收集的数据，避免覆盖
    args = parser.parse_args() # 解析用户从命令行传入的参数
    
    with open(args.config) as file: # 打开 YAML 配置文件
        config = yaml.load(file, Loader=yaml.FullLoader) # 将 YAML 解析为 Python 的字典对象

    exp_name = args.exp_name # 提取实验名

    save_path = './../humanoidverse/logs/delta_a_realdata'  # Default save path, enter folder sim2real to run this (定义收集数据的存储根目录，要求在 sim2real 文件夹下运行脚本)
    save_path = os.path.join(save_path, exp_name) # 根据用户设定的 exp_name 新建子目录
    current_timestamp = time.strftime("%Y%m%d-%H%M%S") # 抓取当前机器系统时间，按年月日-时分秒格式化
    save_path = os.path.join(save_path, current_timestamp) # 将时间戳作为最后一级子目录，确保每次运行不会相互覆盖旧数据
    if not os.path.exists(save_path): # 如果该深层路径还没在文件系统中创建
        os.makedirs(save_path) # 递归创建完整的文件夹架构
        

    # Initialize Unitree SDK (初始化宇树官方的高速通信 SDK，以便绕开繁杂的中间层，直接监听底层数据)
    if config.get("INTERFACE", None): # 检查配置文件中是否指定了网卡名（比如 'eth0' 或 'enp2s0'）
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"]) # 如果有，绑定指定网卡和 DOMAIN_ID 进行初始化（非常重要，防止网卡收不到多播数据）
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"]) # 否则让系统自动选择网卡（可能有坑，多网卡下容易丢包）
    

    # Initialize rclpy and create the DataLogger node (初始化 ROS2 系统)
    rclpy.init(args=None) # 让 ROS2 Python 客户端拉起内核支持
    data_logger = DataLogger(save_path=save_path, config=config) # 实例化前面定义的 DataLogger 这个核心数据收集节点

    try:
        rclpy.spin(data_logger) # 让 ROS2 开始阻塞循环（spin），从而激活所有 ROS Subscriber 回调函数的触发机制。此行后，各种底层数据回调和 timer 将开始欢快地运转
    except KeyboardInterrupt: # 如果在终端强制中断程序 (Ctrl+C)
        data_logger.sigINT_handler(None, None) # 捕获异常，并调用我们自定义的中断清理逻辑（安全退出）
    finally:
        data_logger.destroy_node() # 执行清理，释放 ROS 节点占用的内存和端口
        rclpy.shutdown() # 彻底关闭 ROS2 的后台调度引擎

if __name__ == '__main__':
    main() # 传统起手式，若作为主程序执行则调起 main() 逻辑