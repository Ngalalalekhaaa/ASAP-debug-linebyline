# 导入PyTorch库，用于处理深度学习中的张量（Tensor）数据
import torch
# 从torch模块导入Tensor类型，主要用于代码的类型提示（Type Hinting）
from torch import Tensor
# 导入Path库，用于跨平台的面向对象的文件系统路径操作
from pathlib import Path

# 导入matplotlib的pyplot模块，传统的绘图工具（虽然本代码主要使用plotly，但可能作为备用或遗留导入）
import matplotlib.pyplot as plt
# 导入numpy库，用于高效的数值计算和数组操作
import numpy as np
# 导入defaultdict，用于创建带有默认值（如空列表）的字典，方便记录日志数据
from collections import defaultdict
# 导入Process和Value，用于多进程操作（当前脚本主要使用了threading，这里可能是为了后续扩展或兼容）
from multiprocessing import Process, Value

# 导入plotly主库，用于构建强大的交互式Web图表
import plotly
# 导入plotly的graph_objects，用于构建底层的图表对象（如Scatter折线图）
import plotly.graph_objects as go
# 导入make_subplots，用于在一个画布上创建包含多个子图（Subplots）的网格
from plotly.subplots import make_subplots
# 导入Flask和render_template，用于搭建轻量级的本地Web服务器并渲染HTML模板
from flask import Flask, render_template
# 导入SocketIO，用于实现服务器与前端网页之间的WebSocket双向实时通信
from flask_socketio import SocketIO
# 导入threading模块，用于在后台单独开启一个线程运行Web服务器，不阻塞强化学习的主循环
import threading
# 导入json库，用于将Python数据结构（如图表配置）序列化为JSON格式传给前端
import json
# 导入send_file，用于在Flask路由中直接发送静态文件（如HTML模板）给客户端
from flask import send_file

# 从humanoidverse框架导入评估回调基类RL_EvalCallback，本类将继承它以插入评估生命周期
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
# 导入足式机器人的基础环境类，包含了机器人的物理属性、状态和仿真接口
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
# 导入PPO强化学习算法的训练循环类，用于获取当前策略和环境的引用
from humanoidverse.agents.ppo.ppo import PPO


# 定义图表分析的回调类，主要用于在策略评估期间记录状态并估计速度等信息
class AnalysisPlotLocomotionEstimateVel(RL_EvalCallback):
    """
    函数作用/类作用：
    这是一个用于强化学习评估阶段（Evaluation）的插件/回调类。
    它会在模型测试时（闭环或开环重放）持续抓取机器人的动作（actions）、
    实际关节状态（dof_pos）、速度、力矩以及接触力等数据，并通过WebLogger画出交互式曲线图。
    ASAP关联：
    在 ASAP 中，我们需要对比模型在仿真与真实环境下的跟踪误差。这个类实时可视化了
    机器人的“期望动作(target)”与“实际动作(actual)”的重合度，帮助研究人员直观判断
    Delta Action模型是否起效，以及动作是否导致电机力矩越界。
    """
    # 类型提示：指明training_loop是PPO算法的实例
    training_loop: PPO
    # 类型提示：指明env是LeggedRobotBase足式机器人环境的实例
    env: LeggedRobotBase

    def __init__(self, config, training_loop: PPO):
        """
        函数作用：初始化绘图回调类。
        调用时机/被谁调用：在运行 eval_agent.py 脚本加载模型时，由 Hydra 配置系统或 PPO 类的 _create_eval_callbacks 方法调用实例化。
        """
        # 调用父类RL_EvalCallback的初始化方法
        super().__init__(config, training_loop)
        # 获取当前 PPO 训练/评估循环绑定的物理环境实例
        env: LeggedRobotBase = self.training_loop.env
        # 将环境实例保存为类的成员变量，方便后续获取机器人状态
        self.env = env
        # 获取当前用于推理（Inference）的策略网络（Actor模型）
        self.policy = self.training_loop._get_inference_policy()
        # 获取环境中并行机器人的数量（num_envs，通常在评估时为1或几十个）
        self.num_envs = self.env.num_envs
        # 实例化 WebLogger，传入仿真环境的控制步长（sim_dt），用于计算X轴的时间戳
        self.logger = WebLogger(self.config.sim_dt)
        # 清空/初始化用于存储环境观测数据的缓冲区
        self.reset_buffers()
        # 从配置中读取布尔值，决定是只记录并绘制第0号机器人（单机），还是记录所有机器人的平均值
        self.log_single_robot = self.config.log_single_robot

    def reset_buffers(self):
        """
        函数作用：清空所有记录状态的列表，准备新一轮的评估收集。
        调用时机/被谁调用：在本类的 __init__ 中调用，或者在环境被重置（Reset）时调用。
        """
        # 为每个并行环境初始化一个空列表，用于存储Actor的观测数据
        self.obs_buf = [[] for _ in range(self.num_envs)]
        # 为每个并行环境初始化一个空列表，用于存储Critic的特权观测数据（如果有）
        self.critic_obs_buf = [[] for _ in range(self.num_envs)]
        # 为每个并行环境初始化一个空列表，用于存储Actor输出的动作数据
        self.act_buf = [[] for _ in range(self.num_envs)]

    def on_pre_evaluate_policy(self):
        """
        函数作用：在评估循环正式开始前，提取机器人的物理限制参数并传给绘图器。
        调用时机/被谁调用：被 PPO 的 evaluate_policy 方法在进入评估步进循环之前调用。
        ASAP关联：记录关节位置、速度、力矩的极限值，并在图表上画出边界线。如果运动策略太激进
        （经常触碰或超出黄色极限线），在现实中就会导致电机过热烧毁（论文中提到烧坏了两台G1）。
        """
        # Doing this in two lines because of type annotation issues. (原作者注释：分两行写是为了规避类型注解检查的问题)
        # 获取机器人的关节自由度总数（例如Unitree G1可能是23或29）
        self.robot_num_dofs = self.env.num_dofs
        # 从环境中获取所有关节的位置限制上限和下限，转为numpy数组
        self.log_dof_pos_limits = self.env.dof_pos_limits.cpu().numpy()
        # 从环境中获取所有关节的速度限制，转为numpy数组
        self.log_dof_vel_limits = self.env.dof_vel_limits.cpu().numpy()
        # 从环境中获取所有关节的力矩（Torque）限制，转为numpy数组
        self.log_dof_torque_limits = self.env.torque_limits.cpu().numpy()
        # 将上述三种物理极限范围发送给 WebLogger，以便在前端图表中画出背景辅助线/高亮区
        self.logger.set_robot_limits(self.log_dof_pos_limits, self.log_dof_vel_limits, self.log_dof_torque_limits)
        # 将机器人的总自由度数传递给 WebLogger，用于决定需要生成多少个子图（Subplots）
        self.logger.set_robot_num_dofs(self.robot_num_dofs)

    def on_post_evaluate_policy(self):
        """
        函数作用：在所有评估流程结束后的清理工作。
        调用时机/被谁调用：被 PPO 的 evaluate_policy 方法在退出评估循环后调用。当前此函数为空。
        """
        # 目前无需执行任何结束清理操作
        pass

    def on_pre_eval_env_step(self, actor_state):
        """
        函数作用：在仿真环境执行 env.step() 获取下一步状态【之前】拦截数据，记录当前的动作、速度和各种状态。
        调用时机/被谁调用：被 PPO 算法的评估循环内部，在决定了动作但还未送入物理引擎（Simulator）时调用。
        ASAP关联：动作(actions)在这里代表策略网络输出的期望关节位置（dof_pos_target）。
        ASAP的开环和闭环测试重点就是看这里的期望位置与下一帧的实际位置(self.env.dof_pos)之间的差距。
        """
        # 从策略网络的状态字典中提取出 Actor 的观测数据，并转移到 CPU 内存
        obs: Tensor = actor_state["obs"]["actor_obs"].cpu()
        # 提取 Critic 网络的特权观测数据（通常在Sim-to-Real时用于状态估计），并转移到 CPU 内存
        critic_obs: Tensor = actor_state["obs"]["critic_obs"].cpu()
        # 提取策略网络刚刚输出的动作指令（在位置控制下通常是目标关节角度），转移到 CPU 内存
        actions: Tensor = actor_state["actions"].cpu()
        # 提取状态估计器输出的估计速度（机器人在现实中无法直接获得真实的base_vel，需要估计）
        estimated_vel: Tensor = actor_state["estimator_output"].cpu()

        # 遍历所有并行的环境（通常评估时只需看0号环境，但这里支持多环境记录）
        for i in range(self.num_envs):
            # 将当前时间步的 actor_obs 存入缓存列表
            self.obs_buf[i].append(obs[i])
            # 将当前时间步的 critic_obs 存入缓存列表
            self.critic_obs_buf[i].append(critic_obs[i])
            # 将当前时间步的 动作指令 存入缓存列表
            self.act_buf[i].append(actions[i])

        # Note: we divide by 2.0 because the velocities are scaled by 2.0 in the obs space
        # (原作者注释：注意：我们除以2.0，因为观测空间中的速度被放大了2倍作为特征输入)
        # 如果配置项选择只记录单一机器人（通常是环境索引为0的那个）的数据
        if self.log_single_robot:
            # 组装要记录的字典数据传递给 WebLogger
            self.logger.log_states(
                {
                # 记录策略网络输出的目标关节位置（作为PD控制器的期望位置）
                'dof_pos_target': actions[0].cpu().numpy(),
                # 记录仿真引擎中机器人当前的实际关节位置
                'dof_pos': self.env.dof_pos[0].cpu().numpy(),
                # 记录仿真引擎中机器人当前的实际关节速度
                'dof_vel': self.env.dof_vel[0].cpu().numpy(),
                # 记录底层PD控制器计算出并施加的实际关节力矩
                'dof_torque': self.env.torques[0].cpu().numpy(),
                # 记录上位机给定的X方向线速度指令（遥控器推杆）
                'command_x': self.env.commands[0, 0].item(),
                # 记录上位机给定的Y方向线速度指令
                'command_y': self.env.commands[0, 1].item(),
                # 记录上位机给定的偏航角旋转速度指令
                'command_yaw': self.env.commands[0, 2].item(),
                # 记录真实的X方向躯干线速度
                'base_vel_x': self.env.base_lin_vel[0, 0].item(),
                # 记录真实的Y方向躯干线速度
                'base_vel_y': self.env.base_lin_vel[0, 1].item(),
                # 记录真实的Z方向躯干线速度
                'base_vel_z': self.env.base_lin_vel[0, 2].item(),
                # 记录真实的偏航角旋转速度
                'base_vel_yaw': self.env.base_ang_vel[0, 2].item(),
                # 记录机器人足部传感器获取的Z轴（垂直方向）接触力，用于判断是否踩稳地面
                'contact_forces_z': self.env.contact_forces[0, self.env.feet_indices, 2].cpu().numpy(),
                # 记录神经网络估计出的X方向线速度（需除以预处理缩放系数2.0还原真实物理值）
                'estimated_vel_x': estimated_vel[0, 0] / 2.0,
                # 记录估计出的Y方向线速度
                'estimated_vel_y': estimated_vel[0, 1] / 2.0,
                # 记录估计出的Z方向线速度
                'estimated_vel_z': estimated_vel[0, 2] / 2.0,
                }
            )
        # 如果不限定单机器人，则对所有环境并行运行的机器人的数据求平均值进行记录
        else:
            # log average of all robots (原作者注释：记录所有机器人的平均值)
            self.logger.log_states(
                {
                    # 计算所有机器人目标关节位置的均值
                    'dof_pos_target': actions.mean(dim=0).cpu().numpy(),
                    # 计算所有机器人实际关节位置的均值
                    'dof_pos': self.env.simulator.dof_pos.mean(dim=0).cpu().numpy(),
                    # 计算所有机器人实际关节速度的均值
                    'dof_vel': self.env.simulator.dof_vel.mean(dim=0).cpu().numpy(),
                    # 计算所有机器人实际关节力矩的均值
                    'dof_torque': self.env.torques.mean(dim=0).cpu().numpy(),
                    # 计算X方向指令的均值
                    'command_x': self.env.simulator.commands[:, 0].mean().item(),
                    # 计算Y方向指令的均值
                    'command_y': self.env.simulator.commands[:, 1].mean().item(),
                    # 计算偏航指令的均值
                    'command_yaw': self.env.simulator.commands[:, 2].mean().item(),
                    # 计算实际X线速度的均值
                    'base_vel_x': self.env.base_lin_vel[:, 0].mean().item(),
                    # 计算实际Y线速度的均值
                    'base_vel_y': self.env.base_lin_vel[:, 1].mean().item(),
                    # 计算实际Z线速度的均值
                    'base_vel_z': self.env.base_lin_vel[:, 2].mean().item(),
                    # 计算实际偏航角速度的均值
                    'base_vel_yaw': self.env.base_ang_vel[:, 2].mean().item(),
                    # 计算所有机器人足底接触力的均值
                    'contact_forces_z': self.env.simulator.contact_forces[:, self.env.feet_indices, 2].mean(dim=1).cpu().numpy(),
                    # 计算估计X线速度的均值并还原缩放
                    'estimated_vel_x': estimated_vel[:, 0].mean().item() / 2.0,
                    # 计算估计Y线速度的均值并还原缩放
                    'estimated_vel_y': estimated_vel[:, 1].mean().item() / 2.0,
                    # 计算估计Z线速度的均值并还原缩放
                    'estimated_vel_z': estimated_vel[:, 2].mean().item() / 2.0,
                }
            )
        # 返回未修改的 actor_state 给 PPO 循环继续使用
        return actor_state

    def on_post_eval_env_step(self, actor_state):
        """
        函数作用：在仿真环境执行完 env.step() 之后被调用，用于控制图表刷新频率。
        调用时机/被谁调用：被 PPO 算法评估循环内部，在仿真引擎推演物理效果之后调用。
        """
        # 从状态字典中获取当前的评估步数（Step）
        step = actor_state["step"]
        # 如果当前步数符合配置文件中设定的图表更新间隔（plot_update_interval）
        if (step + 1) % self.config.plot_update_interval == 0:
            # 触发 WebLogger，将收集到的最新数据推送到前端浏览器更新图表
            self.logger.plot_states()
        # 返回状态字典
        return actor_state


# 定义 WebLogger 类，封装了 Flask Web服务器和绘图逻辑
class WebLogger:
    """
    类作用：
    启动一个本地Web服务器，通过 WebSocket 实时向浏览器发送包含机器人运动状态的 Plotly 交互式图表。
    """
    def __init__(self, dt):
        """
        函数作用：初始化日志记录器，搭建Flask服务器和SocketIO实例。
        调用时机/被谁调用：被 AnalysisPlotLocomotionEstimateVel 的 __init__ 方法调用。
        """
        # 使用 defaultdict 创建一个字典，键不存在时自动初始化为空列表，用于存储每个状态的数值随时间的变化
        self.state_log = defaultdict(list)
        # 同理，创建一个字典用于记录每次 episode 的奖励（Rewards）随时间的变化
        self.rew_log = defaultdict(list)
        # 保存仿真环境的时间步长 dt（例如0.02秒/步），用于在绘图时将步数换算成具体的秒数
        self.dt = dt
        # 初始化记录的 episode 总数
        self.num_episodes = 0
        # 创建一个 Flask Web 应用实例
        self.app = Flask(__name__)
        # 使用 Flask 应用实例包装创建 SocketIO 对象，用于 WebSocket 双向实时通讯
        self.socketio = SocketIO(self.app)
        # 初始化线程变量为空，之后用于运行 Flask 服务器而不阻塞主训练/评估进程
        self.thread = None

    def set_robot_limits(self, dof_pos_limits, dof_vel_limits, dof_torque_limits):
        """
        函数作用：接收并存储机器人关节的物理极限阈值。
        调用时机/被谁调用：被 AnalysisPlotLocomotionEstimateVel 的 on_pre_evaluate_policy 调用。
        """
        # 保存关节位置极限
        self.log_dof_pos_limits = dof_pos_limits
        # 保存关节速度极限
        self.log_dof_vel_limits = dof_vel_limits
        # 保存关节力矩极限
        self.log_dof_torque_limits = dof_torque_limits

    def set_robot_num_dofs(self, num_dofs):
        """
        函数作用：接收并存储机器人的自由度总数，以此决定要画多少个子图。
        调用时机/被谁调用：被 AnalysisPlotLocomotionEstimateVel 的 on_pre_evaluate_policy 调用。
        """
        # 保存关节自由度数量（如23）
        self.robot_num_dofs = num_dofs

    def log_state(self, key, value):
        """
        函数作用：记录单一状态名和它的值到字典的末尾。
        调用时机/被谁调用：被本类的 log_states 方法循环调用。
        """
        # 把该变量当前步的 value 添加到名为 key 的列表中
        self.state_log[key].append(value)

    def log_states(self, dict):
        """
        函数作用：批量记录多个状态变量。
        调用时机/被谁调用：被 AnalysisPlotLocomotionEstimateVel 的 on_pre_eval_env_step 每个仿真步调用。
        """
        # 遍历传入字典的键值对，分别记录
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        """
        函数作用：批量记录环境返回的各种细分奖励项（Rewards）。
        调用时机/被谁调用：目前该代码片段中未显式调用，通常用于训练监控或者评估其任务奖励分布。
        """
        # 遍历传入的字典
        for key, value in dict.items():
            # 只有当键名包含 'rew' 时，才将其视作奖励值记录下来（乘以 num_episodes 可能是为了求总和或防止被平均缩小）
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        # 累加记录的 episode 数量
        self.num_episodes += num_episodes

    def reset(self):
        """
        函数作用：彻底清空状态和奖励记录缓存，释放内存。
        调用时机/被谁调用：当环境重启或需要清空重置绘图时调用。
        """
        # 清空状态日志字典
        self.state_log.clear()
        # 清空奖励日志字典
        self.rew_log.clear()

    def plot_states(self):
        """
        函数作用：触发前端图表更新，如果Web服务器线程还没启动则启动它。
        调用时机/被谁调用：被 AnalysisPlotLocomotionEstimateVel 的 on_post_eval_env_step 按时间间隔调用。
        """
        # 如果当前没有挂载运行 Web 服务器的后台线程
        if self.thread is None:
            # 创建一个新的守护线程，目标函数是本类的 _run_server 方法
            self.thread = threading.Thread(target=self._run_server)
            # 启动线程，服务器开始监听浏览器请求
            self.thread.start()
        # 调用内部方法，将现有数据打包更新并推送给图表
        self._update_plot()
    
    def _run_server(self):
        """
        函数作用：定义 Flask 路由规则，并启动 SocketIO 监听循环。
        调用时机/被谁调用：作为目标函数在 self.thread 这个子线程中独立运行。
        """
        # 定义Flask的根路由（URL: http://localhost:5000/）
        @self.app.route('/')
        def index():
            # 当浏览器访问根目录时，返回同目录下的 HTML 模板文件（内含前端接收WebSocket并渲染图表的JS逻辑）
            return send_file('analysis_plot_template.html')

        # 启动 SocketIO 服务，关闭调试模式和自动重载（防止在线程中引发冲突）
        self.socketio.run(self.app, debug=False, use_reloader=False)
    
    def _update_plot(self):
        """
        函数作用：核心绘图函数。根据收集到的日志生成基于 Plotly 的海量子图，并将包含图表信息的 JSON 数据推送到前端。
        调用时机/被谁调用：被 plot_states 方法定期调用。
        ASAP关联：通过这里的详细图表，作者可以直观对比每个关节在使用 ASAP Delta 动作微调前后，位置跟踪轨迹（红线vs蓝线）的贴合度。
        """
        # 拷贝当前的数据日志，防止在多线程迭代期间原始字典被修改导致报错
        log = self.state_log.copy()
        # 计算已记录数据的总时长（步数 * 仿真步长 dt，例如 100步 * 0.02秒 = 2.0秒）
        total_time = len(next(iter(log.values()))) * self.dt
        # 生成一个等差数列表示 X 轴的时间点数组
        time = np.linspace(0, total_time, len(next(iter(log.values()))))

        # 遍历日志字典中的所有键
        for key in log:
            # 如果对应的值类型是纯 Python 列表，为了绘图性能将其转化为 Numpy 数组
            if isinstance(log[key], list):
                log[key] = np.array(log[key])

        # 定义用于绘图的颜色常量（十六进制配色代码）
        BLUE = '#005A9D'    # 蓝色：通常用于表示实际发生的物理数值
        RED = '#DA2513'     # 红色：通常用于表示神经网络输出的目标期望值（Target/Command）
        YELLOW = '#EEDE70'  # 黄色：通常用于表示物理极限区或模型估计的辅助变量

        # 获取机器人的自由度总数，用于批量生成图表
        num_dofs = self.robot_num_dofs

        # 内部函数：用于动态生成 Plotly 所有子图的顶部标题文本
        def get_subplot_titles():
            # 初始化前8个固定用途的子图标题：躯干的 XYZ 和偏航线速度/指令，以及地面的垂直接触力
            titles = [
                'Base velocity x', 'Base velocity y', 'Base velocity yaw', 'Base velocity z',
                'Vertical Contact forces', 'Command velocities x', 'Command velocities y', 'Command velocities yaw'
            ]
            # 遍历每个自由度（如23次）
            for i in range(num_dofs):
                # 针对每个关节（DOF），添加4个子图标题：位置跟踪、速度跟踪、力矩跟踪、以及力矩-速度相图（散点图）
                titles.extend([f'DOF {i} Position', f'DOF {i} Velocity', f'DOF {i} Torque', f'DOF {i} Torque/Velocity'])
            # 返回所有合并好的标题列表
            return titles

        # 计算整个 Dashboard 需要的网格行数。前两行画基座速度和接触力等，后面每个自由度占一行
        num_rows = 2 + num_dofs

        # 创建 Plotly 的网格底图对象（Figure），指定行数、列数（固定为4列），并分配标题
        fig = make_subplots(rows=num_rows, cols=4, subplot_titles=get_subplot_titles())

        # 内部封装函数：简化在特定的行和列添加折线图（Trace）的代码
        def add_trace(x, y, color, row, col, name=None, show_legend=False, line_width=2):
            # 将 go.Scatter（散点/折线对象）添加到制定的子图坐标（row, col）上
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color, width=line_width), name=name, showlegend=show_legend), row=row, col=col)

        # 第1行：绘制躯干的线速度/角速度 以及 对应的网络估计值 和 遥控指令
        # 在第1行第1列：画出X轴物理线速度（蓝线）
        add_trace(time, log["base_vel_x"], BLUE, 1, 1, "Base vel x")
        # 在第1行第1列：画出X轴操作指令（红线）
        add_trace(time, log["command_x"], RED, 1, 1, "Command x")
        # 在第1行第2列：画出Y轴物理线速度（蓝线）
        add_trace(time, log["base_vel_y"], BLUE, 1, 2, "Base vel y")
        # 在第1行第2列：画出Y轴操作指令（红线）
        add_trace(time, log["command_y"], RED, 1, 2, "Command y")
        # 在第1行第3列：画出偏航角的物理角速度（蓝线）
        add_trace(time, log["base_vel_yaw"], BLUE, 1, 3, "Base vel yaw")
        # 在第1行第3列：画出偏航角操作指令（红线）
        add_trace(time, log["command_yaw"], RED, 1, 3, "Command yaw")
        # 在第1行第4列：画出Z轴（上下）的物理线速度（蓝线）
        add_trace(time, log["base_vel_z"], BLUE, 1, 4, "Base vel z")
        
        # 将 Critic/Estimator 估计出的线速度画在相同的图表里（黄线，较细），用于评估状态估计器的准确性
        add_trace(time, log["estimated_vel_x"], YELLOW, 1, 1, "Estimated vel x", line_width=1)
        add_trace(time, log["estimated_vel_y"], YELLOW, 1, 2, "Estimated vel y", line_width=1)
        add_trace(time, log["estimated_vel_z"], YELLOW, 1, 3, "Estimated vel z", line_width=1)

        # 第2行：绘制垂直接触力及复用显示指令
        # 提取各个落脚点的垂直接触力日志记录
        forces = log["contact_forces_z"]
        # 遍历机器人的每只脚（比如双足机器人有2只脚索引）
        for i in range(forces[0].shape[0]):
            # 在第2行第1列的子图中，画出这只脚随时间变化的接触力（判断腾空和落地的相位）
            add_trace(time, [force[i] for force in forces], BLUE, 2, 1, f"Force {i}")

        # 在第2行剩下的子图里单独显示操作杆的 X/Y/Yaw 指令
        add_trace(time, log["command_x"], BLUE, 2, 2, "Command x", True)
        add_trace(time, log["command_y"], BLUE, 2, 3, "Command y", True)
        add_trace(time, log["command_yaw"], BLUE, 2, 4, "Command yaw", True)

        # 内部封装函数：用于在图表背景画出一个半透明的黄色矩形区域，表示机器人的物理极限（超过会报错或烧毁硬件）
        def add_limit_lines(row, col, lower, upper, color=YELLOW):
            # 添加一个覆盖整个时间轴，高度为极限下限到上限的矩形形状，置于折线图图层之下（layer='below'）
            fig.add_shape(type="rect", x0=time[0], x1=time[-1], y0=lower, y1=upper,
                        fillcolor=color, line=dict(width=0), layer='below', row=row, col=col)

        # 第3行及以后：为每个关节单独绘制四个子图分析
        for i in range(num_dofs):
            row = i + 3  # 从图表的第三行开始排布

            # 第1列：关节位置（Position）分析 - 这对ASAP极为关键，体现了网络策略与物理引擎能否完美匹配
            # 画出当前关节的实际角度变化（蓝线）
            add_trace(time, [pos[i] for pos in log["dof_pos"]], BLUE, row, 1, f"DOF {i} pos")
            # 画出网络给出的该关节的目标角度变化（红线）
            add_trace(time, [pos[i] for pos in log["dof_pos_target"]], RED, row, 1, f"DOF {i} pos target")
            # 将该关节的机械限位（上死点/下死点）画为背景高亮区
            add_limit_lines(row, 1, self.log_dof_pos_limits[i, 0], self.log_dof_pos_limits[i, 1])
            
            # 第2列：关节速度（Velocity）分析
            # 画出当前关节的实际角速度变化（蓝线）
            add_trace(time, [vel[i] for vel in log["dof_vel"]], BLUE, row, 2, f"DOF {i} vel")
            # 将该关节电机的最大转速限制画为背景高亮区（通常是对称的，从-limit到+limit）
            add_limit_lines(row, 2, -self.log_dof_vel_limits[i], self.log_dof_vel_limits[i])
            
            # 第3列：关节力矩（Torque）分析
            # 画出当前关节底层控制器施加的实际力矩（蓝线）
            add_trace(time, [torque[i] for torque in log["dof_torque"]], BLUE, row, 3, f"DOF {i} torque")
            # 将该关节电机的最大峰值扭矩限制画为背景高亮区
            add_limit_lines(row, 3, -self.log_dof_torque_limits[i], self.log_dof_torque_limits[i])
            
            # 第4列：力矩-速度相图（Torque/Velocity Curve）
            # 这反映了电机的工作状态空间。电机在高速时往往输出不了大扭矩（反电动势导致）。
            # 用散点图表示（x=速度，y=力矩）
            fig.add_trace(go.Scatter(
                x=[vel[i] for vel in log["dof_vel"]], 
                y=[torque[i] for torque in log["dof_torque"]], 
                mode='markers', 
                marker=dict(color=BLUE, size=2), 
                showlegend=False,
                name=f"DOF {i} Torque/Velocity"
            ), row=row, col=4)

            # 为相图画出边界框（黄线）。超出这个框代表超出了电机的物理性能边界。
            # 画左边界（最小速度边界的垂线）
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=-self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            # 画右边界（最大速度边界的垂线）
            fig.add_shape(type="line", x0=self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

            # 画下边界（最小扭矩边界的水平线）
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=-self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            # 画上边界（最大扭矩边界的水平线）
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

        # 更新整个图表的布局：根据行数自适应高度（每行给300像素），宽度固定为1500像素，并设定大标题
        fig.update_layout(height=300*num_rows, width=1500, title_text="Robot State Plots", showlegend=True)
        
        # 批量更新所有的 X 轴和 Y 轴文本标签
        for i in range(num_rows):
            # 对于每一行的前3列子图，由于是按时间展开的，X轴设为时间"time [s]"
            for j in range(3):
                fig.update_xaxes(title_text="time [s]", row=i+1, col=j+1)
            # 第4列是相图，X轴代表速度，不需要写时间标签
            fig.update_xaxes(title_text="", row=i+1, col=4)
        
        # 手动更新第1、2行关键状态子图的 Y 轴标签
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=1)
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=2)
        fig.update_yaxes(title_text="base ang vel [rad/s]", row=1, col=3)
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=4)
        fig.update_yaxes(title_text="Forces z [N]", row=2, col=1)
        fig.update_yaxes(title_text="Command vel", row=2, col=2)
        fig.update_yaxes(title_text="Command vel", row=2, col=3)
        fig.update_yaxes(title_text="Command vel", row=2, col=4)

        # 批量更新第3行及以后各个关节图表的 Y 轴标签
        for i in range(3, num_rows + 1):
            fig.update_yaxes(title_text="Position [rad]", row=i, col=1)
            fig.update_yaxes(title_text="Velocity [rad/s]", row=i, col=2)
            fig.update_yaxes(title_text="Torque [Nm]", row=i, col=3)
            fig.update_yaxes(title_text="Torque/Velocity", row=i, col=4)

        # 将生成的庞大 Plotly Figure 对象序列化为标准的 JSON 文本字符串
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        # 通过 SocketIO 发送一个名为 'update_plots' 的事件，将 JSON 图表数据推给连接的前端浏览器，完成实时无刷新更新
        self.socketio.emit('update_plots', plot_json)

    def print_rewards(self):
        """
        函数作用：打印累计评估奖励的统计结果（平均值）。
        调用时机/被谁调用：通常在评估循环彻底结束后，被用户手动调用或外部脚本调用以审查性能。
        """
        print("Average rewards per second:")
        # 遍历所有存储的奖励项
        for key, values in self.rew_log.items():
            # 计算该项奖励的均值
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        # 打印总评估局数
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        """
        函数作用：析构函数。在 WebLogger 实例被垃圾回收销毁时，确保释放服务器资源并关闭进程。
        调用时机/被谁调用：Python 垃圾回收器自动调用。
        """
        # 如果后台服务器线程仍在运行
        if self.thread:
            # 命令 SocketIO 停止服务
            self.socketio.stop()
            # 阻塞主进程直到子线程安全关闭完毕，防止僵尸进程
            self.thread.join()