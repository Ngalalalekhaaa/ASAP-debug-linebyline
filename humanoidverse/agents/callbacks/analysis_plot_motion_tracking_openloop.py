import torch
from torch import Tensor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import json
from flask import send_file
# 导入 smpl_sim 中的评估工具函数，用于计算 ASAP 论文中的运动误差指标（如速度、加速度误差等）
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite, compute_error_vel, compute_error_accel
from tqdm import tqdm

# 导入底层强化学习框架和环境依赖
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.agents.ppo.ppo import PPO

class AnalysisPlotMotionTrackingOpenloop(RL_EvalCallback):
    # 【类作用说明】
    # 这是一个继承自 RL_EvalCallback 的回调类（插件）。
    # 它的主要作用是在 PPO 的评估阶段被挂载，用于在每次环境步进前后拦截数据，
    # 记录开环控制下的机器人的状态、动作、完美残差动作 (perfect_delta_a)，
    # 并计算 ASAP 论文中提到的核心评价指标（如 MPJPE），最后通过 WebLogger 进行图表绘制。
    
    training_loop: PPO
    env: LeggedRobotBase

    def __init__(self, config, training_loop: PPO):
        # 【函数说明：初始化函数】
        # 作用：初始化回调类，挂载相关的环境、策略，并实例化基于网页的实时绘图工具 (WebLogger)。
        # 调用者：在 PPO 的 `evaluate_policy()` 函数内部通过 Hydra 的 `instantiate` 被调用（如之前的报错日志所示）。
        super().__init__(config, training_loop)
        env: LeggedRobotBase = self.training_loop.env
        self.env = env
        self.policy = self.training_loop._get_inference_policy() # 获取用于推理（评估）的控制策略
        self.num_envs = self.env.num_envs # 获取并行环境的数量
        self.logger = WebLogger(self.config.sim_dt) # 实例化 Web 日志记录器，传入仿真步长 dt
        self.reset_buffers() # 初始化/重置数据缓存区
        self.log_single_robot = self.config.log_single_robot # 是否只记录并绘制单个机器人的数据
        self.compute_metrics = self.config.compute_metrics # 是否计算复杂的评价指标（如 MPJPE 等）

    def reset_buffers(self):
        # 【函数说明：重置缓存】
        # 作用：清空观测、Critic观测以及动作的缓存列表，为新一轮评估做准备。
        # 调用者：本类的 __init__ 方法，或者在需要重新开始记录时被调用。
        self.obs_buf = [[] for _ in range(self.num_envs)] # Actor 观测历史缓存
        self.critic_obs_buf = [[] for _ in range(self.num_envs)] # Critic 观测历史缓存
        self.act_buf = [[] for _ in range(self.num_envs)] # 策略输出动作的缓存

    def on_pre_evaluate_policy(self):
        # 【函数说明：评估开始前触发】
        # 作用：在正式开始一轮策略评估前，提取机器人的关节物理极限（位置、速度、扭矩），
        # 传给绘图器用于画出限制边界线。如果是定量评估模式，则初始化一系列误差统计容器。
        # 调用者：PPO 评估循环的起始阶段（在开始环境的 reset 和 step 之前）。
        
        # Doing this in two lines because of type annotation issues.
        self.robot_num_dofs = self.env.num_dofs # 机器人的自由度数量（如 G1 的 23 或 29）
        # 提取各个自由度的位置、速度、扭矩极限，并转为 numpy 数组
        self.log_dof_pos_limits = self.env.dof_pos_limits.cpu().numpy()
        self.log_dof_vel_limits = self.env.dof_vel_limits.cpu().numpy()
        self.log_dof_torque_limits = self.env.torque_limits.cpu().numpy()
        
        # 将极限值和自由度数量传递给 WebLogger，用于在 UI 上绘制黄色的物理边界参考线
        self.logger.set_robot_limits(self.log_dof_pos_limits, self.log_dof_vel_limits, self.log_dof_torque_limits)
        self.logger.set_robot_num_dofs(self.robot_num_dofs)
        
        if self.compute_metrics:
            # 初始化一个进度条，长度为 唯一动作数 / 并行环境数 （用于追踪多个模仿动作的测试进度）
            self.pbar = tqdm(range(self.env._motion_lib._num_unique_motions // self.env.num_envs))
            self.success_rate = 0 # 初始化成功率
            self.curr_stpes = 0 # 当前执行的步数
            self.env.start_compute_metrics() # 告诉环境开启指标计算相关的计算流
            
            # 记录环境中机器人的终止（跌倒/失败）状态
            self.terminate_state = torch.zeros(self.env.num_envs, device=self.device)
            self.terminate_memory = [] # 记录每一帧的终止状态历史
            
            # ASAP 论文中的核心误差指标：MPJPE (Mean Per-Joint Position Error)
            self.mpjpe, self.mpjpe_all = [], []
            
            # Ground Truth (参考目标) 和 Prediction (模型实际仿真) 的位置与旋转数据容器
            self.gt_pos, self.gt_pos_all = [], []
            self.gt_rot, self.gt_rot_all = [], []
            self.pred_pos, self.pred_pos_all = [], []
            self.pred_rot, self.pred_rot_all = [], []
            
        
        

    def on_post_evaluate_policy(self):
        # 【函数说明：评估结束后触发】
        # 作用：评估全部跑完后的收尾工作，当前为空。
        # 调用者：PPO 评估循环结束时。
        pass

    def on_pre_eval_env_step(self, actor_state):
        # 【函数说明：每次环境 step() 前触发】
        # 作用：在环境实际执行物理推进前，获取策略计算出的当前动作 (actions) 和状态，
        # 同时提取开环指令 (open_loop_actions) 和 ASAP 论文中的完美残差模型 (perfect_delta_a)，并把它们发给绘图器。
        # 调用者：PPO 评估循环中，调用 `env.step()` 之前。
        
        # 从传入的 actor_state 字典中把张量拉回 CPU
        obs: Tensor = actor_state["obs"]["actor_obs"].cpu()
        critic_obs: Tensor = actor_state["obs"]["critic_obs"].cpu()
        actions: Tensor = actor_state["actions"].cpu()
        

        # 遍历所有并行环境，保存数据到本地 buffer 中
        for i in range(self.num_envs):
            self.obs_buf[i].append(obs[i])
            self.critic_obs_buf[i].append(critic_obs[i])
            self.act_buf[i].append(actions[i])

        if self.log_single_robot:
            # 如果只看 0 号机器人
            # 计算原始的开环动作 (乘以缩放系数)
            open_loop_actions = self.env.get_open_loop_action_at_current_timestep() * self.env.config.robot.control.action_scale
            # 获取完美的 Delta-a (残差动作)，ASAP 中补偿 Sim-to-Real 鸿沟的核心
            perfect_delta_a = self.env._get_perfect_delta_a()
            
            # 将物理状态数据交给 logger，用于在网页端绘制实时曲线
            self.logger.log_states(
                {
                'dof_pos_target': actions[0].cpu().numpy(), # 模型计算出的目标关节位置
                'dof_pos': self.env.dof_pos[0].cpu().numpy(), # 仿真器中机器人的真实关节位置
                'dof_open_loop_actions': open_loop_actions[0].cpu().numpy(), # 预设的开环原始动作
                'dof_perfect_delta_a': perfect_delta_a[0].cpu().numpy(), # ASAP 的残差补偿值
                'dof_vel': self.env.dof_vel[0].cpu().numpy(), # 关节速度
                'dof_torque': self.env.torques[0].cpu().numpy(), # 关节输出扭矩
                'base_vel_x': self.env.base_lin_vel[0, 0].item(), # 基座(根节点) X轴速度
                'base_vel_y': self.env.base_lin_vel[0, 1].item(), # 基座(根节点) Y轴速度
                'base_vel_z': self.env.base_lin_vel[0, 2].item(), # 基座(根节点) Z轴速度
                'base_vel_yaw': self.env.base_ang_vel[0, 2].item(), # 基座(根节点) 偏航角速度
                'contact_forces_z': self.env.contact_forces[0, self.env.feet_indices, 2].cpu().numpy() # 脚部的垂直接触力
                }
            )
        else:
            # log average of all robots (记录并求所有并行环境机器人的平均值，参数逻辑同上)
            self.logger.log_states(
                {
                    'dof_pos_target': actions.mean(dim=0).cpu().numpy(),
                    'dof_pos': self.env.simulator.dof_pos.mean(dim=0).cpu().numpy(),
                    'dof_open_loop_actions': self.env.get_open_loop_action_at_current_timestep().mean(dim=0).cpu().numpy(),
                    'dof_perfect_delta_a': self.env._get_perfect_delta_a().mean(dim=0).cpu().numpy(),
                    'dof_vel': self.env.simulator.dof_vel.mean(dim=0).cpu().numpy(),
                    'dof_torque': self.env.torques.mean(dim=0).cpu().numpy(),
                    'base_vel_x': self.env.base_lin_vel[:, 0].mean().item(),
                    'base_vel_y': self.env.base_lin_vel[:, 1].mean().item(),
                    'base_vel_z': self.env.base_lin_vel[:, 2].mean().item(),
                    'base_vel_yaw': self.env.base_ang_vel[:, 2].mean().item(),
                    'contact_forces_z': self.env.simulator.contact_forces[:, self.env.feet_indices, 2].mean(dim=1).cpu().numpy()
                }
            )
        return actor_state

    def on_post_eval_env_step(self, actor_state):
        # 【函数说明：每次环境 step() 后触发】
        # 作用：环境物理推演完成一步后，提取真实的全局三维姿态和旋转，计算跟踪误差(MPJPE)，
        # 判断机器人是否跌倒失败。当动作播完后，汇总计算整段动作的指标并打印结果。触发 Web 绘图更新。
        # 调用者：PPO 评估循环中，调用 `env.step()` 之后。
        
        step = actor_state["step"]
        
        if self.compute_metrics:
            # 记录参考动作（即人类动捕数据重定向后的动作）的坐标和旋转
            self.gt_pos.append(self.env.extras['ref_body_pos_extend'].cpu().numpy())
            self.gt_rot.append(self.env.extras['ref_body_rot_extend'].cpu().numpy())
            
            # 记录机器人在仿真器里跑出来的实际坐标和旋转
            self.pred_pos.append(self.env._rigid_body_pos_extend.cpu().numpy())
            self.pred_rot.append(self.env._rigid_body_rot_extend.cpu().numpy())
            
            # 计算全局身体位置差值的 L2 范数（乘以1000转为毫米 mm），即 MPJPE 误差指标
            self.mpjpe.append(self.env.dif_global_body_pos.norm(dim=-1).cpu() * 1000)
            
            
            died = actor_state['dones'] # 从环境返回结果看机器人是不是死掉了（跌倒了）
            died[actor_state['extras']["time_outs"]] = False # 如果只是超时（动作放完了），不属于真正跌倒失败
            
            # 判断逻辑：如果当前步数还没有超过动作总帧数，且判定为 died，说明是提前失败了
            termination_state = torch.logical_and(self.curr_stpes <= self.env._motion_lib.get_motion_num_steps() - 1, died) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
            self.terminate_state = torch.logical_or(termination_state, self.terminate_state) # 只要跌倒过一次，这个 episode 就标记为失败
            
            # 这段逻辑用来处理动作的帧长对齐问题，因为库里有多种不同长度的动作同时在多环境并行播放
            if (~self.terminate_state).sum() > 0:
                max_possible_id = self.env._motion_lib._num_unique_motions - 1
                curr_ids = self.env._motion_lib._curr_motion_ids
                if (max_possible_id == curr_ids).sum() > 0: # When you are running out of motions. 
                    bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                    if (~self.terminate_state[:bound]).sum() > 0:
                        curr_max = self.env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
                    else:
                        curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
                else:
                    curr_max = self.env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

                if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
            else:
                curr_max = self.env._motion_lib.get_motion_num_steps().max()
                
            self.curr_stpes += 1
            
            # 如果跑到了动作最大帧数，或者所有并行环境的机器人都已经跌倒了，则进行一次总结算
            if self.curr_stpes >= curr_max or self.terminate_state.sum() == self.env.num_envs:
                
                self.terminate_memory.append(self.terminate_state.cpu().numpy())
                # 计算成功率：1 减去失败机器人的比例
                self.success_rate = (1 - np.concatenate(self.terminate_memory)[: self.env._motion_lib._num_unique_motions].mean())

                # 处理 ASAP 中的 MPJPE (平均单关节位置误差) 序列
                all_mpjpe = torch.stack(self.mpjpe)
                try:
                    assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == self.env.num_envs) # Max should be the same as the number of frames in the motion.
                except:
                    import ipdb; ipdb.set_trace()
                    print('??')

                # 堆叠所有的位置和旋转数组
                all_body_pos_pred = np.stack(self.pred_pos)
                all_body_pos_gt = np.stack(self.gt_pos)
                all_body_rot_pred = np.stack(self.pred_rot)
                all_body_rot_gt = np.stack(self.gt_rot)
                
                # 对有效帧内的数据进行截取平均（忽略第一帧，所以 - 1）
                all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())] # -1 since we do not count the first frame. 
                all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]
                all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]
                all_body_rot_pred = [all_body_rot_pred[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]
                all_body_rot_gt = [all_body_rot_gt[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]


                # 将当次计算的指标追加到全局汇总列表中
                self.mpjpe_all.append(all_mpjpe)
                self.pred_pos_all += all_body_pos_pred
                self.gt_pos_all += all_body_pos_gt
                self.pred_rot_all += all_body_rot_pred
                self.gt_rot_all += all_body_rot_gt

                # 如果所有的独特动作 (unique motions) 都评估完毕了
                if (self.env.start_idx + self.env.num_envs >= self.env._motion_lib._num_unique_motions):
                    terminate_hist = np.concatenate(self.terminate_memory)
                    # 提取成功完成的动作索引
                    succ_idxes = np.nonzero(~terminate_hist[: self.env._motion_lib._num_unique_motions])[0].tolist()

                    # 分别提取“仅成功的动作”的数据和“所有动作”的数据
                    pred_pos_all_succ = [(self.pred_pos_all[:self.env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    gt_pos_all_succ = [(self.gt_pos_all[: self.env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    pred_rot_all_succ = [(self.pred_rot_all[: self.env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    gt_rot_all_succ = [(self.gt_rot_all[: self.env._motion_lib._num_unique_motions])[i] for i in succ_idxes]

                    pred_pos_all = self.pred_pos_all[:self.env._motion_lib._num_unique_motions]
                    gt_pos_all = self.gt_pos_all[: self.env._motion_lib._num_unique_motions]
                    pred_rot_all = self.pred_rot_all[: self.env._motion_lib._num_unique_motions]
                    gt_rot_all = self.gt_rot_all[: self.env._motion_lib._num_unique_motions]
                    
                    # np.sum([i.shape[0] for i in self.pred_pos_all[:self.env._motion_lib._num_unique_motions]])
                    # self.env._motion_lib.get_motion_num_steps().sum()

                    # 记录成功与失败的动捕片段 key
                    failed_keys = self.env._motion_lib._motion_data_keys[terminate_hist[: self.env._motion_lib._num_unique_motions]]
                    success_keys = self.env._motion_lib._motion_data_keys[~terminate_hist[: self.env._motion_lib._num_unique_motions]]
                    # print("failed", self.env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:self.env._motion_lib._num_unique_motions]])
                    
                    # metrics = compute_metrics_lite(pred_pos_all, gt_pos_all, pred_rot_all, gt_rot_all)
                    # metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ, pred_rot_all_succ, gt_rot_all_succ)
                    
                    # 调用第三方库 (smpl_sim) 计算综合评估指标（论文表格里的数据来源）
                    metrics = compute_metrics_lite(pred_pos_all, gt_pos_all)
                    metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                    # 计算指标的平均值准备打印
                    metrics_all_print = {m: np.mean(v) for m, v in metrics.items()}
                    metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                    # 打印最终评测报告到终端
                    print("------------------------------------------")
                    print("------------------------------------------")
                    print(f"Success Rate: {self.success_rate:.10f}")
                    print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                    print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))
                        
                    import ipdb; ipdb.set_trace() # 评测完成，进入断点供开发者分析
                    
 
                # 推进到下一批运动样本
                self.env.forward_motion_samples()
                # 重置跌倒状态为 0 (存活)
                self.terminate_state = torch.zeros(
                    self.lab_env.num_envs, device=self.device
                )

                # 更新进度条
                self.pbar.update(1)
                self.pbar.refresh()
                # 清空当前阶段容器
                self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []
                self.curr_stpes = 0
                    
            # 终端进度条的文本信息更新
            update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {self.env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
            self.pbar.set_description(update_str)
                    
        
        # 按照 config 配置的频率，触发网页端图表的渲染更新
        if (step + 1) % self.config.plot_update_interval == 0:
            self.logger.plot_states()
        return actor_state

class WebLogger:
    # 【类作用说明】
    # 这是一个数据搜集器兼微型 Web 服务器。
    # 它的作用是将回调函数中截获的仿真数据（如动作、速度、扭矩）储存下来，
    # 并使用 Flask 和 SocketIO 在本地启动一个网页，以 Plotly 图表的形式进行实时的、互动式的可视化展示。
    
    def __init__(self, dt):
        # 【函数说明】 初始化数据字典和 Flask 服务器
        # 调用者：AnalysisPlotMotionTrackingOpenloop 插件初始化时。
        self.state_log = defaultdict(list) # 储存机器人物理状态的字典
        self.rew_log = defaultdict(list) # 储存强化学习奖励值的字典
        self.dt = dt # 物理仿真的步长时间
        self.num_episodes = 0
        self.app = Flask(__name__) # 实例化 Flask web 服务器
        self.socketio = SocketIO(self.app) # 实例化 WebSocket 以支持前端实时更新
        self.thread = None # 服务器运行的后台线程

    def set_robot_limits(self, dof_pos_limits, dof_vel_limits, dof_torque_limits):
        # 【函数说明】 设置机器人的物理限制值，供后续画黄色的安全界限区使用
        self.log_dof_pos_limits = dof_pos_limits
        self.log_dof_vel_limits = dof_vel_limits
        self.log_dof_torque_limits = dof_torque_limits

    def set_robot_num_dofs(self, num_dofs):
        # 【函数说明】 记录自由度数，决定要画多少个子图
        self.robot_num_dofs = num_dofs

    def log_state(self, key, value):
        # 【函数说明】 记录单个键值对数据
        self.state_log[key].append(value)

    def log_states(self, dict):
        # 【函数说明】 批量记录字典数据
        # 调用者：AnalysisPlotMotionTrackingOpenloop.on_pre_eval_env_step 中被调用
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        # 【函数说明】 记录每次获得的 reward 值用于绘制奖励曲线
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        # 【函数说明】 清空日志字典
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        # 【函数说明】 绘图的外部入口
        # 如果 Flask 服务器还没开，就开一个子线程启动它，然后调用内部更新图表逻辑
        # 调用者：AnalysisPlotMotionTrackingOpenloop.on_post_eval_env_step 达到一定步数频率时触发
        if self.thread is None:
            self.thread = threading.Thread(target=self._run_server)
            self.thread.start()
        self._update_plot()
    
    def _run_server(self):
        # 【函数说明】 路由并运行 Flask 服务器的进程
        @self.app.route('/')
        def index():
            # 提供包含图表占位符的 HTML 页面
            return send_file('analysis_plot_template.html')

        # 启动 SocketIO 服务
        self.socketio.run(self.app, debug=False, use_reloader=False)
    
    def _update_plot(self):
        # 【函数说明】 核心制图逻辑
        # 作用：把收集到的大量时间序列数据封装成 Plotly 可以识别的 JSON 格式，通过 WebSockets 发给前端网页
        # 调用者：被类内部的 plot_states() 函数调用
        
        log = self.state_log.copy()
        
        # 计算 X 轴的时间序列 (点数 * 步长)
        total_time = len(next(iter(log.values()))) * self.dt
        time = np.linspace(0, total_time, len(next(iter(log.values()))))

        for key in log:
            if isinstance(log[key], list):
                log[key] = np.array(log[key])

        # 定义画图用的标准颜色规范
        BLUE = '#005A9D'
        RED = '#DA2513'
        YELLOW = '#EEDE70'
        PURPLE = '#6B2B73'
        GREEN = '#00A170'

        num_dofs = self.robot_num_dofs

        def get_subplot_titles():
            # 生成 Plotly 子图矩阵的各个小标题
            titles = [
                'Base velocity x', 'Base velocity y', 'Base velocity yaw', 'Base velocity z',
                'Vertical Contact forces', 'Vertical Contact forces', 'Vertical Contact forces', 'Vertical Contact forces'
            ]
            for i in range(num_dofs): # 给每个自由度安排四列：位置、速度、扭矩、扭矩-速度散点图
                titles.extend([f'DOF {i} Position', f'DOF {i} Velocity', f'DOF {i} Torque', f'DOF {i} Torque/Velocity'])
            return titles

        # Calculate number of rows needed (基座速度一行，接触力一行，剩下每个自由度占据一行)
        num_rows = 2 + num_dofs

        # 构建拥有成百上千个小图的 Plotly 网格底板
        fig = make_subplots(rows=num_rows, cols=4, subplot_titles=get_subplot_titles())

        def add_trace(x, y, color, row, col, name=None, show_legend=False):
            # 快捷画线函数
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color), name=name, showlegend=show_legend), row=row, col=col)

        # Base velocities and commands (第 1 行：四列画基座的速度信息)
        add_trace(time, log["base_vel_x"], BLUE, 1, 1, "Base vel x")
        add_trace(time, log["base_vel_y"], BLUE, 1, 2, "Base vel y")
        add_trace(time, log["base_vel_yaw"], BLUE, 1, 3, "Base vel yaw")
        add_trace(time, log["base_vel_z"], BLUE, 1, 4, "Base vel z")

        # Vertical Contact forces (第 2 行：画脚掌接触力)
        forces = log["contact_forces_z"]
        add_trace(time, forces[:,0], BLUE, 2, 1, "Force Left")

        # Command velocities

        def add_limit_lines(row, col, lower, upper, color=YELLOW):
            # 快捷画色块边界函数：超出此黄色框代表打破了物理极限，这是防止 Sim-to-real 失真的关键监视器
            fig.add_shape(type="rect", x0=time[0], x1=time[-1], y0=lower, y1=upper,
                        fillcolor=color, line=dict(width=0), layer='below', row=row, col=col)

        # DOF Positions, Velocities, and Torques (第 3 行及以后：逐个关节画图)
        for i in range(num_dofs):
            row = i + 3  # Start from the third row

            # Position (位置曲线对比：包含预测值、目标值、开环输入、完美残差输入)
            add_trace(time, [pos[i] for pos in log["dof_pos"]], BLUE, row, 1, f"DOF {i} pos")
            add_trace(time, [pos[i] for pos in log["dof_pos_target"]], RED, row, 1, f"DOF {i} pos target")
            add_trace(time, [pos[i] for pos in log["dof_open_loop_actions"]], PURPLE, row, 1, f"DOF {i} open loop actions")
            add_trace(time, [pos[i] for pos in log["dof_perfect_delta_a"]], GREEN, row, 1, f"DOF {i} perfect delta a")
            add_limit_lines(row, 1, self.log_dof_pos_limits[i, 0], self.log_dof_pos_limits[i, 1]) # 画极限边界
            
            # Velocity (速度曲线与边界)
            add_trace(time, [vel[i] for vel in log["dof_vel"]], BLUE, row, 2, f"DOF {i} vel")
            add_limit_lines(row, 2, -self.log_dof_vel_limits[i], self.log_dof_vel_limits[i])
            
            # Torque (扭矩曲线与边界)
            add_trace(time, [torque[i] for torque in log["dof_torque"]], BLUE, row, 3, f"DOF {i} torque")
            add_limit_lines(row, 3, -self.log_dof_torque_limits[i], self.log_dof_torque_limits[i])
            
            # Torque/Velocity curve (扭矩/速度 联合相图散点：反映电机工作区间是否安全)
            fig.add_trace(go.Scatter(
                x=[vel[i] for vel in log["dof_vel"]], 
                y=[torque[i] for torque in log["dof_torque"]], 
                mode='markers', 
                marker=dict(color=BLUE, size=2), 
                showlegend=False,
                name=f"DOF {i} Torque/Velocity"
            ), row=row, col=4)

            # Add velocity limits (为联合相图画出黄色的矩形安全限制框线，上下左右四边)
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=-self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            fig.add_shape(type="line", x0=self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

            # Add torque limits
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=-self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

        # 统一设置图表大布局的尺寸和标题
        fig.update_layout(height=300*num_rows, width=1500, title_text="Robot State Plots", showlegend=True)
        
        # Update x and y axis labels (更新所有 X/Y 坐标轴的单位和文字标签)
        for i in range(num_rows):
            for j in range(3):
                fig.update_xaxes(title_text="time [s]", row=i+1, col=j+1)
            fig.update_xaxes(title_text="", row=i+1, col=4)
        
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=1)
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=2)
        fig.update_yaxes(title_text="base ang vel [rad/s]", row=1, col=3)
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=4)
        fig.update_yaxes(title_text="Forces z [N]", row=2, col=1)


        for i in range(3, num_rows + 1):
            fig.update_yaxes(title_text="Position [rad]", row=i, col=1)
            fig.update_yaxes(title_text="Velocity [rad/s]", row=i, col=2)
            fig.update_yaxes(title_text="Torque [Nm]", row=i, col=3)
            fig.update_yaxes(title_text="Torque/Velocity", row=i, col=4)

        # 将庞大的 Plotly 图表对象编码为 JSON 字符串
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        # 通过 SocketIO 发送事件名为 'update_plots' 的消息推送给网页客户端渲染
        self.socketio.emit('update_plots', plot_json)

    def print_rewards(self):
        # 【函数说明】 计算并向终端打印平均奖励值的效用函数
        # 调用者：非自动触发，可由开发者在需要看控制台日志时调用
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        # 【函数说明】 析构函数，当对象被销毁时，安全地关闭 WebSocket 和后台线程
        if self.thread:
            self.socketio.stop()
            self.thread.join()