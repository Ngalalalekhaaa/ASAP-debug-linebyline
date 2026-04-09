import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from hydra.utils import instantiate
from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
console = Console()

class PPO(BaseAlgo):
    """
    【类说明】PPO (Proximal Policy Optimization) 近端策略优化算法类。
    在机器人控制中，这就是机器人的“学习引擎”和“大脑核心”。
    它负责指导机器人在仿真环境中不断试错，并根据反馈（Reward）安全、稳定地更新神经网络。
    """
    
    # =========================================================================
    # 初始化与配置阶段
    # =========================================================================
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        """
        【函数作用】初始化 PPO 算法实例，绑定环境，设置日志和设备。
        【机器人意义】给机器人分配好训练场地（env），设定好训练大纲（config），并指定是用哪块 GPU 运算（device）。
        【被谁调用】通常被训练的主脚本（如 train.py）在程序启动时调用。
        """
        self.device= device # 运算设备（通常是你的高性能 GPU，如 'cuda:0'）
        self.env = env # 物理仿真环境（包含了几千个并行的机器人，如 IsaacLab 环境）
        self.config = config # PPO 的超参数配置（学习率、步长等）
        self.log_dir = log_dir # 训练日志和模型权重的保存路径
        # Tensorboard 记录器，用于画出 Reward 曲线等指标，每 10 秒刷新一次写入磁盘
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # 耗时统计变量，用于计算机器人在物理环境中的行动时间（collection）和大脑思考学习的时间（learn）
        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self._init_config() # 调用内部函数解析 config 字典

        self.tot_timesteps = 0 # 记录机器人总共与环境交互了多少步（相当于机器人的“年龄”）
        self.tot_time = 0 # 总训练物理时间
        self.current_learning_iteration = 0 # 当前是第几次迭代（第几节训练课）

        # Book keeping (数据记录簿)
        self.ep_infos = [] # 存放每个回合（Episode）的额外信息
        self.rewbuffer = deque(maxlen=100) # 记录最近 100 次回合的总奖励，用于计算平均得分，平滑输出
        self.lenbuffer = deque(maxlen=100) # 记录最近 100 次回合的存活步数（走多远才摔倒）
        
        # 当前所有并行环境（如 4096 个 G1 机器人）的累计奖励，如果在设备上就在 GPU 上
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # 当前所有并行环境的已存活步数
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.eval_callbacks: list[RL_EvalCallback] = [] # 评估时的回调插件列表（比如录像插件）
        self.episode_env_tensors = TensorAverageMeterDict() # 环境反馈的 Tensor 指标平均计算器
        # while True:
        #     _ = self.env.reset_all() # 第一次初始化时，把所有机器人重置到初始站立状态，获取第一帧传感器数据
        #_ = self.env.reset_all()

    def _init_config(self):
        """
        【函数作用】从配置对象中提取 PPO 的各项超参数。
        【机器人意义】解析训练大纲，明确机器人每次要收集多少数据再学习、学习率多大、动作幅度允许改变多少等。
        【被谁调用】仅被当前的 __init__ 函数调用。
        """
        # Env related Config (环境相关配置)
        self.num_envs: int = self.env.config.num_envs # 并行机器人的数量（例如 4096）
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict # 传感器数据的维度字典（包含关节角度、速度、IMU等几项数据及其长度）
        self.num_act = self.env.config.robot.actions_dim # 机器人能执行的动作维度（对于 G1 可能是几十个关节的扭矩或位置目标）

        # Logging related Config (日志配置)
        self.save_interval = self.config.save_interval # 每隔几次迭代保存一次大脑的权重（模型文件 .pt）
        
        # Training related Config (训练节奏配置)
        self.num_steps_per_env = self.config.num_steps_per_env # 每次学习前，机器人在环境里连续走多少步（如 24 步）。总数据量 = 并行数 * 步数。
        self.load_optimizer = self.config.load_optimizer # 恢复训练时，是否连带优化器的状态一起加载（保持动量等）
        self.num_learning_iterations = self.config.num_learning_iterations # 本次程序运行计划训练多少次迭代
        self.init_at_random_ep_len = self.config.init_at_random_ep_len # 初始化时是否让各个环境处于不同的“年龄”进度，防止所有机器人同时摔倒导致数据波动过大

        # Algorithm related Config (PPO 核心算法参数)
        self.desired_kl = self.config.desired_kl # 目标 KL 散度。用于控制新老策略的差异，如果机器人步子迈得太大（学得太激进），用来踩刹车
        self.schedule = self.config.schedule # 学习率调度策略，如 'adaptive' 表示根据 KL 散度自适应调整
        self.actor_learning_rate = self.config.actor_learning_rate # Actor（控制动作的网络）的初始学习率
        self.critic_learning_rate = self.config.critic_learning_rate # Critic（评估状态好坏的网络）的初始学习率
        self.clip_param = self.config.clip_param # PPO 的截断参数（通常是 0.2），防止单次更新动作改变过大，导致机器人突然“发疯”
        self.num_learning_epochs = self.config.num_learning_epochs # 拿收集到的一批数据，反复在神经网络中训练几次（复习几次）
        self.num_mini_batches = self.config.num_mini_batches # 把一批数据分成几个小块来更新网络，防显存炸且更稳定
        self.gamma = self.config.gamma # 折扣因子（如 0.99），决定机器人多看重“未来的奖励”。越接近1越有远见。
        self.lam = self.config.lam # GAE（广义优势估计）的平滑参数（通常 0.95），用于权衡优势估计的方差和偏差。
        self.value_loss_coef = self.config.value_loss_coef # 价值网络 Loss 的权重
        self.entropy_coef = self.config.entropy_coef # 熵正则化权重。越大越鼓励机器人“随机瞎动”探索新动作，防止陷入局部最优（比如只会原地踏步）
        self.max_grad_norm = self.config.max_grad_norm # 梯度裁剪阈值，防止梯度爆炸造成网络崩溃
        self.use_clipped_value_loss = self.config.use_clipped_value_loss # 是否对 Critic 的价值更新也使用截断（类似 PPO 的 clip）

    def setup(self):
        """
        【函数作用】统筹搭建神经网络和经验回放区。
        【机器人意义】给机器人安装“大脑”（Actor和Critic网络），并配发一个“短期记忆本”（Storage）。
        【被谁调用】主脚本在初始化 PPO 对象之后调用。
        """
        logger.info("Setting up PPO")
        self._setup_models_and_optimizer() # 建立大脑
        logger.info(f"Setting up Storage")
        self._setup_storage() # 建立记忆本

    def _setup_models_and_optimizer(self):
        """
        【函数作用】初始化 Actor、Critic 神经网络及对应的 Adam 优化器。
        【机器人意义】
        - Actor：执行者。看着传感器数据（Obs），决定怎么动关节（Action）。
        - Critic：评论家。看着传感器数据（甚至开天眼看真实地形），给当前的局势打分（Value），告诉 Actor 刚才那步走得好不好。
        【被谁调用】被 setup() 调用。
        """
        # 实例化 Actor 网络并放到 GPU 上
        self.actor = PPOActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor, # MLP 隐藏层配置等
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std # 初始探索噪声的标准差
        ).to(self.device)

        # 实例化 Critic 网络并放到 GPU 上
        self.critic = PPOCritic(self.algo_obs_dim_dict,
                                self.config.module_dict.critic).to(self.device)

        # 设置优化器，负责在反向传播后更新网络权重，类似于“肌肉记忆的固化过程”
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    def _setup_storage(self):
        """
        【函数作用】初始化 RolloutStorage，用于存储 PPO 需要的轨迹数据。
        【机器人意义】机器人在仿真中走一步，就把当时看到的（Obs）、做的（Action）、得到的好处（Reward）、是否摔倒（Dones）都记在这个本子上。攒够一页（一步）后供大脑复盘学习。
        【被谁调用】被 setup() 调用。
        """
        # 容量：环境数 x 每环境走多少步（比如 4096 个机器人 x 走 24 步）
        self.storage = RolloutStorage(self.env.num_envs, self.num_steps_per_env)
        
        ## Register obs keys (在记忆本中画好表格，注册需要记录的观测数据字段)
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)
        
        ## Register others (注册强化学习核心要素)
        self.storage.register_key('actions', shape=(self.num_act,), dtype=torch.float) # 实际执行的动作
        self.storage.register_key('rewards', shape=(1,), dtype=torch.float) # 环境给的奖励分数
        self.storage.register_key('dones', shape=(1,), dtype=torch.bool) # 是否摔倒/结束回合
        self.storage.register_key('values', shape=(1,), dtype=torch.float) # Critic 预测的状态价值
        self.storage.register_key('returns', shape=(1,), dtype=torch.float) # 实际算出来的总回报（未来的累计奖励）
        self.storage.register_key('advantages', shape=(1,), dtype=torch.float) # 优势函数（当前动作比平均水平好多少）
        self.storage.register_key('actions_log_prob', shape=(1,), dtype=torch.float) # 执行该动作的概率对数（用于计算PPO的比例率 ratio）
        self.storage.register_key('action_mean', shape=(self.num_act,), dtype=torch.float) # Actor 预测的动作均值
        self.storage.register_key('action_sigma', shape=(self.num_act,), dtype=torch.float) # Actor 的动作探索方差

    def _eval_mode(self):
        """
        【函数作用】将网络切换到评估模式。
        【机器人意义】关掉学习机制和探索，进入考试状态，机器人的动作变得完全确定。
        【被谁调用】evaluate_policy 等评估相关函数。
        """
        self.actor.eval()
        self.critic.eval()

    def _train_mode(self):
        """
        【函数作用】将网络切换到训练模式。
        【机器人意义】开启学习状态（启用 BatchNorm / Dropout 等，启用探索），允许试错。
        【被谁调用】learn() 函数在开始每一轮迭代前调用。
        """
        self.actor.train()
        self.critic.train()

    # =========================================================================
    # 存取款阶段 (Save/Load)
    # =========================================================================
    def load(self, ckpt_path):
        """
        【函数作用】从指定路径加载保存的模型权重。
        【机器人意义】给机器人移植之前的记忆和经验。比如昨天训练到一半停电了，今天把昨天存的 .pt 文件加载进来接着练。
        【被谁调用】主脚本在检测到有指定的 resume_path 时调用。
        """
        # import ipdb; ipdb.set_trace() # 断点调试，被注释了
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device) # 读取权重字典
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"]) # 恢复 Actor 大脑
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"]) # 恢复 Critic 大脑
            if self.load_optimizer: # 如果配置了连带优化器一起恢复
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                # 提取存档中的学习率
                self.actor_learning_rate = loaded_dict['actor_optimizer_state_dict']['param_groups'][0]['lr']
                self.critic_learning_rate = loaded_dict['critic_optimizer_state_dict']['param_groups'][0]['lr']
                self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate) # 生效学习率
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
                logger.info(f"Critic Learning rate: {self.critic_learning_rate}")
            self.current_learning_iteration = loaded_dict["iter"] # 恢复上一次记录的训练轮数，接着往下计数
            return loaded_dict["infos"]

    def save(self, path, infos=None):
        """
        【函数作用】把当前网络参数和优化器状态打包成字典，保存到硬盘。
        【机器人意义】保存脑部切片备份。防止代码崩溃或者为了以后在其他地方直接部署这个机器人。
        【被谁调用】learn() 的主循环中，每隔 save_interval 次迭代调用一次。
        """
        logger.info(f"Saving checkpoint to {path}")
        torch.save({
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)
        
    # =========================================================================
    # 核心训练大循环 (The RL Loop)
    # =========================================================================
    def learn(self):
        """
        【函数作用】PPO 算法的核心流程控制函数。
        【机器人意义】机器人的“完整集训营”过程：去场地跑（Rollout） -> 思考总结（Compute Returns） -> 更新大脑（Training Step） -> 循环往复。
        【被谁调用】主脚本发起训练时调用。
        """
        if self.init_at_random_ep_len:
            # 打乱所有并行机器人的生存步数记录。
            # 防止 4096 个机器人在同一步骤集体摔倒，导致某一时刻的 dones 信号过于集中，影响网络学习的稳定性。
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        # 集训开始前，先拿取所有机器人的第一眼传感器画面
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device) # 数据推送到 GPU
            
        self._train_mode() # 设定为允许学习模式

        num_learning_iterations = self.num_learning_iterations

        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        # 开启循环迭代（训练营一共分为 tot_iter 节课）
        # do not use track, because it will confict with motion loading bar
        # for it in track(range(self.current_learning_iteration, tot_iter), description="Learning Iterations"):
        for it in range(self.current_learning_iteration, tot_iter):
            self.start_time = time.time() # 记录一节课开始的时间

            # Jiawei: Need to return obs_dict to update the obs_dict for the next iteration
            # Otherwise, we will keep using the initial obs_dict for the whole training process
            # 第一阶段：收集数据 (Rollout)。让机器人带着当前大脑在环境里跑几十步，把经历记下来。
            obs_dict =self._rollout_step(obs_dict)

            # 第二阶段：更新大脑 (Training Step)。拿着刚刚记下的记忆本（Storage），进行反向传播更新神经网络参数。
            loss_dict = self._training_step()

            self.stop_time = time.time()
            self.learn_time = self.stop_time - self.start_time # 计算大脑学习花了多少秒

            # Logging (把这节课的成绩单整理好)
            log_dict = {
                'it': it, # 当前第几节课
                'loss_dict': loss_dict, # 学习误差损失
                'collection_time': self.collection_time, # 跑步收集数据耗时
                'learn_time': self.learn_time, # 大脑思考耗时
                'ep_infos': self.ep_infos,
                'rewbuffer': self.rewbuffer, # 奖励缓存池
                'lenbuffer': self.lenbuffer, # 存活长度缓存池
                'num_learning_iterations': num_learning_iterations
            }
            # 第三阶段：打印日志到终端和 Tensorboard
            self._post_epoch_logging(log_dict)
            
            # 第四阶段：按设定的间隔备份保存模型
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            self.ep_infos.clear() # 清空这一轮的特殊信息，准备下一轮
        
        # 训练全部结束，更新总轮数记录，并做最后一次强制保存
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def _actor_rollout_step(self, obs_dict, policy_state_dict):
        """
        【函数作用】Actor 网络在收集数据阶段进行一次前向推理，并记录动作概率等。
        【机器人意义】机器人看着传感器（obs），大脑思考应该怎么动（actions），并且把“做这个动作的自信程度/概率”（log_prob）记下来，以备后面 PPO 计算差异。
        【被谁调用】_rollout_step 内部循环中。
        """
        actions = self._actor_act_step(obs_dict) # Actor 输出具体动作（包含探索噪声）
        policy_state_dict["actions"] = actions
        
        action_mean = self.actor.action_mean.detach() # Actor 认为最完美的动作均值
        action_sigma = self.actor.action_std.detach() # Actor 的探索方差（标准差）
        actions_log_prob = self.actor.get_actions_log_prob(actions).detach().unsqueeze(1) # 这个动作在这个分布下产生的概率对数值
        
        # 把思考的中间结果塞进字典
        policy_state_dict["action_mean"] = action_mean
        policy_state_dict["action_sigma"] = action_sigma
        policy_state_dict["actions_log_prob"] = actions_log_prob

        # 防御性断言，确保张量维度都是 2D： [环境数量, 特征维度]
        assert len(actions.shape) == 2
        assert len(actions_log_prob.shape) == 2
        assert len(action_mean.shape) == 2
        assert len(action_sigma.shape) == 2

        return policy_state_dict

    def _rollout_step(self, obs_dict):
        """
        【函数作用】数据收集循环。在无梯度环境下与模拟器交互指定步数，把所有数据存入 Storage。
        【机器人意义】这就像让机器人在训练场地上连续跑 N 步。在这个过程中，只做动作、记笔记，不进行反思和纠正（因为没有梯度）。
        【被谁调用】learn() 函数调用。
        """
        with torch.inference_mode(): # 关闭 PyTorch 梯度计算，极大节省显存，加快前向推理速度
            # 循环连续执行 N 步（比如 num_steps_per_env = 24 步）
            for i in range(self.num_steps_per_env):
                policy_state_dict = {} # 临时装填这一步策略产生的数据
                
                # 1. Actor 看环境，出动作
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                # 2. Critic 看环境，给出目前的评分（当前局势值多少分）
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                ## Append states to storage (3. 把本回合所有看到、想到的东西都写进“短期记忆本” Storage)
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])
                
                # 4. 把刚刚决定的动作真正下发给物理环境引擎 (Isaac Gym) 执行
                actions = policy_state_dict["actions"]
                actor_state = {}
                actor_state["actions"] = actions
                # 环境时间往前推进一帧，返回新的传感器数据、奖励分、是否摔倒、额外信息
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                
                # 确保新收到的传感器数据转移到正确的设备（GPU）
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # 记录环境中的各种奖励细项（方便排查到底哪项得分高，哪项扣分了）
                self.episode_env_tensors.add(infos["to_log"])
                
                rewards_stored = rewards.clone().unsqueeze(1)
                # 处理超时(time_outs)逻辑：如果回合结束是因为时间到了而不是摔倒了，那我们要补给他一个未来的期望得分（Bootstrapping），否则机器人在倒计时最后几秒就不想走了
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                assert len(rewards_stored.shape) == 2
                
                # 5. 把奖励和结束信号也记在记忆本里
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step() # 记忆本行数指针往下移一行

                self._process_env_step(rewards, dones, infos) # 重置因为摔倒而死掉的机器人的网络隐藏状态（如果有 RNN 的话）

                # 6. Book keeping 记录累加这回合走了多远、拿了多少总分，用于后续在终端画表展示
                if self.log_dir is not None:
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    # 找出哪些机器人这一步摔倒（done）了
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    # 把摔倒机器人的最终得分和存活步数存到统计池里
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    # 清零摔倒机器人的累计分和长度，让它重新开始算
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            # 这个 for 循环结束，代表 N 步跑完了。记录时间。
            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            # 7. 准备用于训练的核心数据：计算 Returns (回报) 和 Advantages (优势)
            # 这里的 last_obs_dict 就是 N 步走完之后的最终画面，用来估算未来还能得多少分
            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'), 
                dones=self.storage.query_key('dones'), 
                rewards=self.storage.query_key('rewards'))
            )
            # 把算好的优势和回报批量写回 Storage
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict # 把当前最新画面传出去，下一次集训接着这张画面练

    def _process_env_step(self, rewards, dones, infos):
        """
        【函数作用】处理环境单步的后续状态更新（处理 Dones）。
        【机器人意义】告诉大脑：“这几个机器人已经摔倒重生了，把它们之前的肌肉记忆或者 RNN 短期记忆全部清零，不要干扰新的回合”。
        【被谁调用】_rollout_step 内部。
        """
        self.actor.reset(dones)
        self.critic.reset(dones)

    def _compute_returns(self, last_obs_dict, policy_state_dict):
        """
        【函数作用】使用广义优势估计 (GAE) 计算每个时刻的回报 (Returns) 和优势 (Advantages)。
        【机器人意义】课后复盘。算一算在刚才跑的 24 步里，哪一步做得特别好（优势为正），哪一步是臭棋（优势为负）。
        - Return：这一步加上未来所有能拿到的分数总和。
        - Advantage = Return - Critic预测的分数。意思是：实际表现比预期表现好多少。好就加大力度做，差就避免。
        【被谁调用】_rollout_step 在收集完所有步数后调用。
        """
        # 获取第 N 步之后的最终价值预估，用于倒推
        last_values= self.critic.evaluate(last_obs_dict["critic_obs"]).detach()
        advantage = 0
        
        # 把这段时间存下的数据拿出来
        values = policy_state_dict['values']
        dones = policy_state_dict['dones']
        rewards = policy_state_dict['rewards']
        
        last_values = last_values.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)
        
        returns = torch.zeros_like(values) # 准备一个空张量存结果
        
        num_steps = returns.shape[0] # 一共跑了多少步
        
        # 时光倒流：从最后一步往前反向推导计算 (GAE 的核心算法)
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values # 如果是最后一步，它的下一步是刚算出来的 last_values
            else:
                next_values = values[step + 1] # 否则它的下一步是之前记下的 value
            
            next_is_not_terminal = 1.0 - dones[step].float() # 如果这步摔倒了（done=1），未来的连锁反应分数就截断为 0
            
            # TD 误差 (TD Error)：当前步真实拿到奖赏 + 未来的预期 - 当前预期的表现
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            # 优势函数累计衰减计算
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step] # 总回报 = 优势 + 预期

        # Compute and normalize the advantages (计算并归一化优势，提高训练稳定性)
        advantages = returns - values
        # 归一化：减去均值，除以标准差。这能确保更新时梯度的分布比较稳定，不会剧烈震荡。
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    # =========================================================================
    # 大脑神经网络更新 (Learning Phase)
    # =========================================================================
    def _training_step(self):
        """
        【函数作用】从 Storage 中提取数据，进行小批量 (Mini-batch) 随机梯度下降，更新 Actor 和 Critic 权重。
        【机器人意义】晚自习时间！机器人把记满的“经验本”翻开，打乱顺序，反复看几遍（Epochs），以此改变大脑中的神经网络连线（权重）。
        【被谁调用】learn() 函数在数据收集完毕后调用。
        """
        loss_dict = self._init_loss_dict_at_training_step() # 初始化三个误差指标(Value, Surrogate, Entropy)为0

        # 从 Storage 中创建一个打乱顺序的批次生成器。
        # 比如一共收集了 10万 条数据，分 4 个 mini_batch，复习 5 个 epoch。
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for policy_state_dict in generator: # 每次抽出一小把记忆
            # Move everything to the device
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
            # 用这一小把记忆去更新一次大脑的权重
            loss_dict = self._update_algo_step(policy_state_dict, loss_dict)

        # 算出本轮更新的平均损失，用于返回在终端打印
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            loss_dict[key] /= num_updates
            
        self.storage.clear() # 这个循环结束后，老记忆就没用了（因为策略已经变了，这叫 On-Policy），清空记忆本准备下一轮
        return loss_dict
    
    def _init_loss_dict_at_training_step(self):
        """
        【函数作用】辅助函数，初始化记录损失的字典。
        【被谁调用】_training_step。
        """
        loss_dict = {}
        loss_dict['Value'] = 0 # 批评家打分估算不准的误差
        loss_dict['Surrogate'] = 0 # Actor动作选择不当的误差（PPO代理损失）
        loss_dict['Entropy'] = 0 # 动作的不确定度（熵）
        return loss_dict
    
    def _update_algo_step(self, policy_state_dict, loss_dict):
        """
        【函数作用】调用具体的 PPO 算法公式更新网络。这层是一个 Wrapper 封装。
        【被谁调用】_training_step。
        """
        loss_dict = self._update_ppo(policy_state_dict, loss_dict)
        return loss_dict

    def _actor_act_step(self, obs_dict):
        """【函数作用】Actor 前向推理。 【被谁调用】_update_ppo, _actor_rollout_step."""
        return self.actor.act(obs_dict["actor_obs"])
    
    def _critic_eval_step(self, obs_dict):
        """【函数作用】Critic 前向推理估值。 【被谁调用】_update_ppo, _rollout_step."""
        return self.critic.evaluate(obs_dict["critic_obs"])
    
    def _update_ppo(self, policy_state_dict, loss_dict):
        """
        【函数作用】***PPO 算法最核心的数学计算逻辑***！计算 Loss 并反向传播 (Backward) 优化器更新步长。
        【机器人意义】这里是真正的“改脑子”环节。通过比较当前的判断和收集数据时的判断，调整神经元连接，使得“做了好动作的概率变大，做坏动作的概率变小”。
        【被谁调用】_update_algo_step。
        """
        # 从批次字典中提取旧的记忆数据
        actions_batch = policy_state_dict['actions'] # 当时做的什么动作
        target_values_batch = policy_state_dict['values'] # 当时评估的状态值
        advantages_batch = policy_state_dict['advantages'] # 算出来的优势（证明这个动作有多好）
        returns_batch = policy_state_dict['returns'] # 算出来的实际总回报
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob'] # 当时做这个动作的旧概率
        old_mu_batch = policy_state_dict['action_mean'] # 旧均值
        old_sigma_batch = policy_state_dict['action_sigma'] # 旧方差

        # 把当时看到的画面塞给 *现在已经更新过一点* 的大脑，看看现在它的反应
        self._actor_act_step(policy_state_dict)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch) # 新大脑对相同动作评估出的新概率
        value_batch = self._critic_eval_step(policy_state_dict) # 新 Critic 给出的新估值
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy # 策略当前的熵（混沌程度）

        # 1. KL 散度与自适应学习率 (Adaptive KL Penalty)
        # 意义：如果发现新大脑的想法和旧大脑差异过大（KL散度超标），说明步子迈大了，强行调小学习率踩刹车，防止学崩。
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                # 高斯分布之间的 KL 散度解析解公式
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    # 差异太大，学习率立刻砍掉 1.5 倍
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    # 差异太小（学得太慢了），学习率放大 1.5 倍
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)

                # 把调整后的学习率强行写入 Adam 优化器
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.actor_learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.critic_learning_rate

        # 2. Surrogate loss (PPO 核心公式：Actor 的裁剪代理损失)
        # Ratio = 新策略选这个动作的概率 / 老策略选这个动作的概率
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        # 没截断的 Loss：优势越大，且新策略比老策略更倾向这个动作，值越小（因为加了负号，优化器是最小化loss）
        surrogate = -torch.squeeze(advantages_batch) * ratio
        # PPO 截断 (Clip)：限制 ratio 只能在 [1 - 0.2, 1 + 0.2] 之间波动。限制单次更新幅度！
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                        1.0 + self.clip_param)
        # 取最大的负号，实际上是取原公式里最小的，这就是 PPO 悲观下界原则
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # 3. Value function loss (Critic 的价值损失，通常是均方误差 MSE)
        # 意义：让 Critic 预测的 value 尽量贴近实际算出来的 returns。
        if self.use_clipped_value_loss:
            # 类似 Actor，对 Critic 也做截断，防止一次更新把评分系统搞崩溃
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        # 4. 合并总 Loss
        entropy_loss = entropy_batch.mean()
        # 总 Actor Loss = 动作代理损失 - 熵乘数 * 熵损失（减号是因为我们要最大化熵，鼓励探索，即最小化负的熵）
        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
        
        critic_loss = self.value_loss_coef * value_loss

        # 5. 反向传播与梯度更新 (PyTorch 标准操作)
        self.actor_optimizer.zero_grad()  # 清空旧梯度
        self.critic_optimizer.zero_grad()
        
        # print("skip backward")
        actor_loss.backward()  # 求导，计算改变哪个权重能降低 actor_loss
        critic_loss.backward() # 同理

        # Gradient step (梯度裁剪：如果求出来的导数太突兀，把它硬切下来，防止网络参数爆炸出现 NaN)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        # 优化器发力：真实地改变神经网络参数！机器人升级了！
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # 累计打印用的数值
        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        return loss_dict

    def set_learning_rate(self, actor_learning_rate, critic_learning_rate):
        """【函数作用】手动设置内部变量学习率。 【被谁调用】恢复存档函数 load()。"""
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate


    @property
    def inference_model(self):
        """【函数作用】对外暴露仅用于推理测试的模型字典。"""
        return {
            "actor": self.actor,
            "critic": self.critic
        }

    # =========================================================================
    # 日志系统 (Logging)
    # =========================================================================
    def _post_epoch_logging(self, log_dict, width=80, pad=35):
        """
        【函数作用】用 Rich 库在终端里漂亮地打印这一轮的成绩单，包括 FPS、均分等。
        【被谁调用】learn() 的主循环里每一轮结束。
        """
        # 统计总步数：每环境步数 * 环境数
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += log_dict['collection_time'] + log_dict['learn_time']
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']

        ep_string = f''
        # 处理从环境 infos 里传上来的其他自定义日志项
        if log_dict['ep_infos']:
            for key in log_dict['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, log_dict['it']) # 写到 tensorboard
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        train_log_dict = {}
        mean_std = self.actor.std.mean() # 当前动作探索随机性还有多大（越到后期越趋近于稳定动作）
        # FPS (Frames Per Second) 算力指标：总步数 / 总耗时
        fps = int(self.num_steps_per_env * self.env.num_envs / (log_dict['collection_time'] + log_dict['learn_time']))
        train_log_dict['fps'] = fps
        train_log_dict['mean_std'] = mean_std.item()

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}

        self._logging_to_writer(log_dict, train_log_dict, env_log_dict)

        # 终端显示的炫酷 UI 字符串拼接
        str = f" \033[1m Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "

        if len(log_dict['rewbuffer']) > 0:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (Collection: {log_dict[
                            'collection_time']:.3f}s, Learning {log_dict['learn_time']:.3f}s)\n"""
                        #   f"""{'Value function loss:':>{pad}} {log_dict['mean_value_loss']:.4f}\n"""
                        #   f"""{'Surrogate loss:':>{pad}} {log_dict['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {train_log_dict['mean_std']:.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(log_dict['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(log_dict['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (collection: {log_dict[
                            'collection_time']:.3f}s, learning {log_dict['learn_time']:.3f}s)\n"""
                        #   f"""{'Value function loss:':>{pad}} {log_dict['mean_value_loss']:.4f}\n"""
                        #   f"""{'Surrogate loss:':>{pad}} {log_dict['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {train_log_dict['mean_std']:.2f}\n""")

        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (log_dict['it'] + 1) * (
                               log_dict['num_learning_iterations'] - log_dict['it']):.1f}s\n""")
        log_string += f"Logging Directory: {self.log_dir}"

        # Use rich Live to update a specific section of the console (终端实时刷新面板)
        with Live(Panel(log_string, title="Training Log"), refresh_per_second=4, console=console):
            # Your training loop or other operations
            pass

    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        """
        【函数作用】把各种指标写入 Tensorboard 二进制文件，方便浏览器端查看训练折线图。
        【被谁调用】_post_epoch_logging 内部。
        """
        # Logging Loss Dict
        for loss_key, loss_value in log_dict['loss_dict'].items():
            self.writer.add_scalar(f'Loss/{loss_key}', loss_value, log_dict['it'])
        self.writer.add_scalar('Loss/actor_learning_rate', self.actor_learning_rate, log_dict['it'])
        self.writer.add_scalar('Loss/critic_learning_rate', self.critic_learning_rate, log_dict['it'])
        self.writer.add_scalar('Policy/mean_noise_std', train_log_dict['mean_std'], log_dict['it'])
        self.writer.add_scalar('Perf/total_fps', train_log_dict['fps'], log_dict['it'])
        self.writer.add_scalar('Perf/collection time', log_dict['collection_time'], log_dict['it'])
        self.writer.add_scalar('Perf/learning_time', log_dict['learn_time'], log_dict['it'])
        if len(log_dict['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(log_dict['rewbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(log_dict['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(log_dict['lenbuffer']), self.tot_time)
        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.writer.add_scalar(k, v, log_dict['it'])

    ##########################################################################################
    # Code for Evaluation (推理测试、考试验收专用阶段)
    # 这部分的代码专门用于训练完毕后，观察机器人走得好不好。在之前交流过，这里再次标注。
    ##########################################################################################

    def env_step(self, actor_state):
        """
        【函数作用】与物理环境单步交互包装器。将下发动作并获取新的观察值。
        【机器人意义】告诉环境：让 G1 按当前的想法动 0.02 秒。返回最新传感器状态。
        【被谁调用】evaluate_policy 纯净循环。
        """
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        """
        【函数作用】获取一个环境画面的示例并转移到 CPU 上打印查看。
        【被谁调用】通常被调试脚本或查看输入维度的测试代码调用。
        """
        # while True:
        #     obs_dict = self.env.reset_all()
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    @torch.no_grad() # 保证整个评估阶段不会算梯度，节省显存
    def evaluate_policy(self):
        """
        【函数作用】独立的无限循环，用于测试或可视化验证已经训好的模型。
        【机器人意义】完全脱掉训练辅助轮。以绝对确定的动作进行仿真渲染或真实机器人的控制指令分发。
        【被谁调用】专门的 evaluate.py 或 play.py 脚本。
        """
        self._create_eval_callbacks() # 挂载插件（如摄像头录屏）
        self._pre_evaluate_policy() # 告诉环境进入考试模式，关掉随机化干扰。reset_all
        actor_state = self._create_actor_state() # 拿个小本本记录测试时的短期状态
        step = 0
        self.eval_policy = self._get_inference_policy() # 提取纯净大脑模型（这个函数输入action_obs,输出动作均值）
        obs_dict = self.env.reset_all() # 机器人归零发车
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device) # 第一步不发力
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state) # 预热算第一步动作
        
        import pprint # 导入漂亮的打印库
        
        while True: # 开启无限演练
            
            # if step==1:
            #     #actor_state = self._create_actor_state() # 拿个小本本记录测试时的短期状态
            #     step = 0
            #     #self.eval_policy = self._get_inference_policy() # 提取纯净大脑模型（这个函数输入action_obs,输出动作均值）
            #     #obs_dict = self.env.reset_all() # 机器人归零发车
            #     init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device) # 第一步不发力
            #     actor_state.update({"obs": obs_dict})#这里的obs_dict是循环外置为0的obs_dict,下面循环中的self.env_step()中的obs_dict是局部变量，在这里访问不到
            #     actor_state.update({"actions": init_actions})
            #     #actor_state = self._pre_eval_env_step(actor_state) # 预热算第一步动作

            
            
            actor_state["step"] = step
            
            actor_state = self._pre_eval_env_step(actor_state) # 脑力运算：看传感器出关节扭矩
            actor_state = self.env_step(actor_state) # 物理运算：物理引擎演进
            actor_state = self._post_eval_env_step(actor_state) # 善后处理：如果插件要截图，就在这截图
            
            # # ================= 全量打印模块 开始 =================
            # print(f"\n========== Step {step} ==========")
            # pprint.pprint(actor_state["actions"], indent=2, width=120)
            # # ================= 全量打印模块 结束 =================
            
            step += 1
        self._post_evaluate_policy() # (实际上这里 while True 执行不到)

    # @torch.no_grad()
    # def collect_delta_a_sim_data(self, num_collect_steps=1000, save_dir="logs_collect_data/delta_a_sim2sim_data/gym2gen"):
    #         """
    #         【函数作用】批量并行数据收集器（变长截断版）。
    #         【防突变机制】自动监听环境的 reset 信号，只收集每个环境从出生到第一次重置之间的纯净连续轨迹。
    #         """
    #         import os
    #         import numpy as np
    #         import torch
    #         from loguru import logger
            
    #         os.makedirs(save_dir, exist_ok=True)
    #         logger.info(f"========== 开始批量收集 Delta-A 仿真连续数据 ==========")
            
    #         self._create_eval_callbacks() 
    #         self._pre_evaluate_policy()
    #         actor_state = self._create_actor_state()
    #         self.eval_policy = self._get_inference_policy()
    #         obs_dict = self.env.reset_all()
    #         init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
    #         actor_state.update({"obs": obs_dict, "actions": init_actions})
    #         actor_state = self._pre_eval_env_step(actor_state)
            
    #         T = num_collect_steps
    #         N = self.env.num_envs
            
    #         buffer_pos = torch.zeros((T, N, 3), device=self.device)
    #         buffer_quat = torch.zeros((T, N, 4), device=self.device)
    #         buffer_lin_vel = torch.zeros((T, N, 3), device=self.device)
    #         buffer_ang_vel = torch.zeros((T, N, 3), device=self.device)
            
    #         buffer_joint_pos = torch.zeros((T, N, self.env.num_dof), device=self.device)
    #         buffer_joint_vel = torch.zeros((T, N, self.env.num_dof), device=self.device)
    #         buffer_action = torch.zeros((T, N, self.num_act), device=self.device)
            
    #         # =========================================================================
    #         # 【新增：生命周期追踪器】
    #         # is_active: 记录当前哪些环境还在执行“第一次”动作 (没有被重置过)
    #         # valid_lengths: 记录每个环境最终成功存活了多少帧
    #         # =========================================================================
    #         is_active = torch.ones(N, dtype=torch.bool, device=self.device)
    #         valid_lengths = torch.zeros(N, dtype=torch.long, device=self.device)

    #         for step in range(T):
    #             actor_state["step"] = step
    #             actor_state = self._pre_eval_env_step(actor_state) 
    #             actions = actor_state["actions"]
                
    #             # 记录数据 (全量记录，后续靠 valid_lengths 截断)
    #             if hasattr(self.env.simulator, "robot_root_states"):
    #                 root_states = self.env.simulator.robot_root_states
    #                 buffer_pos[step] = root_states[:, 0:3]
    #                 buffer_quat[step] = root_states[:, 3:7]
    #                 buffer_lin_vel[step] = root_states[:, 7:10]
    #                 buffer_ang_vel[step] = root_states[:, 10:13]
    #             elif hasattr(self.env, "base_pos"):
    #                 buffer_pos[step] = self.env.base_pos
    #                 buffer_quat[step] = self.env.base_quat
    #                 buffer_lin_vel[step] = self.env.base_lin_vel
    #                 buffer_ang_vel[step] = self.env.base_ang_vel
                
    #             buffer_joint_pos[step] = self.env.simulator.dof_pos
    #             buffer_joint_vel[step] = self.env.simulator.dof_vel
    #             buffer_action[step] = actions.detach()

    #             # 物理引擎演进 (此时底层的重置逻辑可能被触发)
    #             actor_state = self.env_step(actor_state) 
                
    #             # =====================================================================
    #             # 【新增：寿命更新与突变拦截】
    #             # =====================================================================
    #             # 只要环境此时还是 active，它的有效长度就 +1
    #             valid_lengths += is_active.long()
                
    #             # 检查哪些环境在这一步被重置了 (通常在 HumanoidVerse/LeggedGym 中用 reset_buf 表示)
    #             if hasattr(self.env, "reset_buf"):
    #                 # 如果发生了重置，将其 active 状态永久设为 False
    #                 # ~self.env.reset_buf 意思是 "没有重置"
    #                 is_active = is_active & (~self.env.reset_buf)
                
    #             # 如果所有的环境都重置过了（全军覆没，或者都完成了动作），提前结束！
    #             if not is_active.any():
    #                 logger.info(f"所有环境均已结束其首次生命周期，提前终止收集 (停在第 {step+1} 帧)")
    #                 break

    #             actor_state = self._post_eval_env_step(actor_state)
                
    #             if (step + 1) % 50 == 0:
    #                 logger.info(f"收集进度: {step + 1} / {num_collect_steps} 帧, 存活环境数: {is_active.sum().item()}")

    #         logger.info("仿真收集完毕，正在转移张量并进行变长切片...")
    #         np_pos = buffer_pos.detach().cpu().numpy()
    #         np_quat = buffer_quat.detach().cpu().numpy()
    #         np_lin_vel = buffer_lin_vel.detach().cpu().numpy()
    #         np_ang_vel = buffer_ang_vel.detach().cpu().numpy()
    #         np_joint_pos = buffer_joint_pos.detach().cpu().numpy()
    #         np_joint_vel = buffer_joint_vel.detach().cpu().numpy()
    #         np_action = buffer_action.detach().cpu().numpy()
            
    #         dt = self.env.dt if hasattr(self.env, "dt") else 0.02

    #         # =========================================================================
    #         # 【新增：依据每个环境的真实寿命进行切片保存】
    #         # =========================================================================
    #         saved_count = 0
    #         for env_idx in range(N):
    #             # 获取该环境第一次重置前的实际连续帧数
    #             seq_len = valid_lengths[env_idx].item()
                
    #             # 过滤太短的垃圾数据 (比如刚出生不到 10 帧就摔倒了，没有训练价值)
    #             if seq_len < 10:
    #                 continue
                    
    #             # 动态生成该序列的时间戳
    #             np_time = np.arange(seq_len) * dt
                    
    #             # 精确切片：只截取 [0 : seq_len] 的连续纯净数据！
    #             env_pos = np_pos[:seq_len, env_idx, :]
    #             env_quat = np_quat[:seq_len, env_idx, :]
    #             env_lin_vel = np_lin_vel[:seq_len, env_idx, :]
    #             env_ang_vel = np_ang_vel[:seq_len, env_idx, :]
    #             env_joint_pos = np_joint_pos[:seq_len, env_idx, :]
    #             env_joint_vel = np_joint_vel[:seq_len, env_idx, :]
    #             env_action = np_action[:seq_len, env_idx, :]

    #             save_path = os.path.join(save_dir, f"motion_env{env_idx}_len{seq_len}.npz")
                
    #             np.savez(
    #                 save_path,
    #                 time=np_time,
    #                 pos=env_pos,
    #                 quat=env_quat,
    #                 lin_vel=env_lin_vel,
    #                 ang_vel=env_ang_vel,
    #                 joint_pos=env_joint_pos,
    #                 joint_vel=env_joint_vel,
    #                 action=env_action,
    #                 joint_pos_cmd=env_action
    #             )
    #             saved_count += 1

    #         self._post_evaluate_policy()
    #         logger.success(f"========== 收集结束！共成功保存了 {saved_count} 条无突变纯净轨迹 ==========")
    #         logger.success(f"数据已保存在: {os.path.abspath(save_dir)}")
          
    @torch.no_grad()
    def collect_delta_a_sim_data(self, num_collect_steps=1000, save_dir="logs_collect_data/delta_a_sim2sim_data/gym2gen"):
        """
        【函数作用】批量并行数据收集器（变长截断版 + 纯血 PKL 打包）。
        【输出格式】完全复刻 ASAP 官方动作库格式，包含 5 个核心键 (root_trans_offset, pose_aa, dof, root_rot, fps)，并额外附加 Delta-A 需要的 action 键。
        """
        import os
        import numpy as np
        import torch
        import joblib
        from loguru import logger
        from scipy.spatial.transform import Rotation as sRot  # 核心：用于四元数到轴角的高精度转换
        
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"========== 开始批量收集 Delta-A 仿真连续数据 ==========")
        
        self._create_eval_callbacks() 
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)
        
        T = num_collect_steps
        N = self.env.num_envs
        
        # 预分配 GPU 张量缓冲区
        buffer_pos = torch.zeros((T, N, 3), device=self.device)
        buffer_quat = torch.zeros((T, N, 4), device=self.device)
        buffer_joint_pos = torch.zeros((T, N, self.env.num_dof), device=self.device)
        buffer_action = torch.zeros((T, N, self.num_act), device=self.device)
        
        is_active = torch.ones(N, dtype=torch.bool, device=self.device)
        valid_lengths = torch.zeros(N, dtype=torch.long, device=self.device)

        for step in range(T):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state) 
            actions = actor_state["actions"]
            
            # 抓取状态 (IsaacGym 的四元数默认是 xyzw，完美兼容 sRot)
            if hasattr(self.env.simulator, "robot_root_states"):
                root_states = self.env.simulator.robot_root_states
                buffer_pos[step] = root_states[:, 0:3]
                buffer_quat[step] = root_states[:, 3:7]  # xyzw 
            elif hasattr(self.env, "base_pos"):
                buffer_pos[step] = self.env.base_pos
                buffer_quat[step] = self.env.base_quat
            
            buffer_joint_pos[step] = self.env.simulator.dof_pos
            buffer_action[step] = actions.detach()

            # 物理引擎演进
            actor_state = self.env_step(actor_state) 
            
            valid_lengths += is_active.long()
            
            # 监听重置信号
            if hasattr(self.env, "reset_buf"):
                is_active = is_active & (~self.env.reset_buf)
            
            if not is_active.any():
                logger.info(f"所有环境均已结束其首次生命周期，提前终止收集 (停在第 {step+1} 帧)")
                break

            actor_state = self._post_eval_env_step(actor_state)
            
            if (step + 1) % 50 == 0:
                logger.info(f"收集进度: {step + 1} / {num_collect_steps} 帧, 存活环境数: {is_active.sum().item()}")

        logger.info("仿真收集完毕，正在精确合成 ASAP 标准动作库字典 ...")
        # 转移到 CPU 并转为 Numpy
        np_pos = buffer_pos.detach().cpu().numpy()
        np_quat = buffer_quat.detach().cpu().numpy()
        np_joint_pos = buffer_joint_pos.detach().cpu().numpy()
        np_action = buffer_action.detach().cpu().numpy()
        
        dt = self.env.dt if hasattr(self.env, "dt") else 0.02
        fps_val = int(1.0 / dt) if dt > 0 else 50

        # 获取电机的局部旋转轴 (dof_axis)，用于合成 pose_aa
        # 形状通常为 (23, 3)
        dof_axis_np = self._motion_lib.mesh_parsers.dof_axis.cpu().numpy() 
        
        # 判断配置中是否有扩展挂载点 (用于凑够 27 维)
        if hasattr(self.config.robot.motion, "extend_config"):
            num_augment_joint = len(self.config.robot.motion.extend_config)
        else:
            num_augment_joint = 3 # 默认 G1 有 3 个标记点

        dataset_dict = {}
        saved_count = 0
        
        for env_idx in range(N):
            seq_len = valid_lengths[env_idx].item()
            if seq_len < 10:
                continue
                
            # 切片提取该环境的连续数据 (seq_len, dim)
            env_pos = np_pos[:seq_len, env_idx, :]
            env_quat = np_quat[:seq_len, env_idx, :]
            env_joint_pos = np_joint_pos[:seq_len, env_idx, :]
            env_action = np_action[:seq_len, env_idx, :]

            # =========================================================================
            # 【高精度合成 pose_aa】(严格对齐 27 维)
            # =========================================================================
            # 1. 躯干旋转 (Root): 四元数(xyzw) -> 旋转向量(Axis-Angle)
            # 形状由 (seq_len, 3) 变为 (seq_len, 1, 3)
            root_rot_vec = sRot.from_quat(env_quat).as_rotvec() 
            root_rot_vec = root_rot_vec[:, np.newaxis, :]
            
            # 2. 关节旋转 (Joints): 关节角度标量 * 对应电机的 3D 旋转轴
            # (seq_len, 23, 1) * (1, 23, 3) -> 广播计算得到 (seq_len, 23, 3)
            joint_aa = env_joint_pos[:, :, np.newaxis] * dof_axis_np[np.newaxis, :, :]
            
            # 3. 虚拟追踪点填充 (Augment): 补齐零向量
            # (seq_len, 3, 3)
            augment_aa = np.zeros((seq_len, num_augment_joint, 3), dtype=np.float32)
                
            # 4. 沿刚体维度拼接: 1 + 23 + 3 = 27
            # 最终 env_pose_aa 形状 -> (seq_len, 27, 3)
            env_pose_aa = np.concatenate([root_rot_vec, joint_aa, augment_aa], axis=1)

            # =========================================================================
            # 【打包为官方 6 键格式 (剔除 smpl_joints，增加 action)】
            # =========================================================================
            motion_key = f"motion{env_idx}"
            
            dataset_dict[motion_key] = {
                "root_trans_offset": env_pos.astype(np.float32),      # 对应 (119, 3)
                "pose_aa": env_pose_aa.astype(np.float32),            # 对应 (119, 27, 3)
                "dof": env_joint_pos.astype(np.float32),              # 对应 (119, 23)
                "root_rot": env_quat.astype(np.float64),              # 对应 (119, 4)，对齐日志中的 float64
                "fps": fps_val,                                       # 对应 30/50
                "action": env_action.astype(np.float32)               # Delta-A 专属增量键!
            }
            saved_count += 1

        save_path = os.path.join(save_dir, "collected_delta_a_dataset.pkl")
        joblib.dump(dataset_dict, save_path)

        self._post_evaluate_policy()
        logger.success(f"========== 收集结束！共成功打包了 {saved_count} 条纯净轨迹 ==========")
        logger.success(f"终极数据集已保存至: {os.path.abspath(save_path)}")  
        
    def _create_actor_state(self):
        """【函数作用】生成空状态字典。 【被谁调用】evaluate_policy."""
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        """【函数作用】实例化评估所需的 Callback。 【被谁调用】evaluate_policy."""
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.appendset_actor_root_state_tensor(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env=True):
        """【函数作用】考试前网络和环境的模式切换。 【被谁调用】evaluate_policy."""
        self._eval_mode()#保证网络在评估模式，dropout等会关闭。（网络层面，如果没有相关层，删去也没有影响）
        self.env.set_is_evaluating()#实际保证动作选择和开始时间一定。（动作采样层面）
        if reset_env:
            _ = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        """【函数作用】考试结束后的处理。 【被谁调用】evaluate_policy."""
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict):
        """【函数作用】环境单步前的前向推理，计算出下一步死命令(actions)。 【被谁调用】evaluate_policy."""
        actions = self.eval_policy(actor_state["obs"]['actor_obs'])
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        """【函数作用】环境单步后的回调触发（如渲染一帧画面）。 【被谁调用】evaluate_policy."""
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def _get_inference_policy(self, device=None):
        """【函数作用】提取纯净推断网络给外部。 【被谁调用】evaluate_policy."""
        self.actor.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference#输入actor_obs，输出动作的均值