from __future__ import annotations # 允许在定义类之前在类型提示中使用该类的名字（前向引用）
from copy import deepcopy # 引入深拷贝，用于在内存中完全复制一个复杂的对象（如神经网络模型）
from datetime import datetime
import os

import torch # 导入PyTorch深度学习框架核心库
import torch.nn as nn # 导入PyTorch的神经网络模块，用于构建网络层（如全连接层）
from torch.distributions import Normal # 导入正态分布（高斯分布），在机器人连续动作控制中，用于构建动作的概率分布

from .modules import BaseModule # 从当前包的modules模块中导入BaseModule，这是用户自定义的基础神经网络类（通常包含MLP多层感知机的构建逻辑）

class PPOActor(nn.Module):
    # 【类作用】PPO算法的策略网络（Actor），用于输出机器人的连续动作指令（如各个电机的目标角度或扭矩）。
    # 【本质关联】在连续控制任务中，Actor不仅输出一个具体的动作，而是输出一个高斯分布（均值和标准差），用于探索（Exploration）。
    def __init__(self,
                 obs_dim_dict, # 观察维度字典，定义了机器人传感器输入（如关节角度、角速度、目标位置等）的特征维度
                 module_config_dict, # 神经网络架构配置字典（如隐藏层层数、神经元个数、激活函数类型等）
                 num_actions, # 动作维度，即机器人需要控制的执行器数量（如机械臂的关节自由度数量）
                 init_noise_std): # 动作分布的初始标准差，决定了训练初期的探索范围（数值越大，机器人初期随机乱动的幅度越大）
        # 【函数作用】初始化策略网络结构和动作的高斯分布标准差。
        # 【被谁调用】在PPO算法的主体类（如 PPO 训练器或 Agent 类）初始化时被调用。
        super(PPOActor, self).__init__() # 调用父类 nn.Module 的初始化方法，注册为PyTorch的神经网络模块

        # 调用内部方法解析配置文件，将配置文件中的通用占位符替换为真实的机器人动作维度
        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        # 实例化真正的神经网络模型（通常是一个多层感知机MLP），输入是机器人的状态，输出是动作的均值
        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)

        # Action noise: 动作噪声，即高斯分布的标准差（Sigma）。
        # nn.Parameter 使得这个标准差成为神经网络的一个可训练参数，随着强化学习的进行，网络会自动调整它（通常会逐渐减小，让机器人动作从随机探索趋于稳定收敛）。
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions)) 
        self.distribution = None # 初始化动作概率分布对象为空，在执行动作时会被实例化
        # disable args validation for speedup: 禁用正态分布参数的合法性验证（如检查标准差是否大于0），为了在机器人高频控制的训练循环中加快运算速度
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        # 【函数作用】处理网络配置，动态地将配置里的字符串占位符替换成实际数值。
        # 【被谁调用】被当前类的 __init__ 构造函数调用。
        for idx, output_dim in enumerate(module_config_dict['output_dim']): # 遍历配置字典中定义的每一层的输出维度。enumerate(module_config_dict['output_dim'])的作用是：遍历 output_dim 这个列表，同时返回每个元素的索引（idx）和对应的值（output_dim）。
            if output_dim == 'robot_action_dim': # 如果发现输出维度被配置成了字符串 'robot_action_dim'（通常代表网络的最后一层输出）
                module_config_dict['output_dim'][idx] = num_actions # 将其强行替换为真实的动作维度整数（如赋予具体机器人的自由度个数）
        return module_config_dict # 返回修改后的配置字典

    @property # Python装饰器，把方法变成属性调用，隐藏内部的 `actor_module` 变量名
    def actor(self):
        # 【函数作用】获取实际的神经网络模型对象。
        # 【被谁调用】被类内部的方法（如 update_distribution）或其他外部管理权重的代码调用。
        return self.actor_module
    
    @staticmethod # Python静态方法装饰器，不需要实例化类即可调用，不传入 self
    # not used at the moment: 原作者注（当前未使用）
    def init_weights(sequential, scales):
        # 【函数作用】正交初始化（Orthogonal Initialization）神经网络的权重。这在强化学习中极其重要，可以避免深层网络中的梯度消失/爆炸，稳定PPO训练。
        # 【被谁调用】目前未被调用，但通常在创建完网络实例后，被网络构建器或主训练脚本调用来初始化线性层。
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in # 使用正交初始化方法对权重赋值，gain 控制方差的缩放
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))] # 列表推导式：遍历传入的 sequence（如 nn.Sequential），挑出所有全连接层（nn.Linear）进行初始化

    def reset(self, dones=None):
        # 【函数作用】重置网络的内部状态。主要用于包含 RNN/LSTM 等具有记忆功能的时间序列网络。
        # 【被谁调用】在环境交互时（Rollout），如果某个环境的 Episode 结束了（dones=True），由数据收集器（Runner/Worker）调用以清空隐藏状态。
        pass # 由于当前的基础网络（BaseModule）应该是普通的前馈网络（MLP），没有隐藏记忆状态，所以这里直接 pass

    def forward(self):
        # 【函数作用】PyTorch 要求重写的网络前向传播标准接口。
        # 【被谁调用】当使用 `model()` 形式直接调用时。但在当前设计中被废弃了，因为引入了更明确的 act 和 evaluate 接口。
        raise NotImplementedError # 抛出未实现异常。强制开发者不要直接调用 forward，而是去调用 act 或 act_inference
    
    @property # 装饰器，作为只读属性访问
    def action_mean(self):
        # 【函数作用】获取当前动作高斯分布的均值（即神经网络直接输出的动作基准值）。
        # 【被谁调用】在计算PPO的KL散度、记录Tensorboard日志或策略监控时由训练器调用。
        return self.distribution.mean # 返回正态分布的均值张量

    @property # 装饰器
    def action_std(self):
        # 【函数作用】获取当前动作高斯分布的标准差（即机器人动作的随机探索程度）。
        # 【被谁调用】通常用于日志记录，监控机器人的探索空间是否在收敛（std 逐渐变小），由主循环或 Logger 调用。
        return self.distribution.stddev # 返回正态分布的标准差张量
    
    @property # 装饰器
    def entropy(self):
        # 【函数作用】计算当前动作分布的信息熵。熵越大代表机器人越“不确定”，探索性越强。
        # 【本质关联】PPO算法会在损失函数中加入一项“熵奖励（Entropy Bonus）”，鼓励机器人多探索，防止过早陷入局部最优（如机器人发现原地不动能避免惩罚，就再也不动了）。
        # 【被谁调用】PPO的优化器/损失计算模块在反向传播计算总 Loss 时调用。
        return self.distribution.entropy().sum(dim=-1) # 计算熵并在动作维度（最后一个维度）上求和，得到针对单个状态的整体动作分布熵

    def update_distribution(self, actor_obs):
        # 【函数作用】根据当前状态，执行一次神经网络前向传播，计算并更新动作的高斯分布。
        # 【被谁调用】被自身的 act 方法调用。
        mean = self.actor(actor_obs) # 将观察数据输入神经网络（Actor），网络前向传播输出预测的动作均值（mean）
        # self.distribution = Normal(mean, mean*0. + self.std) # 构造PyTorch的标准正态分布对象。mean*0. 是为了通过广播机制将全局可学习的 self.std 扩展成与 mean 形状一致的张量
        print(f"动作均值（mean）张量的统计信息：\n- 形状: {mean.shape}\n- 数值范围: [{mean.min().item()}, {mean.max().item()}]\n- 是否包含 NaN: {torch.isnan(mean).any().item()}\n- 是否包含 Inf: {torch.isinf(mean).any().item()}\n")
        print(f"动作标准差（std）张量的统计信息：\n- 形状: {self.std.shape}\n- 数值范围: [{self.std.min().item()}, {self.std.max().item()}]\n- 是否包含 NaN: {torch.isnan(self.std).any().item()}\n- 是否包含 Inf: {torch.isinf(self.std).any().item()}\n")
        try:
            # 你的原始代码：尝试构建正态分布
            self.distribution = Normal(mean, mean*0. + self.std)
        except ValueError as e:
            # 1. 生成时间戳，作为两个文件的统一后缀，方便对应
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_filename = f"crash_locals_{timestamp}.txt"  # 文件1：文本日志
            pt_filename = f"crash_tensors_{timestamp}.pt"   # 文件2：张量数据
            
            # ========== 生成文件 1：所有的 locals() 文本摘要 ==========
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write("🚨 致命报错拦截：试图用 NaN 构建正态分布！\n")
                f.write(f"报错信息: {e}\n\n")
                f.write("🔍 当前函数内的所有局部变量 (locals) 状态:\n")
                
                for var_name, var_value in locals().items():
                    if isinstance(var_value, torch.Tensor):
                        has_nan = torch.isnan(var_value).any().item()
                        has_inf = torch.isinf(var_value).any().item()
                        f.write(f"👉 {var_name}: [Tensor] 形状 {var_value.shape} | 包含 NaN: {has_nan} | 包含 Inf: {has_inf}\n")
                    else:
                        # 强转字符串并截断前200个字符，防止文本过长
                        f.write(f"👉 {var_name}: {str(var_value)[:200]}\n") 

            # ========== 检查 self.env.rew_buf 是否包含 NaN，并记录对应奖励分量 ==========
            if hasattr(self, "env") and hasattr(self.env, "rew_buf"):
                rew_buf = self.env.rew_buf
                if isinstance(rew_buf, torch.Tensor):
                    nan_mask = torch.isnan(rew_buf)
                    if nan_mask.any().item():
                        nan_ids = torch.nonzero(nan_mask, as_tuple=False).flatten().tolist()
                        print(f"🚨 检测到 self.env.rew_buf 含 NaN，env_ids={nan_ids}")

                        reward_components = {}
                        if hasattr(self.env, "episode_sums") and isinstance(self.env.episode_sums, dict):
                            reward_components.update(self.env.episode_sums)
                        if hasattr(self.env, "extras") and isinstance(self.env.extras, dict):
                            for k, v in self.env.extras.items():
                                if isinstance(v, torch.Tensor) and v.shape[:1] == rew_buf.shape[:1]:
                                    reward_components[f"extras.{k}"] = v

                        with open(txt_filename, "a", encoding="utf-8") as f:
                            f.write("\n\n🧪 self.env.rew_buf NaN 详细排查:\n")
                            f.write(f"nan env ids: {nan_ids}\n")
                            for env_id in nan_ids:
                                f.write(f"\n[env_id={env_id}] rew_buf={rew_buf[env_id].item()}\n")
                                print(f"[env_id={env_id}] rew_buf={rew_buf[env_id].item()}")
                                for name, tensor_val in reward_components.items():
                                    if isinstance(tensor_val, torch.Tensor) and tensor_val.shape[:1] == rew_buf.shape[:1]:
                                        val = tensor_val[env_id].item()
                                        has_nan = torch.isnan(tensor_val[env_id]).item()
                                        has_inf = torch.isinf(tensor_val[env_id]).item()
                                        f.write(f"  - {name}: {val} (NaN={has_nan}, Inf={has_inf})\n")
                                        print(f"  - {name}: {val} (NaN={has_nan}, Inf={has_inf})")
            
            
            # ========== 生成文件 2：所有的张量真实数据 ==========
            tensor_dict = {}
            for var_name, var_value in locals().items():
                if isinstance(var_value, torch.Tensor):
                    # .detach().cpu() 是极其关键的一步，它把张量从 GPU 显存上安全地拷贝到系统内存中
                    tensor_dict[var_name] = var_value.detach().cpu()
            
            torch.save(tensor_dict, pt_filename)
            
            # ========== 终端通知与终止 ==========
            print("\n" + "="*60)
            print("🚨 发生数值崩溃！现场数据已封存：")
            print(f"📄 Locals 文本日志: {os.path.abspath(txt_filename)}")
            print(f"📦 Tensors 真实数据: {os.path.abspath(pt_filename)}")
            print("="*60 + "\n")
            
            raise e # 再次抛出错误，彻底停止程序的错误运行
        

    def act(self, actor_obs, **kwargs):
        # 【函数作用】在训练阶段与环境交互时获取动作。它会根据当前策略分布“采样”出一个带噪声的动作。
        # 【本质关联】这是 RL 收集经验（Rollout）时的核心行为。通过采样，机器人偶尔会尝试意料之外的动作，从而发现更好的运动轨迹。
        # 【被谁调用】被强化学习的 Rollout Worker（数据收集器）在每个环境时间步（Step）中调用。
        self.update_distribution(actor_obs) # 1. 结合当前观察状态，更新机器人的动作概率分布
        return self.distribution.sample() # 2. 从该分布中进行随机采样，返回一个带探索噪声的具体动作指令（张量）
    
    def get_actions_log_prob(self, actions):
        # 【函数作用】计算特定动作在当前策略分布下的对数概率（Log Probability）。
        # 【本质关联】PPO 算法的核心是 Importance Sampling（重要性采样）。我们需要计算之前收集的旧动作在当前新网络下的概率，从而得到新旧策略的比率（Ratio），据此限制策略更新的步长（Clip）。
        # 【被谁调用】被 PPO 的算法核心（PPO Trainer/Optimizer）在计算 Surrogate Loss（替代损失）时调用。
        return self.distribution.log_prob(actions).sum(dim=-1) # 计算每个动作分量的对数概率，并在动作维度上求和（假设各关节动作相互独立，对数概率相加等于总概率相乘）

    def act_inference(self, actor_obs):
        # 【函数作用】在推理/评估阶段（部署到真实机器人或测试时）获取动作。直接返回分布的均值，不进行任何随机采样。
        # 【本质关联】部署机器人时，我们希望它的运动是确定的、平滑的，不需要再探索，因此直接使用网络预测的均值（最优解）。
        # 【被谁调用】被模型的 Evaluator（评估器）或真实机器人控制环（Control Loop）调用。
        actions_mean = self.actor(actor_obs) # 直接前向传播，得到网络预测的最优动作均值
        return actions_mean # 返回均值作为最终下发的物理动作
    
    def to_cpu(self):
        # 【函数作用】将网络模型和参数从 GPU 显存转移到 CPU 内存。
        # 【被谁调用】通常在保存模型权重、多进程数据共享或在没有 GPU 的机器上进行推理时调用。
        self.actor = deepcopy(self.actor).to('cpu') # 对内部的网络模型进行深拷贝，并将其张量转移到 CPU。（注：原代码中self.actor是只读@property，在严格Python语法中这一行赋值可能会抛出AttributeError异常，但这里保持原样）
        self.std.to('cpu') # 将控制方差的可学习参数字典也转移到 CPU



class PPOCritic(nn.Module):
    # 【类作用】PPO算法的价值网络（Critic）。用于评估当前机器人所处状态的“价值”（Value），即预测从当前状态开始，一直到最后能获得的总奖励预期。
    # 【本质关联】如果机器人的状态是“即将摔倒”，Critic 应该输出一个很低的价值分数；如果状态是“稳定向目标奔跑”，应该输出很高的分数。Actor 利用这个分数来判断自己刚才的动作是好是坏。
    def __init__(self,
                 obs_dim_dict, # 观察维度字典，定义Critic所需的状态输入维度（在非对称Actor-Critic架构中，Critic有时会接收比Actor更多的信息，比如特权环境信息）
                 module_config_dict): # Critic神经网络的配置字典
        # 【函数作用】初始化价值评估网络结构。
        # 【被谁调用】由 PPO 训练器/Agent 初始化代码调用。
        super(PPOCritic, self).__init__() # 继承并初始化PyTorch核心网络模块

        self.critic_module = BaseModule(obs_dim_dict, module_config_dict) # 实例化真正的评估神经网络模型（通常输出层只有一个神经元，代表标量价值 V）

    @property # 装饰器
    def critic(self):
        # 【函数作用】获取实际的价值神经网络模型对象。
        # 【被谁调用】类内或外部权重的管理器调用。
        return self.critic_module # 返回 Critic 模型实例
    
    def reset(self, dones=None):
        # 【函数作用】重置价值网络的记忆状态（若包含RNN/GRU）。
        # 【被谁调用】在环境重置（Episode 结束）时被 Rollout Worker 调用。
        pass # 前馈网络无需重置，直接跳过
    
    def evaluate(self, critic_obs, **kwargs):
        # 【函数作用】计算给定状态的价值（Value Estimation）。
        # 【本质关联】用来计算 Advantage（优势函数）。Advantage = 实际得到的回报 - Critic的预测价值。如果 Advantage > 0，说明 Actor 刚才的动作比预期好，Actor需要增加产生该动作的概率。
        # 【被谁调用】
        # 1. 在收集数据时，由数据收集器调用以计算并存储每个状态的 V(s)。
        # 2. 在训练时，由 PPO Trainer 调用，预测新的价值估计以计算 Critic 的 MSE/Huber Loss。
        value = self.critic(critic_obs) # 将观察数据输入给 Critic 网络，前向传播得到标量价值估计
        return value # 返回价值评估分数

# Deprecated: TODO: Let Wenli Fix this 
# 【开发者留言】代码已废弃。待办事项：让Wenli修复这里。（说明这个类目前在代码库中不被推荐使用，可能有遗留bug或者逻辑不兼容）
class PPOActorFixSigma(PPOActor):
    # 【类作用】这是一个修改版的 Actor 策略，它的标准差（Sigma/std）是固定的（或者外部决定的），不需要神经网络将其作为一个可训练的参数去优化。
    # 【本质关联】有时在特殊的机器人控制任务中，我们不希望动作的噪声（探索率）由网络自动学习，而是希望人为制定一个退火（Annealing）计划或者完全取消噪声。
    def __init__(self,                 
                 obs_dim_dict, # 状态观察维度配置
                 network_dict, # 等同于上文的 module_config_dict，网络结构参数
                 network_load_dict, # 网络预训练权重的加载路径或配置字典
                 num_actions,): # 机器人动作自由度数量
        # 【函数作用】初始化策略网络，强制将初始噪声标准差设为0.0。
        # 【被谁调用】被不再使用的旧版训练脚本或特定的评估代码调用。
        super(PPOActorFixSigma, self).__init__(obs_dim_dict, network_dict, network_load_dict, num_actions, 0.0) # 调用父类 PPOActor 的初始化，且硬编码 init_noise_std 为 0.0
        
    def update_distribution(self, obs_dict):
        # 【函数作用】重写父类的方法。不再构建正态分布对象，而是直接提取确定的均值。
        # 【被谁调用】被当前类的 act 方法调用。
        mean = self.actor(obs_dict)['head'] # 前向传播。这里的网络可能稍有不同，它的输出是一个字典，'head' 键对应动作均值张量
        self.distribution = mean # 直接将这个分布对象设为一个确定的张量（均值），破坏了原本的分布逻辑，也就是退化为了确定性策略（Deterministic Policy）

    @property # 装饰器
    def action_mean(self):
        # 【函数作用】获取确定性动作。
        # 【被谁调用】当外部需要提取均值时。
        return self.distribution # 因为 self.distribution 被强行赋值成了张量均值，所以这里直接返回
    
    def get_actions_log_prob(self, actions):
        # 【函数作用】因为变成了确定性策略，无法通过高斯分布计算动作的概率密度（会报错或返回无穷大）。
        # 【被谁调用】如果 PPO 优化器试图调用它，程序应该抛出错误，因为纯确定性策略无法计算概率的比率（Ratio），应该转用 DDPG 等算法。
        raise NotImplementedError # 抛出未实现异常，明确阻断在 PPO 的对数概率计算流程中的使用
    
    def act(self, obs_dict, **kwargs):
        # 【函数作用】执行动作。此时没有任何探索噪声，而是完全贪婪（Greedy）的行为。
        # 【被谁调用】由于没有噪声，通常只能用于测试（Inference），或者依靠环境自身的动力学噪声进行探索。
        self.update_distribution(obs_dict) # 更新当前的预测均值
        return self.distribution # 直接返回均值张量，而不是通过 sample() 采样（因为不再是分布对象，也没有 sample 属性）