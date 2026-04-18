import os
import copy
import torch
from torch import nn
import numpy as np
import random

from typing import Any, List, Dict
from termcolor import colored
from loguru import logger

# ============================================================================
# 函数 1: class_to_dict
# ============================================================================
def class_to_dict(obj) -> dict:
    # 如果传入的对象没有 __dict__ 属性（说明它可能是基本数据类型如 int, str 等，而不是类实例）
    if not hasattr(obj,"__dict__"):
        return obj # 直接返回该对象本身
    
    result = {} # 初始化一个空字典，用于存放转换后的键值对
    
    # 遍历对象的所有属性和方法名称
    for key in dir(obj):
        # 忽略以 "_" 开头的私有属性或内置魔术方法（如 __init__）
        if key.startswith("_"):
            continue
        
        element = []
        # 获取该属性的实际值
        val = getattr(obj, key)
        
        # 如果该属性的值是一个列表
        if isinstance(val, list):
            # 遍历列表中的每一个元素，递归调用 class_to_dict 进行转换
            for item in val:
                element.append(class_to_dict(item))
        else:
            # 如果不是列表，直接递归调用 class_to_dict 转换该值
            element = class_to_dict(val)
        
        # 将转换后的值存入结果字典中
        result[key] = element
        
    return result # 返回最终生成的嵌套字典

"""
【函数说明】：class_to_dict
作用：将一个嵌套的面向对象配置类（Config objects）转换成标准的 Python 字典（Dict）。
在强化学习中，机器人的环境参数、算法超参数通常被写成层级分明的 Python 类。但在记录日志（例如上传到 WandB 或 TensorBoard）或保存配置文件时，我们需要将其序列化为字典格式。
被谁调用：通常在训练脚本（train.py）的初始化阶段被调用，用于将 EnvConfig 和 TrainConfig 转换为字典，保存到日志文件夹中。
"""

# ============================================================================
# 函数 2: pre_process_config
# ============================================================================
def pre_process_config(config) -> None:
    
    # 下面两行是被注释掉的旧代码，原本用于重置观测维度
    # config.robot.policy_obs_dim = -1
    # config.robot.critic_obs_dim = -1
    
    obs_dim_dict = dict() # 用于存储各个观测组的总维度
    # 从配置中获取普通观测组的配置（比如包含关节角度、角速度等）
    _obs_key_list = config.env.config.obs.obs_dict
    # 从配置中获取辅助观测组的配置（比如历史观测 frame stacking，或者特权信息 privileged info）
    _aux_obs_key_list = config.env.config.obs.obs_auxiliary
    
    # 断言：确保配置中“噪声缩放比例”的键名和“观测缩放比例”的键名完全一致，防止漏配
    assert set(config.env.config.obs.noise_scales.keys()) == set(config.env.config.obs.obs_scales.keys())

    # convert obs_dims to list of dicts
    # 扁平化处理：将包含多个字典（.yaml文件中的键值对传入时会自动转换成字典，所以可以用.items()方法）的列表 config.env.config.obs.obs_dims 转换成一个大字典
    each_dict_obs_dims = {k: v for d in config.env.config.obs.obs_dims for k, v in d.items()}
    config.env.config.obs.obs_dims = each_dict_obs_dims # 更新配置中的观测维度字典
    logger.info(f"obs_dims: {each_dict_obs_dims}") # 打印日志
    
    auxiliary_obs_dims = {} # 用于存储辅助观测的维度
    # 遍历所有辅助观测配置
    for aux_obs_key, aux_config in _aux_obs_key_list.items():
        auxiliary_obs_dims[aux_obs_key] = 0 # 初始化该辅助观测的维度为0
        for _key, _num in aux_config.items():
            # 确保辅助观测中依赖的基础观测项，在总观测维度字典中是存在的
            assert _key in config.env.config.obs.obs_dims.keys()
            # 辅助观测维度 = 基础观测项的维度 * 数量（例如历史观测可能堆叠了3帧，_num就是3）
            auxiliary_obs_dims[aux_obs_key] += config.env.config.obs.obs_dims[_key] * _num
    logger.info(f"auxiliary_obs_dims: {auxiliary_obs_dims}")
    
    # 遍历所有主观测配置，计算 Actor 和 Critic 神经网络的输入维度
    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0 # 初始化该观测组维度为0
        for key in obs_config:
            # 如果配置键名以 "_raw" 结尾，去掉这部分以获取真实键名
            if key.endswith("_raw"): key = key[:-4]
            # 如果该项是基础观测（如当前的关节角度）
            if key in config.env.config.obs.obs_dims.keys(): 
                obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[key] # 累加维度
                logger.info(f"{obs_key}: {key} has dim: {config.env.config.obs.obs_dims[key]}")
            # 如果该项是辅助观测（如历史观测）
            else:
                obs_dim_dict[obs_key] += auxiliary_obs_dims[key] # 从辅助观测字典中累加维度
                logger.info(f"{obs_key}: {key} has dim: {auxiliary_obs_dims[key]}")
                
    # 将计算好的所有输入维度保存到 robot 配置中，供后续初始化神经网络（PPO）使用
    config.robot.algo_obs_dim_dict = obs_dim_dict
    logger.info(f"algo_obs_dim_dict: {config.robot.algo_obs_dim_dict}")

    # 下面这段被注释的代码原本是用于自动同步网络输出维度与动作维度（action_dim）的
    # compute action_dim for ppo
    # for agent in config.algo.config.network_dict.keys():
    #     for network in config.algo.config.network_dict[agent].keys():
    #         output_dim = config.algo.config.network_dict[agent][network].output_dim
    #         if output_dim == "action_dim":
    #             config.algo.config.network_dict[agent][network].output_dim = config.env.config.robot.actions_dim
                
    # print the config
    logger.debug(f"PPO CONFIG")
    logger.debug(f"{config.algo.config.module_dict}")
    # logger.debug(f"{config.algo.config.network_dict}")

"""
【函数说明】：pre_process_config
作用：在构建强化学习模型（Actor-Critic 网络）之前，自动计算并汇总观测空间（Observation Space）的总维度。
在机器人学习中，输入给神经网络的观测由很多小部分拼接而成（如：底盘线速度3维+角速度3维+12个关节角度+12个关节速度+...=几十维）。如果有历史堆叠（History），维度会成倍增加。这个函数会自动遍历你的配置表，把所有启用传感器的维度加起来，防止你手动算错。
被谁调用：在环境或算法初始化阶段（如 PPO/DDPG 算法实例化前），由主控脚本调用，因为神经网络的全连接层(Linear Layer)初始化时必须知道精准的 input_dim。
"""

# ============================================================================
# 函数 3: parse_observation
# ============================================================================
def parse_observation(cls: Any, 
                      key_list: List, 
                      buf_dict: Dict, 
                      obs_scales: Dict, 
                      noise_scales: Dict,
                      current_noise_curriculum_value: Any) -> None:
    """ Parse observations for the legged_robot_base class
    （为四足/腿足机器人基类解析观测值）
    """

    # 遍历需要获取的观测键名列表（如 ["base_lin_vel", "dof_pos", ...]）
    for obs_key in key_list:#obs_key是观测键名，如"base_lin_vel"。
        # 如果键名以 "_raw" 结尾，代表我们需要获取未经噪声污染的“干净/原始”数据
        if obs_key.endswith("_raw"):
            obs_key = obs_key[:-4] # 去掉 "_raw" 后缀，得到真正的观测名
            obs_noise = 0. # 对于原始数据，噪声设为0
        else:
            # 否则，计算当前应该加入的噪声大小。
            # noise_scales 包含每个变量的基准噪声幅度，current_noise_curriculum_value 是课程学习的比例（随训练加深逐渐变大）
            obs_noise = noise_scales[obs_key] * current_noise_curriculum_value
        
        # print(f"obs_key: {obs_key}, obs_noise: {obs_noise}")
        
        # 核心黑魔法：getattr 动态调用函数。
        # 假设 obs_key 是 "dof_pos"，这里等价于调用了 cls._get_obs_dof_pos().clone()
        # 它要求机器人类（cls）中必须实现这些特定的数据获取方法。 clone() 防止原数据被修改。
        actor_obs = getattr(cls, f"_get_obs_{obs_key}")().clone()#也不一定是actor_obs？。
        
        # 获取该观测值的缩放系数（归一化系数，将物理量缩放到类似 [-1, 1] 的区间以利于神经网络训练）
        obs_scale = obs_scales[obs_key]
        
        # Yuanhang 留下的注释：使用 rand_like (均匀分布 0-1) 替代 randn_like (正态分布 N~[0,1])
        # 之前的代码（被注释）：buf_dict[obs_key] = actor_obs * obs_scale + (torch.randn_like(actor_obs)* 2. - 1.) * obs_noise
        # 当前代码原理：
        # torch.rand_like(actor_obs) 生成 0 到 1 的随机张量。
        # * 2. - 1. 将其变换为 -1 到 1 之间的均匀分布噪声。
        # 再乘以 obs_noise 得到实际添加的物理噪声量。
        # 公式解释： (原始观测 + 均匀噪声) * 缩放系数。 注意：这里是加了噪声后再乘缩放系数！
        buf_dict[obs_key] = (actor_obs + (torch.rand_like(actor_obs)* 2. - 1.) * obs_noise) * obs_scale#先加噪再乘放缩系数

"""
【函数说明】：parse_observation
作用：数据流的核心枢纽。它的功能是：1. 从仿真器中读取机器人的状态数据；2. 加入 Domain Randomization (域随机化/传感器噪声) 以增强策略的鲁棒性（Sim2Real 必备）；3. 乘以 scale 缩放系数进行数据归一化。
特别亮点：采用了“噪声课程学习”（current_noise_curriculum_value），即在训练初期让机器人学“干净”的环境，训练后期逐渐加大传感器的噪声（如关节抖动、IMU漂移），这能大幅提高真机部署的成功率。
被谁调用：每一帧（step）仿真中，被机器人基类（如 `legged_robot.py` 中的 `compute_observations` 方法）调用，生成的 `buf_dict` 就是最终喂给 Actor 神经网络的输入 Tensor。
"""

# ============================================================================
# 函数 4: export_policy_as_jit
# ============================================================================
def export_policy_as_jit(actor_critic, path):
    # 如果 Actor-Critic 模型中包含 'memory_a' 属性，说明它是一个带记忆的循环神经网络（RNN/LSTM/GRU）
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU (代码注释提示：当前假设是LSTM，未来待办添加GRU支持)
        # 使用专门的导出器类（定义在下面）包装模型
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path) # 执行导出
    else: 
        # 如果不是RNN，说明是普通的多层感知机（MLP）
        os.makedirs(path, exist_ok=True) # 创建目标文件夹
        path = os.path.join(path, 'policy_1.pt') # 设定保存文件名为 policy_1.pt
        
        # 深拷贝 Actor 策略网络（忽略 Critic，因为部署时只需要 Actor 来输出动作），并移动到 CPU 上
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        # 将 PyTorch 模型编译为 TorchScript，脱离 Python 依赖
        traced_script_module = torch.jit.script(model)
        # 保存为 .pt 文件
        traced_script_module.save(path)

"""
【函数说明】：export_policy_as_jit
作用：模型部署（Deployment）。把在 Python+PyTorch 环境下训练好的神经网络，打包编译成 TorchScript 格式（.pt 文件）。
为什么要这么做：真狗/真机控制系统的底层通常是 C++（为了追求极致的实时性和低延迟，如 1000Hz 的控制频率）。C++ 无法直接运行 Python 代码，但可以通过 LibTorch 加载 TorchScript 文件来进行神经网络的推理（Inference）。
被谁调用：在训练结束时，或者训练过程中定期保存检查点（Checkpoints）时，被环境的 `save` 方法或者主控台的回调函数调用。
"""

# ============================================================================
# 类 5: PolicyExporterLSTM
# ============================================================================
class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        # 拷贝 Actor 决策网络
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent # 记录是否为循环网络标志
        
        # 提取并拷贝 LSTM 的核心结构
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu() # 移至 CPU（真机推理通常在CPU上运行即可满足实时性要求）
        
        # 注册缓冲区 (Buffers)：hidden_state (隐藏状态) 和 cell_state (细胞状态)
        # Register_buffer 的作用是让这些张量随模型一起保存，但不作为网络参数进行梯度更新。
        # 这里的维度设定为 (num_layers, batch_size=1, hidden_size)
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    # 前向传播函数：在真机部署时，C++ 每次获取新观测就会调用这个前向传播
    def forward(self, x):
        # 1. 扩展维度（增加序列长度维度=1），并传入当前的隐状态(h, c)通过 LSTM 层
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        
        # 2. 【核心逻辑】更新状态。把当前的隐状态保存下来，留给下一个时间步（下一步仿真/现实使用）
        self.hidden_state[:] = h
        self.cell_state[:] = c
        
        # 3. 将 LSTM 的输出去维后，送入普通的 Actor MLP 网络，输出最终的动作命令（如目标关节角度）
        return self.actor(out.squeeze(0))

    # 使用 @torch.jit.export 装饰器，表示这个方法也要一起导出到 C++ 环境中
    @torch.jit.export
    def reset_memory(self):
        # 重置隐藏状态和细胞状态为全 0
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        # 导出逻辑，与非 LSTM 版本类似
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        # 编译自己（包含了 actor, memory, 和自动状态更新的 forward 逻辑）为 TorchScript
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

"""
【类说明】：PolicyExporterLSTM
作用：为带状态的神经网络（RNN/LSTM）提供一种特殊的封装容器，以实现完美的 C++ 部署。
由于 LSTM 在每次推理时都需要输入上一帧的“隐藏状态”，如果直接导出原模型，C++ 端每次都要手动维护并传递这个隐状态变量，非常麻烦且容易出错。
这个类的巧妙之处在于：它把 LSTM 的隐藏状态（hidden_state）定义为了模型内部的 buffer。在它的 `forward` 函数中，每次自动读取内部状态，并自动更新它。这样对于外部（C++）调用者来说，就和调用普通无状态 MLP 一样，只需要不断输入当前帧的 observation，就能拿到 action。
新增的 `reset_memory()` 方法供 C++ 端在机器人摔倒、重启、进入新回合（Episode）时调用，清空记忆。
被谁调用：只在 python 端被上面定义的 `export_policy_as_jit` 函数在识别到有 LSTM 结构时实例化并调用。
"""