import torch # 导入 PyTorch 深度学习框架
import torch.nn as nn # 导入 PyTorch 神经网络模块
import torch.optim as optim # 导入 PyTorch 优化器模块

# 导入底层 PPO 算法所需的各类模块：Actor、Critic、数据回放缓冲区等
from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask # 导入基础环境任务类
from humanoidverse.agents.base_algo.base_algo import BaseAlgo # 导入基础算法类
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback # 导入评估阶段的回调基类
from humanoidverse.utils.average_meters import TensorAverageMeterDict # 导入用于统计平均值的工具

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter # 导入 Tensorboard 用于可视化训练曲线
import time # 导入时间模块，用于计算训练耗时
import os # 导入操作系统接口模块
import statistics # 导入统计学模块
from collections import deque # 导入双端队列，用于平滑记录最近几十个 episode 的奖励
from hydra.utils import instantiate # 导入 Hydra 实例化工具，用于根据 yaml 动态创建对象
from loguru import logger # 导入高级日志打印工具
from rich.progress import track # 导入富文本进度条工具
from rich.console import Console # 导入富文本控制台工具
from rich.panel import Panel # 导入富文本面板工具
from rich.live import Live # 导入富文本实时更新工具
console = Console() # 实例化 rich 控制台


from humanoidverse.agents.ppo.ppo import PPO # 导入标准的 PPO 算法基类
from humanoidverse.envs.base_task.base_task import BaseTask # （重复导入）基础环境任务类
from pathlib import Path # 导入面向对象的文件路径处理工具
from omegaconf import OmegaConf # 导入 OmegaConf，用于处理 Hydra 的 yaml 配置字典
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking # 导入运动跟踪环境基类
from humanoidverse.utils.helpers import pre_process_config # 导入配置预处理工具
from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # （重复导入）基础算法类
from hydra.utils import instantiate # （重复导入）Hydra 实例化工具


class PPODeltaA(PPO):
    # 【类作用说明：微调最终策略】
    # 结合 ASAP 论文，这是用于第二阶段最终策略微调（RL Fine-tuning）的 PPO 类。
    # 它的作用是：加载并冻结已经训练好的 Delta-A 模型（负责弥补 Sim-to-Real 鸿沟），
    # 在这个“被增强的仿真环境”中，重新训练/微调机器人的主运动控制策略。
    # 这样训练出来的主策略可以直接部署到真实机器人上，无需再携带 Delta-A 模型。
    
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        # 【函数说明：初始化函数】
        # 作用：初始化微调训练器，定位、解析、实例化并加载已经训练好的 Delta-A 模型（残差模型），然后冻结其参数。
        # 被谁调用：在 train.py 主入口中，由 Hydra 实例化 `algo` 时被调用。
        
        super().__init__(env, config, log_dir, device) # 调用父类 PPO 初始化正在训练的主策略（Actor-Critic 网络）

        # 指定了预训练好的 Delta-A 模型的路径
        if config.policy_checkpoint is not None:
            has_config = True 
            checkpoint = Path(config.policy_checkpoint) 
            config_path = checkpoint.parent / "config.yaml" # 寻找 Delta-A 模型的配置文件
            if not config_path.exists():
                config_path = checkpoint.parent.parent / "config.yaml"
                if not config_path.exists():
                    has_config = False 
                    logger.error(f"Could not find config path: {config_path}") 

            if has_config:
                logger.info(f"Loading training config file from {config_path}") # 打印日志：加载 Delta-A 模型的配置
                with open(config_path) as file:
                    policy_config = OmegaConf.load(file) 

                if policy_config.eval_overrides is not None:
                    policy_config = OmegaConf.merge(
                        policy_config, policy_config.eval_overrides
                    )
            
        pre_process_config(policy_config) 
        
        # 动态实例化这个已训练好的 Delta-A 策略对象
        self.loaded_policy: BaseAlgo = instantiate(policy_config.algo, env=env, device=device, log_dir=None)
        self.loaded_policy.algo_obs_dim_dict = policy_config.env.config.robot.algo_obs_dim_dict
        self.loaded_policy.setup() #初始化神经网络，权重和rollout_buffer对象

        # 正式加载 Delta-A 的 .pt 权重文件
        self.loaded_policy.load(config.policy_checkpoint) 

        
        # import ipdb; ipdb.set_trace() 
        self.loaded_policy._eval_mode() # 强制将 Delta-A 模型切换为评估模式 (即 eval()，关闭 Dropout 等)

        # 遍历已加载的 Delta-A 模型的所有参数
        for name, param in self.loaded_policy.actor.actor_module.module.named_parameters():
            # 核心步骤：冻结 Delta-A 模型！在微调主策略时，代表“物理动力学补偿”的 Delta-A 必须保持固定。
            param.requires_grad = False 
            # print(f"Parameter name: {name}, Parameter: {param}, Requires Grad: {param.requires_grad}")
        
        # import ipdb; ipdb.set_trace()   
        # 提取 Delta-A 用于纯推理的前向传播函数句柄
        self.loaded_policy.eval_policy = self.loaded_policy._get_inference_policy()#self.loaded_policy.eval_policy被赋值为了一个函数的名字（没有括号），需要输入actor_obs，输出动作的均值
        



    def _rollout_step(self, obs_dict):
        # 【函数说明：单步数据采集 (Rollout)】
        # 作用：在仿真交互中同时运作两个网络：
        #       1. 当前正在微调的主策略网络，输出基准动作 `actions` (即 \pi(s))。
        #       2. 被冻结的 Delta-A 网络，输出残差补偿动作 `actions_closed_loop` (即 \pi^\Delta(s, a))。
        #       将这两个动作发给底层环境，环境会将它们叠加应用，并将交互结果存入 Storage 用于更新主策略。
        # 被谁调用：父类 PPO 的 `train()` 循环中。
        
        # import ipdb; ipdb.set_trace()
        # self._eval_mode()
        # self.eval_policy = self._get_inference_policy()
        with torch.inference_mode(): # 开启推理模式，数据采集不计算梯度
            for i in range(self.num_steps_per_env): 
                # Compute the actions and values
                # actions = self.actor.act(obs_dict["actor_obs"]).detach()

                policy_state_dict = {} 
                # 1. 正在训练的主网络 (Actor) 观测状态，输出名义控制指令 actions
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                # 正在训练的主网络 (Critic) 评估当前状态的价值 Value
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values 

                ## Append states to storage 
                for obs_key in obs_dict.keys():#这一步观测值放入Rollout列表这一步的位置
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():#这一步动作（实际动作，均值，标准差，对数概率）放入Rollout列表这一步的位置
                    self.storage.update_key(obs_, policy_state_dict[obs_])
                    
                actions = policy_state_dict["actions"] # 提取主网络输出的动作
                actor_state = {}
                actor_state["actions"] = actions # 主策略动作放入字典
                
                ################ inference the policy ################
                # 2. 调用冻结的 Delta-A 模型，根据专门给它准备的观测组输出“残差补偿动作”
                policy_output = self.loaded_policy.eval_policy(obs_dict['closed_loop_actor_obs']).detach()
                actor_state["actions_closed_loop"] = policy_output # 把 Delta-A 补偿存为 'actions_closed_loop' 放入字典

                ######################################################

                # 传入环境。在环境底层(_compute_torques)会发生: torques = PD(actions + actions_closed_loop)
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                # critic_obs = privileged_obs if privileged_obs is not None else obs
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().unsqueeze(1) 
                
                # 时间超时的自举 (Bootstrapping) 补偿
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                assert len(rewards_stored.shape) == 2 
                
                # 存入回放缓冲区
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                # 记录 Tensorboard 指标
                if self.log_dir is not None:
                    # Book keeping 
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode']) 
                    self.cur_reward_sum += rewards 
                    self.cur_episode_length += 1 
                    new_ids = (dones > 0).nonzero(as_tuple=False) 
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            # prepare data for training 

            # 计算多步回报 (Returns) 和优势 (Advantages)，这部分将专门用于优化正在训练的主策略
            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'), #由3个键值对（值是长度24的列表）组成的字典
                dones=self.storage.query_key('dones'), 
                rewards=self.storage.query_key('rewards'))
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict 
 
 
    @torch.no_grad() # 保证整个评估阶段不会算梯度，节省显存
    def evaluate_policy(self):
        """
        【函数作用】独立的无限循环，用于测试或可视化验证已经训好的模型。
        【机器人意义】完全脱掉训练辅助轮。以绝对确定的动作进行仿真渲染或真实机器人的控制指令分发。
        【被谁调用】专门的 evaluate.py 或 play.py 脚本。
        """
        self._create_eval_callbacks() # 挂载插件（如摄像头录屏）
        self._pre_evaluate_policy() # 告诉环境进入考试模式，关掉随机化干扰。reset_all会有一次。
        actor_state = self._create_actor_state() # 拿个小本本记录测试时的短期状态
        step = 0
        self.eval_policy = self._get_inference_policy() # 先设actor网络为eval模式，然后提取纯净大脑模型（这个函数输入action_obs,输出动作均值）
        obs_dict = self.env.reset_all() # 机器人归零发车
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device) # 第一步不发力
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state) # 预热算第一步动作。（相当于reset_all从初始位置执行全0动作0.02s后的状态，通过该状态推出的第一步动作）
        
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

            
            
        ################ inference the policy ################
            # 2. 调用冻结的 Delta-A 模型，根据专门给它准备的观测组输出“残差补偿动作”
            policy_output = self.loaded_policy.eval_policy(obs_dict['closed_loop_actor_obs']).detach()
            actor_state["actions_closed_loop"] = policy_output # 把 Delta-A 补偿存为 'actions_closed_loop' 放入字典

            ######################################################

            
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
   

    def _pre_eval_env_step(self, actor_state: dict):
        # 【函数说明：评估阶段单步前置处理】
        # 作用：在策略评估（测试）模式下被调用。依然是“双网络同时工作”。
        #       1. 微调后的主网络计算基础控制。
        #       2. 冻结的 Delta-A 网络计算物理补偿。
        # 被谁调用：在评估主循环或模型测试脚本执行物理推演之前被调用。
        
        # 1. 微调中的主策略推理出基础动作
        actions = self.eval_policy(actor_state["obs"]['actor_obs'])
        # 2. 冻结的 Delta-A 策略推理出用于修正的残差动作
        actions_closed_loop = self.loaded_policy.eval_policy(actor_state['obs']['closed_loop_actor_obs']).detach()

        actor_state.update({"actions": actions, "actions_closed_loop": actions_closed_loop})
        
        # 触发下游回调（如误差分析画图工具）
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state