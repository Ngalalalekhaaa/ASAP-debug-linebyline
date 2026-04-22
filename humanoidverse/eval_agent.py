import os
import sys
from pathlib import Path

# Hydra 是一个管理复杂配置文件的利器。机器人RL配置极多（奖励权重、环境物理参数、网络结构等），通常用Hydra管理。
import hydra 
from hydra.utils import instantiate # 动态实例化：根据配置文件里的路径，直接生成对应的 Python 对象
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf # 处理 Hydra 配置字典的库
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
#from utils.config_utils import * # noqa: E402, F403 #和下面重复导入了，已经被你注释掉

# add argparse arguments (添加命令行参数解析)

from humanoidverse.utils.config_utils import * # noqa: E402, F403
from loguru import logger # 一个非常好用的彩色日志库

import threading # 引入多线程。为了在机器人仿真的同时监听键盘输入，必须开新线程，否则会卡死仿真主循环
# from pynput import keyboard # pynput用于捕获键盘按键（注释掉了可能在其他地方导入了）

import os
import random
import numpy as np
# import torch

def set_seed(seed=42):
    """
    固定所有常见的随机数种子，以确保实验的可重复性。
    """
    # 1. 操作系统级别的随机性（影响哈希等）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Python 内置的 random 模块
    random.seed(seed)
    
    # 3. NumPy
    np.random.seed(seed)
    
    # 4. PyTorch
    torch.manual_seed(seed)
    
    # 5. 如果你使用了 GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
        
        # 针对 cuDNN 的额外设置，确保每次计算过程一致
        # 注意：这可能会稍微降低一点训练速度
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

# 在 main.py 或训练脚本最开头调用
# set_seed(42)


# ============================================================================
# 函数 1: on_press
# ============================================================================
def on_press(key, env):
    try:
        # 如果按下 'n' 键，切换到下一个任务（多任务强化学习场景下使用，比如从走路切换到跑步）
        if key.char == 'n':
            env.next_task()
            logger.info("Moved to the next task.")
            
        # 下面是 Force Control（外力扰动控制），用于测试机器人强化学习策略的鲁棒性（Robustness）
        # 在真机测试前，我们通常会在仿真里推拉机器人，看看它能不能保持平衡不摔倒。
        if hasattr(key, 'char'):
            if key.char == '1':
                # 给左手在 Z 轴（通常是垂直向上或向前）施加正向的力（+1.0 N）
                env.apply_force_tensor[:, env.left_hand_link_index, 2] += 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '2':
                # 给左手施加反向的力（-1.0 N）
                env.apply_force_tensor[:, env.left_hand_link_index, 2] -= 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '3':
                # 给右手施加正向的力
                env.apply_force_tensor[:, env.right_hand_link_index, 2] += 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
            elif key.char == '4':
                # 给右手施加反向的力
                env.apply_force_tensor[:, env.right_hand_link_index, 2] -= 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
    except AttributeError:
        # 捕获特殊按键（如 Shift, Ctrl 等没有 char 属性的键）带来的异常，忽略它们
        pass

"""
【函数详细解析：on_press】
- 作用：定义了当键盘按下时，系统应该做什么反应。它是人机交互的回调函数。
- RL背景：在评估 (Evaluation) 阶段，我们不仅要看机器人按预定路线走，还要测试它的“抗扰动能力”（Perturbation Recovery）。通过给特定的连杆（Link，这里是左右手）注入力（Force Tensor），模拟真机环境中被人推搡或撞到障碍物的情况。如果策略（Policy）训练得好，机器人会调整步伐恢复平衡；如果不好就会摔倒（Done）。
- 被谁调用：被下面的 `listen_for_keypress` 函数中的 `keyboard.Listener` 捕获到按键动作时自动调用。
"""

# ============================================================================
# 函数 2: listen_for_keypress
# ============================================================================
def listen_for_keypress(env):
    try:
        from pynput import keyboard
    except Exception as e:
        logger.warning(f"Keyboard listener disabled (pynput unavailable): {e}")
        return

    # 使用 with 语句启动键盘监听器，传入上面定义的 on_press 规则，使用 lambda 把 env 传进去
    with keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
        listener.join() # 阻塞当前线程，持续监听，直到程序结束

"""
【函数详细解析：listen_for_keypress】
- 作用：启动键盘监听循环。
- RL背景：因为仿真器的 `env.step()` 是一个死循环，如果在主线程里监听键盘，画面就会卡住。所以这个函数会被放到一个独立的子线程（Daemon Thread）中运行，实现一边渲染机器人动作，一边实时读取你的按键。
- 被谁调用：在 `main` 函数的中后部，作为 `threading.Thread(target=listen_for_keypress)` 的目标函数被调用。
"""

# ----------------- 之前的注释代码 -----------------
# from humanoidverse.envs.base_task.base_task import BaseTask
# from humanoidverse.envs.base_task.omnih2o_cfg import OmniH2OCfg

# ============================================================================
# 函数 3: main
# ============================================================================
# @hydra.main 装饰器：指定默认配置文件夹为 "config"，默认配置文件为 "base_eval.yaml"
@hydra.main(config_path="config", config_name="base_eval", version_base="1.2")
def main(override_config: OmegaConf):
    
    # set_seed(42)
    
    # ---------------- 1. 日志系统初始化 ----------------
    # logging to hydra log file: 将日志写到 hydra 自动生成的运行目录下的 eval.log 中
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove() # 移除默认的 loguru 输出端
    logger.add(hydra_log_path, level="DEBUG") # 记录详细的 DEBUG 级别日志到文件

    # 从环境变量获取控制台日志输出级别（默认 INFO），并设置输出到终端屏幕 (sys.stdout)，开启颜色
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    # 配置标准的 python logging 库，并用 HydraLoggerBridge 桥接到我们统一的日志管理系统中
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    # os.chdir(hydra.utils.get_original_cwd()) # 切换到原始工作目录（注释掉了）

    # ---------------- 2. 配置项解析与合并 ----------------
    # 评估机器人时，必须使用和训练时一模一样的物理环境配置，否则神经网络的输入输出维度/含义会对不上！
    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint) # 提取模型权重的路径
        config_path = checkpoint.parent / "config.yaml" # 尝试在 checkpoint 同级目录寻找训练时的配置文件
        
        # 如果同级目录没有，尝试去上一级目录找（不同的框架保存 ckpt 的目录层级习惯不同）
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        # 如果找到了训练时的配置
        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file) # 加载训练配置

            # 将评估时的覆盖参数（如：设置 num_envs=1 只看一台机器人，headless=False 开启可视化画面）融合进去
            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            # 把命令行里传进来的其他覆盖参数也合并进来
            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        # 如果没有给定 checkpoint，处理单纯的参数覆盖逻辑
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config
            
    # ---------------- 3. 仿真器选择与初始化 ----------------
    # 获取配置中使用的仿真器类型（目前主流的是老的 IsaacGym 和新的基于 Omniverse 的 IsaacSim/Lab）
    simulator_type = config.simulator['_target_'].split('.')[-1]
    
    if simulator_type == 'IsaacSim':
        # 如果是 IsaacSim，启动 Omniverse 的 AppLauncher 流程
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args # 修正 sys.argv 供内部调用
        # 强行用配置中的参数覆盖命令行参数
        args_cli.num_envs = config.num_envs # 环境数量（评估通常只开极少量的环境用于渲染）
        args_cli.seed = config.seed # 随机种子
        args_cli.env_spacing = config.env.config.env_spacing # 多个机器人之间的站位间距
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless # 是否无头模式（关闭图形渲染界面以节省算力，评估通常设为 False 看动画）

        app_launcher = AppLauncher(args_cli) # 启动 Isaac Sim
        simulation_app = app_launcher.app
        
    if simulator_type == 'IsaacGym':
        import isaacgym # 导入老的 IsaacGym 库，必须在 torch 之前导入以防底层 C++ 内存冲突
        
    # ---------------- 4. 算法与环境准备 ----------------
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config # 预处理配置（如计算观测维度）
    import torch
    # 导入导出工具：将 PyTorch 模型导出为真机 C++ 可读的 JIT 或 ONNX 格式
    from humanoidverse.utils.inference_helpers import export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx

    pre_process_config(config) # 计算 actor 和 critic 网络的维度大小

    # 指定运算设备：如果有配置就用配置的，否则有显卡用显卡 CUDA，没有用 CPU
    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 创建并保存本次评估的配置备份日志，方便后续查阅
    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving eval logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    # 提取当前权重的 step 步数/编号，用于给渲染出的视频文件命名
    ckpt_num = config.checkpoint.split('/')[-1].split('_')[-1].split('.')[0]
    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
    config.env.config.ckpt_dir = str(checkpoint.parent) # 记录 ckpt 目录
    
    # 核心：根据配置文件实例化强化学习环境（比如包含了地形、机器人模型、传感器等的仿真世界）
    env = instantiate(config.env, device=device)

    # # 构造一个全0的动作输入（因为刚重置，还没有网络输出动作）
    # actions = torch.zeros(env.num_envs, env.dim_actions, device=env.device, requires_grad=False)
    # actor_state = {}
    # actor_state["actions"] = actions
    
    # # 驱动仿真器走一步，从而获得刚重置后的第一个环境观测值（Observation）
    # # 返回的是初始状态的观测，提供给策略网络进行第一次推理
    # for i in range(1000):
    #     obs_dict, _, _, _ = env.step(actor_state)

    # ---------------- 5. 启动键盘监听线程 ----------------
    # 启动上文定义的键盘监听守护线程。daemon=True 表示主程序结束时这个线程会跟着被强制销毁
    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()

    # ---------------- 6. 实例化算法并加载权重 ----------------
    # 实例化算法类（通常是 PPO 评估器）
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup() # 初始化网络参数
    algo.load(config.checkpoint) # 核心：将保存在硬盘上的 .pt 模型权重加载进 Actor 神经网络

    # ---------------- 7. 模型导出部署阶段 (Sim2Real 必备) ----------------
    EXPORT_POLICY = False # 是否导出为 TorchScript JIT 格式
    EXPORT_ONNX = True # 是否导出为 ONNX 格式（主流的跨平台、跨语言推理框架）

    # 当启用 save_motion 采集时，导出 ONNX 会触发一次额外 reset_all（get_example_obs），
    # 可能污染“首生命”采样统计，因此默认跳过导出。
    if bool(getattr(config.env.config, "save_motion", False)):
        EXPORT_ONNX = False
        logger.info("save_motion=True: disable ONNX export in eval to avoid extra warmup reset.")

    checkpoint_path = str(checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # 定位根目录，并创建导出的目标文件夹 exported/
    HV_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exported_policy_path = os.path.join(HV_ROOT_DIR, checkpoint_dir, 'exported')
    os.makedirs(exported_policy_path, exist_ok=True)
    
    # 构建导出的文件名
    exported_policy_name = checkpoint_path.split('/')[-1]
    exported_onnx_name = exported_policy_name.replace('.pt', '.onnx')

    # 将只包含推理逻辑的 Actor 网络（脱离训练引擎）打包导出
    if EXPORT_POLICY:
        # export_policy_as_jit(algo.alg.actor_critic, exported_policy_path, exported_policy_name)
        logger.info('Exported policy as jit script to: ', os.path.join(exported_policy_path, exported_policy_name))
        
    if EXPORT_ONNX:
        # 导出 ONNX 通常需要一个 Dummy Input（假输入）来描绘网络的流图结构
        example_obs_dict = algo.get_example_obs() # 拿一组当前环境的观测样本作为输入形状参考
        export_policy_as_onnx(algo.inference_model, exported_policy_path, exported_onnx_name, example_obs_dict)
        
        # 开发者留下的注释：当前暂时注释掉了状态估计器（Estimator）的联合导出，因为存在冲突。
        # (注：真机中经常无法直接获取准确底盘速度，需要用 Estimator 网络预测速度，再喂给 Policy 网络)
        # Jiawei: don't have a better way for conflicts now; need to comment out the following line for force control
        # export_policy_and_estimator_as_onnx(algo.inference_model, exported_policy_path, exported_onnx_name, example_obs_dict)
        logger.info(f'Exported policy as onnx to: {os.path.join(exported_policy_path, exported_onnx_name)}')

    # ---------------- 8. 启动死循环，开始评估 ----------------
    # 进入仿真的主循环：不断收集环境观测值 -> 神经网络输出关节命令 -> 环境执行命令步进仿真
    
    algo.evaluate_policy()
    #algo.collect_delta_a_sim_data()
    # algo.evaluate_policy_for_mpjpe(
    #     pkl_path=config.get("mpjpe_eval_pkl_path", None),
    #     motion_keys=config.get("mpjpe_eval_motion_keys", None),
    #     max_steps=config.get("mpjpe_eval_max_steps", None),
    #     fallback_env_id=config.get("mpjpe_eval_fallback_env_id", 0),
    #     csv_path=config.get("mpjpe_eval_csv_path", None),
    #     step_csv_path=config.get("mpjpe_eval_step_csv_path", None),
    #     use_deltaa_compensation=config.get("mpjpe_eval_use_deltaa_compensation", True),
    # )

# 标准 python 执行入口
if __name__ == "__main__":
    main()

"""
【函数详细解析：main】
- 作用：策略评估的总司令部。负责调度配置文件、环境、神经网络、以及导出模块。
- RL背景：当你训练了几天几夜，终于得到一个 `model_9999.pt` 文件后，你会运行这套脚本。它做两件核心的事：
    1. 还原现场：把当时的训练配置完美重现，让你通过可视化窗口（Rendering）看到机器人的表现。
    2. 提纯封包：训练时网络很庞大（有Actor、Critic、Optimizer），但真机只需要用来决策的 Actor。脚本会将 Actor 剥离出来，并转化成 ONNX 等可以在真机 C++ / ROS 系统下以微秒级极速运行的轻量级格式。
- 被谁调用：直接通过终端命令行作为主程序执行。例如 `python eval.py checkpoint=runs/Exp01/model_1000.pt headless=False`。
"""