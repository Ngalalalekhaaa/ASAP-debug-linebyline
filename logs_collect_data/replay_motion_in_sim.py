# pyright: reportMissingImports=false
import os
import re
import time
from pathlib import Path
from typing import Optional

import hydra
import joblib
import numpy as np
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# 这行会注册项目中的 Hydra resolver / 配置工具（按项目既有写法保留）
from humanoidverse.utils.config_utils import *  # noqa: F401,F403
from humanoidverse.utils.helpers import pre_process_config


def _to_torch(x, device, dtype):
    import torch

    return torch.as_tensor(x, device=device, dtype=dtype)


def _to_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(v)


def _infer_motion_env_id(motion_key: str) -> Optional[int]:
    m = re.fullmatch(r"motion(\d+)", str(motion_key))
    if m is None:
        return None
    return int(m.group(1))


def _grid_origin_from_env_id(env_id: int, env_spacing: float, source_num_envs: int):
    import math

    num_cols = int(math.floor(math.sqrt(source_num_envs)))
    num_cols = max(num_cols, 1)
    row = env_id // num_cols
    col = env_id % num_cols
    return np.array([env_spacing * row, env_spacing * col, 0.0], dtype=np.float32)


def _pick_motion_key(motion_data: dict, motion_key: Optional[str], motion_index: int):
    keys = list(motion_data.keys())
    if len(keys) == 0:
        raise ValueError("动作文件为空，没有可用动作。")

    if motion_key is not None:
        if motion_key not in motion_data:
            raise KeyError(f"指定的 motion_key 不存在: {motion_key}, 可选: {keys}")
        return motion_key

    motion_index = max(0, min(int(motion_index), len(keys) - 1))
    return keys[motion_index]


def _ensure_required_groups(cfg: DictConfig):
    """
    某些命令只给了 +simulator/+exp/+robot/+terrain/+obs，
    但 env/legged_base.yaml 还依赖 ${domain_rand} 和 ${rewards}。
    这里给出稳妥默认，避免 InterpolationKeyError。
    """
    project_root = Path(__file__).resolve().parents[1]
    config_root = project_root / "humanoidverse" / "config"

    if "domain_rand" not in cfg or cfg.domain_rand is None:
        dr_cfg = OmegaConf.load(str(config_root / "domain_rand" / "NO_domain_rand.yaml"))
        cfg.domain_rand = dr_cfg.domain_rand
        logger.warning("未提供 domain_rand，自动使用: domain_rand/NO_domain_rand")

    if "rewards" not in cfg or cfg.rewards is None:
        rw_cfg = OmegaConf.load(str(config_root / "rewards" / "motion_tracking" / "reward_motion_tracking_basic.yaml"))
        cfg.rewards = rw_cfg.rewards
        logger.warning("未提供 rewards，自动使用: rewards/motion_tracking/reward_motion_tracking_basic")


@hydra.main(version_base=None, config_path="../humanoidverse/config", config_name="base")
def main(cfg: DictConfig):
    # 让脚本默认可视化模式更友好
    cfg.headless = False
    cfg.num_envs = int(cfg.get("replay_num_envs", 1))
    _ensure_required_groups(cfg)

    # 读取用户传入参数（通过 Hydra 的 +key=value）
    motion_file = cfg.get("visualize_motion_file", None)
    motion_key = cfg.get("motion_key", None)
    motion_index = int(cfg.get("motion_index", 0))
    loop = _to_bool(cfg.get("loop", True), default=True)
    recenter_nonzero_to_origin = _to_bool(cfg.get("recenter_nonzero_to_origin", False), default=False)
    source_num_envs_for_origin = int(cfg.get("source_num_envs_for_origin", 4096))

    if motion_file is None:
        raise ValueError("请通过 +visualize_motion_file=... 指定 pkl 文件路径。")

    motion_file = str(Path(motion_file).expanduser().resolve())
    if not os.path.exists(motion_file):
        raise FileNotFoundError(f"动作文件不存在: {motion_file}")

    # 若用户使用 +exp=motion_tracking，会在环境初始化时先加载 robot.motion.motion_file。
    # 这里强制改为当前回放文件，避免默认动作库格式不匹配导致初始化报错。
    if "robot" in cfg and cfg.robot is not None:
        if "motion" not in cfg.robot or cfg.robot.motion is None:
            cfg.robot.motion = OmegaConf.create({})
        cfg.robot.motion.motion_file = motion_file

    simulator_type = cfg.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacGym":
        # 必须先于 torch 导入
        import isaacgym  # noqa: F401

    import torch

    pre_process_config(cfg)

    if cfg.get("device", None):
        device = cfg.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using simulator: {simulator_type}, device: {device}")
    logger.info(f"Loading motion file: {motion_file}")

    env = instantiate(cfg.env, device=device)
    env.reset_all()

    motion_data = joblib.load(motion_file)
    key = _pick_motion_key(motion_data, motion_key, motion_index)#key是motionx。也就是第x个动作的键。
    motion = motion_data[key]#第i个动作的数据，是一个字典，包含了根位置、根旋转、关节位置等信息。

    root_trans = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    dof = np.asarray(motion["dof"], dtype=np.float32)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float32)

    num_frames = int(root_trans.shape[0])
    if dof.shape[0] != num_frames or root_rot.shape[0] != num_frames:
        raise ValueError("root_trans_offset / root_rot / dof 帧数不一致，无法回放。")

    root_lin_vel = np.asarray(motion.get("root_lin_vel", np.zeros((num_frames, 3), dtype=np.float32)), dtype=np.float32)
    root_ang_vel = np.asarray(motion.get("root_ang_vel", np.zeros((num_frames, 3), dtype=np.float32)), dtype=np.float32)
    dof_vel = np.asarray(motion.get("dof_vel", np.zeros_like(dof, dtype=np.float32)), dtype=np.float32)

    fps = float(motion.get("fps", 50.0))
    frame_dt = 1.0 / max(fps, 1e-6)

    logger.info(f"Replay motion: {key}")
    logger.info(f"Frames: {num_frames}, fps: {fps}")
    logger.info(f"Replay num_envs: {cfg.num_envs}")
    logger.info(f"recenter_nonzero_to_origin: {recenter_nonzero_to_origin}")
    if recenter_nonzero_to_origin:
        logger.info(f"source_num_envs_for_origin: {source_num_envs_for_origin}")
    logger.info("按 Ctrl+C 可退出。")

    sim = env.simulator
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # 预分配张量，避免循环里频繁 new tensor
    root_state = torch.zeros((env.num_envs, 13), device=env.device, dtype=torch.float32)
    dof_state = torch.zeros((env.num_envs, sim.num_dof, 2), device=env.device, dtype=torch.float32)
    zero_torque = torch.zeros((env.num_envs, sim.num_dof), device=env.device, dtype=torch.float32)

    # 用于“把非0号环境平移到原点附近显示”的偏移量。
    # 规则：env0 不动；env1..N 的根位置额外加上 -env_origins[env_id]。
    env_root_recenter_offsets = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    if recenter_nonzero_to_origin and env.num_envs > 1:
        if hasattr(env, "env_origins"):
            env_origins = env.env_origins.to(device=env.device, dtype=torch.float32)
            env_root_recenter_offsets[1:] = -env_origins[1:]
        else:
            logger.warning("当前环境没有 env_origins，无法执行 recenter_nonzero_to_origin。")

    # 单环境回放时，优先使用“数据本身”做平移：
    # 把当前 motion 的首帧根位置 XY 平移到原点（Z 不变），
    # 这样不依赖采集时 num_envs / spacing，任何 motionK 都能稳定显示在原点附近。
    if recenter_nonzero_to_origin and env.num_envs == 1:
        first_root = _to_torch(root_trans[0], env.device, torch.float32)
        env_root_recenter_offsets[0, 0:2] = -first_root[0:2]
        logger.info(
            f"single-env recenter by first-frame XY, shift={env_root_recenter_offsets[0].detach().cpu().tolist()}"
        )

    while True:
        for i in range(num_frames):
            # 1) 写入目标根状态
            root_state[:, 0:3] = _to_torch(root_trans[i], env.device, torch.float32) + env_root_recenter_offsets
            root_state[:, 3:7] = _to_torch(root_rot[i], env.device, torch.float32)
            root_state[:, 7:10] = _to_torch(root_lin_vel[i], env.device, torch.float32)
            root_state[:, 10:13] = _to_torch(root_ang_vel[i], env.device, torch.float32)

            sim.robot_root_states[env_ids] = root_state
            if hasattr(sim, "all_root_states") and sim.all_root_states.shape[0] == env.num_envs:
                sim.all_root_states[env_ids] = root_state

            # 2) 写入目标关节状态
            dof_state[:, :, 0] = _to_torch(dof[i], env.device, torch.float32)
            dof_state[:, :, 1] = _to_torch(dof_vel[i], env.device, torch.float32)

            # 3) 强制覆盖仿真状态
            sim.set_actor_root_state_tensor(env_ids, sim.all_root_states if hasattr(sim, "all_root_states") else root_state)
            sim.set_dof_state_tensor(env_ids, dof_state.view(-1, 2))

            # 4) 小步推进并渲染（用于刷新 viewer）
            sim.apply_torques_at_dof(zero_torque)
            sim.simulate_at_each_physics_step()
            env.render(sync_frame_time=(simulator_type == "IsaacGym"))

            if simulator_type != "IsaacGym":
                time.sleep(frame_dt)

        if not loop:
            break


if __name__ == "__main__":
    main()
