# from time import time
# from warnings import WarningMessage
# import numpy as np
# import os

# from humanoidverse.utils.torch_utils import *
# # from isaacgym import gymtorch, gymapi, gymutil

# import torch
# from torch import Tensor
# from typing import Tuple, Dict
# from rich.progress import Progress

# from humanoidverse.envs.env_utils.general import class_to_dict
# from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
# from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
# # from humanoidverse.envs.env_utils.command_generator import CommandGenerator
# from scipy.stats import vonmises


# class LeggedRobotLocomotion(LeggedRobotBase):
#     def __init__(self, config, device):
#         self.init_done = False
#         super().__init__(config, device)
#         self._init_gait_params()
#         self.upper_left_arm_dof_names = self.config.robot.upper_left_arm_dof_names
#         self.upper_right_arm_dof_names = self.config.robot.upper_right_arm_dof_names
#         self.upper_left_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_left_arm_dof_names]
#         self.upper_right_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_right_arm_dof_names]
#         self.hips_dof_id = [self.simulator._body_list.index(link) - 1 for link in self.config.robot.motion.hips_link] # Yuanhang: -1 for the base link (pelvis)
#         self.init_done = True
    
#     def _init_buffers(self):
#         super()._init_buffers()
#         self.commands = torch.zeros(
#             (self.num_envs, 4), dtype=torch.float32, device=self.device
#         )
#         self.command_ranges = self.config.locomotion_command_ranges

#     def _init_gait_params(self):
#         # Initialize the normalized period of the swing phase
#         self.a_swing = 0.0 # start of the swing phase
#         self.b_swing = 0.5 # end of the swing phase
#         self.a_stance = 0.5 # start of the stance phase
#         self.b_stance = 1.0 # end of the stance phase
#         self.kappa = 4.0 # shared variance in Von Mises 
#         self.left_offset = 0.0 # left foot offset
#         self.right_offset = 0.5 # right foot offset
        
#         self.left_feet_height = torch.zeros(self.num_envs, device=self.device) # left feet height
#         self.right_feet_height = torch.zeros(self.num_envs, device=self.device) # right feet height
        
#         self.phase_time = torch.zeros(self.num_envs, dtype=torch.float32, requires_grad=False, device=self.device)
#         self.phase_time_np = np.zeros(self.num_envs, dtype=np.float32)
#         self.phase_left = (self.phase_time + self.left_offset) % 1
#         self.phase_right = (self.phase_time + self.right_offset) % 1
#         self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

#         # Initialize the gait period
#         if hasattr(self.config.rewards, "gait_period"):
#             if not self.config.rewards.gait_period:
#                 self.T = self.config.rewards.gait_period # gait period in seconds # gait period in seconds
#             else:
#                 self.T = 1. # gait period in seconds
#         else:
#             self.T = 1.
        
#         if hasattr(self.config.rewards, "gait_period"):
#             # Randomize the gait phase time
#             if self.config.obs.use_phase:
#                 self.phi_offset = np.random.rand(self.num_envs)*self.T
#             else: 
#                 self.phi_offset = np.zeros(self.num_envs)
#         else:
#             self.phi_offset = np.zeros(self.num_envs)
#         # Initialize the target arm joint positions
#         self.swing_arm_joint_pos = torch.tensor([-1.04, 0.0, 0.0, 1.57,
#                                                 0.0, 0.0, 0.0], device=self.device, dtype=torch.float, requires_grad=False)
#         self.stance_arm_joint_pos = torch.tensor([0.757, 0.0, 0.0, 1.57,
#                                                 0.0, 0.0, 0.0], device=self.device, dtype=torch.float, requires_grad=False)
#         print("phi_offset: ", self.phi_offset)
    

#     def _setup_simulator_control(self):
#         self.simulator.commands = self.commands

#     def _update_tasks_callback(self):
#         """ Callback called before computing terminations, rewards, and observations
#             Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
#         """
#         # 
#         super()._update_tasks_callback()

#         # commands
#         if not self.is_evaluating:
#             env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
#             self._resample_commands(env_ids)
#         forward = quat_apply(self.base_quat, self.forward_vec)
#         heading = torch.atan2(forward[:, 1], forward[:, 0])
#         self.commands[:, 2] = torch.clip(
#             0.5 * wrap_to_pi(self.commands[:, 3] - heading), 
#             self.command_ranges["ang_vel_yaw"][0], 
#             self.command_ranges["ang_vel_yaw"][1]
#         )
    
#     def _post_physics_step(self):
#         super()._post_physics_step()
#         self.update_phase_time()

#     def update_phase_time(self):
#         # Update the phase time
#         self.phase_time_np = self._calc_phase_time()
#         self.phase_time = torch.tensor(self.phase_time_np, device=self.device, dtype=torch.float, requires_grad=False)
#         self.phase_left = (self.phase_time + self.left_offset) % 1
#         self.phase_right = (self.phase_time + self.right_offset) % 1
#         self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
    
#     def _resample_commands(self, env_ids):
#         self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
#         self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
#         self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)

#         # set small commands to zero
#         self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


#     def _reset_tasks_callback(self, env_ids):
#         super()._reset_tasks_callback(env_ids)
#         if not self.is_evaluating:
#             self._resample_commands(env_ids)

#     def set_is_evaluating(self, command=None):
#         super().set_is_evaluating()
#         self.commands = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
#         # TODO: haotian: adding command configuration
#         if command is not None:
#             self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands

#     ########################### TRACKING REWARDS ###########################

#     def _reward_tracking_lin_vel(self):
#         # Tracking of linear velocity commands (xy axes)
#         lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
#         return torch.exp(-lin_vel_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
#     def _reward_tracking_ang_vel(self):
#         # Tracking of angular velocity commands (yaw) 
#         ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
#         return torch.exp(-ang_vel_error/self.config.rewards.reward_tracking_sigma.ang_vel)

#     ########################### PENALTY REWARDS ###########################

#     def _reward_penalty_lin_vel_z(self):
#         # Penalize z axis base linear velocity
#         return torch.square(self.base_lin_vel[:, 2])
    
#     def _reward_penalty_ang_vel_xy(self):
#         # Penalize xy axes base angular velocity
#         return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
#     def _reward_penalty_ang_vel_xy_torso(self):
#         # Penalize xy axes base angular velocity

#         torso_ang_vel = quat_rotate_inverse(self.simulator._rigid_body_rot[:, self.torso_index], self.simulator._rigid_body_ang_vel[:, self.torso_index])
#         return torch.sum(torch.square(torso_ang_vel[:, :2]), dim=1)
    

#     def _reward_penalty_feet_contact_forces(self):
#         # penalize high contact forces
#         return torch.sum((torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) -  self.config.rewards.locomotion_max_contact_force).clip(min=0.), dim=1)

#     ########################### FEET REWARDS ###########################

#     def _reward_feet_air_time(self):
#         # Reward long steps
#         # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
#         contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
#         contact_filt = torch.logical_or(contact, self.last_contacts) 
#         self.last_contacts = contact
#         first_contact = (self.feet_air_time > 0.) * contact_filt
#         self.feet_air_time += self.dt
#         rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
#         rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
#         self.feet_air_time *= ~contact_filt
#         return rew_airTime
    
#     def _reward_penalty_in_the_air(self):
#         contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
#         contact_filt = torch.logical_or(contact, self.last_contacts) 
#         first_foot_contact = contact_filt[:,0]
#         second_foot_contact = contact_filt[:,1]
#         reward = ~(first_foot_contact | second_foot_contact)
#         return reward



#     def _reward_penalty_stumble(self):
#         # Penalize feet hitting vertical surfaces
#         return torch.any(torch.norm(self.simulator.contact_forces[:, self.feet_indices, :2], dim=2) >\
#              5 *torch.abs(self.simulator.contact_forces[:, self.feet_indices, 2]), dim=1)




#     def _reward_base_height(self):
#         # Penalize base height away from target

#         base_height = self.simulator.robot_root_states[:, 2]
#         return torch.square(base_height - self.config.rewards.desired_base_height)


    
#     def _reward_feet_ori(self):
#         left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
#         left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
#         right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
#         right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
#         return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

#     def _reward_penalty_feet_slippage(self):
#         # assert self.simulator._rigid_body_vel.shape[1] == 20
#         foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
#         return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    
    
#     def _reward_penalty_feet_height(self):
#         # Penalize base height away from target
#         feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2]
#         dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
#         dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
#         return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 
    
#     def _reward_penalty_feet_swing_height(self):
#         contact = torch.norm(self.simulator.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
#         height_error = torch.square(self.simulator._rigid_body_pos[:, self.feet_indices, 2] - \
#                                     self.config.rewards.feet_height_target) * ~contact
#         return torch.sum(height_error, dim=(1))
    
#     def _reward_penalty_close_feet_xy(self):
#         # returns 1 if two feet are too close
#         left_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[0], :2]
#         right_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[1], :2]
#         feet_distance_xy = torch.norm(left_foot_xy - right_foot_xy, dim=1)
#         return (feet_distance_xy < self.config.rewards.close_feet_threshold) * 1.0
    

#     def _reward_penalty_close_knees_xy(self):
#         # returns 1 if two knees are too close
#         left_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[0], :2]
#         right_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[1], :2]
#         self.knee_distance_xy = torch.norm(left_knee_xy - right_knee_xy, dim=1)
#         return (self.knee_distance_xy < self.config.rewards.close_knees_threshold)* 1.0
    

#     def _reward_upperbody_joint_angle_freeze(self):
#         # returns keep the upper body joint angles close to the default
#         assert self.config.robot.has_upper_body_dof
#         deviation = torch.abs(self.simulator.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:,self.upper_dof_indices])
#         return torch.sum(deviation, dim=1)
    
#     def _reward_penalty_hip_pos(self):
#         # Penalize the hip joints (only roll and yaw)
#         hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
#         hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
#         return torch.sum(torch.square(hip_pos), dim=1)
    
#     ########################### GAIT REWARDS ###########################
#     def _calc_phase_time(self):
#         # Calculate the phase time
#         episode_length_np = self.episode_length_buf.cpu().numpy()
#         phase_time = (episode_length_np * self.dt + self.phi_offset) % self.T / self.T
#         return phase_time

#     def _reward_contact(self):
#         res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
#         for i in range(2): # left and right feet
#             is_stance = self.leg_phase[:, i] < 0.55
#             contact = self.simulator.contact_forces[:, self.feet_indices[i], 2] > 1
#             res += ~(contact ^ is_stance)
#         return res
    
#     def calculate_phase_expectation(self, phi, offset=0, phase="swing"):
#         """
#         Calculate the expectation value of I_i(φ).

#         Parameters:
#         phi (float): The given phase time.
#         offset (float): The offset of the phase time.

#         Returns:
#         float: The expectation value of I_i(φ).
#         """
#         # print("phase_time: ", phi)
#         phi = (phi + offset) % 1
#         phi *= 2 * np.pi
#         # Create Von Mises distribution objects for A_i and B_i
#         if phase == "swing":
#             dist_A = vonmises(self.kappa, loc=2 * np.pi * self.a_swing)
#             dist_B = vonmises(self.kappa, loc=2 * np.pi * self.b_swing)
#         else:
#             dist_A = vonmises(self.kappa, loc=2 * np.pi * self.a_stance)
#             dist_B = vonmises(self.kappa, loc=2 * np.pi * self.b_stance)
#         # Calculate P(A_i < φ) and P(B_i < φ)
#         P_A_less_phi = dist_A.cdf(phi)
#         P_B_less_phi = dist_B.cdf(phi)
#         # Calculate P(A_i < φ < B_i)
#         P_A_phi_B = P_A_less_phi * (1 - P_B_less_phi)
#         # Calculate the expectation value of I_i
#         E_I_i = P_A_phi_B

#         return E_I_i
    
#     def _reward_gait_period(self):
#         """
#         Jonah Siekmann, et al. "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"
#         paper link: https://arxiv.org/abs/2011.01387
#         """
#         # Calculate the expectation value of I_i of left and right feet
#         E_I_l_swing = self.calculate_phase_expectation(self.phase_time_np, offset=self.left_offset, phase="swing")
#         E_I_l_stance = self.calculate_phase_expectation(self.phase_time_np, offset=self.left_offset, phase="stance")
#         E_I_r_swing = self.calculate_phase_expectation(self.phase_time_np, offset=self.right_offset, phase="swing")
#         E_I_r_stance = self.calculate_phase_expectation(self.phase_time_np, offset=self.right_offset, phase="stance")
#         # print("E_I_l_swing: ", E_I_l_swing, ", E_I_r_swing: ", E_I_r_swing)
#         # print("E_I_l_stance: ", E_I_l_stance, ", E_I_r_stance: ", E_I_r_stance)
#         ## Convert to tensor
#         E_I_l_swing = torch.tensor(E_I_l_swing, device=self.device, dtype=torch.float, requires_grad=False)
#         E_I_r_swing = torch.tensor(E_I_r_swing, device=self.device, dtype=torch.float, requires_grad=False)
#         E_I_l_stance = torch.tensor(E_I_l_stance, device=self.device, dtype=torch.float, requires_grad=False)
#         E_I_r_stance = torch.tensor(E_I_r_stance, device=self.device, dtype=torch.float, requires_grad=False)
#         # Get the contact forces and velocities of the feet, and the velocities of the arm ee
#         Ff_left = torch.norm(self.simulator.contact_forces[:, self.feet_indices[0], :], dim=-1) # left foot contact force
#         Ff_right = torch.norm(self.simulator.contact_forces[:, self.feet_indices[1], :], dim=-1) # right foot contact force
#         vf_left = torch.norm(self.simulator._rigid_body_vel[:, self.feet_indices[0], :], dim=-1) # left foot velocity
#         vf_right = torch.norm(self.simulator._rigid_body_vel[:, self.feet_indices[1], :], dim=-1) # right foot velocity
#         # print("Ff_left: ", Ff_left, ", Ff_right: ", Ff_right)
#         # print("vf_left: ", vf_left, ", vf_right: ", vf_right)
#         reward_gait = E_I_l_swing * torch.exp(-Ff_left**2) + E_I_r_swing * torch.exp(-Ff_right**2) + \
#                       E_I_l_stance * torch.exp(-200*vf_left**2) + E_I_r_stance * torch.exp(-200*vf_right**2)
#         # Sum up the gait reward
#         return reward_gait

#     ######################### Observations #########################
#     def _get_obs_command_lin_vel(self):
#         return self.commands[:, :2]
    
#     def _get_obs_command_ang_vel(self):
#         return self.commands[:, 2:3]
    
#     def _get_obs_phase_time(self):
#         return self.phase_time.unsqueeze(1)
    
#     def _get_obs_sin_phase(self):
#         return torch.sin(2 * np.pi * self.phase_time).unsqueeze(1)
    
#     def _get_obs_cos_phase(self):
#         return torch.cos(2 * np.pi * self.phase_time).unsqueeze(1)




from time import time
from warnings import WarningMessage
import numpy as np
import os

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from humanoidverse.envs.env_utils.general import class_to_dict
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
# from humanoidverse.envs.env_utils.command_generator import CommandGenerator
from scipy.stats import vonmises

from pathlib import Path # 路径处理
from termcolor import colored # 终端彩色打印
from loguru import logger # 高级日志打印库

from scipy.spatial.transform import Rotation as sRot # 用于四元数和旋转向量/欧拉角之间的转换
import joblib # 用于读取或保存 .pkl 数据集


# [RL注释] LeggedRobotLocomotion类继承自LeggedRobotBase，代表这是一个针对双足/多足机器人移动（Locomotion）任务的强化学习环境。
# [RL注释] 它定义了智能体（Agent）所处环境的具体规则，包括如何生成指令、计算奖励以及获取观测。
class LeggedRobotLocomotion(LeggedRobotBase):
    
    # [RL注释] 函数作用：初始化环境实例，加载机器人配置，获取关键关节的索引。
    # [RL注释] 被谁调用：由强化学习框架（如rsl_rl）或训练脚本在创建环境实例（env = LeggedRobotLocomotion(...)）时调用。
    def __init__(self, config, device):
        # [RL注释] 标记初始化未完成
        self.init_done = False
        # [RL注释] 调用父类初始化，完成物理引擎、地形、基础张量的设置
        super().__init__(config, device)
        # [RL注释] 初始化步态（Gait）参数，用于生成周期性的运动参考信号
        self._init_gait_params()
        # [RL注释] 从配置中获取左臂和右臂自由度（DOF）的名称
        self.upper_left_arm_dof_names = self.config.robot.upper_left_arm_dof_names
        self.upper_right_arm_dof_names = self.config.robot.upper_right_arm_dof_names
        # [RL注释] 根据名称查找到对应的左臂和右臂DOF在全局驱动器数组中的索引
        self.upper_left_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_left_arm_dof_names]
        self.upper_right_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_right_arm_dof_names]
        # [RL注释] 获取髋关节（Hips）在物理仿真刚体列表中的索引。减1是因为第0个通常是环境/世界基座
        self.hips_dof_id = [self.simulator._body_list.index(link) - 1 for link in self.config.robot.motion.hips_link] # Yuanhang: -1 for the base link (pelvis)
        # [RL注释] 标记初始化完成
        self.init_done = True
        
        self._init_save_motion() # 如果配置了要保存动作，初始化相关的显存缓冲区

        
    
    
    
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

    
    # [RL注释] 函数作用：初始化强化学习过程中的各种张量缓冲区（Buffer），用于在GPU上高效存储和更新状态。
    # [RL注释] 被谁调用：由基类 LeggedRobotBase 的 __init__ 函数内部调用。
    def _init_buffers(self):
        # [RL注释] 先初始化父类的缓冲区（如关节位置、速度、接触力等）
        super()._init_buffers()
        # [RL注释] 初始化速度指令缓冲区，维度为 [环境数量, 4]，分别代表 [线速度x, 线速度y, 角速度yaw, 目标朝向heading]
        self.commands = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        # [RL注释] 从配置中读取指令的采样范围（即期望机器人达到的最大/最小速度）
        self.command_ranges = self.config.locomotion_command_ranges

    # [RL注释] 函数作用：设置机器人行走的周期性步态参数（如支撑相、摆动相的起止时间及相位）。
    # [RL注释] 被谁调用：被本类的 __init__ 函数调用。
    def _init_gait_params(self):
        # Initialize the normalized period of the swing phase
        # [RL注释] 归一化步态周期 (0到1)。摆动相（脚在空中）从0开始，到0.5结束
        self.a_swing = 0.0 # start of the swing phase
        self.b_swing = 0.5 # end of the swing phase
        # [RL注释] 支撑相（脚落地受力）从0.5开始，到1.0结束
        self.a_stance = 0.5 # start of the stance phase
        self.b_stance = 1.0 # end of the stance phase
        # [RL注释] Von Mises分布（圆周上的正态分布）的集中度参数kappa，用于平滑地计算步态奖励
        self.kappa = 4.0 # shared variance in Von Mises 
        # [RL注释] 左右脚相位相差0.5，形成交替迈步的经典双足步态
        self.left_offset = 0.0 # left foot offset
        self.right_offset = 0.5 # right foot offset
        
        # [RL注释] 初始化左右脚在当前环境下的高度记录张量
        self.left_feet_height = torch.zeros(self.num_envs, device=self.device) # left feet height
        self.right_feet_height = torch.zeros(self.num_envs, device=self.device) # right feet height
        
        # [RL注释] 初始化每个环境的全局相位时间张量（用于RL策略网络观测）
        self.phase_time = torch.zeros(self.num_envs, dtype=torch.float32, requires_grad=False, device=self.device)
        # [RL注释] 保存对应相位时间的NumPy版本数组（方便用scipy等CPU库计算）
        self.phase_time_np = np.zeros(self.num_envs, dtype=np.float32)
        # [RL注释] 分别计算左脚和右脚当前的局部相位（模1，确保在[0,1)区间）
        self.phase_left = (self.phase_time + self.left_offset) % 1
        self.phase_right = (self.phase_time + self.right_offset) % 1
        # [RL注释] 将左右脚相位拼接成特征输入
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        # Initialize the gait period
        # [RL注释] 设定步态周期时长(秒)。如果有配置则使用配置值，否则默认1.0秒
        if hasattr(self.config.rewards, "gait_period"):
            if not self.config.rewards.gait_period:
                self.T = self.config.rewards.gait_period # gait period in seconds # gait period in seconds
            else:
                self.T = 1. # gait period in seconds
        else:
            self.T = 1.
        
        # [RL注释] 为打破多个环境之间的同步性（Domain Randomization），给每个环境初始化一个随机相位偏移
        if hasattr(self.config.rewards, "gait_period"):
            # Randomize the gait phase time
            if self.config.obs.use_phase:
                self.phi_offset = np.random.rand(self.num_envs)*self.T
            else: 
                self.phi_offset = np.zeros(self.num_envs)
        else:
            self.phi_offset = np.zeros(self.num_envs)
        # Initialize the target arm joint positions
        # [RL注释] 摆动期手臂关节的参考目标位置（用于姿态正则化，防止手臂乱甩）
        self.swing_arm_joint_pos = torch.tensor([-1.04, 0.0, 0.0, 1.57,
                                                0.0, 0.0, 0.0], device=self.device, dtype=torch.float, requires_grad=False)
        # [RL注释] 支撑期手臂关节的参考目标位置
        self.stance_arm_joint_pos = torch.tensor([0.757, 0.0, 0.0, 1.57,
                                                0.0, 0.0, 0.0], device=self.device, dtype=torch.float, requires_grad=False)
        print("phi_offset: ", self.phi_offset)
    

    # [RL注释] 函数作用：将RL环境层的指令引用传递给底层的物理模拟器类。
    # [RL注释] 被谁调用：通常在物理引擎初始化及环境连结完成后调用。
    def _setup_simulator_control(self):
        self.simulator.commands = self.commands

    # [RL注释] 函数作用：在每次物理步进(physics step)之前调用的回调，用于更新高层任务状态，如重采样指令和计算朝向角。
    # [RL注释] 被谁调用：由基类的 step() 函数在物理引擎仿真 step_physics() 之前循环调用。
    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        # [RL注释] 执行父类的回调（可能包含更新地形、外部推力扰动等）
        super()._update_tasks_callback()

        # commands
        # [RL注释] 如果在训练模式下（非评估模式）
        if not self.is_evaluating:
            # [RL注释] 检查哪些环境到了需要重新采样指令的时间点（指令不能每帧变，通常几秒换一次）
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            # [RL注释] 对这些环境重采样速度指令
            self._resample_commands(env_ids)
        # [RL注释] 计算机器人当前在世界坐标系下的前向向量
        forward = quat_apply(self.base_quat, self.forward_vec)
        # [RL注释] 基于前向向量计算当前的航向角 (Yaw)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        # [RL注释] 将航向角误差（期望航向 - 当前航向）转换为期望的角速度指令，并做截断限制
        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading), 
            self.command_ranges["ang_vel_yaw"][0], 
            self.command_ranges["ang_vel_yaw"][1]
        )
    
    # # [RL注释] 函数作用：在物理引擎完成一步模拟之后调用的回调，用于更新后处理数据（比如时间流逝造成的相位更新）。
    # # [RL注释] 被谁调用：由基类的 step() 函数在 step_physics() 完成之后调用。
    # def _post_physics_step(self):
    #     super()._post_physics_step()
    #     self.update_phase_time()


    # # def _post_physics_step(self):
    #     """
    #     【函数作用】：只记录每个环境“第一次生命”（首次 reset 前）的连续数据，避免重置后轨迹突变。
    #     【被谁调用】：每次 env.step 结尾被调用。
    #     """
    #     # 先执行父类后处理：刷新状态、计算奖励、终止判断、重置等基础流程都在父类里完成。
    #     super()._post_physics_step()
    #     self.update_phase_time()

    #     # ========================= 评估期奖励追踪（新增） =========================
    #     # 目标：在 eval 时记录“每一步奖励”与“整回合 return（含终止步奖励）”，并直接打印到终端用于比较模型。
    #     # 说明：
    #     # 1) self.rew_buf 是本步总奖励（已包含各项奖励与 termination 项）；
    #     # 2) 在父类 super()._post_physics_step() 内，reset 已经执行完，但 reset_buf/time_out_buf 仍可用于识别本步结束的 episode；
    #     # 3) 因此在这里统计不会漏掉“最后一步奖励”。
    #     if self.is_evaluating:
    #         # 懒初始化：仅在首次进入评估步时创建缓存。
    #         if not hasattr(self, "_eval_return_tracker_ready"):
    #             # 当前 episode 的累计回报（按 env 独立累计）。
    #             self._eval_episode_returns = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    #             # 当前 episode 的逐步奖励序列（按 env 独立 list）。
    #             self._eval_episode_step_rewards = [[] for _ in range(self.num_envs)]
    #             # 每个 env 已完成了多少个 episode（用于编号）。
    #             self._eval_episode_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    #             # 仅保留轻量统计用于终端显示（不落盘）。
    #             self._eval_completed_returns = []
    #             self._eval_completed_lengths = []

    #             self._eval_return_tracker_ready = True

    #         # 逐步累积：将本步奖励计入每个 env 的当前 episode。
    #         step_rewards_cpu = self.rew_buf.detach().cpu()
    #         for env_idx in range(self.num_envs):
    #             rew_i = float(step_rewards_cpu[env_idx].item())
    #             self._eval_episode_returns[env_idx] += rew_i
    #             self._eval_episode_step_rewards[env_idx].append(rew_i)

    #         # 检测本步结束的 episode（timeout 或其它终止）。
    #         done_ids = torch.nonzero(self.reset_buf.bool(), as_tuple=False).flatten().tolist()
    #         for env_idx in done_ids:
    #             # 整回合 return（已包含最后一步奖励）。
    #             ep_return = float(self._eval_episode_returns[env_idx].item())
    #             # 回合长度：使用 last_episode_length_buf（避免 reset 后被清零）。
    #             ep_length = int(self.last_episode_length_buf[env_idx].item()) if hasattr(self, "last_episode_length_buf") else len(self._eval_episode_step_rewards[env_idx])
    #             # 回合结束类型：优先标注 timeout，其它归为 terminated。
    #             done_type = "timeout" if bool(self.time_out_buf[env_idx].item()) else "terminated"
    #             # episode 编号。
    #             ep_idx = int(self._eval_episode_index[env_idx].item())

    #             # 汇总到全局已完成回合统计（用于实时均值显示）。
    #             self._eval_completed_returns.append(ep_return)
    #             self._eval_completed_lengths.append(ep_length)
    #             recent_n = 20#最近20回合的均值
    #             recent_returns = self._eval_completed_returns[-recent_n:]
    #             recent_lengths = self._eval_completed_lengths[-recent_n:]
    #             recent_return_mean = float(np.mean(recent_returns)) if len(recent_returns) > 0 else 0.0
    #             recent_length_mean = float(np.mean(recent_lengths)) if len(recent_lengths) > 0 else 0.0
    #             total_finished = len(self._eval_completed_returns)

                


    #             # 控制台打印：每个结束回合输出一行摘要 + 最近20回合均值。
    #             print(colored(
    #                 f"[eval-return] env={env_idx} ep={ep_idx} done={done_type} len={ep_length} return={ep_return:.6f} | "
    #                 f"recent{min(recent_n, total_finished)}_mean_return={recent_return_mean:.6f} "
    #                 f"recent{min(recent_n, total_finished)}_mean_len={recent_length_mean:.2f} total_finished={total_finished}",
    #                 "cyan"
    #             ))

    #             # 清空该 env 当前 episode 缓存，开始下一回合统计。
    #             self._eval_episode_returns[env_idx] = 0.0
    #             self._eval_episode_step_rewards[env_idx] = []
    #             self._eval_episode_index[env_idx] += 1
    #     # ======================================================================

    #     # 若未开启保存动作，直接退出，不做任何额外记录。
    #     if not self.save_motion:
    #         # 提前返回，避免无意义的 CPU 拷贝与列表追加。
    #         return

    #     # 懒初始化：第一次进入时创建“首生命期”追踪器
    #     if not hasattr(self, "_first_life_active"):
    #         # 标记每个 env 是否仍处于“首生命”阶段（True 表示还没经历第一次 reset）。
    #         self._first_life_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    #         # 标记是否已经完成并保存过首生命数据（防止重复 dump）。
    #         self._first_life_saved = False
    #         # 记录首生命录制已经走了多少步（全局步计数，不是每个 env 单独计数）。
    #         self._first_life_step_counter = 0
    #         # 记录每个 env 首次 reset 发生在第几步，-1 表示尚未发生。
    #         self._first_life_reset_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
    #         # 记录每个 env 首次 reset 的原因字符串（诊断用）。
    #         self._first_life_reset_reason = ["" for _ in range(self.num_envs)]
    #         # 为每个 key 建立“按 env 分桶”的列表缓存：每个 env 各自有一段连续轨迹。
    #         self.motions_for_saving = {
    #             'root_trans_offset': [[] for _ in range(self.num_envs)],
    #             'pose_aa': [[] for _ in range(self.num_envs)],
    #             'dof': [[] for _ in range(self.num_envs)],
    #             'root_rot': [[] for _ in range(self.num_envs)],
    #             'actor_obs': [[] for _ in range(self.num_envs)],
    #             'action': [[] for _ in range(self.num_envs)],
    #             'terminate': [[] for _ in range(self.num_envs)],
    #             'root_lin_vel': [[] for _ in range(self.num_envs)],
    #             'root_ang_vel': [[] for _ in range(self.num_envs)],
    #             'dof_vel': [[] for _ in range(self.num_envs)],
    #         }
    #         # 单独缓存每个 env 的 motion time 序列，便于后续按 env 导出。
    #         self.motion_times_buf = [[] for _ in range(self.num_envs)]

    #     # 已经保存过首生命数据，不再重复记录
    #     if self._first_life_saved:
    #         # 已完成一次导出后，后续 step 全部直接返回。
    #         return

    #     # 计算当前参考动作时间：episode 已走步数 * dt + 该 env 的动作起始偏移。
    #     motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times

    #     # 当前时刻快照（CPU），供按 env 切分记录
    #     # 根平移位置（x,y,z）拷贝到 CPU，后续逐 env append。
    #     root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
    #     # print(f"写入.pkl的root_trans:{root_trans}")
    #     # 按 simulator 后端提取根四元数字段（不同后端四元数布局可能不同）。
    #     if self.config.simulator.config.name == "isaacgym":
    #         # IsaacGym 使用 [x,y,z,w] 存在 3:7。
    #         root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
    #     elif self.config.simulator.config.name == "isaacsim":
    #         # IsaacSim 这里按当前代码约定重排到 [x,y,z,w]。
    #         root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu()
    #     elif self.config.simulator.config.name == "genesis":
    #         # Genesis 与 IsaacGym 一致读取 3:7。
    #         root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
    #     else:
    #         # 未支持后端直接抛异常，避免静默写错数据。
    #         raise NotImplementedError

    #     # 把根四元数转成旋转向量（axis-angle 的 rotvec 形式），用于构造 pose_aa。
    #     root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float()
    #     # 当前关节位置拷贝到 CPU。
    #     dof = self.simulator.dof_pos.cpu()
    #     # 拼接 pose_aa：根旋转向量 + 每个 dof 的轴角 + augment 关节补零。
    #     pose_aa = torch.cat([
    #         root_rot_vec[:, None, :],
    #         self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None],
    #         torch.zeros((self.num_envs, self.num_augment_joint, 3)),
    #     ], axis=1)

    #     # 只记录“未发生首次 reset”的环境；本步 reset 的环境不写入（避免突变帧）
    #     # 先取当前“首生命活跃”掩码副本。
    #     active_mask = self._first_life_active.clone()#active_mask:当前这一帧需要记录的环境序号
    #     if hasattr(self, "reset_buf"):
    #         # 本步已 reset 的 env 置为 False，避免把突变帧写入轨迹。
    #         active_mask = active_mask & (~self.reset_buf.bool())

    #     # 把活跃掩码转为 Python 列表索引，便于 for 循环逐 env 追加。
    #     active_ids = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
    #     # 对每个仍在首生命且本步未 reset 的 env，逐 key 追加一帧数据。
    #     for env_idx in active_ids:
    #         # 根平移一帧。
    #         self.motions_for_saving['root_trans_offset'][env_idx].append(root_trans[env_idx].clone())
    #         # 根旋转四元数一帧。
    #         self.motions_for_saving['root_rot'][env_idx].append(root_rot[env_idx].clone())
    #         # 关节位置一帧。
    #         self.motions_for_saving['dof'][env_idx].append(dof[env_idx].clone())
    #         # 轴角姿态一帧。
    #         self.motions_for_saving['pose_aa'][env_idx].append(pose_aa[env_idx].clone())
    #         # actor 动作一帧（detach+cpu，避免图和显存绑定）。
    #         self.motions_for_saving['action'][env_idx].append(self.actions[env_idx].detach().cpu().clone())
    #         # actor 观测一帧。
    #         self.motions_for_saving['actor_obs'][env_idx].append(self.obs_buf_dict['actor_obs'][env_idx].detach().cpu().clone())
    #         # 终止标记一帧（当步通常为 False，因为 active_mask 已过滤 reset）。
    #         self.motions_for_saving['terminate'][env_idx].append(self.reset_buf[env_idx].detach().cpu().clone())
    #         # 关节速度一帧。
    #         self.motions_for_saving['dof_vel'][env_idx].append(self.simulator.dof_vel[env_idx].detach().cpu().clone())
    #         # 根线速度一帧。
    #         self.motions_for_saving['root_lin_vel'][env_idx].append(self.simulator.robot_root_states[env_idx, 7:10].detach().cpu().clone())
    #         # 根角速度一帧。
    #         self.motions_for_saving['root_ang_vel'][env_idx].append(self.simulator.robot_root_states[env_idx, 10:13].detach().cpu().clone())
    #         # 对应参考时间戳一帧。
    #         self.motion_times_buf[env_idx].append(motion_times[env_idx].detach().cpu().clone())

    #     # 诊断：记录每个 env 首次“掉出首生命”的步数和原因
    #     if hasattr(self, "reset_buf"):
    #         # 找到“这一步首次从活跃变为 reset”的 env（活跃且 reset_buf=True）。
    #         just_reset_mask = self._first_life_active & self.reset_buf.bool()#前面实际赋值这一帧是将self._first_life_active克隆到了active_mask，所以前者还没有改变
    #         # 转成 env id 列表。#这一帧just_reset_mask对应的环境在上面没有被赋值
    #         just_reset_ids = torch.nonzero(just_reset_mask, as_tuple=False).flatten().tolist()

    #         # 逐个刚 reset 的 env 记录首次 reset 信息。
    #         for env_idx in just_reset_ids:
    #             # 若此前已记录过首次 reset，直接跳过。
    #             if self._first_life_reset_step[env_idx].item() >= 0:
    #                 # continue 保证“首次”语义不被覆盖。
    #                 continue

    #             # 收集本 env 首次 reset 的命中原因（可多项并存）。
    #             reasons = []

    #             # timeout 原因
    #             if hasattr(self, "time_out_buf") and bool(self.time_out_buf[env_idx].item()):
    #                 # 命中超时（自然结束）。
    #                 reasons.append("timeout")

    #             # motion_far 原因（motion tracking 任务特有）
    #             if self.config.termination.terminate_when_motion_far:
    #                 # 重新按当前阈值计算该 env 是否 motion_far（用于诊断解释）。
    #                 motion_far_now = bool(torch.any(torch.norm(self.dif_global_body_pos[env_idx], dim=-1) > self.terminate_when_motion_far_threshold).item())
    #                 if motion_far_now:
    #                     # 命中 motion_far。
    #                     reasons.append("motion_far")

    #             # base 通用终止条件
    #             if self.config.termination.terminate_by_contact:
    #                 # 非允许触碰体发生大接触力则记 contact。
    #                 by_contact = bool(torch.any(torch.norm(self.simulator.contact_forces[env_idx, self.termination_contact_indices, :], dim=-1) > 1.).item())
    #                 if by_contact:
    #                     # 命中 contact。
    #                     reasons.append("contact")

    #             if self.config.termination.terminate_by_gravity:
    #                 # 判断重力投影 x 分量是否越界。
    #                 by_gravity_x = bool(torch.any(torch.abs(self.projected_gravity[env_idx, 0:1]) > self.config.termination_scales.termination_gravity_x).item())
    #                 # 判断重力投影 y 分量是否越界。
    #                 by_gravity_y = bool(torch.any(torch.abs(self.projected_gravity[env_idx, 1:2]) > self.config.termination_scales.termination_gravity_y).item())
    #                 if by_gravity_x or by_gravity_y:
    #                     # 命中姿态越界。
    #                     reasons.append("gravity")

    #             if self.config.termination.terminate_by_low_height:
    #                 # 判断基座高度是否低于最小高度阈值。
    #                 by_low_height = bool(torch.any(self.simulator.robot_root_states[env_idx, 2:3] < self.config.termination_scales.termination_min_base_height).item())
    #                 if by_low_height:
    #                     # 命中低高度终止。
    #                     reasons.append("low_height")

    #             if len(reasons) == 0:
    #                 # 若未匹配到显式规则，归类为未知或概率型终止。
    #                 reasons.append("unknown_or_probabilistic")

    #             # 记录首次 reset 发生的全局录制步。
    #             self._first_life_reset_step[env_idx] = self._first_life_step_counter
    #             # 记录首次 reset 原因（多原因用 + 连接）。
    #             self._first_life_reset_reason[env_idx] = "+".join(reasons)

    #             # 打印该 env 首次 reset 的诊断信息，便于追查为何丢轨迹。
    #             print(colored(
    #                 f"[first-life-diag] env={env_idx} first_reset_at_step={self._first_life_step_counter} "
    #                 f"ep_len={int(self.episode_length_buf[env_idx].item())} reason={self._first_life_reset_reason[env_idx]}",
    #                 "yellow"
    #             ))

    #     # 更新“首生命仍存活”掩码
    #     if hasattr(self, "reset_buf"):
    #         # 把本步 reset 的 env 从首生命活跃集合中永久移除。
    #         self._first_life_active = self._first_life_active & (~self.reset_buf.bool())
    #     # 全局录制步计数 +1。
    #     self._first_life_step_counter += 1

    #     # 若所有 env 首生命都结束，则可触发导出。
    #     all_first_life_finished = not bool(self._first_life_active.any().item())
    #     # 兜底上限：即使仍有 env 未 reset，超过 1000 步也强制导出。
    #     reached_max_steps = self._first_life_step_counter >= 1000

    #     # 满足任一导出条件时，组装并保存数据。
    #     if all_first_life_finished or reached_max_steps:
    #         # dump_data 结构：motion0/motion1/... 每个 env 一条轨迹。
    #         dump_data = {}
    #         # 统计最终真正写入的 env 数。
    #         saved_envs = 0
    #         # 要保存的所有键。
    #         keys_to_save = list(self.motions_for_saving.keys())
    #         # 与旧逻辑对齐：丢弃前 3 帧热身。
    #         warmup_trim = 0

    #         # 诊断：导出前打印每个 env 的首生命采样长度与保留情况
    #         # 打印总览标题。
    #         print(colored("[first-life-diag] per-env summary before dump:", "cyan"))
    #         # 逐 env 打印 seq_len 与是否会被保留。
    #         for env_idx in range(self.num_envs):
    #             # 该 env 首生命已缓存帧数。
    #             seq_len_i = len(self.motion_times_buf[env_idx])
    #             # 是否通过 warmup_trim 过滤。
    #             keep_i = seq_len_i > warmup_trim
    #             # 读取首次 reset 步（若存在诊断字段）。
    #             reset_step_i = int(self._first_life_reset_step[env_idx].item()) if hasattr(self, "_first_life_reset_step") else -1
    #             # 读取首次 reset 原因（若存在诊断字段）。
    #             reset_reason_i = self._first_life_reset_reason[env_idx] if hasattr(self, "_first_life_reset_reason") else ""
    #             # 输出该 env 的汇总诊断。
    #             print(colored(
    #                 f"  env={env_idx} seq_len={seq_len_i} keep={keep_i} warmup_trim={warmup_trim} "
    #                 f"first_reset_step={reset_step_i} reason={reset_reason_i if reset_reason_i else 'not_reset_yet'}",
    #                 "cyan"
    #             ))

    #         # 正式按 env 组装 dump_data。
    #         for env_idx in range(self.num_envs):
    #             # 当前 env 的轨迹长度。
    #             seq_len = len(self.motion_times_buf[env_idx])
    #             # 与旧逻辑保持一致：跳过最前面的 3 帧
    #             if seq_len <= warmup_trim:
    #                 # 太短则不写入（避免空数组或无意义片段）。
    #                 continue

    #             # motion{env_idx} 作为该 env 的导出键。
    #             motion_key = f"motion{env_idx}"
    #             # 对每个字段做 stack 后转 numpy，时间维在前。
    #             dump_data[motion_key] = {
    #                 key: torch.stack(self.motions_for_saving[key][env_idx][warmup_trim:], dim=0).numpy()
    #                 for key in keys_to_save
    #             }
    #             # 保存对齐后的 motion_times。
    #             dump_data[motion_key]['motion_times'] = torch.stack(self.motion_times_buf[env_idx][warmup_trim:], dim=0).numpy()
    #             # 记录帧率（1/dt）。
    #             dump_data[motion_key]['fps'] = 1 / self.dt
    #             # 成功保存 env 计数 +1。
    #             saved_envs += 1

    #         if saved_envs > 0:
    #             # 至少一个 env 有效，写入 pkl。
    #             joblib.dump(dump_data, f'{self.save_motion_dir}.pkl')
    #             # 打印成功信息与有效 env 数。
    #             print(colored(f"Saved first-life motion data to {self.save_motion_dir}.pkl (envs={saved_envs})", 'green'))
    #         else:
    #             # 没有任何 env 满足最小长度，提示未保存。
    #             print(colored("No valid first-life data collected; nothing saved.", 'yellow'))

    #         # 标记已保存，防止重复进入导出逻辑。
    #         self._first_life_saved = True
    #         # 按当前脚本设计，保存后立即退出进程。
    #         import sys
    #         # 结束运行，避免继续评估循环。
    #         sys.exit()
 
    def _post_physics_step(self):
        """
        【函数作用】：只记录每个环境“第一次生命”（首次 reset 前）的连续数据，避免重置后轨迹突变。
        【被谁调用】：每次 env.step 结尾被调用。
        """
        # 先执行父类后处理：刷新状态、计算奖励、终止判断、重置等基础流程都在父类里完成。
        super()._post_physics_step()
        self.update_phase_time()

        # ========================= 评估期奖励追踪 =========================
        if self.is_evaluating:
            if not hasattr(self, "_eval_return_tracker_ready"):
                self._eval_episode_returns = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                self._eval_episode_step_rewards = [[] for _ in range(self.num_envs)]
                self._eval_episode_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                self._eval_completed_returns = []
                self._eval_completed_lengths = []
                self._eval_return_tracker_ready = True

            step_rewards_cpu = self.rew_buf.detach().cpu()
            for env_idx in range(self.num_envs):
                rew_i = float(step_rewards_cpu[env_idx].item())
                self._eval_episode_returns[env_idx] += rew_i
                self._eval_episode_step_rewards[env_idx].append(rew_i)

            done_ids = torch.nonzero(self.reset_buf.bool(), as_tuple=False).flatten().tolist()
            for env_idx in done_ids:
                ep_return = float(self._eval_episode_returns[env_idx].item())
                ep_length = int(self.last_episode_length_buf[env_idx].item()) if hasattr(self, "last_episode_length_buf") else len(self._eval_episode_step_rewards[env_idx])
                done_type = "timeout" if bool(self.time_out_buf[env_idx].item()) else "terminated"
                ep_idx = int(self._eval_episode_index[env_idx].item())

                self._eval_completed_returns.append(ep_return)
                self._eval_completed_lengths.append(ep_length)
                recent_n = 20
                recent_returns = self._eval_completed_returns[-recent_n:]
                recent_lengths = self._eval_completed_lengths[-recent_n:]
                recent_return_mean = float(np.mean(recent_returns)) if len(recent_returns) > 0 else 0.0
                recent_length_mean = float(np.mean(recent_lengths)) if len(recent_lengths) > 0 else 0.0
                total_finished = len(self._eval_completed_returns)

                print(colored(
                    f"[eval-return] env={env_idx} ep={ep_idx} done={done_type} len={ep_length} return={ep_return:.6f} | "
                    f"recent{min(recent_n, total_finished)}_mean_return={recent_return_mean:.6f} "
                    f"recent{min(recent_n, total_finished)}_mean_len={recent_length_mean:.2f} total_finished={total_finished}",
                    "cyan"
                ))

                self._eval_episode_returns[env_idx] = 0.0
                self._eval_episode_step_rewards[env_idx] = []
                self._eval_episode_index[env_idx] += 1
        # ======================================================================

        if not self.save_motion:
            return

        if not hasattr(self, "_first_life_active"):
            self._first_life_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._first_life_saved = False
            self._first_life_step_counter = 0
            self._first_life_reset_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            self._first_life_reset_reason = ["" for _ in range(self.num_envs)]
            
            self.motions_for_saving = {
                'root_trans': [[] for _ in range(self.num_envs)], # [Locomotion修改] 去掉 offset 语义，改为纯粹的 root_trans
                # 'pose_aa': [[] for _ in range(self.num_envs)],  # [IL注释] 依赖于 mesh_parsers 和参考动作，去掉
                'dof': [[] for _ in range(self.num_envs)],
                'root_rot': [[] for _ in range(self.num_envs)],
                'actor_obs': [[] for _ in range(self.num_envs)],
                'action': [[] for _ in range(self.num_envs)],
                'terminate': [[] for _ in range(self.num_envs)],
                'root_lin_vel': [[] for _ in range(self.num_envs)],
                'root_ang_vel': [[] for _ in range(self.num_envs)],
                'dof_vel': [[] for _ in range(self.num_envs)],
            }
            # [Locomotion修改] 变量名从 motion_times_buf 改为 sim_times_buf，脱离 motion 跟踪语义
            self.sim_times_buf = [[] for _ in range(self.num_envs)]

        if self._first_life_saved:
            return

        # [Locomotion修改] 纯环境仿真时间，不再叠加参考动作的 self.motion_start_times
        sim_times = (self.episode_length_buf) * self.dt 

        root_trans = self.simulator.robot_root_states[:, 0:3].cpu()

        if self.config.simulator.config.name == "isaacgym":
            root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
        elif self.config.simulator.config.name == "isaacsim":
            root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu()
        elif self.config.simulator.config.name == "genesis":
            root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
        else:
            raise NotImplementedError

        # [IL注释] 模仿学习用于将四元数转为轴角形式进行重定向对齐，Locomotion 不需要
        # root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float()
        
        dof = self.simulator.dof_pos.cpu()

        # [IL注释] 拼接 pose_aa 依赖于 self._motion_lib.mesh_parsers.dof_axis，Locomotion中不需要这个复杂的转换
        # pose_aa = torch.cat([
        #     root_rot_vec[:, None, :],
        #     self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None],
        #     torch.zeros((self.num_envs, self.num_augment_joint, 3)),
        # ], axis=1)

        active_mask = self._first_life_active.clone()
        if hasattr(self, "reset_buf"):
            active_mask = active_mask & (~self.reset_buf.bool())

        active_ids = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
        
        for env_idx in active_ids:
            self.motions_for_saving['root_trans'][env_idx].append(root_trans[env_idx].clone())
            self.motions_for_saving['root_rot'][env_idx].append(root_rot[env_idx].clone())
            self.motions_for_saving['dof'][env_idx].append(dof[env_idx].clone())
            # self.motions_for_saving['pose_aa'][env_idx].append(pose_aa[env_idx].clone()) # [IL注释]
            self.motions_for_saving['action'][env_idx].append(self.actions[env_idx].detach().cpu().clone())
            self.motions_for_saving['actor_obs'][env_idx].append(self.obs_buf_dict['actor_obs'][env_idx].detach().cpu().clone())
            self.motions_for_saving['terminate'][env_idx].append(self.reset_buf[env_idx].detach().cpu().clone())
            self.motions_for_saving['dof_vel'][env_idx].append(self.simulator.dof_vel[env_idx].detach().cpu().clone())
            self.motions_for_saving['root_lin_vel'][env_idx].append(self.simulator.robot_root_states[env_idx, 7:10].detach().cpu().clone())
            self.motions_for_saving['root_ang_vel'][env_idx].append(self.simulator.robot_root_states[env_idx, 10:13].detach().cpu().clone())
            self.sim_times_buf[env_idx].append(sim_times[env_idx].detach().cpu().clone())

        if hasattr(self, "reset_buf"):
            just_reset_mask = self._first_life_active & self.reset_buf.bool()
            just_reset_ids = torch.nonzero(just_reset_mask, as_tuple=False).flatten().tolist()

            for env_idx in just_reset_ids:
                if self._first_life_reset_step[env_idx].item() >= 0:
                    continue

                reasons = []

                if hasattr(self, "time_out_buf") and bool(self.time_out_buf[env_idx].item()):
                    reasons.append("timeout")

                # [IL注释] 基于与参考动作偏差过大的终止条件，Locomotion 没有此概念
                # if self.config.termination.terminate_when_motion_far:
                #     motion_far_now = bool(torch.any(torch.norm(self.dif_global_body_pos[env_idx], dim=-1) > self.terminate_when_motion_far_threshold).item())
                #     if motion_far_now:
                #         reasons.append("motion_far")

                if self.config.termination.terminate_by_contact:
                    by_contact = bool(torch.any(torch.norm(self.simulator.contact_forces[env_idx, self.termination_contact_indices, :], dim=-1) > 1.).item())
                    if by_contact:
                        reasons.append("contact")

                if self.config.termination.terminate_by_gravity:
                    by_gravity_x = bool(torch.any(torch.abs(self.projected_gravity[env_idx, 0:1]) > self.config.termination_scales.termination_gravity_x).item())
                    by_gravity_y = bool(torch.any(torch.abs(self.projected_gravity[env_idx, 1:2]) > self.config.termination_scales.termination_gravity_y).item())
                    if by_gravity_x or by_gravity_y:
                        reasons.append("gravity")

                if self.config.termination.terminate_by_low_height:
                    by_low_height = bool(torch.any(self.simulator.robot_root_states[env_idx, 2:3] < self.config.termination_scales.termination_min_base_height).item())
                    if by_low_height:
                        reasons.append("low_height")

                if len(reasons) == 0:
                    reasons.append("unknown_or_probabilistic")

                self._first_life_reset_step[env_idx] = self._first_life_step_counter
                self._first_life_reset_reason[env_idx] = "+".join(reasons)

                print(colored(
                    f"[first-life-diag] env={env_idx} first_reset_at_step={self._first_life_step_counter} "
                    f"ep_len={int(self.episode_length_buf[env_idx].item())} reason={self._first_life_reset_reason[env_idx]}",
                    "yellow"
                ))

        if hasattr(self, "reset_buf"):
            self._first_life_active = self._first_life_active & (~self.reset_buf.bool())
            
        self._first_life_step_counter += 1
        all_first_life_finished = not bool(self._first_life_active.any().item())
        reached_max_steps = self._first_life_step_counter >= 1000

        if all_first_life_finished or reached_max_steps:
            dump_data = {}
            saved_envs = 0
            keys_to_save = list(self.motions_for_saving.keys())
            warmup_trim = 0

            print(colored("[first-life-diag] per-env summary before dump:", "cyan"))
            for env_idx in range(self.num_envs):
                seq_len_i = len(self.sim_times_buf[env_idx])
                keep_i = seq_len_i > warmup_trim
                reset_step_i = int(self._first_life_reset_step[env_idx].item()) if hasattr(self, "_first_life_reset_step") else -1
                reset_reason_i = self._first_life_reset_reason[env_idx] if hasattr(self, "_first_life_reset_reason") else ""
                print(colored(
                    f"  env={env_idx} seq_len={seq_len_i} keep={keep_i} warmup_trim={warmup_trim} "
                    f"first_reset_step={reset_step_i} reason={reset_reason_i if reset_reason_i else 'not_reset_yet'}",
                    "cyan"
                ))

            for env_idx in range(self.num_envs):
                seq_len = len(self.sim_times_buf[env_idx])
                if seq_len <= warmup_trim:
                    continue

                motion_key = f"motion{env_idx}"
                dump_data[motion_key] = {
                    key: torch.stack(self.motions_for_saving[key][env_idx][warmup_trim:], dim=0).numpy()
                    for key in keys_to_save
                }
                dump_data[motion_key]['sim_times'] = torch.stack(self.sim_times_buf[env_idx][warmup_trim:], dim=0).numpy()
                dump_data[motion_key]['fps'] = 1 / self.dt
                saved_envs += 1

            if saved_envs > 0:
                joblib.dump(dump_data, f'{self.save_motion_dir}.pkl')
                print(colored(f"Saved first-life motion data to {self.save_motion_dir}.pkl (envs={saved_envs})", 'green'))
            else:
                print(colored("No valid first-life data collected; nothing saved.", 'yellow'))

            self._first_life_saved = True
            import sys
            sys.exit()

    # [RL注释] 函数作用：推进并更新强化学习控制论中的“步态相位时间”。
    # [RL注释] 被谁调用：被 _post_physics_step() 回调调用。
    def update_phase_time(self):
        # Update the phase time
        # [RL注释] 计算当前CPU数组形式的相位
        self.phase_time_np = self._calc_phase_time()
        # [RL注释] 将计算得到的相位转换为GPU张量供神经网络或奖励系统使用
        self.phase_time = torch.tensor(self.phase_time_np, device=self.device, dtype=torch.float, requires_grad=False)
        # [RL注释] 更新左右脚的当前相位（0~1之间）
        self.phase_left = (self.phase_time + self.left_offset) % 1
        self.phase_right = (self.phase_time + self.right_offset) % 1
        # [RL注释] 拼合后供观测器（Observation）读取
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
    
    # [RL注释] 函数作用：随机生成环境期望机器人的运动指令（线速度和偏航角），这是强化学习的训练目标随机化过程。
    # [RL注释] 被谁调用：在 _update_tasks_callback() 判定到达时间间隔时，或在 _reset_tasks_callback() 环境重置时调用。
    def _resample_commands(self, env_ids):
        # [RL注释] 在给定的范围内均匀分布随机生成X轴线速度（前进/后退）
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        # [RL注释] 在给定的范围内均匀分布随机生成Y轴线速度（侧移）
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        # [RL注释] 随机生成机器人的期望目标航向角（Heading）
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        # [RL注释] “死区”处理：如果生成的线速度指令过小（范数<0.2），直接将其置为0，教导机器人学会安静站立
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    # [RL注释] 函数作用：在某些环境结束（如机器人摔倒、达到最大步数）并重置时的回调。
    # [RL注释] 被谁调用：由基类中的 reset_idx() 函数调用。
    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        # [RL注释] 重置环境时，同时为该环境随机换一个新指令，防止过拟合到某一特定任务
        if not self.is_evaluating:
            self._resample_commands(env_ids)

    # [RL注释] 函数作用：切换到评估（Testing/Eval）模式，冻结随机指令探索，可以手动固定给定指令进行测试。
    # [RL注释] 被谁调用：由外部评估脚本（比如play.py或evaluate.py）手动调用。
    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        # [RL注释] 清零所有环境指令
        self.commands = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        
        # [RL注释] 如果没有传参数，去配置中寻找默认的评估指令
        if command is None and getattr(self.config, "eval_command", None) is not None:
            command = self.config.eval_command
            
        # TODO: haotian: adding command configuration
        # [RL注释] 将测试指令覆盖到所有环境中（只覆盖前三项：线速度x, 线速度y, 角速度yaw）
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands

    ########################### TRACKING REWARDS ###########################
    # [RL注释] 这部分的函数用于计算跟踪奖励（Reward）。RL的目标是最大化奖励。这里的奖励促使机器人速度紧贴指令速度。

    # [RL注释] 函数作用：计算XY平面的线速度跟踪奖励。
    # [RL注释] 被谁调用：被基类 Reward Manager 的 compute_reward() 自动遍历调用。
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # [RL注释] 计算实际XY速度和指令XY速度差异的平方和（误差）
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # [RL注释] 使用负指数函数将误差映射到 (0, 1] 之间。误差为0时奖励为1，误差越大奖励越接近0。
        return torch.exp(-lin_vel_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
    # [RL注释] 函数作用：计算偏航（Yaw）角速度跟踪奖励。
    # [RL注释] 被谁调用：同上，基类 Reward Manager调用。
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        # [RL注释] 计算实际角速度Z和指令角速度Z的误差平方
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # [RL注释] 负指数映射
        return torch.exp(-ang_vel_error/self.config.rewards.reward_tracking_sigma.ang_vel)

    ########################### PENALTY REWARDS ###########################
    # [RL注释] 这部分的函数用于计算惩罚（返回正值，在配置文件中乘上负权重变成惩罚）。用于约束不期望的动态行为，产生自然的步态。

    # [RL注释] 函数作用：惩罚基座在Z轴（垂直方向）的线速度。希望机器人身体平稳前行，不要上下大幅度颠簸。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    # [RL注释] 函数作用：惩罚基座在XY轴的角速度（Roll和Pitch）。希望机器人保持水平，不发生侧倾和前后俯仰。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    # [RL注释] 函数作用：特别惩罚躯干（Torso）部位在XY轴的角速度，让上半身更稳定。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_ang_vel_xy_torso(self):
        # Penalize xy axes base angular velocity

        # [RL注释] 利用逆旋转计算出躯干局部坐标系下的角速度
        torso_ang_vel = quat_rotate_inverse(self.simulator._rigid_body_rot[:, self.torso_index], self.simulator._rigid_body_ang_vel[:, self.torso_index])
        return torch.sum(torch.square(torso_ang_vel[:, :2]), dim=1)
    
    # [RL注释] 函数作用：惩罚过大的足端地面接触力。防止机器人的脚步“猛烈砸地”，保护硬件。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_feet_contact_forces(self):
        # penalize high contact forces
        # [RL注释] 计算当前受力超过设定阈值（max_contact_force）的部分进行惩罚
        return torch.sum((torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) -  self.config.rewards.locomotion_max_contact_force).clip(min=0.), dim=1)

    ########################### FEET REWARDS ###########################
    # [RL注释] 这部分函数专注于足端的奖励/惩罚，用于塑造特定的落脚、抬脚、滞空等行为特征。

    # [RL注释] 函数作用：奖励较长的脚步滞空时间。促使机器人迈大步，而不是快速小碎步蹭地。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # [RL注释] 获取当前脚的Z向接触力是否大于1N（判定为接触）
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        # [RL注释] 考虑到PhysX引擎在网格碰撞时接触力可能有噪声，通过和上一步状态求逻辑或来滤波
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        # [RL注释] 判定是否是空中飞跃后“第一次”落地
        first_contact = (self.feet_air_time > 0.) * contact_filt
        # [RL注释] 滞空时间累加仿真步长dt
        self.feet_air_time += self.dt
        # [RL注释] 仅在脚着地的一瞬间，根据其经历的滞空时间给予奖励（扣除0.5秒的基础阈值）
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        # [RL注释] 如果没有运动指令（站立不动），则取消该奖励
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        # [RL注释] 落地后，将滞空时间清零
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    # [RL注释] 函数作用：双脚腾空惩罚（避免机器人像兔子一样双脚同时跳跃）。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_in_the_air(self):
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_foot_contact = contact_filt[:,0]
        second_foot_contact = contact_filt[:,1]
        # [RL注释] 如果左脚和右脚都没有接触地面，则触发惩罚
        reward = ~(first_foot_contact | second_foot_contact)
        return reward


    # [RL注释] 函数作用：惩罚脚碰到垂直障碍物或发生横向碰撞（即绊倒）。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_stumble(self):
        # Penalize feet hitting vertical surfaces
        # [RL注释] 如果脚部受到的水平接触力(XY)大于垂直接触力(Z)的5倍，判定为被绊到了
        return torch.any(torch.norm(self.simulator.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.simulator.contact_forces[:, self.feet_indices, 2]), dim=1)



    # [RL注释] 函数作用：惩罚基座高度偏离设定的目标高度。使机器人保持半蹲或直立的特定姿态高度。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_base_height(self):
        # Penalize base height away from target

        # [RL注释] 获取基座Z轴绝对高度，并与期望值求差的平方
        base_height = self.simulator.robot_root_states[:, 2]
        return torch.square(base_height - self.config.rewards.desired_base_height)


    # [RL注释] 函数作用：惩罚脚底板的倾斜。希望机器人脚掌平整落地，而不是用脚尖或脚跟滑行。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_feet_ori(self):
        # [RL注释] 将重力向量转换到左脚和右脚的局部坐标系
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        # [RL注释] 取局部重力向量在XY平面的分量作为误差，越大代表脚掌越不水平
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    # [RL注释] 函数作用：惩罚脚在受力支撑时的滑动。希望脚踩实后保持绝对静止（滑动说明摩擦力不够或控制不好）。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_feet_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        # [RL注释] 提取脚部线速度
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        # [RL注释] 如果脚的接触力大于1N（说明踩在地上了），但此时脚还有速度，则对这个速度进行惩罚
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    
    # [RL注释] 函数作用：惩罚脚部最低高度偏离设定值。通常用于引导机器人在特定地形下保持足够净空。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_feet_height(self):
        # Penalize base height away from target
        feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2]
        # [RL注释] 计算脚高与期望值的差值
        dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        # [RL注释] 提供一个0.02米的容差（Deadzone），在这个范围内不惩罚
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 
    
    # [RL注释] 函数作用：惩罚摆动腿在空中的高度偏离目标值。用于规范机器人跨步时抬脚多高。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_feet_swing_height(self):
        # [RL注释] 检测是否有接触力
        contact = torch.norm(self.simulator.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        # [RL注释] 仅对没有接触地面（在摆动相）的腿，计算其高度偏离目标的平方误差
        height_error = torch.square(self.simulator._rigid_body_pos[:, self.feet_indices, 2] - \
                                    self.config.rewards.feet_height_target) * ~contact
        return torch.sum(height_error, dim=(1))
    
    # [RL注释] 函数作用：惩罚双脚在XY平面上距离过近。防止因为模型步态收敛导致自我干涉（俗称“绊脚”打架）。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_close_feet_xy(self):
        # returns 1 if two feet are too close
        left_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[0], :2]
        right_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[1], :2]
        # [RL注释] 计算左右脚间距，如果小于设定的安全阈值则输出1作为触发惩罚
        feet_distance_xy = torch.norm(left_foot_xy - right_foot_xy, dim=1)
        return (feet_distance_xy < self.config.rewards.close_feet_threshold) * 1.0
    
    # [RL注释] 函数作用：惩罚双膝靠得太近（内八字碰膝盖）。同样用于避免刚体自我碰撞。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_close_knees_xy(self):
        # returns 1 if two knees are too close
        left_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[0], :2]
        right_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[1], :2]
        self.knee_distance_xy = torch.norm(left_knee_xy - right_knee_xy, dim=1)
        return (self.knee_distance_xy < self.config.rewards.close_knees_threshold)* 1.0
    
    # [RL注释] 函数作用：惩罚上半身关节偏离默认姿态（如手臂乱挥）。使运动更像人，更美观。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_upperbody_joint_angle_freeze(self):
        # returns keep the upper body joint angles close to the default
        assert self.config.robot.has_upper_body_dof
        # [RL注释] 计算上半身各DOF角度与默认角度之差的绝对值之和
        deviation = torch.abs(self.simulator.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:,self.upper_dof_indices])
        return torch.sum(deviation, dim=1)
    
    # [RL注释] 函数作用：惩罚髋关节产生不必要的Roll（横滚）和Yaw（偏航）角度，促使髋关节主要在Pitch（俯仰，前后迈步）上发力。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_penalty_hip_pos(self):
        # Penalize the hip joints (only roll and yaw)
        hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
        hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
        return torch.sum(torch.square(hip_pos), dim=1)
    
    ########################### GAIT REWARDS ###########################
    # [RL注释] 这一部分是核心的相位驱动步态生成逻辑（Gait Rewards），这使得机器人不用运动学反解也能走出类似人类的交替步伐。

    # [RL注释] 函数作用：利用经过的回合长度计算当前系统处于步态的哪个时间相位百分比（0.0 到 1.0）。
    # [RL注释] 被谁调用：内部函数 update_phase_time() 调用。
    def _calc_phase_time(self):
        # Calculate the phase time
        episode_length_np = self.episode_length_buf.cpu().numpy()
        phase_time = (episode_length_np * self.dt + self.phi_offset) % self.T / self.T
        return phase_time

    # [RL注释] 函数作用：强制物理环境中的着地情况匹配步态时钟相位的期望情况。例如，相位要求脚在地上，它就该有接触力。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(2): # left and right feet
            # [RL注释] 判断根据相位计算当前脚是否应该处于支撑相
            is_stance = self.leg_phase[:, i] < 0.55
            # [RL注释] 获取当前脚真实的接触状态
            contact = self.simulator.contact_forces[:, self.feet_indices[i], 2] > 1
            # [RL注释] 如果真实状态与期望状态不符（使用XOR异或，再取反代表匹配），则给予奖励
            res += ~(contact ^ is_stance)
        return res
    
    # [RL注释] 函数作用：计算当前相位时间在给定的Von Mises连续分布下的概率期望值。用于产生平滑且周期性的连续奖励权重。
    # [RL注释] 被谁调用：_reward_gait_period() 调用。
    def calculate_phase_expectation(self, phi, offset=0, phase="swing"):
        """
        Calculate the expectation value of I_i(φ).

        Parameters:
        phi (float): The given phase time.
        offset (float): The offset of the phase time.

        Returns:
        float: The expectation value of I_i(φ).
        """
        # print("phase_time: ", phi)
        # [RL注释] 计算带有左/右脚偏移的实际相位
        phi = (phi + offset) % 1
        phi *= 2 * np.pi
        # Create Von Mises distribution objects for A_i and B_i
        # [RL注释] 为摆动相(swing)或支撑相(stance)的起点A和终点B构建冯·米塞斯概率分布
        if phase == "swing":
            dist_A = vonmises(self.kappa, loc=2 * np.pi * self.a_swing)
            dist_B = vonmises(self.kappa, loc=2 * np.pi * self.b_swing)
        else:
            dist_A = vonmises(self.kappa, loc=2 * np.pi * self.a_stance)
            dist_B = vonmises(self.kappa, loc=2 * np.pi * self.b_stance)
        # Calculate P(A_i < φ) and P(B_i < φ)
        # [RL注释] 分布的累积分布函数(CDF)，用于平滑判定相位是否处于A和B之间
        P_A_less_phi = dist_A.cdf(phi)
        P_B_less_phi = dist_B.cdf(phi)
        # Calculate P(A_i < φ < B_i)
        P_A_phi_B = P_A_less_phi * (1 - P_B_less_phi)
        # Calculate the expectation value of I_i
        # [RL注释] 得到最终连续可微的指标函数期望值（0到1之间），用来当做奖励分配的权重
        E_I_i = P_A_phi_B

        return E_I_i
    
    # [RL注释] 函数作用：核心步态奖励，参照Jonah Siekmann等人的论文。它根据平滑的时钟期望将接触力和足底速度的惩罚/奖励在支撑相和摆动相动态分配。
    # [RL注释] 被谁调用：Reward Manager
    def _reward_gait_period(self):
        """
        Jonah Siekmann, et al. "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"
        paper link: https://arxiv.org/abs/2011.01387
        """
        # Calculate the expectation value of I_i of left and right feet
        # [RL注释] 获取左右脚分别在摆动相和支撑相情况下的权重期望期望（由时钟相位驱动）
        E_I_l_swing = self.calculate_phase_expectation(self.phase_time_np, offset=self.left_offset, phase="swing")
        E_I_l_stance = self.calculate_phase_expectation(self.phase_time_np, offset=self.left_offset, phase="stance")
        E_I_r_swing = self.calculate_phase_expectation(self.phase_time_np, offset=self.right_offset, phase="swing")
        E_I_r_stance = self.calculate_phase_expectation(self.phase_time_np, offset=self.right_offset, phase="stance")
        # print("E_I_l_swing: ", E_I_l_swing, ", E_I_r_swing: ", E_I_r_swing)
        # print("E_I_l_stance: ", E_I_l_stance, ", E_I_r_stance: ", E_I_r_stance)
        ## Convert to tensor
        # [RL注释] 将CPU numpy数组转为GPU Tensor
        E_I_l_swing = torch.tensor(E_I_l_swing, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_r_swing = torch.tensor(E_I_r_swing, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_l_stance = torch.tensor(E_I_l_stance, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_r_stance = torch.tensor(E_I_r_stance, device=self.device, dtype=torch.float, requires_grad=False)
        # Get the contact forces and velocities of the feet, and the velocities of the arm ee
        # [RL注释] 获取真实的物理反馈：足底压力和足底速度
        Ff_left = torch.norm(self.simulator.contact_forces[:, self.feet_indices[0], :], dim=-1) # left foot contact force
        Ff_right = torch.norm(self.simulator.contact_forces[:, self.feet_indices[1], :], dim=-1) # right foot contact force
        vf_left = torch.norm(self.simulator._rigid_body_vel[:, self.feet_indices[0], :], dim=-1) # left foot velocity
        vf_right = torch.norm(self.simulator._rigid_body_vel[:, self.feet_indices[1], :], dim=-1) # right foot velocity
        # print("Ff_left: ", Ff_left, ", Ff_right: ", Ff_right)
        # print("vf_left: ", vf_left, ", vf_right: ", vf_right)
        # [RL注释] 计算步态总奖励：
        # 摆动期（swing）：期望脚腾空，受力为0（Ff^2小则奖励大）；
        # 支撑期（stance）：期望脚踩实，相对速度为0（vf^2小则奖励大）。
        reward_gait = E_I_l_swing * torch.exp(-Ff_left**2) + E_I_r_swing * torch.exp(-Ff_right**2) + \
                      E_I_l_stance * torch.exp(-200*vf_left**2) + E_I_r_stance * torch.exp(-200*vf_right**2)
        # Sum up the gait reward
        return reward_gait

    ######################### Observations #########################
    # [RL注释] 这一部分用于构建智能体的“眼睛和神经”。把环境中的数据送入神经网络的前向传播中以输出关节力矩（Actions）。
    
    # [RL注释] 函数作用：将线速度指令作为观测（Observation）向量的一部分暴露给策略网络，使得网络知道“我要去哪”。
    # [RL注释] 被谁调用：被基类 Observation Manager 的 compute_observations() 循环调用。
    def _get_obs_command_lin_vel(self):
        return self.commands[:, :2]
    
    # [RL注释] 函数作用：将角速度指令作为观测暴露给策略网络。
    # [RL注释] 被谁调用：Observation Manager
    def _get_obs_command_ang_vel(self):
        return self.commands[:, 2:3]
    
    # [RL注释] 函数作用：输出当前归一化的相位时间（直接将时间本身作为特征输入网络有助策略理解周期）。
    # [RL注释] 被谁调用：Observation Manager
    def _get_obs_phase_time(self):
        return self.phase_time.unsqueeze(1)
    
    # [RL注释] 函数作用：提取相位时间的Sin值，用于将跳变的周期(1突变到0)变成平滑且连续的圆周坐标。
    # [RL注释] 被谁调用：Observation Manager
    def _get_obs_sin_phase(self):
        return torch.sin(2 * np.pi * self.phase_time).unsqueeze(1)
    
    # [RL注释] 函数作用：提取相位时间的Cos值，配合Sin值完整地向网络描述当前位于步态周期的哪一点。
    # [RL注释] 被谁调用：Observation Manager
    def _get_obs_cos_phase(self):
        return torch.cos(2 * np.pi * self.phase_time).unsqueeze(1)