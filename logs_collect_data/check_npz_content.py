import numpy as np
import os

def view_npz_contents(file_path, print_preview=True, num_steps=50):
    """
    查看 .npz 文件的详细内容。
    
    参数:
        file_path (str): .npz 文件的路径。
        print_preview (bool): 是否打印数组的预览数据。
        num_steps (int): 预览时打印的时间步（第一维）数量。默认打印前 50 步。
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 '{file_path}'")
        return

    try:
        # 加载 .npz 文件。allow_pickle=True 在包含对象数组时是必需的
        data = np.load(file_path, allow_pickle=True)
        
        print(f"✅ 成功加载文件: {file_path}")
        print("=" * 80)
        
        # data.files 包含了所有的键名列表
        keys = data.files
        print(f"📦 文件中包含 {len(keys)} 个数组 (Keys):")
        print(keys)
        print("=" * 80)
        
        # 遍历打印每个数组的具体信息
        for key in keys:
            arr = data[key]
            print(f"🔹 键名 (Key): '{key}'")
            
            # 判断是否是 numpy 数组
            if isinstance(arr, np.ndarray):
                print(f"   - 形状 (Shape) : {arr.shape}")
                print(f"   - 数据类型 (Dtype): {arr.dtype}")
                
                # 打印数据预览
                if print_preview:
                    # 如果数组维度大于0，按第一维（通常是时间步）进行切片
                    if arr.ndim > 0:
                        preview_data = arr[:num_steps]
                    else:
                        preview_data = arr
                    
                    # 临时设置 numpy 的打印选项，保证 50 行不会被省略号 (...) 截断
                    # threshold=np.inf 取消省略号，linewidth=150 让每行能显示更多列，suppress=True 抑制科学计数法
                    with np.printoptions(threshold=np.inf, linewidth=150, suppress=True):
                        print(f"   - 数据预览 (前 {num_steps} 步):\n{preview_data}")
            else:
                # 处理存储了非 numpy 数组对象的情况
                print(f"   - 数据类型: {type(arr)}")
                if print_preview:
                    print(f"   - 数据预览:\n{arr}")
                    
            print("-" * 80)
            
    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
    finally:
        # 释放文件句柄
        if 'data' in locals() and hasattr(data, 'close'):
            data.close()

# ==========================================
# 使用示例：
# ==========================================
if __name__ == "__main__":
    # 请将下面的路径替换为你实际的 .npz 文件路径
    sample_file = "logs_collect_data/delta_a_sim2sim_data/motion_env0_len196.npz" 
    
    # 运行查看函数
    view_npz_contents(sample_file, print_preview=True, num_steps=1)
    
    
HYDRA_FULL_ERROR=1 \
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=train_delta_a_open_loop \
+domain_rand=NO_domain_rand \
+rewards=motion_tracking/delta_a/reward_delta_a_openloop \
rewards.reward_scales.penalty_minimal_action_norm=-0.1 \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=delta_a/open_loop \
num_envs=8192 \
project_name=DeltaA_Training \
robot.motion.motion_file="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/mtnum3001_addmotorandobsnoise_20260506174210.pkl" \
env.config.max_episode_length_s=1 \
+device=cuda:2 \
env.config.resample_motion_when_training=True \
env.config.resample_time_interval_s=10000 \
experiment_name=obsopenloop_mtfile_mtnum3001_addmotorandobsnoise \
log_task_name=maxepisodelengths1 \
+rewards.reward_scales.mpjpe=0 \
robot.control.action_clip_value=20.0 \
algo.config.clip_param=0.2 \
> 0506/obsopenloop_mtfile_mtnum3001_addmotorandobsnoise_maxepisodelengths1.log 2>&1 &


+checkpoint=logs/0505/20260505_022853-obsopenloop_mtfile_mtnum1_no_addobsnoise-train_without_obsnoise_and_actionclip20-g1_29dof_anneal_23dof/model_4600.pt
python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=train_delta_a_open_loop \
+domain_rand=NO_domain_rand \
+rewards=motion_tracking/delta_a/reward_delta_a_openloop \
rewards.reward_scales.normalized_penalty_minimal_action_norm=0.1 \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=delta_a/open_loop \
num_envs=1 \
project_name=DeltaA_Training \
robot.motion.motion_file="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/mtnum1_no_addobsnoise_20260505015851.pkl" \
env.config.max_episode_length_s=1 \
+device=cuda:0 \
env.config.resample_motion_when_training=True \
env.config.resample_time_interval_s=10000 \
experiment_name=test \
log_task_name=test \
+rewards.reward_scales.mpjpe=0 \
robot.control.action_clip_value=20.0 \
algo.config.clip_param=0.2 \
+checkpoint=logs/DeltaA_Training/20260505_204236-obsopenloop_mtfile_mtnum3000_addobsnoise-maxepisodelengths0p25-g1_29dof_anneal_23dof/model_1800.pt \
headless=False

python humanoidverse/eval_agent.py \
+checkpoint=logs/MotionTracking/xxxxxxxx_xxxxxxx-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_5800.pt

+checkpoint=logs/DeltaA_Training/20260505_203915-obsopenloop_mtfile_mtnum3000_addobsnoise-maxepisodelengths0p5-g1_29dof_anneal_23dof/model_14100.pt \
+checkpoint=logs/0505/20260505_022853-obsopenloop_mtfile_mtnum1_no_addobsnoise-train_without_obsnoise_and_actionclip20-g1_29dof_anneal_23dof/model_4600.pt \
+checkpoint=logs/DeltaA_Training/20260504_143613-obsopenloop-from_start-g1_29dof_anneal_23dof/model_24000.pt \


rewards.reward_scales.penalty_minimal_action_norm=-0.1 \
+anklePR=True \
+checkpoint=logs/DeltaA_Training/20260503_015440-trybesteffect_origin-from_start_obsall_actionclip5-g1_29dof_anneal_23dof/model_11000.pt \
teleop_body_position_extend
robot.motion.motion_file="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/mtnum3000_addobsnoise_20260426212346.pkl" \
robot.motion.motion_file="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/timeoutonly_mtnum128_20260413184401.pkl" \
+checkpoint=logs/DeltaA_Training/20260413_204909-openloopDeltaA_training-delta_a-g1_29dof_anneal_23dof/model_17300.pt \
+checkpoint="logs/DeltaA_Training/20260427_091950-mtnum3000_addobsnoise_20260426212346-actionorm_m0p1_mpjpe_0-g1_29dof_anneal_23dof/model_12700.pt" \
log_task_name=after_20260427_091950-mtnum3000_addobsnoise_20260426212346-actionorm_m0p1_12700pt_with_2th_defenses \
robot.control.action_clip_value=5.0 \
    
+rewards.reward_scales.teleop_body_position_extend=1.5 \
rewards.reward_scales.teleop_body_position_extend=5 \

rewards.reward_scales.normalized_penalty_minimal_action_norm=0.1 \
experiment_name=trybesteffect_change_teleopbodypositionextendto2_and_normalizedpenaltyminimalactionnorm0p1 \






# HYDRA_FULL_ERROR=1 \
# nohup python humanoidverse/train_agent.py \
# +simulator=isaacgym \
# +exp=train_delta_a_closed_loop \
# algo.config.policy_checkpoint="logs/DeltaA_Training/20260412_102648-openloopDeltaA_training-delta_a-g1_29dof_anneal_23dof/model_10000.pt" \
# +domain_rand=NO_domain_rand_finetune_with_deltaA \
# +rewards=motion_tracking/reward_motion_tracking_dm_simfinetuning \
# +robot=g1/g1_29dof_anneal_23dof \
# +terrain=terrain_locomotion_plane \
# +obs=delta_a/train_policy_with_delta_a \
# num_envs=4096 \
# project_name=DeltaA_Finetune \
# experiment_name=finetune_with_deltaA \
# robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
# env.config.add_extra_action=True \
# +checkpoint="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_12300.pt" \
# domain_rand.push_robots=False \
# env.config.noise_to_initial_level=1 \
# rewards.reward_penalty_curriculum=True \
# +device=cuda:7 \
# algo.config.save_interval=100 \
# algo.config.num_learning_iterations=20000 \
# > closeloopDeltaA_10000ptopen0412mtnum2_12300pt0208_cuda7.log 2>&1 &


__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py \
+checkpoint=logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_12300.pt \
+simulator=genesis \
+env.config.save_motion=True \
+env.config.save_note=mtnum3001_addmotorandobsnoise \
+env.config.eval_timestamp=$(date +%Y%m%d_%H%M%S) \
+num_envs=3001 \
+rewards.reward_scales.mpjpe=1 \
+headless=True \
+domain_rand=domain_rand_for_collect_data \
+device=cuda:1


python humanoidverse/eval_agent.py \
+checkpoint=logs/0506/DF/20260505_200911-tryifcanfinetune_with_openloopobs-deltaamodel_05041436_16900pt_closeloopaddobsnoise-g1_29dof_anneal_23dof/model_100.pt \
+simulator=genesis \
+num_envs=1 \
+rewards.reward_scales.mpjpe=1 \
+headless=False \
+device=cuda:0


+checkpoint=logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_12300.pt \

#+opt=eval_analysis_plot_motion_tracking_openloop




__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python logs_collect_data/replay_motion_in_sim.py \
+simulator=genesis \
+exp=motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=motion_tracking/delta_a/reward_delta_a_openloop \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
+visualize_motion_file="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/test_mtnum4_addmotorandobsnoise_20260506170604.pkl" \
+replay_num_envs=1 \
+recenter_nonzero_to_origin=True \
+motion_key=motion3 \
+rewards.reward_scales.mpjpe=1


+experiment_name=
+log_task_name=


# HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py \
# +checkpoint=logs/DeltaA_Training/20260410_105832-openloopDeltaA_training-delta_a-g1_29dof_anneal_23dof/model_5100.pt \
# +mpjpe_eval_pkl_path=logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/timeoutonly_mtnum2_20260413154610.pkl \
# +mpjpe_eval_csv_path=logs_eval/TEST/mpjpe_summary.csv \
# +mpjpe_eval_step_csv_path=logs_eval/TEST/mpjpe_steps.csv \
# +mpjpe_eval_use_deltaa_compensation=True \
# +device=cuda:0 \
# robot.motion.motion_file=mpjpe_eval_pkl_path



HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py \
+device=cuda:0 \
+checkpoint="logs/MotionTracking/20260505_021818-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_500.pt" \
+simulator=genesis \
+rewards.reward_scales.mpjpe=1


+robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
+rewards.reward_scales.mpjpe=1
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia 
+simulator=genesis \


+checkpoint="logs/DeltaA_Finetune/20260503_104035-finetune_withobsall_1mtfile-delta_a-g1_29dof_anneal_23dof/model_1485.pt" \
+robot.motion.motion_file="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/test_withoutresetallbeforeseteval_mtnum1_20260424105728.pkl" \
+robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
+robot.motion.motion_file="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/test3_mtnum1_add_postphysstepsforresetall_son_20260424175110.pkl" \

 
    
    
+checkpoint=logs/DeltaA_Training/20260413_204233-openloopDeltaA_training-delta_a-g1_29dof_anneal_23dof/model_50900.pt
+mpjpe_eval_pkl_path=logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/timeoutonly_mtnum1_20260413154405.pkl \
+mpjpe_eval_csv_path=logs_eval/TEST/mpjpe_summary.csv \
+mpjpe_eval_step_csv_path=logs_eval/TEST/mpjpe_steps.csv \
+mpjpe_eval_use_deltaa_compensation=False \
+device=cuda:0 \
+robot.motion.motion_file='${mpjpe_eval_pkl_path}'


python humanoidverse/eval_agent.py \
+simulator=isaacgym \
+exp=train_delta_a_closed_loop \
+domain_rand=NO_domain_rand_finetune_with_deltaA \
+rewards=motion_tracking/reward_motion_tracking_dm_simfinetuning \
algo.config.policy_checkpoint="logs/0505/20260505_022853-obsopenloop_mtfile_mtnum1_no_addobsnoise-train_without_obsnoise_and_actionclip20-g1_29dof_anneal_23dof/model_4600.pt" \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=delta_a/train_policy_with_delta_a \
+num_envs=1 \
+project_name=DeltaA_Finetune \
experiment_name=finetune_with_deltaA \
robot.motion.motion_file="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/mtnum1_no_addobsnoise_20260505015851.pkl" \
+checkpoint="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_12300.pt" \
domain_rand.push_robots=False \
env.config.noise_to_initial_level=0 \
rewards.reward_penalty_curriculum=True \
+device=cuda:0 \
env.config.termination.terminate_by_gravity=False \
+rewards.reward_scales.mpjpe=1 \
env.config.add_extra_action=True \
+headless=False
    
    
+anklePR=True


algo.config.policy_checkpoint="logs/DeltaA_Training/20260412_102648-openloopDeltaA_training-delta_a-g1_29dof_anneal_23dof/model_10000.pt" \
algo.config.policy_checkpoint="logs/DeltaA_Training/20260413_204233-openloopDeltaA_training-delta_a-g1_29dof_anneal_23dof/model_50900.pt" \
algo.config.policy_checkpoint="logs/DeltaA_Training/20260416_171312-after_20260413_204909_17300pt_penaltyMinimalAction0p5-check_stop_error_num3-g1_29dof_anneal_23dof/model_82900.pt" \
algo.config.policy_checkpoint="logs/DeltaA_Training/20260418_230309-after_20260413_204909_17300pt_penaltyMinimalAction0p5-actionnorm_m0p5-g1_29dof_anneal_23dof/model_10000.pt"
algo.config.policy_checkpoint="logs/DeltaA_Training/20260418_230419-after_20260413_204909_17300pt_penaltyMinimalAction0p5-actionnorm_m0p5-g1_29dof_anneal_23dof/model_10000.pt" \
algo.config.policy_checkpoint="logs/DeltaA_Training/20260427_091950-mtnum3000_addobsnoise_20260426212346-actionorm_m0p1_mpjpe_0-g1_29dof_anneal_23dof/model_3300.pt" \
algo.config.policy_checkpoint="logs/DeltaA_Training/20260428_122356-test_deltaaTraining-openloop_addobs_all-g1_29dof_anneal_23dof/model_4900.pt" \
algo.config.policy_checkpoint="logs/DeltaA_Training/20260430_173653-trybesteffect-from_start_obsall_actionclip5-g1_29dof_anneal_23dof/model_6600.pt" \
algo.config.policy_checkpoint="logs/0503/20260503_020504-trybesteffect_use_normalized_penalty_minimal_action_norm-from_start_obsall_actionclip5_normalpman0p1-g1_29dof_anneal_23dof/model_5500.pt" \

    
robot.motion.motion_file="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/test5_mtnum4_allsame_20260426141944.pkl" \
openloop_addobs_all


HYDRA_FULL_ERROR=1 \
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=train_delta_a_closed_loop \
algo.config.policy_checkpoint="logs/DeltaA_Training/20260504_143613-obsopenloop-from_start-g1_29dof_anneal_23dof/model_16900.pt" \
+domain_rand=NO_domain_rand_finetune_with_deltaA \
+rewards=motion_tracking/reward_motion_tracking_dm_simfinetuning \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=delta_a/train_policy_with_delta_a \
num_envs=8192 \
project_name=DeltaA_Finetune \
experiment_name=tryifcanfinetune_with_openloopobs \
log_task_name=deltaamodel_05041436_16900pt_closeloopaddobsnoise \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
env.config.add_extra_action=True \
+checkpoint="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_12300.pt" \
domain_rand.push_robots=False \
env.config.noise_to_initial_level=1 \
rewards.reward_penalty_curriculum=True \
+device=cuda:0 \
algo.config.save_interval=100 \
algo.config.num_learning_iterations=100000 \
> 0505/DeltaA_Finetune_tryifcanfinetune_with_openloopobs_deltaamodel_05041436_16900pt_closeloopaddobsnoise.log 2>&1 &

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=train_delta_a_closed_loop \
algo.config.policy_checkpoint="logs/DeltaA_Training/20260504_143613-obsopenloop-from_start-g1_29dof_anneal_23dof/model_3000.pt" \
+domain_rand=NO_domain_rand_finetune_with_deltaA \
+rewards=motion_tracking/reward_motion_tracking_dm_simfinetuning \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=delta_a/train_policy_with_delta_a \
num_envs=1 \
project_name=DeltaA_Finetune \
experiment_name=test \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
env.config.add_extra_action=True \
+checkpoint="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_12300.pt" \
domain_rand.push_robots=False \
env.config.noise_to_initial_level=1 \
rewards.reward_penalty_curriculum=True \
+device=cuda:0 \
algo.config.save_interval=100 \
algo.config.num_learning_iterations=3000 \
headless=False

algo.config.policy_checkpoint="logs/DeltaA_Training/20260504_143613-obsopenloop-from_start-g1_29dof_anneal_23dof/model_24000.pt" \
+checkpoint="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_12300.pt" \



robot.motion.motion_file="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/test_mtnum1_addobsnoise_20260426210721.pkl" \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \




HYDRA_FULL_ERROR=1 \
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=motion_tracking/reward_motion_tracking_dm_2real \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
num_envs=4096 \
project_name=MotionTracking \
experiment_name=MotionTracking_CR7 \
robot.motion.motion_file="logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/mtnum1_no_addobsnoise_20260505015851.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_penalty_degree=0.00001 \
env.config.resample_motion_when_training=False \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
robot.asset.self_collisions=0 \
+device=cuda:4 \
+simulator=genesis \
> 0505/MotionTracking_MotionTracking_CR7_mtnum1_no_addobsnoise_genesis.log 2>&1 &
