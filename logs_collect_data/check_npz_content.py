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
    
    
# nohup python humanoidverse/train_agent.py \
#   +simulator=isaacgym \
#   +exp=train_delta_a_open_loop \
#   +domain_rand=NO_domain_rand \
#   +rewards=motion_tracking/delta_a/reward_delta_a_openloop \
#   +robot=g1/g1_29dof_anneal_23dof \
#   +terrain=terrain_locomotion_plane \
#   +obs=delta_a/open_loop \
#   num_envs=4096 \
#   project_name=DeltaA_Training \
#   experiment_name=openloopDeltaA_training \
#   robot.motion.motion_file="/home/liuhenghui/ASAP/logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/first_life_20260409220350.pkl" \
#   env.config.max_episode_length_s=1.0 \
#   rewards.reward_scales.penalty_minimal_action_norm=-0.1 \
#   +device=cuda:1 \
#   env.config.resample_motion_when_training=True \
#   env.config.resample_time_interval_s=10000 \
#   > openloopDeltaA_training_motionfile20260409220350_cuda1.log 2>&1 &