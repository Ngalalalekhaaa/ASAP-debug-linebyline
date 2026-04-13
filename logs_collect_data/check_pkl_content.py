import os
import joblib
import numpy as np
import torch

def view_pkl_contents(file_path, num_rows=5):
    """
    查看 HumanoidVerse 动作库 .pkl 文件的详细结构和数据。
    
    参数:
        file_path (str): .pkl 文件的路径。
        num_rows (int): 预览时打印的数据行数（时间步数）。
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 '{file_path}'")
        return

    try:
        print(f"🔄 正在加载文件: {file_path} ...")
        # ASAP 框架底层使用的是 joblib 来保存和读取 pkl
        data = joblib.load(file_path)
        
        print("✅ 加载成功！")
        print("=" * 80)
        print(f"📦 顶层数据类型: {type(data)}")
        
        if not isinstance(data, dict):
            print("⚠️ 警告: 该 pkl 文件顶层不是一个字典，无法按常规框架解析。")
            return
            
        top_keys = list(data.keys())
        print(f"🔑 顶层动作键名 (包含 {len(top_keys)} 个动作): {top_keys}")
        print("=" * 80)

        # 设置 PyTorch 的打印选项，防止被省略号截断
        torch.set_printoptions(profile="full", linewidth=150, sci_mode=False)

        # 遍历顶层的每一个动作（通常只有一个，例如 'motion0' 或特定的视频名）
        for motion_key, motion_data in data.items():
            print(f"\n🎬 【动作片段】: '{motion_key}'")
            
            if not isinstance(motion_data, dict):
                print(f"  - 格式异常: 期待字典，实际为 {type(motion_data)}")
                continue
                
            print(f"  - 包含 {len(motion_data.keys())} 个属性特征。")
            print("-" * 80)

            # 遍历该动作下的所有属性 (如 root_trans_offset, pose_aa, fps 等)
            for attr_key, attr_val in motion_data.items():
                print(f"🔹 属性名 (Key): '{attr_key}'")
                
                # 如果是 Numpy 数组
                if isinstance(attr_val, np.ndarray):
                    print(f"   - 格式 : Numpy Array")
                    print(f"   - 形状 : {attr_val.shape}")
                    print(f"   - 类型 : {attr_val.dtype}")
                    
                    if attr_val.ndim > 0:
                        preview = attr_val[:num_rows]
                    else:
                        preview = attr_val
                        
                    # 临时设置 numpy 打印选项
                    with np.printoptions(threshold=np.inf, linewidth=150, suppress=True):
                        print(f"   - 数据预览 (前 {num_rows} 行):\n{preview}")

                # 如果是 PyTorch 张量
                elif torch.is_tensor(attr_val):
                    print(f"   - 格式 : PyTorch Tensor")
                    print(f"   - 形状 : {list(attr_val.shape)}")
                    print(f"   - 类型 : {attr_val.dtype}")
                    
                    if attr_val.ndim > 0:
                        preview = attr_val[:num_rows]
                    else:
                        preview = attr_val
                    print(f"   - 数据预览 (前 {num_rows} 行):\n{preview}")

                # 如果是列表
                elif isinstance(attr_val, list):
                    print(f"   - 格式 : List")
                    print(f"   - 长度 : {len(attr_val)}")
                    print(f"   - 数据预览 (前 {num_rows} 项): {attr_val[:num_rows]}")

                # 如果是标量 (如 int, float, bool)
                else:
                    print(f"   - 格式 : {type(attr_val).__name__}")
                    print(f"   - 数据值 : {attr_val}")
                    
                print("-" * 60)

    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")

# ==========================================
# 使用示例与参数调整
# ==========================================
if __name__ == "__main__":
    # 1. 指定你的 .pkl 文件路径
    pkl_file_path = "/home/luckysir/ASAP/logs/MotionTracking/20260208_021503-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/timeoutonly_mtnum8_20260413155020.pkl"
    
    # 2. 调整你想查看的行数 (时间步数)
    # 把这里的 5 改成你想看的数字，比如 20, 50, 100
    display_rows = 3 
    
    view_pkl_contents(pkl_file_path, num_rows=display_rows)