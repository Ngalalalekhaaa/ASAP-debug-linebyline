import os
import glob
import subprocess
import re
import pandas as pd

def run_auto_eval():
    # 1. 设定 pt 文件夹路径 (请修改为你真实的 pt 文件夹路径)
    ckpt_dir = "logs/DeltaA_Training/20260509_101041-obsopenloop_mtfile_mtnum3002_05071618domainrandinpaper9900pt_nodmwhencollect_20260509004754_pmanm0p1_actionclip20-from_start-g1_29dof_anneal_23dof/"
    
    # 2. 获取所有的 pt 文件并按 epoch/step 数字进行自然排序
    pt_files = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    def extract_step(filepath):
        match = re.search(r'model_(\d+)\.pt', filepath)
        return int(match.group(1)) if match else 0
    pt_files.sort(key=extract_step)

    print(f"Found {len(pt_files)} checkpoint files. Starting evaluation...")
    
    results_dict = {}

    # 3. 循环遍历执行评估
    for pt_file in pt_files:
        model_name = os.path.basename(pt_file)
        print(f"\n[{model_name}] Evaluating...")
        
        # 构建完整的命令列表
        cmd = [
            "python", "humanoidverse/eval_agent.py",
            "+simulator=isaacgym",
            "+exp=train_delta_a_open_loop",
            "+domain_rand=NO_domain_rand",
            "+rewards=motion_tracking/delta_a/reward_delta_a_openloop",
            "rewards.reward_scales.normalized_penalty_minimal_action_norm=0.1",
            "+robot=g1/g1_29dof_anneal_23dof",
            "+terrain=terrain_locomotion_plane",
            "+obs=delta_a/open_loop",
            "+num_envs=4",
            "+project_name=DeltaA_Training",
            'robot.motion.motion_file="logs/MotionTracking/20260507_161819-CR7amass_pretraindomainrandinpaper-motion_tracking-g1_29dof_anneal_23dof/motions/test_mtnum4_3002CR7_nodmwhencollect_20260513161808.pkl"',
            "env.config.max_episode_length_s=8",
            "+device=cuda:0",
            "env.config.resample_motion_when_training=True",
            "env.config.resample_time_interval_s=10000",
            "experiment_name=test",
            "log_task_name=test",
            "+rewards.reward_scales.mpjpe=1",
            "robot.control.action_clip_value=20.0",
            "algo.config.clip_param=0.2",
            f"+checkpoint={pt_file}",  # 动态替换当前的 pt 文件
            "+headless=True"           # 强烈推荐批量评估时开启无头模式加速
        ]

        # 执行命令并捕获输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        mpjpe_val = None
        # 实时读取终端输出，寻找目标标识符
        for line in process.stdout:
            # 如果你想在终端看到完整的评估日志，可以取消注释下一行
            # print(line, end="") 
            if "==== EVAL_RESULT_MPJPE:" in line:
                try:
                    # 解析后面的数值
                    val_str = line.split("==== EVAL_RESULT_MPJPE:")[1].replace("====", "").strip()
                    mpjpe_val = float(val_str)
                except ValueError:
                    print(f"[{model_name}] Error parsing mpjpe value from line: {line}")

        process.wait() # 等待子进程（Isaac Gym 环境）完全退出

        # 记录结果
        if mpjpe_val is not None:
            print(f"[{model_name}] Finish! MPJPE = {mpjpe_val:.5f}")
            results_dict[model_name] = mpjpe_val
        else:
            print(f"[{model_name}] Failed! Metric not found.")
            results_dict[model_name] = None

    # 4. 汇总并保存结果到 CSV
    print("\n================ FINAL RESULTS ================")
    for k, v in results_dict.items():
        val = f"{v:.5f}" if v is not None else "Failed"
        print(f"{k}: {val}")
        
    df = pd.DataFrame(list(results_dict.items()), columns=['Checkpoint', 'MPJPE'])
    df.to_csv("eval_mpjpe_results.csv", index=False)
    print("\nResults saved to eval_mpjpe_results.csv")

if __name__ == "__main__":
    run_auto_eval()