import math
import os
import pickle

import numpy as np
import pandas as pd

from utils.Environment import CounterfactualPathEvaluator

modelist = ['GSD', 'GG', 'TS', 'TG']

MODE_VEL_MEAN = {'GG': 30.14, 'GSD': 13.32, 'TS': 50.01, 'TG': 95.65}
MODE_VEL_STD  = {'GG': 20.13, 'GSD': 15.45, 'TS': 29.61, 'TG': 56.29}


def speed_score(velocity, mode):
    mu = MODE_VEL_MEAN[mode]
    sigma_sq = max(MODE_VEL_STD[mode] ** 2, 1.0)
    return math.exp(-0.5 * (velocity - mu) ** 2 / sigma_sq)


def evaluate_all(traj_path='data/artificial_od_single.csv',
                 model_path=r'PathModel\sac_actor_ep5000.pth',
                 map_path='data/hex_grid.pkl',
                 output_dir='SimpleModeResult',
                 max_samples=None):
    os.makedirs(output_dir, exist_ok=True)

    print("加载数据和模型...")
    traj = pd.read_csv(traj_path)
    if max_samples:
        traj = traj.head(max_samples)
    with open(map_path, 'rb') as f:
        hex_mapdata_raw = pickle.load(f)
    from utils.hex_utils import code_to_mode_matrices
    mode_maps = code_to_mode_matrices(
        {k: int(v['code']) if isinstance(v, dict) else int(v) for k, v in hex_mapdata_raw.items()}
    )

    evaluator = CounterfactualPathEvaluator(
        model_path=model_path,
        hex_mapdata_raw=hex_mapdata_raw,
        mode_maps=mode_maps,
        fov=5,
        distance_threshold=1.0,
    )

    total = len(traj)
    correct = 0
    labeled = 0
    per_mode_correct = {m: 0 for m in modelist}
    per_mode_total = {m: 0 for m in modelist}

    log_path = os.path.join(output_dir, 'eval_log.txt')
    log = open(log_path, 'w', encoding='utf-8')

    print(f"共 {total} 条轨迹，开始评估...")
    for i in range(total):
        row = traj.iloc[[i]].copy()
        true_mode = str(row['mode'].iat[0]).strip() if 'mode' in row.columns else None
        traj_id = str(row['ID'].iat[0]).strip() if 'ID' in row.columns else f"row_{i}"

        # 单模式 rollout
        single_match, single_success, single_progress = [], [], []
        for m in modelist:
            r = evaluator.evaluate(row, [m])
            single_match.append(float(r["multi_match_rate"]))
            single_success.append(float(r["success"]))
            single_progress.append(float(r["progress_score"]))

        # 速度分布评分
        vel = 0.0
        if 'distance_m' in row.columns and 'time' in row.columns:
            t = float(row['time'].iat[0])
            if t > 0:
                vel = float(row['distance_m'].iat[0]) / t
        single_speed = [speed_score(vel, m) for m in modelist]

        # 综合打分：match * speed
        final_scores = [single_match[j] * single_speed[j] for j in range(len(modelist))]

        # 预测：综合分最高的模式
        best_idx = int(np.argmax(final_scores))
        best_mode = modelist[best_idx]

        is_correct = int(best_mode == true_mode) if true_mode in modelist else None

        if true_mode in modelist:
            labeled += 1
            if is_correct:
                correct += 1
            per_mode_total[true_mode] += 1
            if is_correct:
                per_mode_correct[true_mode] += 1

        sm_str = "  ".join(f"{m}:{single_match[j]:.3f}" for j, m in enumerate(modelist))
        sp_str = "  ".join(f"{m}:{single_speed[j]:.3f}" for j, m in enumerate(modelist))
        mark = "OK" if is_correct == 1 else ("MISS" if is_correct == 0 else "?")
        log.write(
            f"[{i+1:05d}/{total}] ID={traj_id}  true={true_mode}  pred={best_mode}  "
            f"{mark}  vel={vel:.1f}\n"
            f"  match=[{sm_str}]\n"
            f"  speed=[{sp_str}]\n"
        )

        if (i + 1) % 1 == 0 or i == 0:
            acc = correct / labeled * 100 if labeled > 0 else 0
            print(f"  [{i+1}/{total}] ID={traj_id} true={true_mode} pred={best_mode} {mark} "
                  f"vel={vel:.1f} acc={acc:.1f}%")
            log.flush()

    acc_overall = correct / labeled * 100 if labeled > 0 else 0
    print(f"\n{'='*60}")
    print(f"总体准确率: {acc_overall:.2f}% ({correct}/{labeled})")
    for m in modelist:
        t = per_mode_total[m]
        c = per_mode_correct[m]
        a = c / t * 100 if t > 0 else 0
        print(f"  {m}: {a:.2f}% ({c}/{t})")
    print(f"{'='*60}")

    log.write(f"\n{'='*60}\n")
    log.write(f"总体准确率: {acc_overall:.2f}% ({correct}/{labeled})\n")
    for m in modelist:
        t = per_mode_total[m]
        c = per_mode_correct[m]
        a = c / t * 100 if t > 0 else 0
        log.write(f"  {m}: {a:.2f}% ({c}/{t})\n")
    log.write(f"{'='*60}\n")
    log.close()
    print(f"结果已保存到 {output_dir}/eval_log.txt")


if __name__ == '__main__':
    evaluate_all()
