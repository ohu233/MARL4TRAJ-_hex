import heapq
import math
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

modelist = ['GSD', 'GG', 'TS', 'TG']

MODE_VEL_MEAN = {'GG': 30.14, 'GSD': 13.32, 'TS': 50.01, 'TG': 95.65}
MODE_VEL_STD  = {'GG': 20.13, 'GSD': 15.45, 'TS': 29.61, 'TG': 56.29}

HEX_DIRECTIONS = [
    (0, -1, +1),
    (-1, 0, +1),
    (-1, +1, 0),
    (0, +1, -1),
    (+1, 0, -1),
    (+1, -1, 0),
]


def hex_distance(c1, c2):
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]), abs(c1[2] - c2[2]))


def hex_neighbors(q, r, s):
    return [(q + dq, r + dr, s + ds) for dq, dr, ds in HEX_DIRECTIONS]


def _hex_ring_offsets(radius):
    if radius == 0:
        return [(0, 0, 0)]
    offsets = []
    d_start = HEX_DIRECTIONS[0]
    q, r, s = d_start[0] * radius, d_start[1] * radius, d_start[2] * radius
    edge_order = [2, 3, 4, 5, 0, 1]
    for d_idx in edge_order:
        dq, dr, ds = HEX_DIRECTIONS[d_idx]
        for _ in range(radius):
            offsets.append((q, r, s))
            q += dq
            r += dr
            s += ds
    return offsets


def find_nearest_road_cell(center_q, center_r, center_s, mapdata_dict, max_radius=30):
    for ring_r in range(max_radius + 1):
        for dq, dr, ds in _hex_ring_offsets(ring_r):
            key = (center_q + dq, center_r + dr, center_s + ds)
            if mapdata_dict.get(key, 0) != 0:
                return key
    return None


def code_to_mode_matrices(hex_mapdata):
    mode_matrices = {m: {} for m in modelist}
    for cube, code in hex_mapdata.items():
        if (code >> 6) & 1 == 1:
            mode_matrices['TG'][cube] = 1
        if (code >> 1) & 1 == 1:
            mode_matrices['TS'][cube] = 1
        if (code >> 3) & 1 == 1 or (code >> 4) & 1 == 1:
            mode_matrices['GG'][cube] = 1
        if (code >> 2) & 1 == 1 or (code >> 5) & 1 == 1:
            mode_matrices['GSD'][cube] = 1
    return mode_matrices


def speed_score(velocity, mode):
    mu = MODE_VEL_MEAN[mode]
    sigma_sq = max(MODE_VEL_STD[mode] ** 2, 1.0)
    return math.exp(-0.5 * (velocity - mu) ** 2 / sigma_sq)


def astar(road_map, start, goal, max_nodes=50000):
    if start == goal:
        return [start]
    if road_map.get(start, 0) == 0 or road_map.get(goal, 0) == 0:
        return None
    open_set = [(hex_distance(start, goal), 0, start)]
    g_score = {start: 0}
    came_from = {}
    closed = set()
    nodes = 0
    while open_set and nodes < max_nodes:
        _, _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        nodes += 1
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        for nb in hex_neighbors(*current):
            if nb in closed or road_map.get(nb, 0) == 0:
                continue
            new_g = g_score[current] + 1
            if new_g < g_score.get(nb, float('inf')):
                g_score[nb] = new_g
                came_from[nb] = current
                heapq.heappush(open_set, (new_g + hex_distance(nb, goal), new_g, nb))
    return None


def eval_single_mode(road_map, hex_start, hex_end, distance_threshold=1.0):
    initial_distance = max(float(hex_distance(hex_start, hex_end)), 1.0)
    road_start = find_nearest_road_cell(*hex_start, road_map, max_radius=30)
    road_end = find_nearest_road_cell(*hex_end, road_map, max_radius=30)
    if road_start is None or road_end is None:
        return 0.0, 0, 0.0
    snap_dist_start = hex_distance(hex_start, road_start)
    snap_dist_end = hex_distance(hex_end, road_end)
    path = astar(road_map, road_start, road_end)
    if path is None:
        return 0.0, 0, 0.0
    astar_len = len(path) - 1
    total_len = snap_dist_start + astar_len + snap_dist_end
    match_rate = astar_len / total_len if total_len > 0 else 0.0
    final_distance = hex_distance(road_end, hex_end)
    success = 1 if final_distance <= distance_threshold else 0
    progress = max(0.0, min(1.0, (initial_distance - final_distance) / initial_distance))
    return match_rate, success, progress


# ============================================================
# Viterbi
# ============================================================

def build_transition_matrix(stay_prob=0.9):
    """固定转移概率矩阵：对角线 stay_prob，其余均分。"""
    n = len(modelist)
    off = (1.0 - stay_prob) / (n - 1)
    mat = np.full((n, n), off, dtype=np.float64)
    np.fill_diagonal(mat, stay_prob)
    return mat


def speed_change_score_matrix(vel_prev, vel_cur, sigma=30.0):
    """计算速度变化得分矩阵。如果 vel_cur - vel_prev 与 mode_i→mode_j 的期望速度差一致，得分高。"""
    vel_diff = vel_cur - vel_prev
    sc = np.zeros((len(modelist), len(modelist)), dtype=np.float64)
    for i, m_prev in enumerate(modelist):
        for j, m_cur in enumerate(modelist):
            expected_diff = MODE_VEL_MEAN[m_cur] - MODE_VEL_MEAN[m_prev]
            sc[i][j] = math.exp(-0.5 * (vel_diff - expected_diff) ** 2 / (sigma ** 2))
    return sc


def viterbi_decode(scores_seq, trans_mat, vel_seq=None, init_prob=None):
    """
    scores_seq: list of [score_mode0, score_mode1, ...] 每段的观测分数
    trans_mat: 4x4 转移概率矩阵
    vel_seq: list of float, 每段的速度（可选，用于速度变化信号）
    init_prob: 1x4 初始概率，默认均匀
    返回: list of mode indices
    """
    n = len(scores_seq)
    n_modes = len(modelist)
    if init_prob is None:
        init_prob = np.ones(n_modes) / n_modes

    # 取 log 避免下溢
    log_trans = np.log(trans_mat + 1e-12)
    log_init = np.log(init_prob + 1e-12)

    # V[t][j] = max log-prob of reaching state j at time t
    V = np.full((n, n_modes), -np.inf)
    backptr = np.zeros((n, n_modes), dtype=int)

    # t=0
    for j in range(n_modes):
        obs = max(scores_seq[0][j], 1e-8)
        V[0][j] = log_init[j] + np.log(obs)

    # t>0
    for t in range(1, n):
        for j in range(n_modes):
            obs = max(scores_seq[t][j], 1e-8)
            log_obs = np.log(obs)
            if vel_seq and vel_seq[t] > 0 and vel_seq[t - 1] > 0:
                sc = speed_change_score_matrix(vel_seq[t - 1], vel_seq[t])
                candidates = V[t - 1] + log_trans[:, j] + np.log(sc[:, j] + 1e-12) + log_obs
            else:
                candidates = V[t - 1] + log_trans[:, j] + log_obs
            backptr[t][j] = int(np.argmax(candidates))
            V[t][j] = candidates[backptr[t][j]]

    # 回溯
    path = [0] * n
    path[-1] = int(np.argmax(V[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = int(backptr[t + 1][path[t + 1]])
    return path


# ============================================================
# 主流程
# ============================================================

def evaluate_all(traj_path='data/artificial_od_mult.csv',
                 map_path='data/hex_grid.pkl',
                 output_dir='SimpleModeResult',
                 max_samples=None,
                 eval_mode=True):
    os.makedirs(output_dir, exist_ok=True)

    print("加载数据...")
    traj = pd.read_csv(traj_path)
    if max_samples:
        traj = traj.head(max_samples)
    with open(map_path, 'rb') as f:
        hex_mapdata_raw = pickle.load(f)

    code_dict = {k: int(v['code']) if isinstance(v, dict) else int(v) for k, v in hex_mapdata_raw.items()}
    mode_maps = code_to_mode_matrices(code_dict)

    total = len(traj)
    print(f"共 {total} 条轨迹")

    # 统计转移概率
    trans_mat = build_transition_matrix()
    print(f"转移概率矩阵:")
    for i, m in enumerate(modelist):
        row = "  ".join(f"{m2}:{trans_mat[i][j]:.3f}" for j, m2 in enumerate(modelist))
        print(f"  {m} -> [{row}]")

    # ====== 第一遍：计算每段分数 ======
    print("\n第一遍：计算每段分数...")
    # 按 ID 分组存储
    id_groups = defaultdict(list)  # {id: [(global_idx, scores, true_mode), ...]}
    all_results = []  # [(scores, independent_pred, true_mode, traj_id, vel), ...]

    for i in range(total):
        row = traj.iloc[i]
        true_mode = str(row.get('mode', '')).strip() if 'mode' in traj.columns else None
        traj_id = str(row.get('ID', f'row_{i}')).strip()

        hex_start = (int(round(row['locxo'])), int(round(row['locyo'])), int(round(row['loczo'])))
        hex_end = (int(round(row['locxd'])), int(round(row['locyd'])), int(round(row['loczd'])))

        single_match, single_success, single_progress = [], [], []
        for m in modelist:
            mr, sc, pg = eval_single_mode(mode_maps[m], hex_start, hex_end)
            single_match.append(mr)
            single_success.append(sc)
            single_progress.append(pg)

        vel = 0.0
        if 'distance_m' in traj.columns and 'time' in traj.columns:
            t = float(row['time'])
            if t > 0:
                vel = float(row['distance_m']) / t
        single_speed = [speed_score(vel, m) for m in modelist]

        scores = [single_match[j] * single_speed[j] for j in range(len(modelist))]

        all_results.append({
            'scores': scores,
            'match': single_match,
            'speed': single_speed,
            'true_mode': true_mode,
            'traj_id': traj_id,
            'vel': vel,
        })
        id_groups[traj_id].append(i)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{total}] scored")

    # ====== 第二遍：Viterbi 解码 ======
    print("\n第二遍：Viterbi 解码...")
    viterbi_preds = [None] * total

    for traj_id, indices in id_groups.items():
        scores_seq = [all_results[idx]['scores'] for idx in indices]
        vel_seq = [all_results[idx]['vel'] for idx in indices]
        if len(indices) == 1:
            viterbi_preds[indices[0]] = int(np.argmax(scores_seq[0]))
        else:
            path = viterbi_decode(scores_seq, trans_mat, vel_seq=vel_seq)
            for k, idx in enumerate(indices):
                viterbi_preds[idx] = path[k]

    # ====== 统计 ======
    log_path = os.path.join(output_dir, 'eval_log.txt')
    log = open(log_path, 'w', encoding='utf-8')

    correct = 0
    labeled = 0
    per_mode = {m: {'correct': 0, 'total': 0} for m in modelist}
    has_mode = eval_mode and 'mode' in traj.columns

    for i in range(total):
        r = all_results[i]
        true_mode = r['true_mode']
        traj_id = r['traj_id']
        pred = modelist[viterbi_preds[i]]

        is_ok = None
        if has_mode and true_mode in modelist:
            is_ok = int(pred == true_mode)
            labeled += 1
            if is_ok:
                correct += 1
            per_mode[true_mode]['total'] += 1
            if is_ok:
                per_mode[true_mode]['correct'] += 1

        sm_str = "  ".join(f"{m}:{r['match'][j]:.3f}" for j, m in enumerate(modelist))

        if has_mode:
            mark = "OK" if is_ok == 1 else ("MISS" if is_ok == 0 else "?")
            log.write(
                f"[{i+1:05d}/{total}] ID={traj_id}  true={true_mode}  pred={pred}  "
                f"{mark}  vel={r['vel']:.1f}\n"
                f"  match=[{sm_str}]\n"
            )
            acc = correct / labeled * 100 if labeled > 0 else 0
            print(f"  [{i+1}/{total}] ID={traj_id} true={true_mode} pred={pred} {mark} "
                  f"vel={r['vel']:.1f} acc={acc:.1f}%")
            print(f"    match=[{sm_str}]")
        else:
            log.write(
                f"[{i+1:05d}/{total}] ID={traj_id}  pred={pred}  "
                f"vel={r['vel']:.1f}\n"
                f"  match=[{sm_str}]\n"
            )
            print(f"  [{i+1}/{total}] ID={traj_id} pred={pred} vel={r['vel']:.1f}%")
            print(f"    match=[{sm_str}]")

    # ====== 汇总 ======
    print(f"\n{'='*60}")
    if has_mode and labeled > 0:
        acc_overall = correct / labeled * 100
        print(f"准确率: {acc_overall:.2f}% ({correct}/{labeled})")
        for m in modelist:
            c = per_mode[m]['correct']
            t = per_mode[m]['total']
            a = c / t * 100 if t > 0 else 0
            print(f"  {m}: {a:.2f}% ({c}/{t})")
        log.write(f"\n准确率: {acc_overall:.2f}% ({correct}/{labeled})\n")
        for m in modelist:
            c = per_mode[m]['correct']
            t = per_mode[m]['total']
            a = c / t * 100 if t > 0 else 0
            log.write(f"  {m}: {a:.2f}% ({c}/{t})\n")
    else:
        print("部署模式，无准确率统计")
        log.write("\n部署模式，无准确率统计\n")
    print(f"{'='*60}")

    log.close()
    print(f"结果已保存到 {output_dir}/eval_log.txt")


if __name__ == '__main__':
    evaluate_all()
