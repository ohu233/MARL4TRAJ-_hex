import heapq
import math
import os
import pickle

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
    """A* 搜索，只在 road_map 上的格子间移动。返回路径列表或 None。"""
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
    """
    对单个模式做 A* 评估。
    返回 (match_rate, success, progress)。
    """
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


def evaluate_all(traj_path='data/artificial_od_single.csv',
                 map_path='data/hex_grid.pkl',
                 output_dir='SimpleModeResult',
                 max_samples=None):
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
    correct = 0
    labeled = 0
    per_mode_correct = {m: 0 for m in modelist}
    per_mode_total = {m: 0 for m in modelist}

    log_path = os.path.join(output_dir, 'eval_log.txt')
    log = open(log_path, 'w', encoding='utf-8')

    print(f"共 {total} 条轨迹，开始评估...")
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

        final_scores = [single_match[j] * single_speed[j] for j in range(len(modelist))]
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

        mark = "OK" if is_correct == 1 else ("MISS" if is_correct == 0 else "?")
        sm_str = "  ".join(f"{m}:{single_match[j]:.3f}" for j, m in enumerate(modelist))
        sp_str = "  ".join(f"{m}:{single_speed[j]:.3f}" for j, m in enumerate(modelist))
        log.write(
            f"[{i+1:05d}/{total}] ID={traj_id}  true={true_mode}  pred={best_mode}  "
            f"{mark}  vel={vel:.1f}\n"
            f"  match=[{sm_str}]\n"
            f"  speed=[{sp_str}]\n"
        )

        if True:
            acc = correct / labeled * 100 if labeled > 0 else 0
            mr_str = "  ".join(f"{m}:{single_match[j]:.3f}" for j, m in enumerate(modelist))
            sp_str = "  ".join(f"{m}:{single_speed[j]:.3f}" for j, m in enumerate(modelist))
            print(f"  [{i+1}/{total}] ID={traj_id} true={true_mode} pred={best_mode} {mark} "
                  f"vel={vel:.1f} acc={acc:.1f}%")
            print(f"    match=[{mr_str}]  speed=[{sp_str}]")
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
