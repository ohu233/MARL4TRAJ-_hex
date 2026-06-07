# 作为基线对比
import numpy as np
import pandas as pd
import pickle

"""
本文件完成和强化学习方法对比的三个baseline，分别为规则类方法，seq2seq的序列预测方法，HMM方法。 
输入： （1）序列：data_lower_train_random.csv ,列名为[ID,locx_o,locy_o,locx_d,locy_d,mode,time,distance （2）地图信息：GridModesAdjacentRealworld.pkl
输出： 对于每一条，输出一个预测的x.y坐标序列，起点和终点分别为locx_o,locy_o和locx_d,locy_d 
评价指标： 匹配度 预测出的序列与真实模式对应率。

"""

# 地图信息导入
from utils.hex_utils import (
    load_hex_mapdata, load_hex_mapdata_raw, code_to_mode_matrices,
    HEX_DIRECTIONS, hex_distance, hex_neighbors,
    get_hex_neighborhood,
    HEX_RADIUS,
)

hex_mapdata_raw = load_hex_mapdata_raw('data/hex_grid.pkl')
hex_code_dict = {k: int(v['code']) for k, v in hex_mapdata_raw.items()}

shuffle_traj = pd.read_csv('data/data_lower_train_random.csv')
train_traj, test_traj = shuffle_traj[:10000], shuffle_traj[10000:]

from utils.tools import mapdata_to_modelmatrix, get_patch
mode_matrix_dict = code_to_mode_matrices(hex_code_dict)

# 1. 规则类方法：对于每一条，输出一个预测的x.y坐标序列，起点和终点分别为locx_o,locy_o和locx_d,locy_d
def rule_based_method(traj, mode_matrix_dict):
    import numpy as np
    from utils.tools import get_patch

    start = (int(traj['locx_o']), int(traj['locy_o']))
    end = (int(traj['locx_d']), int(traj['locy_d']))
    mode = traj['mode']
    mode_matrix = mode_matrix_dict[mode]
    current = start
    trajectory = [current]

    def get_octant(dx, dy):
        if dx == 0 and dy > 0:
            return 4  # 正上
        if dx == 0 and dy < 0:
            return 5  # 正下
        if dy == 0 and dx > 0:
            return 6  # 正右
        if dy == 0 and dx < 0:
            return 7  # 正左
        if dx > 0 and dy > 0:
            return 0  # 第一象限
        if dx < 0 and dy > 0:
            return 1  # 第二象限
        if dx < 0 and dy < 0:
            return 2  # 第三象限
        if dx > 0 and dy < 0:
            return 3  # 第四象限

    offsets = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]

    while np.linalg.norm(np.array(current) - np.array(end)) > 1:
        x, y = current
        dx, dy = end[0] - x, end[1] - y
        octant = get_octant(dx, dy)
        # 正方向，直接插值到终点或边界
        if octant in [4, 5, 6, 7]:
            if octant == 4:  # 正上
                bx, by = x, min(y + 2, len(mode_matrix[0]) - 1, end[1])
            elif octant == 5:  # 正下
                bx, by = x, max(y - 2, 0, end[1])
            elif octant == 6:  # 正右
                bx, by = min(x + 2, len(mode_matrix) - 1, end[0]), y
            else:  # 正左
                bx, by = max(x - 2, 0, end[0]), y
            steps = max(abs(bx - x), abs(by - y))
            for t in range(1, steps + 1):
                inter_x = int(round(x + (bx - x) * t / steps))
                inter_y = int(round(y + (by - y) * t / steps))
                if (inter_x, inter_y) != trajectory[-1]:
                    trajectory.append((inter_x, inter_y))
            current = (bx, by)
            continue

        neighbors = get_patch(mode_matrix, x, y, size=5)
        candidates = []
        for idx, val in enumerate(neighbors):
            if val == 1:
                i, j = offsets[idx]
                nx, ny = x + i, y + j
                if (i, j) != (0, 0):
                    if get_octant(i, j) == octant:
                        dist = np.linalg.norm([nx - end[0], ny - end[1]])
                        candidates.append((nx, ny, dist))
        if candidates:
            next_cell = min(candidates, key=lambda x: x[2])
            steps = max(abs(next_cell[0] - x), abs(next_cell[1] - y))
            for t in range(1, steps + 1):
                inter_x = int(round(x + (next_cell[0] - x) * t / steps))
                inter_y = int(round(y + (next_cell[1] - y) * t / steps))
                if (inter_x, inter_y) != trajectory[-1]:
                    trajectory.append((inter_x, inter_y))
            current = (next_cell[0], next_cell[1])
        else:
            # 没有同象限可行点，插值到5x5边界
            dir_vec = np.array([dx, dy], dtype=float)
            if np.linalg.norm(dir_vec) == 0:
                break
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            max_step = 2
            bx = int(round(x + dir_vec[0] * max_step))
            by = int(round(y + dir_vec[1] * max_step))
            bx = min(max(bx, 0), len(mode_matrix) - 1)
            by = min(max(by, 0), len(mode_matrix[0]) - 1)
            steps = max(abs(bx - x), abs(by - y))
            for t in range(1, steps + 1):
                inter_x = int(round(x + (bx - x) * t / steps))
                inter_y = int(round(y + (by - y) * t / steps))
                if (inter_x, inter_y) != trajectory[-1]:
                    trajectory.append((inter_x, inter_y))
            current = (bx, by)

    if trajectory[-1] != end:
        trajectory.append(end)
    print('done', traj['ID'], 'mode:', mode, 'start:', start, 'end:', end, 'trajectory length:', len(trajectory))
    return trajectory

# 2. A star方法：基于地图信息和起点终点坐标，使用A star方法进行路径规划，输出一个预测的x.y坐标序列，起点和终点分别为locx_o,locy_o和locx_d,locy_d
#  启发函数用欧式距离，代价函数用曼哈顿距离，约束条件为软约束，在满足模式要求的格子上代价为1，在不满足模式要求的格子上代价为10
def a_star_method(traj, mode_matrix_dict)->list:
    import heapq

    start = (int(traj['locx_o']), int(traj['locy_o']))
    end = (int(traj['locx_d']), int(traj['locy_d']))
    mode = traj['mode']
    mode_matrix = mode_matrix_dict[mode]

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, end), 0, start, [start]))
    closed_set = set()

    path = []
    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current in closed_set:
            continue
        closed_set.add(current)

        if current == end:
            return path

        x, y = current
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if 0 <= neighbor[0] < len(mode_matrix) and 0 <= neighbor[1] < len(mode_matrix[0]):
                    if neighbor in closed_set:
                        continue
                    new_cost = cost + (1 if mode_matrix[neighbor[0]][neighbor[1]] == 1 else 2)
                    heapq.heappush(open_set, (new_cost + heuristic(neighbor, end), new_cost, neighbor, path + [neighbor]))

    return path  # 如果没有找到路径，返回当前路径



# 4. 测试与评价模块
def eval(traj_df, method_eval):
    """
    该函数用来评估方法性能，输出在测试集上的四种模式匹配度，再从测试集中，不同模式各抽取5条画出预测轨迹和路网图
    method: function, 规则类方法、seq2seq方法或HMM方法
    return：df,在原df基础上增加一列pred_path，表示预测的路径；dict,四种模式的平均匹配度；可视化图像，展示不同模式的预测轨迹和路网图
    """
    total_count = 0
    match_rate_dict = {'TG':0, 'GG':0, 'GSD':0, 'TS':0}
    count_dict = {'TG':0, 'GG':0, 'GSD':0, 'TS':0}
    method = method_eval # 直接传函数

    new_traj_df = traj_df.copy()
    new_traj_df['pred_path'] = None
    for idx, traj in traj_df.iterrows():
        match = 0
        pred_path = method(traj, mode_matrix_dict)
        total = len(pred_path)
        mode = traj['mode']
        new_traj_df.at[idx, 'pred_path'] = pred_path
        for x, y in pred_path:
            if mode_matrix_dict[mode][x][y] == 1:
                match += 1
        match_rate_dict[mode] += match / total
        count_dict[mode] += 1
        total_count += 1
        if total_count % 100 == 0:
            avg_rate = {m: (match_rate_dict[m] / count_dict[m] if count_dict[m] > 0 else 0) for m in match_rate_dict}
            print(f'已处理{total_count}条，当前平均匹配度：{avg_rate}')

    for mode in match_rate_dict:
        match_rate_dict[mode] /= len(traj_df[traj_df['mode'] == mode])

    # 随机抽取不同模式的轨迹进行可视化

    import matplotlib.pyplot as plt
    from utils.geo_utils import grid_to_mercator, grid_bounds_to_mercator
    from utils.basemap import add_osm_basemap, USE_OSM_BASEMAP

    sample_trajs = traj_df.groupby('mode').apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)

    for mode in sample_trajs['mode'].unique():
        fig, ax = plt.subplots(figsize=(8, 8))
        mode_trajs = sample_trajs[sample_trajs['mode'] == mode]

        # 收集所有预测路径点以确定范围
        all_xs, all_ys = [], []
        pred_paths = {}
        for idx, traj in mode_trajs.iterrows():
            pred_path = method_eval(traj, mode_matrix_dict)
            pred_paths[idx] = pred_path
            xs, ys = zip(*pred_path)
            all_xs.extend(xs)
            all_ys.extend(ys)

        x_min, x_max = min(all_xs) - 2, max(all_xs) + 2
        y_min, y_max = min(all_ys) - 2, max(all_ys) + 2

        mxmin, mxmax, mymin, mymax = grid_bounds_to_mercator(x_min, x_max, y_min, y_max)
        ax.set_xlim(mxmin, mxmax)
        ax.set_ylim(mymin, mymax)

        if USE_OSM_BASEMAP:
            add_osm_basemap(ax, alpha=0.5)

        for idx, pred_path in pred_paths.items():
            traj = mode_trajs.loc[idx]
            xs, ys = zip(*pred_path)
            merc_x, merc_y = grid_to_mercator(np.array(xs), np.array(ys))
            ax.plot(merc_x, merc_y, marker='o',
                    label=f'Traj ID {traj["ID"]}', markersize=1, linewidth=2)

        ax.set_title(f'Predicted Trajectories for Mode {mode}')
        ax.set_xlabel('Web Mercator X')
        ax.set_ylabel('Web Mercator Y')
        ax.legend()
        ax.grid(True)
        plt.show()

    return new_traj_df, match_rate_dict



if __name__ == "__main__":
    rule_based_traj_df, rule_based_match = eval(test_traj, rule_based_method)
    print('Rule-based method match rate by mode: ', rule_based_match)
    rule_based_traj_df.to_csv('rule_based_traj_df.csv', index=False)

    a_traj_df, a_match = eval(test_traj, a_star_method)
    print('A star method match rate by mode: ', a_match)
    a_traj_df.to_csv('a_traj_df.csv', index=False)

