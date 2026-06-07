"""可视化 Rule-Based_AStar 的预测结果。
对每条 OD 对用 A* 在预测模式路网上寻路，画出实际路径。
同一 ID 的多条 OD 聚合为一张图。
"""

import math
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import requests
from PIL import Image
from io import BytesIO

from TestPath import (
    build_hex_lonlat_index, hex_to_mercator,
    add_amap_basemap, _get_zoom_for_bounds,
)
import importlib
_rba = importlib.import_module("Rule-Based_AStar")
astar = _rba.astar
find_nearest_road_cell = _rba.find_nearest_road_cell
code_to_mode_matrices = _rba.code_to_mode_matrices

modelist = ['GSD', 'GG', 'TS', 'TG']
MODE_COLORS = {"TG": "purple", "GG": "blue", "GSD": "green", "TS": "red"}


def plot_pred_od_pairs(pred_csv, map_path='data/hex_grid.pkl', save_dir='VisPred_results'):
    os.makedirs(save_dir, exist_ok=True)

    # 加载地图索引
    with open(map_path, 'rb') as f:
        hex_mapdata_raw = pickle.load(f)
    build_hex_lonlat_index(hex_mapdata_raw)

    from utils.hex_utils import code_to_mode_matrices
    mode_maps = code_to_mode_matrices(
        {k: int(v['code']) if isinstance(v, dict) else int(v) for k, v in hex_mapdata_raw.items()}
    )

    df = pd.read_csv(pred_csv)
    total = len(df)
    print(f"加载 {total} 条 OD 对，{df['ID'].nunique()} 个 ID")

    # 按 ID 分组
    id_groups = defaultdict(list)
    for i in range(total):
        row = df.iloc[i]
        traj_id = str(row['ID'])
        pred_mode = str(row.get('mode', ''))
        hex_start = (int(row['locxo']), int(row['locyo']), int(row['loczo']))
        hex_end = (int(row['locxd']), int(row['locyd']), int(row['loczd']))
        id_groups[traj_id].append({
            'index': i,
            'pred_mode': pred_mode,
            'hex_start': hex_start,
            'hex_end': hex_end,
            'vel': float(row.get('velocity', 0)),
        })

    # 为每个 ID 画一张聚合图
    for traj_id, items in id_groups.items():
        # 计算 mercator 范围
        all_mx, all_my = [], []
        for item in items:
            for h in [item['hex_start'], item['hex_end']]:
                merc = hex_to_mercator(*h)
                if merc is not None:
                    all_mx.append(merc[0])
                    all_my.append(merc[1])

        if not all_mx:
            print(f"  [SKIP] ID={traj_id} 无法映射到地图")
            continue

        padding = 300
        merc_bounds = (min(all_mx) - padding, max(all_mx) + padding,
                       min(all_my) - padding, max(all_my) + padding)

        fig, ax = plt.subplots(figsize=(10, 9))
        zoom = _get_zoom_for_bounds(*merc_bounds)
        add_amap_basemap(ax, merc_bounds, zoom=zoom)

        # 叠加预测模式对应路网
        pred_modes = set(item['pred_mode'] for item in items if item['pred_mode'] in modelist)
        xmin, xmax, ymin, ymax = merc_bounds
        for mode_name in sorted(pred_modes):
            if mode_name not in mode_maps:
                continue
            road_hexes = []
            for cube, val in mode_maps[mode_name].items():
                if val == 1:
                    m = hex_to_mercator(cube[0], cube[1], cube[2])
                    if m is not None:
                        if xmin <= m[0] <= xmax and ymin <= m[1] <= ymax:
                            road_hexes.append(m)
            if road_hexes:
                rx = [r[0] for r in road_hexes]
                ry = [r[1] for r in road_hexes]
                ax.scatter(rx, ry, s=1.5, c=MODE_COLORS.get(mode_name, 'gray'),
                           marker='s', alpha=0.2, zorder=2,
                           label=f'road_{mode_name}')

        # 画每条 OD 的 A* 路径（同一 ID 内链式共享路网锚点）
        items.sort(key=lambda x: x['index'])
        prev_road_end = None
        for item in items:
            s_merc = hex_to_mercator(*item['hex_start'])
            e_merc = hex_to_mercator(*item['hex_end'])
            if s_merc is None or e_merc is None:
                continue

            color = MODE_COLORS.get(item['pred_mode'], 'gray')
            pred = item['pred_mode']
            road_map = mode_maps.get(pred, {})

            # 链式：首段吸附起点，后续段用上一段终点
            if prev_road_end:
                road_start = prev_road_end
            else:
                road_start = find_nearest_road_cell(*item['hex_start'], road_map, max_radius=30)
            road_end = find_nearest_road_cell(*item['hex_end'], road_map, max_radius=30)
            path = None
            if road_start and road_end:
                path = astar(road_map, road_start, road_end)

            if path:
                # 画吸附段（起点→路网，当前模式颜色虚线）
                if prev_road_end is None:
                    start_on_road = hex_to_mercator(*road_start)
                    if start_on_road:
                        ax.plot([s_merc[0], start_on_road[0]], [s_merc[1], start_on_road[1]],
                                color=color, linewidth=1, alpha=0.6, linestyle='--', zorder=3)
                # 画 A* 路径
                px, py = [], []
                for p in path:
                    merc = hex_to_mercator(*p)
                    if merc:
                        px.append(merc[0])
                        py.append(merc[1])
                if px:
                    ax.plot(px, py, color=color, linewidth=2.5, alpha=0.9, zorder=4)
                # 画吸附段（路网→终点，当前模式颜色虚线）
                end_on_road = hex_to_mercator(*road_end)
                if end_on_road:
                    ax.plot([end_on_road[0], e_merc[0]], [end_on_road[1], e_merc[1]],
                            color=color, linewidth=1, alpha=0.6, linestyle='--', zorder=3)
                prev_road_end = road_end
            else:
                # A* 失败，画直线
                ax.plot([s_merc[0], e_merc[0]], [s_merc[1], e_merc[1]],
                        color=color, linewidth=1.5, alpha=0.6, linestyle=':', zorder=4)
                prev_road_end = None


        # 模式统计
        mode_counts = defaultdict(int)
        for item in items:
            mode_counts[item['pred_mode']] += 1
        mode_str = ", ".join(f"{m}:{c}" for m, c in sorted(mode_counts.items()))

        ax.set_title(f"ID={traj_id}  segments={len(items)}  pred=[{mode_str}]")
        ax.legend(fontsize=8, loc='upper right')
        ax.axis('off')

        save_path = os.path.join(save_dir, f"{traj_id}.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f"  ID={traj_id} -> {save_path}")

    print(f"\n共 {len(id_groups)} 张图保存到 {save_dir}/")


if __name__ == '__main__':
    import sys
    pred_csv = 'data\dataset_20230917_nanjing_to_gaochun_lishui_with_hex_downsampled_od_pred.csv'
    if pred_csv is None:
        # 自动找最新的 _pred.csv
        import glob
        candidates = sorted(glob.glob("data/*_pred.csv"))
        if candidates:
            pred_csv = candidates[-1]
            print(f"自动选择: {pred_csv}")
        else:
            print("未找到 *_pred.csv 文件，请指定路径: python VisualizePred.py <pred.csv>")
            exit(1)
    plot_pred_od_pairs(pred_csv)
