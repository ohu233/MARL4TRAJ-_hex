import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Tuple, Union
import warnings
from PIL import Image
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

from utils.geo_utils import grid_to_mercator, full_grid_bounds_mercator
from utils.basemap import set_ax_extent, set_full_extent, USE_OSM_BASEMAP
from utils.hex_utils import (
    get_hex_neighborhood, get_hex_neighborhood_multi,
    code_to_mode_matrices, load_hex_mapdata,
    MODE_LIST, HEX_RADIUS,
)

MODE_LIST = ["GSD", "GG", "TS", "TG"]


def mapdata_to_modelmatrix(mapdata: Union[dict, Dict], *args, **kwargs):
    """
    将地图数据转为 per-mode 的二值查找表。

    支持两种输入格式:
      1. hex dict: {(q, r, s) → code}  → 调用 code_to_mode_matrices
      2. 旧矩形 grid dict: {key → [...]}  → 旧逻辑 (向后兼容)

    对于 hex 输入，忽略 n_row/n_col 参数。
    """
    # 检测输入类型：检查第一个 key 是否为 cube 坐标 tuple
    first_key = next(iter(mapdata)) if mapdata else None
    if isinstance(first_key, tuple) and len(first_key) == 3:
        # hex dict 输入
        return code_to_mode_matrices(mapdata)

    # 旧矩形 grid 输入 — 向后兼容
    n_row = args[0] if args else kwargs.get('n_row', 529)
    n_col = args[1] if len(args) > 1 else kwargs.get('n_col', 564)

    modelmatrix = {"TG": [[0 for _ in range(n_row)] for _ in range(n_col)],
                    "GG": [[0 for _ in range(n_row)] for _ in range(n_col)],
                    "GSD": [[0 for _ in range(n_row)] for _ in range(n_col)],
                    "TS": [[0 for _ in range(n_row)] for _ in range(n_col)]
    }
    for k, v in mapdata.items():
        try:
            if v[4] & 1 == 1 or v[4] >> 1 & 1 == 1:
                modelmatrix['TG'][k[0]][k[1]] = 1
            if v[4] & 1 == 1 or v[4] >> 6 & 1 == 1 or v[4] >> 1 & 1 == 1:
                modelmatrix['TS'][k[0]][k[1]] = 1
            if v[4] >> 3 & 1 == 1:
                modelmatrix['GG'][k[0]][k[1]] = 1
            if v[4] >> 2 & 1 == 1 or v[4] >> 5 & 1 == 1:
                modelmatrix['GSD'][k[0]][k[1]] = 1
        except:
            print('Input Data Out of Range: ', k, v, 'Map Size: ', n_row, n_col)
    return modelmatrix


def get_patch(modelmatrix, x, y, size=3) -> list:
    """
    以 (x,y) 为中心的邻域提取。

    支持两种 modelmatrix 格式:
      1. hex dict: {(q,r,s) → value} → 调用 get_hex_neighborhood
      2. 旧 2D list: list[list[int]] → 旧逻辑 (向后兼容)

    可通过 `isinstance(modelmatrix, dict)` 自动检测。
    """
    # 自动检测：第一个元素是否为 cube 坐标 tuple
    if isinstance(modelmatrix, dict):
        first_key = next(iter(modelmatrix)) if modelmatrix else None
        if isinstance(first_key, tuple) and len(first_key) == 3:
            # hex dict
            radius = (size - 1) // 2
            return get_hex_neighborhood(modelmatrix, x, y, 0, radius=radius).tolist()

    # 旧 2D list 逻辑
    try:
        xmax = len(modelmatrix)
        ymax = len(modelmatrix[0])
    except:
        print('Input Data Out of Range When Getting Patch: ', type(modelmatrix), x, y)
        return [0 for _ in range(size * size)]

    patch = []
    n = (size - 1) // 2
    for dx in range(-n, n + 1):
        for dy in range(-n, n + 1):
            nx, ny = int(x) + dx, int(y) + dy
            if 0 <= nx < xmax and 0 <= ny < ymax:
                patch.append(modelmatrix[nx][ny])
            else:
                patch.append(0)
    return patch


def state_to_vector(state: Dict, mode_list: List[str] = None) -> np.ndarray:
    """
    将 PathEnv 的 dict state 编码成 1D 向量（hex 版本）:

    [init_pos(3), target(3), cur(3), rel(3), mode_onehot(4), candidate_onehot(4), patch_flat]
    总维度 = 20 + len(patch)
    """
    if mode_list is None:
        mode_list = MODE_LIST

    init_pos = np.array(state['current_position'], dtype=np.float32)
    target_pos = np.array(state['remaining_distance'], dtype=np.float32)
    cur_pos = np.array(state['previous_remaining_distance'], dtype=np.float32)
    rel_dis = np.array(state['total_distance'], dtype=np.float32)

    # 确保是 3 维向量（cube 坐标），pad 或截断
    for arr, name in [(init_pos, 'current_position'),
                        (target_pos, 'remaining_distance'),
                        (cur_pos, 'previous_remaining_distance'),
                        (rel_dis, 'total_distance')]:
        if arr.shape[0] < 3:
            arr = np.pad(arr, (0, 3 - arr.shape[0]), constant_values=0)
        elif arr.shape[0] > 3:
            arr = arr[:3]

    mode_onehot = np.zeros(len(mode_list), dtype=np.float32)
    current_mode = state['current_mode']
    if isinstance(current_mode, (list, tuple, np.ndarray)):
        for m in current_mode:
            if m in mode_list:
                mode_onehot[mode_list.index(m)] = 1.0
    else:
        if current_mode in mode_list:
            mode_onehot[mode_list.index(current_mode)] = 1.0

    candidate_onehot = np.zeros(len(mode_list), dtype=np.float32)
    candidate_modes = state.get('candidate_modes', set())
    for m in candidate_modes:
        if m in mode_list:
            candidate_onehot[mode_list.index(m)] = 1.0

    patch = np.array(state['patch'], dtype=np.float32).reshape(-1)

    vec = np.concatenate([init_pos, target_pos, cur_pos, rel_dis,
                          mode_onehot, candidate_onehot, patch], axis=0)
    return vec


def state_to_vector_hex(state: Dict, mode_list: List[str] = None) -> np.ndarray:
    """
    Hex 专用版（显式 hex 版本，与 state_to_vector 逻辑相同但语义更清晰）。
    vec_dim = 20 (4 个 3D cube 向量 + 2 个 4D onehot)
    """
    return state_to_vector(state, mode_list)


def calculate_match_rate(traj_list: list, mapdata) -> float:
    """
    计算轨迹点在路网上的匹配率。

    mapdata 支持:
      - hex dict: {(q,r,s) → value}  直接 dict.get 查询
      - 2D np.ndarray: 旧矩形 grid  → mapdata[x, y] 查询
    """
    if not traj_list:
        return 0.0

    on_road = 0
    total = 0

    if isinstance(mapdata, dict):
        # hex dict
        for p in traj_list:
            if p is None or len(p) < 2:
                continue
            q, r = int(round(p[0])), int(round(p[1]))
            s = -(q + r) if len(p) < 3 else int(round(p[2]))
            total += 1
            if mapdata.get((q, r, s), 0) != 0:
                on_road += 1
    else:
        # 旧 2D array
        x_max, y_max = mapdata.shape[0], mapdata.shape[1]
        for p in traj_list:
            if p is None or len(p) < 2:
                continue
            x, y = int(round(p[0])), int(round(p[1]))
            total += 1
            if 0 <= x < x_max and 0 <= y < y_max:
                if mapdata[x, y] != 0:
                    on_road += 1

    if total == 0:
        return 0.0
    return on_road / total


def plt_multi_map(modes: List[str], use_hex: bool = True):
    """
    在底图上叠加指定 modes 的道路点。
    use_hex=True: 使用 hex_grid.pkl
    use_hex=False: 使用旧 GridModesAdjacentRealworld.pkl
    """
    if modes is None:
        raise ValueError("modes 不能为空，例如 ['TG', 'GG']")

    valid_modes = {"TG", "GG", "GSD", "TS"}
    invalid = [m for m in modes if m not in valid_modes]
    if invalid:
        raise ValueError(f"不支持的 mode: {invalid}，可选: {sorted(valid_modes)}")

    mode_colors = {
        "TG": "orange", "GG": "blue", "GSD": "green", "TS": "red",
    }

    if use_hex:
        from utils.hex_utils import hex_to_mercator
        hex_mapdata = load_hex_mapdata()
        mode_mats = code_to_mode_matrices(hex_mapdata)

        fig, ax = plt.subplots(figsize=(20, 20))

        for mode in modes:
            active = [(q, r, s) for (q, r, s), v in mode_mats[mode].items() if v == 1]
            if active:
                qs = np.array([c[0] for c in active])
                rs = np.array([c[1] for c in active])
                ss = np.array([c[2] for c in active])
                merc_x, merc_y = hex_to_mercator(hex_mapdata, qs, rs, ss)
                ax.scatter(merc_x, merc_y, s=5, c=mode_colors[mode],
                           marker='+', linewidths=0.3, alpha=0.7)

        set_full_extent(ax)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig('intermediate_fig.png')
        return fig, ax

    else:
        # 旧逻辑
        with open(r"data\GridModesAdjacentRealworld.pkl", "rb") as f:
            mapdata = pickle.load(f)
        matrice = mapdata_to_modelmatrix(mapdata, 529, 564)

        ref_matrix = np.array(matrice["TG"])
        x_max, y_max = ref_matrix.shape[0], ref_matrix.shape[1]

        fig, ax = plt.subplots(figsize=(20, 20))

        for mode in modes:
            matrix = np.array(matrice[mode])
            points = np.argwhere(matrix == 1)
            if points.size > 0:
                grid_x = points[:, 0]
                grid_y = points[:, 1]
                merc_x, merc_y = grid_to_mercator(grid_x, grid_y)
                ax.scatter(merc_x, merc_y, s=5, c=mode_colors[mode],
                           marker='+', linewidths=0.3, alpha=0.7)

        set_full_extent(ax)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig('intermediate_fig.png')
        return fig, ax
