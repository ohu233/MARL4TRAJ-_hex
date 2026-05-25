"""
平顶六边形（flat-top hexagon）网格几何模块。

坐标约定（与 提取六边形网格坐标及道路信息.py 一致）:
  - 平顶六边形，尖朝左右，side_len = 200
  - 立方体坐标 (q, r, s)，满足 q + r + s = 0
  - q = (cx * 2/3) / 200  → q+ = 东
  - r = (-cx/3 + √3/3 * cy) / 200  → r+ = 北
  - hex_grid.pkl key: (q, r, s) 元组, value: {'lon', 'lat', 'code'}
"""

import pickle
import numpy as np

# ============================================================
# 常量
# ============================================================
SIDE_LENGTH = 200
HEX_RADIUS = 199            # max cube distance from hex center
TOTAL_CELLS = 3 * SIDE_LENGTH**2 - 3 * SIDE_LENGTH + 1  # 119401

# 平顶六边形 6 个方向（cube 坐标偏移量）
#      |z/s
#      |
#      |
#      /\
#     /  \
#    /    \
#   y/r    x/q
HEX_DIRECTIONS =  [
    (1, 0, -1),   # 0: NE
    (1, -1, 0),   # 1: E
    (0, -1, 1),   # 2: SE
    (-1, 0, 1),   # 3: SW
    (-1, 1, 0),   # 4: W
    (0, 1, -1),   # 5: NW
    (0, 0, 0),    # 6: self
]

# action → radius-1 邻域中的索引
# ring 顺序从 东 开始: [dir4, dir5, dir0, dir1, dir2, dir3]
# neighbor 列表: [center] + ring → [center, dir4, dir5, dir0, dir1, dir2, dir3]
# 所以 action i 在 neighbor 中的位置 = ring.index(HEX_DIRECTIONS[i]) + 1
ACTION_TO_HEX_IDX = {0: 3, 1: 4, 2: 5, 3: 6, 4: 1, 5: 2}

# 道路模式位掩码（与 GridModesAdjacentRealworld.pkl / tools.py 一致）
# 县道、普铁、省道、高速收费站、高速、国道、高铁、火车站
MODE_LIST = ['GSD', 'GG', 'TS', 'TG']


# ============================================================
# 基础六边形几何
# ============================================================
def hex_distance(c1, c2):
    """六边形距离（cube distance）"""
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]), abs(c1[2] - c2[2]))


def hex_is_valid(q, r, s, radius=HEX_RADIUS):
    """检查 (q, r, s) 是否在以原点为中心、半径为 radius 的六边形内"""
    return max(abs(q), abs(r), abs(s)) <= radius


def hex_add(c, d):
    """cube 坐标加法"""
    return (c[0] + d[0], c[1] + d[1], c[2] + d[2])


def hex_sub(c1, c2):
    """cube 坐标减法"""
    return (c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2])


def hex_neighbors(q, r, s):
    """返回 (q, r, s) 的 6 个邻居"""
    return [(q + dq, r + dr, s + ds) for dq, dr, ds in HEX_DIRECTIONS]


# ============================================================
# 六边形邻域生成（替代原 get_patch）
# ============================================================
def _hex_ring_offsets(radius):
    """
    返回"恰好距离中心 radius 步"的所有 cube 偏移量，按固定顺序排列。
    从 HEX_DIRECTIONS[4] 方向起始点开始，绕 6 条边走一圈。
    """
    if radius == 0:
        return [(0, 0, 0)]
    offsets = []
    # 起始点：从中心向 dir4 走 radius 步
    d_start = HEX_DIRECTIONS[4]  # (-1, +1, 0)
    q, r, s = d_start[0] * radius, d_start[1] * radius, d_start[2] * radius
    # 沿 6 个方向各走 radius 步
    for d_idx in range(6):
        dq, dr, ds = HEX_DIRECTIONS[d_idx]
        for _ in range(radius):
            offsets.append((q, r, s))
            q += dq
            r += dr
            s += ds
    return offsets


def get_hex_neighborhood(mapdata_dict, q, r, s, radius=3):
    """
    返回以 (q, r, s) 为中心、radius 范围内的六边形邻域。

    缺失格子（pkl 中不存在）→ 特征填 0。

    参数:
        mapdata_dict: dict, {cube → value}，value 是标量（如 0/1 或 code）
        q, r, s: 中心 cube 坐标
        radius: 邻域半径

    返回:
        features: np.ndarray (N_cells,) 一维，每格一个标量值
        N_cells = 3*radius² + 3*radius + 1
    """
    n_cells = 3 * radius**2 + 3 * radius + 1
    features = np.zeros(n_cells, dtype=np.float32)

    idx = 0
    for ring_r in range(radius + 1):
        for dq, dr, ds in _hex_ring_offsets(ring_r):
            key = (q + dq, r + dr, s + ds)
            features[idx] = float(mapdata_dict.get(key, 0))
            idx += 1

    return features


def get_hex_neighborhood_multi(mapdata_dict, q, r, s, radius=3):
    """
    返回邻域节点坐标列表和特征矩阵（用于 GNN）。

    返回:
        nodes: list of (q, r, s)，长度为 N_cells
        features: np.ndarray (N_cells,)
        mask: np.ndarray (N_cells,) bool，True=在 mapdata 中存在
    """
    n_cells = 3 * radius**2 + 3 * radius + 1
    nodes = []
    features = np.zeros(n_cells, dtype=np.float32)
    mask = np.zeros(n_cells, dtype=bool)

    idx = 0
    for ring_r in range(radius + 1):
        for dq, dr, ds in _hex_ring_offsets(ring_r):
            key = (q + dq, r + dr, s + ds)
            nodes.append(key)
            if key in mapdata_dict:
                features[idx] = float(mapdata_dict[key])
                mask[idx] = True
            idx += 1

    return nodes, features, mask


# ============================================================
# 邻接矩阵（用于 GNN）
# ============================================================
def build_hex_adjacency(nodes, radius):
    """
    给定六边形邻域节点列表，构建边索引。
    两个节点有边 ⇔ cube 距离 = 1。

    返回:
        edge_index: np.ndarray (2, num_edges)，int64
    """
    # 建立 node → idx 映射
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    edges = []
    for i, node in enumerate(nodes):
        for nb in hex_neighbors(*node):
            j = node_to_idx.get(nb)
            if j is not None:
                edges.append((i, j))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    edge_index = np.array(edges, dtype=np.int64).T  # (2, num_edges)
    return edge_index


def get_fixed_edge_index(radius):
    """
    获取固定邻域的 edge_index（假设所有节点都存在时）。
    用于 GNN 的前向传播，所有 patch 共享同一邻接结构。
    """
    n_cells = 3 * radius**2 + 3 * radius + 1
    nodes = []
    for ring_r in range(radius + 1):
        nodes.extend(_hex_ring_offsets(ring_r))
    return build_hex_adjacency(nodes, radius)


# ============================================================
# 地图数据加载与转换
# ============================================================
def load_hex_mapdata(pkl_path='data/hex_grid.pkl'):
    """
    加载 hex_grid.pkl 全量数据（仅 code）。

    返回:
        dict: {(q, r, s) → code}
    """
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    return {k: int(v['code']) for k, v in raw.items()}


def load_hex_mapdata_raw(pkl_path='data/hex_grid.pkl'):
    """
    加载 hex_grid.pkl 全量原始数据（含 lon, lat, code）。

    返回:
        dict: {(q, r, s) → {'lon': ..., 'lat': ..., 'code': ...}}
    """
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def code_to_mode_matrices(hex_mapdata):
    """
    将 {cube → code} 位掩码转为 {mode_name: {cube → 0/1}}。

    位掩码逻辑与 tools.mapdata_to_modelmatrix 一致:
      TG:  code&1==1 or code>>1&1==1     (bits 0, 1)
      TS:  code&1==1 or code>>6&1==1 or code>>1&1==1  (bits 0, 1, 6)
      GG:  code>>3&1==1                  (bit 3)
      GSD: code>>2&1==1 or code>>5&1==1  (bits 2, 5)
    """
    mode_matrices = {m: {} for m in MODE_LIST}

    for cube, code in hex_mapdata.items():
        if code & 1 == 1 or (code >> 1) & 1 == 1:
            mode_matrices['TG'][cube] = 1
        if code & 1 == 1 or (code >> 6) & 1 == 1 or (code >> 1) & 1 == 1:
            mode_matrices['TS'][cube] = 1
        if (code >> 3) & 1 == 1:
            mode_matrices['GG'][cube] = 1
        if (code >> 2) & 1 == 1 or (code >> 5) & 1 == 1:
            mode_matrices['GSD'][cube] = 1

    return mode_matrices


# ============================================================
# 坐标转换（用于可视化）
# ============================================================
def hex_to_lonlat(hex_mapdata, q, r, s):
    """从 hex_mapdata 中查找 (q,r,s) 的经纬度"""
    key = (int(q), int(r), int(s))
    entry = hex_mapdata.get(key)
    if entry is not None and hasattr(entry, 'get'):
        # 原始 pkl 格式: {cube: {'lon':..., 'lat':..., 'code':...}}
        return entry.get('lon', 0), entry.get('lat', 0)
    return 0, 0


def hex_to_mercator(hex_mapdata_raw, q_arr, r_arr, s_arr):
    """
    六边形 cube 坐标 → Web Mercator (EPSG:3857)。
    通过 hex_grid.pkl 的原始 dict 查找 lon/lat。

    参数:
        hex_mapdata_raw: 原始 pkl dict {(q,r,s): {'lon','lat','code'}}
        q_arr, r_arr, s_arr: 坐标数组

    返回:
        merc_x, merc_y: np.ndarray
    """
    import pyproj
    wgs84 = pyproj.CRS.from_epsg(4326)
    merc = pyproj.CRS.from_epsg(3857)
    transformer = pyproj.Transformer.from_crs(wgs84, merc, always_xy=True)

    q_arr = np.asarray(q_arr).ravel()
    r_arr = np.asarray(r_arr).ravel()
    s_arr = np.asarray(s_arr).ravel()

    lons = np.zeros(len(q_arr))
    lats = np.zeros(len(q_arr))

    for i in range(len(q_arr)):
        key = (int(q_arr[i]), int(r_arr[i]), int(s_arr[i]))
        entry = hex_mapdata_raw.get(key)
        if entry is not None:
            lons[i] = entry['lon']
            lats[i] = entry['lat']

    mx, my = transformer.transform(lons, lats)
    return mx, my


def hex_bounds_mercator(hex_mapdata_raw, q_range=None, r_range=None, s_range=None):
    """
    六边形区域 → Web Mercator 范围。
    返回 (xmin, xmax, ymin, ymax)
    """
    if q_range is not None:
        # 用 corner 点计算
        pass

    # 取所有点的 lon/lat 边界
    lons = [v['lon'] for v in hex_mapdata_raw.values()]
    lats = [v['lat'] for v in hex_mapdata_raw.values()]

    import pyproj
    wgs84 = pyproj.CRS.from_epsg(4326)
    merc = pyproj.CRS.from_epsg(3857)
    transformer = pyproj.Transformer.from_crs(wgs84, merc, always_xy=True)

    corners_lon = [min(lons), max(lons), min(lons), max(lons)]
    corners_lat = [min(lats), min(lats), max(lats), max(lats)]
    mx, my = transformer.transform(corners_lon, corners_lat)

    return min(mx), max(mx), min(my), max(my)
