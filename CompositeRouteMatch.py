import argparse
import heapq
import math
import os
import pickle
from io import BytesIO
from collections import Counter, defaultdict

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

import pandas as pd
import requests
from PIL import Image


MODE_LIST = ["GSD", "GG", "TS", "TG"]
MODE_ORDER = {mode: i for i, mode in enumerate(MODE_LIST)}
MODE_COLORS = {"GSD": "green", "GG": "blue", "TS": "red", "TG": "purple"}

HEX_DIRECTIONS = [
    (0, -1, +1),
    (-1, 0, +1),
    (-1, +1, 0),
    (0, +1, -1),
    (+1, 0, -1),
    (+1, -1, 0),
]

DEFAULT_INPUT_CSV = "data/dataset_20230917_nanjing_to_gaochun_lishui_with_hex_downsampled_od.csv"
DEFAULT_MAP_PATH = "data/hex_grid_NanJing.pkl"
DEFAULT_OUTPUT_DIR = "CompositeRoute_results"

DEFAULT_K = 12
DEFAULT_MAX_SNAP_RADIUS = 30
DEFAULT_DETOUR_RATIO = 2.5
DEFAULT_DETOUR_WEIGHT = 1.0
DEFAULT_EMISSION_WEIGHT = 1.0
DEFAULT_MAX_EXPANSIONS = 50000
AMAP_TILE_URL = "https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"
WEB_MERCATOR_HALF_WORLD = 20037508.342789244
WEB_MERCATOR_WORLD = 40075016.68557849

ROAD_MODES = {"GSD", "GG"}
RAIL_MODES = {"TS", "TG"}


def hex_distance(c1, c2):
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]), abs(c1[2] - c2[2]))


def hex_neighbors(q, r, s):
    return [(q + dq, r + dr, s + ds) for dq, dr, ds in HEX_DIRECTIONS]


def hex_ring_offsets(radius):
    if radius == 0:
        return [(0, 0, 0)]

    offsets = []
    q, r, s = 0, -radius, radius
    edge_order = [2, 3, 4, 5, 0, 1]
    for d_idx in edge_order:
        dq, dr, ds = HEX_DIRECTIONS[d_idx]
        for _ in range(radius):
            offsets.append((q, r, s))
            q += dq
            r += dr
            s += ds
    return offsets


def normalize_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def node_cell(node):
    return node[:3]


def node_mode(node):
    return node[3]


def out_of_china(lon, lat):
    return lon < 72.004 or lon > 137.8347 or lat < 0.8293 or lat > 55.8271


def transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y
    ret += 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def transform_lon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y
    ret += 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lon, lat):
    if out_of_china(lon, lat):
        return lon, lat
    a = 6378245.0
    ee = 0.00669342162296594323
    dlat = transform_lat(lon - 105.0, lat - 35.0)
    dlon = transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    return lon + dlon, lat + dlat


def lonlat_to_mercator(lon, lat):
    gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
    x = gcj_lon * WEB_MERCATOR_HALF_WORLD / 180.0
    y = math.log(math.tan((90.0 + gcj_lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * WEB_MERCATOR_HALF_WORLD / 180.0
    return x, y


def mercator_to_lonlat(mx, my):
    lon = mx / WEB_MERCATOR_HALF_WORLD * 180.0
    lat = my / WEB_MERCATOR_HALF_WORLD * 180.0
    lat = 180.0 / math.pi * (2.0 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lon, lat


def lonlat_to_tile(lon, lat, zoom):
    tx = int((lon + 180.0) / 360.0 * (1 << zoom))
    lat_rad = math.radians(lat)
    ty = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (1 << zoom))
    return tx, ty


def tile_to_mercator_bounds(tx, ty, zoom):
    n = 1 << zoom
    left = tx / n * WEB_MERCATOR_WORLD - WEB_MERCATOR_HALF_WORLD
    right = (tx + 1) / n * WEB_MERCATOR_WORLD - WEB_MERCATOR_HALF_WORLD
    top = WEB_MERCATOR_HALF_WORLD - ty / n * WEB_MERCATOR_WORLD
    bottom = WEB_MERCATOR_HALF_WORLD - (ty + 1) / n * WEB_MERCATOR_WORLD
    return left, right, bottom, top


def zoom_for_bounds(xmin, xmax, ymin, ymax, fig_width_px=900):
    span = max(xmax - xmin, ymax - ymin)
    if span <= 0:
        return 15
    m_per_px = span / fig_width_px
    zoom = int(math.log2(WEB_MERCATOR_WORLD / (256 * m_per_px))) - 1
    return max(10, min(17, zoom))


def fetch_amap_tile(tx, ty, zoom):
    url = AMAP_TILE_URL.format(s=(tx + ty) % 4 + 1, x=tx, y=ty, z=zoom)
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGB")
    except (requests.RequestException, OSError):
        return None
    return None


def add_amap_basemap(ax, merc_bounds, zoom):
    xmin, xmax, ymin, ymax = merc_bounds
    left_lon, top_lat = mercator_to_lonlat(xmin, ymax)
    right_lon, bottom_lat = mercator_to_lonlat(xmax, ymin)
    tl_tx, tl_ty = lonlat_to_tile(left_lon, top_lat, zoom)
    br_tx, br_ty = lonlat_to_tile(right_lon, bottom_lat, zoom)

    for ty in range(tl_ty, br_ty + 1):
        for tx in range(tl_tx, br_tx + 1):
            tile_img = fetch_amap_tile(tx, ty, zoom)
            if tile_img is None:
                continue
            t_left, t_right, t_bottom, t_top = tile_to_mercator_bounds(tx, ty, zoom)
            ax.imshow(
                tile_img,
                extent=[t_left, t_right, t_bottom, t_top],
                aspect="auto",
                alpha=0.85,
                interpolation="bilinear",
                zorder=0,
            )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")


def mode_switch_penalty(mode_a, mode_b):
    if mode_a == mode_b:
        return 0.0
    pair = {mode_a, mode_b}
    if pair == {"GSD", "GG"}:
        return 15.0     # 公路间切换
    if pair == {"TS", "TG"}:
        return 25.0     # 铁路间切换
    if (mode_a in ROAD_MODES and mode_b in RAIL_MODES) or (
        mode_a in RAIL_MODES and mode_b in ROAD_MODES
    ):
        return 20.0     # 公路↔铁路
    return 30.0


def code_to_modes(code):
    modes = []
    if (code >> 2) & 1 == 1 or (code >> 5) & 1 == 1:
        modes.append("GSD")
    if (code >> 3) & 1 == 1 or (code >> 4) & 1 == 1:
        modes.append("GG")
    if (code >> 1) & 1 == 1:
        modes.append("TS")
    if (code >> 6) & 1 == 1:
        modes.append("TG")
    return modes


def load_composite_road(map_path):
    with open(map_path, "rb") as f:
        raw = pickle.load(f)

    mode_cells = {mode: set() for mode in MODE_LIST}
    cell_modes = {}
    lonlat_index = {}

    for cube, value in raw.items():
        cell = tuple(int(x) for x in cube)
        code = int(value["code"]) if isinstance(value, dict) else int(value)
        if isinstance(value, dict) and "lon" in value and "lat" in value:
            lonlat_index[cell] = (float(value["lon"]), float(value["lat"]))
        modes = code_to_modes(code)
        if not modes:
            continue
        ordered_modes = sorted(modes, key=lambda m: MODE_ORDER[m])
        cell_modes[cell] = ordered_modes
        for mode in ordered_modes:
            mode_cells[mode].add(cell)

    return mode_cells, cell_modes, lonlat_index


def extract_signal_points(df, target_id):
    df = df.copy()
    df["_norm_id"] = df["ID"].map(normalize_id)
    group = df[df["_norm_id"] == normalize_id(target_id)].copy()
    if group.empty:
        available = sorted(df["_norm_id"].dropna().unique().tolist())
        preview = ", ".join(available[:20])
        raise ValueError(f"ID={target_id} not found. Available IDs include: {preview}")

    group["_row_order"] = range(len(group))
    group = group.sort_values("_row_order")

    points = []
    for _, row in group.iterrows():
        points.append((int(row["locxo"]), int(row["locyo"]), int(row["loczo"])))

    last = group.iloc[-1]
    points.append((int(last["locxd"]), int(last["locyd"]), int(last["loczd"])))
    return points, group.drop(columns=["_norm_id", "_row_order"])


def find_candidates(point, cell_modes, k=DEFAULT_K, max_radius=DEFAULT_MAX_SNAP_RADIUS):
    candidates = []
    seen = set()
    for radius in range(max_radius + 1):
        for dq, dr, ds in hex_ring_offsets(radius):
            cell = (point[0] + dq, point[1] + dr, point[2] + ds)
            modes = cell_modes.get(cell)
            if not modes:
                continue
            for mode in modes:
                node = (cell[0], cell[1], cell[2], mode)
                if node in seen:
                    continue
                seen.add(node)
                candidates.append(
                    {
                        "node": node,
                        "q": cell[0],
                        "r": cell[1],
                        "s": cell[2],
                        "mode": mode,
                        "snap_dist": radius,
                    }
                )
        if len(candidates) >= k:
            break

    candidates.sort(key=lambda c: (c["snap_dist"], MODE_ORDER[c["mode"]], c["q"], c["r"], c["s"]))
    return candidates[:k]


def in_corridor(cell, start_cell, goal_cell, margin):
    for idx in range(3):
        lo = min(start_cell[idx], goal_cell[idx]) - margin
        hi = max(start_cell[idx], goal_cell[idx]) + margin
        if cell[idx] < lo or cell[idx] > hi:
            return False
    return True


def composite_neighbors(node, mode_cells, cell_modes, start_cell, goal_cell, corridor_margin):
    cell = node_cell(node)
    mode = node_mode(node)

    for nb in hex_neighbors(*cell):
        if not in_corridor(nb, start_cell, goal_cell, corridor_margin):
            continue
        if nb in mode_cells[mode]:
            yield (nb[0], nb[1], nb[2], mode), 1.0

    for other_mode in cell_modes.get(cell, []):
        if other_mode != mode:
            yield (cell[0], cell[1], cell[2], other_mode), mode_switch_penalty(mode, other_mode)

    for nb in hex_neighbors(*cell):
        if not in_corridor(nb, start_cell, goal_cell, corridor_margin):
            continue
        for other_mode in cell_modes.get(nb, []):
            if other_mode != mode:
                yield (
                    nb[0],
                    nb[1],
                    nb[2],
                    other_mode,
                ), 1.0 + mode_switch_penalty(mode, other_mode)


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_stats(path):
    if not path:
        return 0, 0
    move_len = 0
    switch_count = 0
    for prev, cur in zip(path, path[1:]):
        if node_cell(prev) != node_cell(cur):
            move_len += hex_distance(node_cell(prev), node_cell(cur))
        if node_mode(prev) != node_mode(cur):
            switch_count += 1
    return move_len, switch_count


def astar_composite(
    start,
    goal,
    mode_cells,
    cell_modes,
    max_expansions=DEFAULT_MAX_EXPANSIONS,
):
    if start == goal:
        return {"cost": 0.0, "path": [start], "move_len": 0, "switch_count": 0}

    start_cell = node_cell(start)
    goal_cell = node_cell(goal)
    straight = hex_distance(start_cell, goal_cell)
    corridor_margin = max(20, min(90, int(straight * 1.5) + 10))

    open_set = [(hex_distance(start_cell, goal_cell), 0.0, start)]
    came_from = {}
    best_g = {start: 0.0}
    closed = set()
    expansions = 0

    while open_set and expansions < max_expansions:
        _, g_cost, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        expansions += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            move_len, switch_count = path_stats(path)
            return {
                "cost": g_cost,
                "path": path,
                "move_len": move_len,
                "switch_count": switch_count,
            }

        for nb, step_cost in composite_neighbors(
            current, mode_cells, cell_modes, start_cell, goal_cell, corridor_margin
        ):
            if nb in closed:
                continue
            new_g = g_cost + step_cost
            if new_g < best_g.get(nb, float("inf")):
                best_g[nb] = new_g
                came_from[nb] = current
                h = hex_distance(node_cell(nb), goal_cell)
                heapq.heappush(open_set, (new_g + h, new_g, nb))

    return None


def transition_key(prev_node, cur_node):
    return prev_node, cur_node


def transition_result(
    prev_node,
    cur_node,
    prev_signal,
    cur_signal,
    mode_cells,
    cell_modes,
    cache,
    detour_ratio=DEFAULT_DETOUR_RATIO,
    detour_weight=DEFAULT_DETOUR_WEIGHT,
):
    key = transition_key(prev_node, cur_node)
    if key not in cache:
        cache[key] = astar_composite(prev_node, cur_node, mode_cells, cell_modes)
    result = cache[key]
    if result is None:
        return None

    signal_dist = max(hex_distance(prev_signal, cur_signal), 1)
    excess = max(0.0, result["move_len"] - signal_dist * detour_ratio)
    detour_cost = excess * detour_weight
    total_cost = result["cost"] + detour_cost
    return {
        "cost": total_cost,
        "path_cost": result["cost"],
        "detour_cost": detour_cost,
        "path": result["path"],
        "move_len": result["move_len"],
        "switch_count": result["switch_count"],
    }


def viterbi_match(
    signal_points,
    candidates_by_point,
    mode_cells,
    cell_modes,
    emission_weight=DEFAULT_EMISSION_WEIGHT,
):
    n = len(signal_points)
    dp = []
    backptr = []
    trans_cache = {}
    trans_choice = {}

    first_scores = []
    for cand in candidates_by_point[0]:
        first_scores.append(emission_weight * cand["snap_dist"])
    dp.append(first_scores)
    backptr.append([None] * len(candidates_by_point[0]))

    unreachable_segments = set()

    for i in range(1, n):
        cur_dp = [float("inf")] * len(candidates_by_point[i])
        cur_back = [None] * len(candidates_by_point[i])
        for cur_idx, cur_cand in enumerate(candidates_by_point[i]):
            emission = emission_weight * cur_cand["snap_dist"]
            for prev_idx, prev_cand in enumerate(candidates_by_point[i - 1]):
                if math.isinf(dp[i - 1][prev_idx]):
                    continue
                trans = transition_result(
                    prev_cand["node"],
                    cur_cand["node"],
                    signal_points[i - 1],
                    signal_points[i],
                    mode_cells,
                    cell_modes,
                    trans_cache,
                )
                if trans is None:
                    continue
                cost = dp[i - 1][prev_idx] + trans["cost"] + emission
                if cost < cur_dp[cur_idx]:
                    cur_dp[cur_idx] = cost
                    cur_back[cur_idx] = prev_idx
                    trans_choice[(i, cur_idx)] = trans
            if cur_back[cur_idx] is None:
                unreachable_segments.add(i)
        dp.append(cur_dp)
        backptr.append(cur_back)

    best_last_idx = min(range(len(dp[-1])), key=lambda idx: dp[-1][idx])
    if math.isinf(dp[-1][best_last_idx]):
        raise RuntimeError("No fully connected candidate sequence found for this ID.")

    matched_indices = [None] * n
    matched_indices[-1] = best_last_idx
    for i in range(n - 1, 0, -1):
        prev_idx = backptr[i][matched_indices[i]]
        if prev_idx is None:
            raise RuntimeError(f"Backtrace failed at signal index {i}.")
        matched_indices[i - 1] = prev_idx

    matched_candidates = [
        candidates_by_point[i][matched_indices[i]]
        for i in range(n)
    ]

    segment_paths = []
    unreachable_count = 0
    total_transition_cost = 0.0
    total_detour_cost = 0.0
    total_move_len = 0
    total_path_switches = 0

    for i in range(1, n):
        cur_idx = matched_indices[i]
        trans = trans_choice.get((i, cur_idx))
        if trans is None:
            unreachable_count += 1
            segment_paths.append(None)
            continue
        segment_paths.append(trans["path"])
        total_transition_cost += trans["path_cost"]
        total_detour_cost += trans["detour_cost"]
        total_move_len += trans["move_len"]
        total_path_switches += trans["switch_count"]

    return {
        "total_cost": dp[-1][best_last_idx],
        "matched_candidates": matched_candidates,
        "segment_paths": segment_paths,
        "unreachable_count": unreachable_count,
        "unreachable_candidate_segments": len(unreachable_segments),
        "total_transition_cost": total_transition_cost,
        "total_detour_cost": total_detour_cost,
        "total_move_len": total_move_len,
        "total_path_switches": total_path_switches,
    }


def flatten_route(segment_paths):
    route = []
    for seg_idx, path in enumerate(segment_paths, start=1):
        if not path:
            continue
        for step_idx, node in enumerate(path):
            if route and route[-1]["node"] == node:
                continue
            route.append(
                {
                    "segment_index": seg_idx,
                    "step_in_segment": step_idx,
                    "node": node,
                    "q": node[0],
                    "r": node[1],
                    "s": node[2],
                    "mode": node[3],
                }
            )
    return route


def cell_to_lonlat(cell, lonlat_index):
    return lonlat_index.get(tuple(int(x) for x in cell))


def cell_to_mercator(cell, lonlat_index):
    ll = cell_to_lonlat(cell, lonlat_index)
    if ll is None:
        return None
    return lonlat_to_mercator(ll[0], ll[1])


def plot_route_png(target_id, signal_points, matched_rows, route_rows, lonlat_index, mode_cells, output_dir):
    import matplotlib.pyplot as plt

    safe_id = normalize_id(target_id)
    fig, ax = plt.subplots(figsize=(8, 12))

    signal_xy = []
    for point in signal_points:
        merc = cell_to_mercator(point, lonlat_index)
        if merc is not None:
            signal_xy.append(merc)

    route_xy_all = []
    for row in route_rows:
        merc = cell_to_mercator((row["q"], row["r"], row["s"]), lonlat_index)
        if merc is not None:
            route_xy_all.append(merc)

    all_xy = signal_xy + route_xy_all
    if all_xy:
        padding = 500.0
        xmin = min(p[0] for p in all_xy) - padding
        xmax = max(p[0] for p in all_xy) + padding
        ymin = min(p[1] for p in all_xy) - padding
        ymax = max(p[1] for p in all_xy) + padding
        merc_bounds = (xmin, xmax, ymin, ymax)
        zoom = zoom_for_bounds(*merc_bounds)
        add_amap_basemap(ax, merc_bounds, zoom)

        for mode in MODE_LIST:
            xs = []
            ys = []
            for cell in mode_cells.get(mode, []):
                merc = cell_to_mercator(cell, lonlat_index)
                if merc is None:
                    continue
                if xmin <= merc[0] <= xmax and ymin <= merc[1] <= ymax:
                    xs.append(merc[0])
                    ys.append(merc[1])
            if xs:
                ax.scatter(
                    xs,
                    ys,
                    s=3,
                    color=MODE_COLORS.get(mode, "gray"),
                    alpha=0.18,
                    marker="s",
                    linewidths=0,
                    label=f"road_{mode}",
                    zorder=1,
                )

    if signal_xy:
        ax.plot(
            [p[0] for p in signal_xy],
            [p[1] for p in signal_xy],
            color="black",
            linestyle="--",
            linewidth=1.2,
            marker="o",
            markersize=3,
            alpha=0.65,
            label="signal",
            zorder=4,
        )

    matched_xy = []
    for row in matched_rows:
        merc = cell_to_mercator((row["matched_q"], row["matched_r"], row["matched_s"]), lonlat_index)
        if merc is not None:
            matched_xy.append(merc)
    if matched_xy:
        ax.scatter(
            [p[0] for p in matched_xy],
            [p[1] for p in matched_xy],
            s=16,
            color="dimgray",
            alpha=0.75,
            label="matched",
            zorder=5,
        )

    used_labels = set()
    current_mode = None
    current_xy = []
    for row in route_rows:
        merc = cell_to_mercator((row["q"], row["r"], row["s"]), lonlat_index)
        if merc is None:
            continue
        mode = row["mode"]
        if current_mode is None:
            current_mode = mode
            current_xy = [merc]
            continue
        if mode == current_mode:
            current_xy.append(merc)
            continue
        if len(current_xy) >= 2:
            label = current_mode if current_mode not in used_labels else None
            ax.plot(
                [p[0] for p in current_xy],
                [p[1] for p in current_xy],
                color=MODE_COLORS.get(current_mode, "gray"),
                linewidth=2.4,
                alpha=0.9,
                label=label,
                zorder=6,
            )
            used_labels.add(current_mode)
        current_mode = mode
        current_xy = [current_xy[-1], merc] if current_xy else [merc]

    if current_mode is not None and len(current_xy) >= 2:
        label = current_mode if current_mode not in used_labels else None
        ax.plot(
            [p[0] for p in current_xy],
            [p[1] for p in current_xy],
            color=MODE_COLORS.get(current_mode, "gray"),
            linewidth=2.4,
            alpha=0.9,
            label=label,
            zorder=6,
        )

    if signal_xy:
        ax.scatter(signal_xy[0][0], signal_xy[0][1], s=70, c="lime", edgecolors="black", label="start", zorder=8)
        ax.scatter(signal_xy[-1][0], signal_xy[-1][1], s=70, c="orange", marker="X", edgecolors="black", label="end", zorder=8)

    ax.set_title(f"Composite route ID={safe_id}")
    ax.set_xlabel("Web Mercator X")
    ax.set_ylabel("Web Mercator Y")
    ax.grid(False)
    ax.legend(fontsize=8, loc="best")
    ax.axis("off")

    png_path = os.path.join(output_dir, f"{safe_id}_route.png")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return png_path


def build_output_rows(target_id, signal_points, match_result, output_dir, lonlat_index, mode_cells):
    os.makedirs(output_dir, exist_ok=True)
    safe_id = normalize_id(target_id)

    matched_rows = []
    matched = match_result["matched_candidates"]
    for idx, (signal, cand) in enumerate(zip(signal_points, matched)):
        matched_rows.append(
            {
                "signal_index": idx,
                "signal_q": signal[0],
                "signal_r": signal[1],
                "signal_s": signal[2],
                "matched_q": cand["q"],
                "matched_r": cand["r"],
                "matched_s": cand["s"],
                "snap_dist": cand["snap_dist"],
            }
        )

    route_rows = flatten_route(match_result["segment_paths"])
    png_path = plot_route_png(
        target_id,
        signal_points,
        matched_rows,
        route_rows,
        lonlat_index,
        mode_cells,
        output_dir,
    )

    matched_mode_switches = 0
    for prev, cur in zip(matched, matched[1:]):
        if prev["mode"] != cur["mode"]:
            matched_mode_switches += 1

    total_snap_dist = sum(c["snap_dist"] for c in matched)

    combined_rows = []
    for row in matched_rows:
        combined_rows.append({"record_type": "matched_point", "ID": safe_id, **row})

    for row in route_rows:
        combined_rows.append(
            {
                "record_type": "route_cell",
                "ID": safe_id,
                "segment_index": row["segment_index"],
                "step_in_segment": row["step_in_segment"],
                "q": row["q"],
                "r": row["r"],
                "s": row["s"],
            }
        )

    summary_row = {
        "record_type": "summary",
        "ID": safe_id,
        "signals": len(signal_points),
        "matched_points": len(matched),
        "route_steps": len(route_rows),
        "total_cost": round(match_result["total_cost"], 3),
        "total_snap_dist": total_snap_dist,
        "total_move_len": match_result["total_move_len"],
        "total_transition_cost": round(match_result["total_transition_cost"], 3),
        "total_detour_cost": round(match_result["total_detour_cost"], 3),
        "matched_mode_switches": matched_mode_switches,
        "path_mode_switches": match_result["total_path_switches"],
        "unreachable_segments": match_result["unreachable_count"],
        "unreachable_candidate_segments": match_result["unreachable_candidate_segments"],
        "visualization": png_path,
    }
    combined_rows.append(summary_row)

    route_cells = [(row["q"], row["r"], row["s"]) for row in route_rows]
    return {
        "combined_rows": combined_rows,
        "route_cells": route_cells,
        "png_path": png_path,
    }


def match_id(args):
    mode_cells, cell_modes, lonlat_index = load_composite_road(args.map_path)
    df = pd.read_csv(args.input_csv)
    return match_one_id(args.id, df, mode_cells, cell_modes, lonlat_index, args)


def match_one_id(target_id, df, mode_cells, cell_modes, lonlat_index, args):
    signal_points, _ = extract_signal_points(df, target_id)

    candidates_by_point = []
    for idx, point in enumerate(signal_points):
        candidates = find_candidates(point, cell_modes, k=args.k, max_radius=args.max_snap_radius)
        if not candidates:
            raise RuntimeError(f"No road candidates found near signal index {idx}: {point}")
        candidates_by_point.append(candidates)

    result = viterbi_match(signal_points, candidates_by_point, mode_cells, cell_modes)
    return build_output_rows(target_id, signal_points, result, args.output_dir, lonlat_index, mode_cells)


def list_ids(input_csv):
    df = pd.read_csv(input_csv, usecols=["ID"])
    ids = []
    seen = set()
    for value in df["ID"]:
        tid = normalize_id(value)
        if tid in seen:
            continue
        seen.add(tid)
        ids.append(tid)
    return ids


def grid_key(cell):
    return f"{int(cell[0])},{int(cell[1])},{int(cell[2])}"


def write_aggregate_outputs(results, failed, args):
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    freq = Counter()

    for tid, result in results:
        all_rows.extend(result["combined_rows"])
        for cell in result["route_cells"]:
            freq[grid_key(cell)] += 1

    for tid, reason in failed:
        all_rows.append({"record_type": "failed", "ID": tid, "error": reason})

    all_routes_path = os.path.join(args.output_dir, "all_routes.csv")
    pd.DataFrame(all_rows).to_csv(all_routes_path, index=False, encoding="utf-8")

    freq_rows = [
        {"grid": grid, "frequency": count}
        for grid, count in sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    ]
    freq_path = os.path.join(args.output_dir, "grid_frequency.csv")
    pd.DataFrame(freq_rows).to_csv(freq_path, index=False, encoding="utf-8")

    return {
        "all_routes": all_routes_path,
        "grid_frequency": freq_path,
    }


def match_all_ids(args):
    mode_cells, cell_modes, lonlat_index = load_composite_road(args.map_path)
    df = pd.read_csv(args.input_csv)
    ids = []
    seen = set()
    for value in df["ID"]:
        tid = normalize_id(value)
        if tid in seen:
            continue
        seen.add(tid)
        ids.append(tid)

    outputs = []
    failed = []
    for idx, tid in enumerate(ids, start=1):
        print(f"[{idx}/{len(ids)}] matching ID={tid}")
        try:
            outputs.append((tid, match_one_id(tid, df, mode_cells, cell_modes, lonlat_index, args)))
        except Exception as exc:
            failed.append((tid, str(exc)))
            print(f"  [FAILED] ID={tid}: {exc}")

    paths = write_aggregate_outputs(outputs, failed, args)
    return outputs, failed, paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match one ID's signal points on a composite hex road network."
    )
    parser.add_argument("id", nargs="?", default=None, help="Trajectory ID to match. Omit to match all IDs.")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--map-path", default=DEFAULT_MAP_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--max-snap-radius", type=int, default=DEFAULT_MAX_SNAP_RADIUS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.id is None:
        outputs, failed, paths = match_all_ids(args)
        print(f"all routes: {paths['all_routes']}")
        print(f"grid frequency: {paths['grid_frequency']}")
        print(f"success: {len(outputs)}")
        print(f"failed: {len(failed)}")
        return

    result = match_id(args)
    paths = write_aggregate_outputs([(normalize_id(args.id), result)], [], args)
    print(f"visualization: {result['png_path']}")
    print(f"all routes: {paths['all_routes']}")
    print(f"grid frequency: {paths['grid_frequency']}")


if __name__ == "__main__":
    main()
