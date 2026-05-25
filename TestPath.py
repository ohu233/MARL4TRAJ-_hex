import os
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from utils.SoftActorCritic import DiscreteSACAgent, SACConfig
from utils.Environment import PathEnv
from utils.tools import state_to_vector, calculate_match_rate
from utils.hex_utils import hex_distance

# ========== 配置 ==========
TRAJ_CSV = 'data/artificial_od_all.csv'
MODEL_PATH = "PathModel\sac_actor_ep2000_withGNN_withCurri.pth"
SAVE_DIR = "TestPath_results"
FOV = 3
USE_GNN = True
MAX_STEPS = 300
SAVE_FIGURES = True

# True: 测试时使用 row['mode'] 作为唯一选中模式
# False: 保持环境原有随机 mode 采样
USE_ROW_MODE_FROM_DATA = True

MODE_COLORS = {"TG": "orange", "GG": "blue", "GSD": "green", "TS": "red"}

# ============================================================
# 高德地图瓦片底图
# ============================================================
import math
import requests
from io import BytesIO
from PIL import Image
import pyproj

_WGS84 = pyproj.CRS.from_epsg(4326)
_MERC = pyproj.CRS.from_epsg(3857)
_TRANSFORMER = pyproj.Transformer.from_crs(_WGS84, _MERC, always_xy=True)

AMAP_TILE_URL = (
    "https://webrd0{s}.is.autonavi.com/appmaptile"
    "?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"
)

# hex→lonlat 缓存
_hex_lonlat_cache = {}

def build_hex_lonlat_index(hex_mapdata_raw):
    """从 hex_grid.pkl 原始数据建立 {(q,r,s): (lon, lat)} 索引。"""
    _hex_lonlat_cache.clear()
    for k, v in hex_mapdata_raw.items():
        _hex_lonlat_cache[tuple(int(c) for c in k)] = (v['lon'], v['lat'])

def hex_to_lonlat(q, r, s):
    """cube 坐标 → WGS84 (lon, lat)，找不到则返回 None。"""
    key = (int(round(q)), int(round(r)), int(round(s)))
    return _hex_lonlat_cache.get(key)

def hex_to_mercator(q, r, s):
    """cube 坐标 → Web Mercator (mx, my)。"""
    ll = hex_to_lonlat(q, r, s)
    if ll is None:
        return None
    mx, my = _TRANSFORMER.transform(ll[0], ll[1])
    return mx, my

def lonlat_to_tile(lon, lat, zoom):
    """WGS84 → 高德瓦片 (tx, ty)。"""
    tx = int((lon + 180) / 360 * (1 << zoom))
    lat_rad = math.radians(lat)
    ty = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * (1 << zoom))
    return tx, ty

def tile_to_mercator_bounds(tx, ty, zoom):
    """瓦片 (tx, ty) → Web Mercator 范围 (left, right, bottom, top)。"""
    n = 1 << zoom
    left = tx / n * 40075016.68557849 - 20037508.342789244
    right = (tx + 1) / n * 40075016.68557849 - 20037508.342789244
    top = 20037508.342789244 - ty / n * 40075016.68557849
    bottom = 20037508.342789244 - (ty + 1) / n * 40075016.68557849
    return left, right, bottom, top

def fetch_amap_tile(tx, ty, zoom):
    """下载单张高德瓦片，返回 PIL Image 或 None。"""
    import random
    url = AMAP_TILE_URL.format(s=random.randint(1, 4), x=tx, y=ty, z=zoom)
    try:
        resp = requests.get(url, timeout=5, headers={
            "User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content))
    except Exception:
        pass
    return None

def add_amap_basemap(ax, merc_bounds, zoom=13):
    """
    在 matplotlib axes 上叠加高德瓦片底图。
    merc_bounds: (xmin, xmax, ymin, ymax) Web Mercator 坐标。
    """
    xmin, xmax, ymin, ymax = merc_bounds
    merc_to_wgs = pyproj.Transformer.from_crs(_MERC, _WGS84, always_xy=True)

    # 左上角 → 瓦片
    tl_tx, tl_ty = lonlat_to_tile(*merc_to_wgs.transform(xmin, ymax), zoom)
    # 右下角 → 瓦片
    br_tx, br_ty = lonlat_to_tile(*merc_to_wgs.transform(xmax, ymin), zoom)

    # 在 mercator 坐标系下拼瓦片
    for ty in range(tl_ty, br_ty + 1):
        for tx in range(tl_tx, br_tx + 1):
            t_left, t_right, t_bottom, t_top = tile_to_mercator_bounds(tx, ty, zoom)
            tile_img = fetch_amap_tile(tx, ty, zoom)
            if tile_img is not None:
                ax.imshow(tile_img, extent=[t_left, t_right, t_bottom, t_top],
                          aspect='auto', alpha=0.85, interpolation='bilinear')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')


def _compute_merc_bounds(traj_list, padding_meters=200):
    """根据 hex 轨迹列表计算 Web Mercator 范围。"""
    all_mx, all_my = [], []
    for traj in traj_list:
        for p in traj:
            merc = hex_to_mercator(p[0], p[1], p[2])
            if merc is not None:
                all_mx.append(merc[0])
                all_my.append(merc[1])
    if not all_mx:
        return 0, 1, 0, 1
    return (min(all_mx) - padding_meters, max(all_mx) + padding_meters,
            min(all_my) - padding_meters, max(all_my) + padding_meters)


def _get_zoom_for_bounds(xmin, xmax, ymin, ymax, fig_width_px=600):
    """根据范围自动选合适的缩放级别。"""
    span = max(xmax - xmin, ymax - ymin)
    if span <= 0:
        return 15
    # 每像素约覆盖的米数
    m_per_px = span / fig_width_px
    # 经验公式
    zoom = int(np.log2(40075016.68557849 / (256 * m_per_px))) - 1
    return max(10, min(17, zoom))


def plot_trajectory(traj_hex, hex_end, mode_str, selected_mode, save_path,
                    success_flag, match_rate, trans_count):
    """绘制单条 hex 轨迹，高德瓦片底图。"""
    # hex → mercator
    mxs, mys = [], []
    for p in traj_hex:
        merc = hex_to_mercator(p[0], p[1], p[2])
        if merc is not None:
            mxs.append(merc[0])
            mys.append(merc[1])
    end_merc = hex_to_mercator(hex_end[0], hex_end[1], hex_end[2])

    if not mxs:
        return

    padding = 300
    merc_bounds = (min(mxs) - padding, max(mxs) + padding,
                   min(mys) - padding, max(mys) + padding)

    color = MODE_COLORS.get(mode_str, "C0")
    fig, ax = plt.subplots(figsize=(8, 7))

    # 高德底图
    zoom = _get_zoom_for_bounds(*merc_bounds)
    add_amap_basemap(ax, merc_bounds, zoom=zoom)

    ax.plot(mxs, mys, marker='o', markersize=3, color=color,
            linewidth=2, alpha=0.9, zorder=4)
    ax.scatter(mxs[0], mys[0], c=color, marker='o', s=100,
               edgecolors='red', linewidths=2, zorder=6, label='start')
    if end_merc is not None:
        ax.scatter(end_merc[0], end_merc[1], c=color, marker='X', s=120,
                   edgecolors='black', linewidths=1.5, zorder=6, label='goal')
    ax.scatter(mxs[-1], mys[-1], c='black', marker='^', s=80,
               linewidths=1.5, zorder=6, label='agent_end')

    ax.set_title(f"Mode={mode_str}, Sel={selected_mode}, "
                 f"Succ={success_flag}, Match={match_rate:.2f}, Trans={trans_count}")
    ax.legend(fontsize=7, loc='upper right')
    ax.axis('off')

    if SAVE_FIGURES:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_combined_for_id(items, tid, save_dir):
    """为单个 ID 生成轨迹聚合图，高德瓦片底图。"""
    combined_dir = os.path.join(save_dir, "combined_by_id")
    os.makedirs(combined_dir, exist_ok=True)

    all_trajs = [item["traj"] for item in items]
    merc_bounds = _compute_merc_bounds(all_trajs)

    fig, ax = plt.subplots(figsize=(10, 9))
    zoom = _get_zoom_for_bounds(*merc_bounds)
    add_amap_basemap(ax, merc_bounds, zoom=zoom)

    prev_end = None
    for item in items:
        traj = item["traj"]
        mxs, mys = [], []
        for p in traj:
            merc = hex_to_mercator(p[0], p[1], p[2])
            if merc is not None:
                mxs.append(merc[0])
                mys.append(merc[1])

        color = MODE_COLORS.get(item["mode"], "C0")
        ax.plot(mxs, mys, marker='o', markersize=2, color=color,
                linewidth=1.5, alpha=0.8, zorder=4)

        if mxs:
            ax.scatter(mxs[0], mys[0], c=color, marker='o', s=60, zorder=5)
            ax.scatter(mxs[-1], mys[-1], c=color, marker='x', s=60, zorder=5)

        if prev_end is not None:
            pe = hex_to_mercator(prev_end[0], prev_end[1], prev_end[2])
            ss = hex_to_mercator(item["start_hex"][0], item["start_hex"][1],
                                 item["start_hex"][2])
            if pe is not None and ss is not None:
                ax.plot([pe[0], ss[0]], [pe[1], ss[1]], linestyle="--",
                        color="gray", linewidth=1, alpha=0.7, zorder=3)
        prev_end = item["end_hex"]

    # 首段起点和末段终点高亮
    first = items[0]
    last = items[-1]
    fs = hex_to_mercator(first["start_hex"][0], first["start_hex"][1],
                          first["start_hex"][2])
    le = hex_to_mercator(last["end_hex"][0], last["end_hex"][1],
                          last["end_hex"][2])
    if fs is not None:
        ax.scatter(fs[0], fs[1], c=MODE_COLORS.get(first["mode"], "C0"),
                   marker='o', s=140, edgecolors='red', linewidths=2,
                   zorder=6, label='start')
    if le is not None:
        ax.scatter(le[0], le[1], c=MODE_COLORS.get(last["mode"], "C0"),
                   marker='X', s=160, edgecolors='black', linewidths=2,
                   zorder=6, label='end')

    ax.set_title(f"ID={tid}, segments={len(items)}")
    ax.legend(fontsize=8, loc='upper right')
    ax.axis('off')

    save_path = os.path.join(combined_dir, f"{tid}.png")
    if SAVE_FIGURES:
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[Combined] ID={tid}, {len(items)} segments -> {save_path}")


def load_env(traj_df, use_row_mode_from_data: bool = False, fov: int = 3):
    from utils.hex_utils import load_hex_mapdata_raw
    mapdata = load_hex_mapdata_raw('data/hex_grid.pkl')
    build_hex_lonlat_index(mapdata)
    env = PathEnv(
        train_mode=not use_row_mode_from_data,
        mapdata=mapdata,
        traj=traj_df,
        FOV=fov,
        distance_threshold=1.0,
    )
    return env


def load_agent(env, model_path: str, use_gnn: bool = True):
    cfg = SACConfig(device="cpu")
    device = torch.device(cfg.device)
    agent = DiscreteSACAgent(vec_dim=20, hex_radius=env.FOV, action_dim=6,
                              cfg=cfg, use_gnn=use_gnn, in_channels=5)
    state_dict = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()
    env.traj_cnt = 0
    return agent


def run_eval(env, agent, traj_df, max_steps: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    episodes = len(traj_df)
    records = []
    success_list = []
    reward_list = []
    match_list = []
    trans_list = []

    # ID 聚合图缓存
    current_id = None
    id_buffer = []

    def _flush_id_buffer():
        if id_buffer and current_id is not None:
            plot_combined_for_id(id_buffer, current_id, save_dir)
            id_buffer.clear()

    for ep in range(episodes):
        row = traj_df.iloc[ep]
        row_id = str(row['ID'])
        real_mode = str(row['mode']).strip()
        hex_end = (float(row['locxd']), float(row['locyd']), float(row['loczd']))

        if current_id is not None and row_id != current_id:
            _flush_id_buffer()
        current_id = row_id

        if USE_ROW_MODE_FROM_DATA:
            env.selected_mode = np.array([real_mode])

        state = env.reset()
        hex_start = env.hex_start
        total_reward = 0.0
        success_flag = 0
        step_count = 0

        # 记录轨迹（hex cube 坐标）
        traj = [hex_start]

        for _ in range(max_steps):
            s_vec = state_to_vector(state)
            a = agent.select_action(s_vec, evaluate=True)
            state, r, done, succ = env.step(int(a))
            total_reward += float(r)
            step_count += 1
            traj.append(env.hex_start)

            if succ:
                success_flag = 1

            if done:
                break

        final_dist = hex_distance(env.hex_start, hex_end)
        match_rate = calculate_match_rate(traj, env.multi_mapdata)

        success_list.append(success_flag)
        reward_list.append(total_reward)
        match_list.append(match_rate)
        trans_list.append(env.min_trans_count)

        records.append({
            "episode": ep,
            "ID": row_id,
            "real_mode": real_mode,
            "selected_mode": "+".join(env.selected_mode),
            "reward": float(total_reward),
            "success": success_flag,
            "match_rate": float(match_rate),
            "min_trans": int(env.min_trans_count),
            "steps": step_count,
            "final_dist": float(final_dist),
            "start_hex": hex_start,
            "end_hex": env.hex_start,
            "traj": traj,
            "mode": real_mode,
        })

        id_buffer.append(records[-1])

        # ====== 画单条轨迹 ======
        if SAVE_FIGURES:
            ep_dir = os.path.join(save_dir, "episodes")
            os.makedirs(ep_dir, exist_ok=True)
            fname = (f"{row_id}_{ep:04d}_succ{success_flag}"
                     f"_match{match_rate:.2f}.png")
            plot_trajectory(traj, hex_end, real_mode,
                            "+".join(env.selected_mode),
                            os.path.join(ep_dir, fname),
                            success_flag, match_rate,
                            env.min_trans_count)

        if (ep + 1) % 100 == 0:
            print(f"[{ep+1}/{episodes}] "
                  f"avg_reward={np.mean(reward_list[-100:]):.1f}, "
                  f"succ_rate={np.mean(success_list[-100:])*100:.1f}%, "
                  f"match_rate={np.mean(match_list[-100:])*100:.1f}%, "
                  f"avg_trans={np.mean(trans_list[-100:]):.1f}")

    _flush_id_buffer()

    # ====== 保存 CSV（traj 转 json 字符串） ======
    df = pd.DataFrame(records)
    df["traj"] = df["traj"].apply(
        lambda t: json.dumps([[float(p[0]), float(p[1]), float(p[2])] for p in t],
                             ensure_ascii=False))
    csv_path = os.path.join(save_dir, "traj_records.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved traj records to: {csv_path}")

    # ====== 汇总 ======
    print("=" * 60)
    print(f"Episodes    : {episodes}")
    print(f"Avg reward  : {np.mean(reward_list):.3f}")
    print(f"Success rate: {np.mean(success_list) * 100:.2f}%")
    print(f"Avg match   : {np.mean(match_list) * 100:.2f}%")
    print(f"Avg trans   : {np.mean(trans_list):.1f}")
    print(f"Avg steps   : {np.mean([r['steps'] for r in records]):.1f}")
    print("=" * 60)

    # ====== 按真实 mode 分组统计 ======
    df['success'] = success_list
    df['match'] = match_list
    print("\n=== Per-mode stats ===")
    for m in ['TG', 'GG', 'GSD', 'TS']:
        sub = df[df['real_mode'] == m]
        if len(sub) > 0:
            print(f"  {m}: count={len(sub)}, "
                  f"succ={sub['success'].mean()*100:.1f}%, "
                  f"match={sub['match'].mean()*100:.1f}%")

    return df


if __name__ == "__main__":
    traj_df = pd.read_csv(TRAJ_CSV)
    print(f"Loaded {len(traj_df)} trajectories from {TRAJ_CSV}")

    env = load_env(traj_df, use_row_mode_from_data=USE_ROW_MODE_FROM_DATA, fov=FOV)

    print(f"Loading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        # 尝试找最新模型
        import glob
        candidates = sorted(glob.glob("PathModel/sac_actor_ep*.pth"))
        if candidates:
            MODEL_PATH = candidates[-1]
            print(f"Using latest model: {MODEL_PATH}")
        else:
            print("No trained model found. Train first (TrainPath.py), or place a .pth in PathModel/")
            exit(1)

    agent = load_agent(env, MODEL_PATH, use_gnn=USE_GNN)
    print(f"Running evaluation, saving to {SAVE_DIR}/")
    run_eval(env, agent, traj_df, max_steps=MAX_STEPS, save_dir=SAVE_DIR)
