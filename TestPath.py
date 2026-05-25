import os
import pickle
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

from utils.SoftActorCritic import (
    DiscreteSACAgent,
    SACConfig,
)
from utils.Environment import PathEnv
from utils.tools import state_to_vector, calculate_match_rate, plt_multi_map
from utils.geo_utils import grid_to_mercator, grid_bounds_to_mercator
from utils.basemap import add_osm_basemap

# ========== 配置 ==========
traj_test = pd.read_csv('data\data_lower_test_filtered.csv')
EPISODES = len(traj_test)
MAX_STEPS = 300
MODEL_PATH = "PathModel\sac_actor_ep5000_withConv_withCurri.pth"
SAVE_DIR = None
FOV = 3
USE_GNN = True

# True: 测试时每个 episode 使用 row['mode']，不随机
# False: 保持环境原有随机 mode 采样
USE_ROW_MODE_FROM_DATA = True

# True: 使用 OSM 瓦片底图（需联网，坐标用 Web Mercator）
# False: 使用 figure 文件夹路网图作为底图（离线，坐标用 grid）
USE_OSM_BASEMAP = False
SAVE_FIGURES = True
# ========== figure 底图相关（仅 USE_OSM_BASEMAP=False 时生效） ==========
MAP_ROW, MAP_COL = 529, 564
FIGURE_DIR = "figure"
DEFAULT_BACKIMG_PATH = "figur/all_modes_js.png"

MODE_COLORS = {
    "TG": "purple",
    "GG": "blue",
    "GSD": "green",
    "TS": "red",
}

MODE_ORDER = ["TG", "GG", "GSD", "TS"]


def _normalize_modes(selected_mode, fallback_mode=None):
    """将 selected_mode 统一为排序后的 mode 列表，与 figure 文件命名一致。"""
    if selected_mode is None:
        modes = [fallback_mode] if fallback_mode is not None else []
    elif isinstance(selected_mode, (list, tuple, np.ndarray, set)):
        modes = [str(m).strip() for m in selected_mode]
    else:
        modes = [str(selected_mode).strip()]

    valid = [m for m in modes if m in MODE_ORDER]
    if not valid and fallback_mode is not None and fallback_mode in MODE_ORDER:
        valid = [fallback_mode]

    return sorted(set(valid), key=lambda x: MODE_ORDER.index(x))


def _load_mode_background(selected_mode, fallback_mode=None, cache=None):
    """加载 mode 对应的底图，带缓存。"""
    if cache is None:
        cache = {}

    modes = _normalize_modes(selected_mode, fallback_mode=fallback_mode)
    key = tuple(modes) if modes else ("__default__",)

    if key in cache:
        return cache[key]

    if modes:
        fig_name = "+".join(modes) + ".png"
        candidate = os.path.join(FIGURE_DIR, fig_name)
        if os.path.exists(candidate):
            cache[key] = mpimg.imread(candidate)
            return cache[key]

    cache[key] = mpimg.imread(DEFAULT_BACKIMG_PATH)
    print(f"[WARN] 背景图不存在，使用默认底图: modes={modes}")
    return cache[key]


def mode_legend_handles():
    return [
        Line2D([0], [0], color="purple", lw=2, label="TG"),
        Line2D([0], [0], color="blue", lw=2, label="GG"),
        Line2D([0], [0], color="green", lw=2, label="GSD"),
        Line2D([0], [0], color="red", lw=2, label="TS"),
    ]


def load_env(traj_df, use_row_mode_from_data: bool = False, fov: int = 3):
    """和训练时保持一致的 PathEnv 配置。"""
    from utils.hex_utils import load_hex_mapdata_raw
    mapdata = load_hex_mapdata_raw('data/hex_grid.pkl')

    env = PathEnv(
        train_mode=not use_row_mode_from_data,
        mapdata=mapdata,
        traj=traj_df,
        FOV=fov,
        distance_threshold=1.0,
    )
    return env


def load_agent(env, model_path: str, use_gnn: bool = True):
    """创建 DiscreteSACAgent，并加载 actor 权重。"""
    cfg = SACConfig(device="cpu")
    device = torch.device(cfg.device)

    agent = DiscreteSACAgent(vec_dim=20, hex_radius=env.FOV, action_dim=6,
                              cfg=cfg, use_gnn=use_gnn, in_channels=5)
    state_dict = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()

    env.traj_cnt = 0
    return agent


def run_eval_with_plots(env, agent, traj_df, episodes: int,
                        max_steps: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    all_trajs = []
    ep_rewards = []
    ep_success_flags = []
    ep_trans = []
    traj_records = []

    bg_cache = {}

    current_id = None
    id_buffer = []

    def _flush_id_buffer():
        if id_buffer:
            _plot_combined_for_id(id_buffer, current_id, save_dir)
            id_buffer.clear()

    for ep in range(episodes):
        row_idx = ep
        if row_idx >= len(traj_df):
            break
        row = traj_df.iloc[row_idx]
        row_id = str(row['ID'])

        if current_id is not None and row_id != current_id:
            _flush_id_buffer()
        current_id = row_id

        mode = str(row['mode']).strip()

        if USE_ROW_MODE_FROM_DATA:
            env.selected_mode = np.array([mode], dtype=object)

        state = env.reset()

        delta = np.array([row['locx_o'], row['locy_o']], dtype=float)
        end_xy = np.array([row['locx_d'], row['locy_d']], dtype=float)
        mode = row['mode']

        grid_pos = np.array(state['current_position'], dtype=float)
        actual_pos = grid_pos + delta
        traj = [actual_pos.copy()]
        total_reward = 0.0
        success_flag = 0

        for t in range(max_steps):
            s_vec = state_to_vector(state)
            a = agent.select_action(s_vec, evaluate=True)
            next_state, r, done, success = env.step(int(a))

            grid_pos = np.array(next_state['current_position'], dtype=float)
            actual_pos = grid_pos + delta
            traj.append(actual_pos.copy())

            total_reward += float(r)
            if success:
                success_flag = 1

            state = next_state

            if done:
                break

        traj_arr = np.array(traj)
        print(calculate_match_rate(traj_arr.tolist(), env.multi_mapdata))

        item = {
            "traj": traj_arr,
            "mode": str(mode),
            "id": str(row['ID']),
            "start_xy": traj_arr[0].copy(),
            "end_xy": end_xy.copy(),
        }
        all_trajs.append(item)
        id_buffer.append(item)
        ep_rewards.append(total_reward)
        ep_success_flags.append(success_flag)
        ep_trans.append(env.min_trans_count)

        traj_list = [[float(p[0]), float(p[1])] for p in traj_arr]
        traj_records.append({
            "episode": ep,
            "order": int(row['order']) if 'order' in row else int(row_idx),
            "reward": float(total_reward),
            "success": int(success_flag),
            "match": calculate_match_rate(traj_arr.tolist(), env.multi_mapdata),
            "min_trans": int(env.min_trans_count),
            "mode": 1 if mode in env.selected_mode else 0,
            "traj": json.dumps(traj_list, ensure_ascii=False),
        })

        # ====== 画当前 episode 的轨迹图 ======
        try:
            mode_str = str(mode).strip()
            xs = traj_arr[:, 0]
            ys = traj_arr[:, 1]
            x_min_local, x_max_local = xs.min() - 2, xs.max() + 2
            y_min_local, y_max_local = ys.min() - 2, ys.max() + 2

            order_idx = int(row['order']) if 'order' in row else ep
            if USE_OSM_BASEMAP:
                _plot_episode_osm(traj_arr, end_xy, mode_str, row_id, order_idx,
                                  success_flag,
                                  x_min_local, x_max_local, y_min_local, y_max_local,
                                  env, save_dir)
            else:
                _plot_episode_figure(traj_arr, end_xy, mode_str, row_id, order_idx,
                                     success_flag,
                                     x_min_local, x_max_local, y_min_local, y_max_local,
                                     env, bg_cache, save_dir)
        except Exception as e:
            print(f"Error plotting ID={row_id} order={order_idx}: {e}")

    _flush_id_buffer()

    # ====== 保存 CSV ======
    df = pd.DataFrame(traj_records)
    csv_path = os.path.join(save_dir, "traj_records.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved traj records CSV to: {csv_path}")

    print("=" * 60)
    print("Avg reward   : {:.3f}".format(np.mean(ep_rewards)))
    print("Success rate : {:.2f}%".format(np.mean(ep_success_flags) * 100))
    print("Avg trans    : {:.1f}".format(np.mean(ep_trans)))
    print("=" * 60)

    return all_trajs


# ============================================================
# OSM 底图版本：坐标转 Web Mercator，叠加 OSM 瓦片
# ============================================================
def _plot_episode_osm(traj_arr, end_xy, mode_str, row_id, idx, success_flag,
                       x_min_local, x_max_local, y_min_local, y_max_local,
                       env, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))

    mxmin, mxmax, mymin, mymax = grid_bounds_to_mercator(
        x_min_local, x_max_local, y_min_local, y_max_local
    )
    ax.set_xlim(mxmin, mxmax)
    ax.set_ylim(mymin, mymax)

    add_osm_basemap(ax, alpha=0.8)
    ax.set_aspect("equal")

    traj_color = MODE_COLORS.get(mode_str, "C0")
    merc_x, merc_y = grid_to_mercator(traj_arr[:, 0], traj_arr[:, 1])
    ax.plot(merc_x, merc_y, color=traj_color, alpha=0.35, linewidth=1.5)

    start_mx, start_my = grid_to_mercator(traj_arr[0, 0], traj_arr[0, 1])
    end_mx, end_my = grid_to_mercator(end_xy[0], end_xy[1])
    agent_end_mx, agent_end_my = grid_to_mercator(traj_arr[-1, 0], traj_arr[-1, 1])

    ax.scatter(start_mx, start_my,
               c=traj_color, marker='o', s=18,
               edgecolors='red', linewidths=1.5, label='start')
    ax.scatter(end_mx, end_my,
               c=traj_color, marker='x', s=22,
               linewidths=1.5, label='end')
    ax.scatter(agent_end_mx, agent_end_my,
               c='black', marker='^', s=15, linewidths=1, label='agent_end')

    start_handle = Line2D([0], [0], marker='o', color='w', label='start',
                  markerfacecolor=traj_color, markeredgecolor='red',
                  markersize=5, linewidth=0)
    end_handle = Line2D([0], [0], marker='x', color=traj_color, label='end',
                markersize=5, linewidth=0)

    ax.legend(handles=mode_legend_handles() + [start_handle, end_handle], loc='best',
              fontsize=7)

    ax.set_xlabel("Web Mercator X")
    ax.set_ylabel("Web Mercator Y")
    ax.set_title(
        f"ID={row_id}, idx={idx}, "
        f"Succ={success_flag == 1}, Mode={mode_str}, "
        f"Selected Mode={env.selected_mode}, Trans={env.min_trans_count}"
    )

    filename = (
        f"{row_id}_{idx:02d}"
        f"_succ_{success_flag}"
        f"_match_{calculate_match_rate(traj_arr.tolist(), env.multi_mapdata):.2f}"
        ".png"
    )
    ep_path = os.path.join(save_dir, filename)
    if SAVE_FIGURES:
        plt.savefig(ep_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"[ID={row_id} idx={idx}] saved: {ep_path}")


# ============================================================
# figure 底图版本：grid 坐标直接绘图，从 figure 中切片底图
# ============================================================
def _plot_episode_figure(traj_arr, end_xy, mode_str, row_id, idx, success_flag,
                          x_min_local, x_max_local, y_min_local, y_max_local,
                          env, bg_cache, save_dir):
    episode_bg = _load_mode_background(
        selected_mode=getattr(env, "selected_mode", None),
        fallback_mode=mode_str,
        cache=bg_cache,
    )

    height, width = episode_bg.shape[0], episode_bg.shape[1]
    ratio = ((height / MAP_ROW) * (width / MAP_COL)) ** 0.5

    x_min_idx = int(max(0, x_min_local * ratio))
    x_max_idx = int(min(width, x_max_local * ratio))
    y_min_idx = int(max(0, y_min_local * ratio))
    y_max_idx = int(min(height, y_max_local * ratio))

    sliced_img = episode_bg[
        height - y_max_idx: height - y_min_idx,
        x_min_idx:x_max_idx
    ]

    plt.figure(figsize=(6, 5))
    plt.imshow(
        sliced_img,
        extent=[x_min_local, x_max_local, y_min_local, y_max_local],
        alpha=0.5
    )

    traj_color = MODE_COLORS.get(mode_str, "C0")
    plt.plot(traj_arr[:, 0], traj_arr[:, 1],
             marker='o', markersize=2, color=traj_color)

    plt.scatter(traj_arr[0, 0], traj_arr[0, 1],
                c=traj_color, marker='o', s=100,
                edgecolors='red', linewidths=2, label='start')
    plt.scatter(end_xy[0], end_xy[1],
                c=traj_color, marker='x', s=120,
                linewidths=2, label='end')

    agent_end = traj_arr[-1]
    plt.scatter(agent_end[0], agent_end[1],
                c='black', marker='^', s=60, linewidths=1.5, label='agent_end')

    start_handle = Line2D([0], [0], marker='o', color='w', label='start',
                  markerfacecolor=traj_color, markeredgecolor='red',
                  markersize=8, linewidth=0)
    end_handle = Line2D([0], [0], marker='x', color=traj_color, label='end',
                markersize=8, linewidth=0)

    plt.legend(handles=mode_legend_handles() + [start_handle, end_handle], loc='best')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(
        f"ID={row_id}, idx={idx}, "
        f"Succ={success_flag == 1}, Mode={mode_str}, "
        f"Selected Mode={env.selected_mode}, Trans={env.min_trans_count}"
    )
    plt.grid(True)

    filename = (
        f"{row_id}_{idx:02d}"
        f"_succ_{success_flag}"
        f"_match_{calculate_match_rate(traj_arr.tolist(), env.multi_mapdata):.2f}"
        ".png"
    )
    ep_path = os.path.join(save_dir, filename)
    if SAVE_FIGURES:
        plt.savefig(ep_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"[ID={row_id} idx={idx}] saved: {ep_path}")


def _plot_combined_for_id(items, tid, save_dir):
    """为单个 ID 生成轨迹聚合图。"""
    combined_dir = os.path.join(save_dir, "combined_by_id")
    os.makedirs(combined_dir, exist_ok=True)

    all_xs = []
    all_ys = []
    for item in items:
        all_xs.extend(item["traj"][:, 0].tolist())
        all_ys.extend(item["traj"][:, 1].tolist())

    x_min = min(all_xs) - 2
    x_max = max(all_xs) + 2
    y_min = min(all_ys) - 2
    y_max = max(all_ys) + 2

    if USE_OSM_BASEMAP:
        _plot_combined_osm(items, tid, x_min, x_max, y_min, y_max, combined_dir)
    else:
        _plot_combined_figure(items, tid, x_min, x_max, y_min, y_max, combined_dir)


def _plot_combined_osm(items, tid, x_min, x_max, y_min, y_max, combined_dir):
    """OSM 版本聚合图。"""
    fig, ax = plt.subplots(figsize=(8, 7))
    mxmin, mxmax, mymin, mymax = grid_bounds_to_mercator(x_min, x_max, y_min, y_max)
    ax.set_xlim(mxmin, mxmax)
    ax.set_ylim(mymin, mymax)

    add_osm_basemap(ax, alpha=0.5)
    ax.set_aspect("equal")

    prev_end = None
    for item in items:
        traj = item["traj"]
        mode_str = item["mode"]
        color = MODE_COLORS.get(mode_str, "C0")

        merc_x, merc_y = grid_to_mercator(traj[:, 0], traj[:, 1])
        ax.plot(merc_x, merc_y, color=color, alpha=0.35, linewidth=1.5)

        od_x = np.array([traj[0, 0], traj[-1, 0]])
        od_y = np.array([traj[0, 1], traj[-1, 1]])
        od_mx, od_my = grid_to_mercator(od_x, od_y)
        ax.scatter(od_mx[0], od_my[0], c=color, marker='o', s=12, zorder=4)
        ax.scatter(od_mx[1], od_my[1], c=color, marker='x', s=14, zorder=4)

        if prev_end is not None:
            seg_start = item["start_xy"]
            pmx, pmy = grid_to_mercator(prev_end[0], prev_end[1])
            smx, smy = grid_to_mercator(seg_start[0], seg_start[1])
            ax.plot([pmx, smx], [pmy, smy],
                    linestyle="--", color="gray", linewidth=0.8, alpha=0.7)
        prev_end = item["end_xy"]

    first_start = items[0]["start_xy"]
    fs_mx, fs_my = grid_to_mercator(first_start[0], first_start[1])
    ax.scatter(fs_mx, fs_my,
               c=MODE_COLORS.get(items[0]["mode"], "C0"),
               marker="o", s=18, edgecolors="red", linewidths=1.5,
               zorder=5, label="start")

    last_end = items[-1]["end_xy"]
    le_mx, le_my = grid_to_mercator(last_end[0], last_end[1])
    ax.scatter(le_mx, le_my,
               c=MODE_COLORS.get(items[-1]["mode"], "C0"),
               marker="x", s=22, linewidths=1.5, zorder=5, label="end")

    handles = mode_legend_handles()
    start_handle = Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor="gray", markeredgecolor="red",
        markersize=5, linewidth=0, label="start",
    )
    end_handle = Line2D(
        [0], [0], marker="x", color="gray",
        markersize=5, linewidth=0, label="end",
    )
    ax.legend(handles=handles + [start_handle, end_handle], loc="best", fontsize=7)

    ax.set_xlabel("Web Mercator X")
    ax.set_ylabel("Web Mercator Y")
    ax.set_title(f"ID={tid}, segments={len(items)}")

    save_path = os.path.join(combined_dir, f"{tid}.png")
    if SAVE_FIGURES:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Combined] ID={tid}, {len(items)} segments → {save_path}")


def _plot_combined_figure(items, tid, x_min, x_max, y_min, y_max, combined_dir):
    """figure 版本聚合图。"""
    used_modes = list({item["mode"] for item in items})
    bg_img = _load_mode_background(selected_mode=used_modes)
    height, width = bg_img.shape[0], bg_img.shape[1]
    ratio = ((height / MAP_ROW) * (width / MAP_COL)) ** 0.5

    x_min_idx = int(max(0, x_min * ratio))
    x_max_idx = int(min(width, x_max * ratio))
    y_min_idx = int(max(0, y_min * ratio))
    y_max_idx = int(min(height, y_max * ratio))

    sliced_img = bg_img[
        height - y_max_idx: height - y_min_idx,
        x_min_idx: x_max_idx,
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(
        sliced_img,
        extent=[x_min, x_max, y_min, y_max],
        alpha=0.5,
    )

    prev_end = None
    for item in items:
        traj = item["traj"]
        mode_str = item["mode"]
        color = MODE_COLORS.get(mode_str, "C0")

        ax.plot(
            traj[:, 0], traj[:, 1],
            marker="o", markersize=2, color=color, linewidth=1.5,
        )

        if prev_end is not None:
            seg_start = item["start_xy"]
            ax.plot(
                [prev_end[0], seg_start[0]],
                [prev_end[1], seg_start[1]],
                linestyle="--", color="gray", linewidth=0.8, alpha=0.7,
            )
        prev_end = item["end_xy"]

    first_start = items[0]["start_xy"]
    ax.scatter(
        first_start[0], first_start[1],
        c=MODE_COLORS.get(items[0]["mode"], "C0"),
        marker="o", s=120, edgecolors="red", linewidths=2,
        zorder=5, label="start",
    )

    last_end = items[-1]["end_xy"]
    ax.scatter(
        last_end[0], last_end[1],
        c=MODE_COLORS.get(items[-1]["mode"], "C0"),
        marker="x", s=120, linewidths=2, zorder=5, label="end",
    )

    handles = mode_legend_handles()
    start_handle = Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor="gray", markeredgecolor="red",
        markersize=8, linewidth=0, label="start",
    )
    end_handle = Line2D(
        [0], [0], marker="x", color="gray",
        markersize=8, linewidth=0, label="end",
    )
    ax.legend(handles=handles + [start_handle, end_handle], loc="best")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"ID={tid}, segments={len(items)}")
    ax.grid(True)

    save_path = os.path.join(combined_dir, f"{tid}.png")
    if SAVE_FIGURES:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Combined] ID={tid}, {len(items)} segments → {save_path}")


if __name__ == "__main__":
    if SAVE_DIR is None:
        base = os.path.splitext(os.path.basename(MODEL_PATH))[0]
        SAVE_DIR = os.path.join(base + "_test")

    traj_test = pd.read_csv('data\data_lower_test_filtered.csv')

    env = load_env(traj_test, use_row_mode_from_data=USE_ROW_MODE_FROM_DATA, fov=FOV)
    agent = load_agent(env, MODEL_PATH, use_gnn=USE_GNN)

    run_eval_with_plots(
        env, agent,
        traj_df=traj_test,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        save_dir=SAVE_DIR,
    )
