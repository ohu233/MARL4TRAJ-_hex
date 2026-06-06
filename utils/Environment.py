import pickle
import numpy as np
import pandas as pd
import torch

from utils.SoftActorCritic import DiscreteSACAgent, SACConfig
from utils.tools import state_to_vector
from utils.hex_utils import (
    HEX_DIRECTIONS, ACTION_TO_HEX_IDX,
    hex_distance, hex_is_valid, hex_add, hex_sub,
    get_hex_neighborhood, load_hex_mapdata, load_hex_mapdata_raw,
    code_to_mode_matrices,
    find_nearest_road_cell, find_k_nearest_road_cells,
    build_bfs_distance_field, build_bfs_distance_field_from_multiple,
    HEX_RADIUS, _hex_ring_offsets,
)

"""
1. 多路网环境下避免换乘应该安排在下层PathEnv中实现，而不是ModeEnv中：
    - 安排在上层容易使上层任务耦合过度，奖励设计等趋于复杂
    - 安排在下层可以增强可解释性，更符合直觉，并能够避免下层过于类似搜索算法，且没有搜索算法性能优秀，失去强化学习的意义
    - 安排在下层可提高扩展性，后续引入换乘点等概念时，更容易在下层实现，而不需要修改上层逻辑
2. ModeEnv的逻辑需要大改：
    - 上层最好选择一次性输入一整条轨迹作为一个ep，对一整条轨迹进行模式选择
    - 奖励设计重点考虑（路径长度加权）：综合所选路网匹配度（判断是否匹配）、成功率、速度匹配、最大路网匹配度（或1，2位路网匹配度之差：越大说明越高概率是单模式），换乘次数（错误路网下，换乘次数一般多于正确路网）、模式数量惩罚
"""

modelist = ['GSD', 'GG', 'TS', 'TG']

class PathEnv:
    '''
    Train：（单独训练）在随机扰动的Mode选择下，进行路径恢复，输出Path
    Test：（协同部署）接收ModeAgent传递的Mode
    '''
    def __init__(self,
                 selected_mode: np.ndarray = None,                  # 选择的路网
                 train_mode: bool = True,                           # 训练模式
                 mapdata: dict = None,                              # 地图数据
                 traj: pd.DataFrame = None,                         # 轨迹数据
                 FOV: int = 1,                                      # 观测半径
                 distance_threshold: float = 1.0,                   # 成功判定的距离阈值
                 bfs_search_radius: int = 10,                       # BFS 搜索半径
                 bfs_max_nodes: int = 50000,                        # BFS 距离场最大节点数
                 reward_alpha: float = 0.3,                         # 势能场 cube 距离变化系数 α·ΔD_cube
                 reward_beta: float = 0.7,                          # 势能场路网距离变化系数 β·ΔD_net
                 step_penalty: float = 0.1,                         # 每步代价，抑制绕路
                 offroad_penalty: float = 8.0,                      # 离开选中路网的惩罚
                 adsorption_radius: int = 20,                       # 吸附集合 C_D 的搜索半径
                 adsorption_K: int = 3,                             # C_D 中保留的最近路网点数量
                 ):

        self.selected_mode = selected_mode
        self.train_mode = train_mode
        self.traj = traj
        self.traj_cnt = 0
        self.FOV = FOV
        self.distance_threshold = distance_threshold
        self.bfs_search_radius = bfs_search_radius
        self.bfs_max_nodes = bfs_max_nodes
        self._bfs_dist = None
        self._road_end = None
        self._C_D = None

        # 势能场奖励参数
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.step_penalty = float(step_penalty)
        self.offroad_penalty = float(offroad_penalty)
        self.adsorption_radius = adsorption_radius
        self.adsorption_K = adsorption_K

        # 加载 hex 地图数据
        first_key = next(iter(mapdata)) if mapdata else None
        first_value = mapdata.get(first_key) if first_key is not None else None
        if not (
            isinstance(first_key, tuple)
            and len(first_key) == 3
            and isinstance(first_value, dict)
            and 'code' in first_value
        ):
            raise ValueError(
                "PathEnv requires raw hex mapdata: "
                "{(q, r, s): {'lon': ..., 'lat': ..., 'code': ...}}"
            )
        self.hex_mapdata_raw = mapdata
        code_dict = {k: int(v['code']) for k, v in mapdata.items()}
        self.mapdata = code_to_mode_matrices(code_dict)

        if self.selected_mode is None:
            self.selected_mode = np.array(modelist)
        self.min_mode_count = 1
        self.max_mode_count = len(modelist)

    def _patch_or_zero(self, mode, q, r, s) -> np.ndarray:
        """未选中的 mode 返回全零 FOV，避免噪声干扰。"""
        if mode in self.selected_mode:
            return get_hex_neighborhood(self.mapdata[mode], q, r, s, radius=self.FOV)
        else:
            n_cells = 3 * self.FOV**2 + 3 * self.FOV + 1    # FOV内栅格数量

            return np.zeros(n_cells, dtype=np.float32)

    def _bfs_gradient_patch(self, pos) -> np.ndarray:
        """Return a local BFS descent field aligned with the hex patch ordering."""
        pos_key = (int(round(pos[0])), int(round(pos[1])), int(round(pos[2])))
        current_dist = self._adsorption_distance_to_C_D(pos_key)
        values = []

        for ring_r in range(self.FOV + 1):
            for dq, dr, ds in _hex_ring_offsets(ring_r):
                key = (pos_key[0] + dq, pos_key[1] + dr, pos_key[2] + ds)
                if self.multi_mapdata.get(key, 0) == 0 or self._bfs_dist is None:
                    values.append(-1.0)
                    continue
                cell_dist = self._bfs_dist.get(key)
                if cell_dist is None:
                    values.append(-1.0)
                    continue
                values.append(float(np.clip(current_dist - cell_dist, -1.0, 1.0)))

        return np.asarray(values, dtype=np.float32)

    def _build_patch(self, pos) -> list:
        return (
            get_hex_neighborhood(self.multi_mapdata, *pos, radius=self.FOV).tolist() +
            self._patch_or_zero('GSD', *pos).tolist() +
            self._patch_or_zero('GG', *pos).tolist() +
            self._patch_or_zero('TS', *pos).tolist() +
            self._patch_or_zero('TG', *pos).tolist() +
            self._bfs_gradient_patch(pos).tolist()
        )

    def reset(self):

        self.step_cnt = 0

        # 计算当前轨迹索引
        current_traj_idx = self.traj_cnt % len(self.traj)

        # 训练模式下随机选择 mode 组合（必然包含真实mode）
        if self.train_mode:
            num_modes = np.random.randint(self.min_mode_count, self.max_mode_count + 1)
            # 获取真实mode
            real_mode = str(self.traj.loc[current_traj_idx, 'mode']).strip()
            if real_mode in modelist:
                # 确保真实mode在可选模式中，并从全量模式中随机选择其他模式
                remaining_modes = [m for m in modelist if m != real_mode]
                extra_num = min(num_modes - 1, len(remaining_modes))
                extra_modes = np.random.choice(remaining_modes, size=max(0, extra_num), replace=False)
                if extra_num > 0 and len(extra_modes) > 0:
                    self.selected_mode = np.concatenate([[real_mode], extra_modes])
                else:
                    self.selected_mode = np.array([real_mode])
            else:
                self.selected_mode = np.random.choice(modelist, size=num_modes, replace=False)
        else:
            if self.selected_mode is None or len(self.selected_mode) == 0:
                self.selected_mode = np.array(modelist)

        # 读取起点/终点 cube 坐标
        row = self.traj.iloc[current_traj_idx]
        hex_start, hex_end = self._read_hex_coords(row)
        # 获取栅格距离
        if 'distance_cells' in row:
            self.episode_distance_cells = float(row['distance_cells'])
        else:   # 避免字段缺失
            self.episode_distance_cells = float(hex_distance(hex_start, hex_end))

        self.hex_start = hex_start
        self.hex_end = hex_end

        # 构建 multi_mapdata（dict 合并）
        self.multi_mapdata = {} # (x, y, z) = 1
        for mode in self.selected_mode:
            mode_dict = self.mapdata[mode]
            for cube in mode_dict:
                self.multi_mapdata[cube] = 1

        # 搜索起终点最近的路网接触点（用于单点兼容）
        road_start = find_nearest_road_cell(
            *hex_start, self.multi_mapdata, max_radius=self.bfs_search_radius
        )
        self._road_end = find_nearest_road_cell(
            *hex_end, self.multi_mapdata, max_radius=self.bfs_search_radius
        )

        # 构建终点吸附集合 C_D（K 个最近路网点）
        cd_candidates = find_k_nearest_road_cells(
            *hex_end, self.multi_mapdata,
            max_radius=self.adsorption_radius,
            K=self.adsorption_K
        )
        # 获取多源距离场 源的hex coods
        self._C_D = [c for c in cd_candidates if self.multi_mapdata.get(c, 0) != 0]

        # 从 C_D 多源 BFS 构建路网距离场：dict
        if self._C_D:
            self._bfs_dist = build_bfs_distance_field_from_multiple(
                self._C_D, self.multi_mapdata, max_nodes=self.bfs_max_nodes
            )
        else:
            self._bfs_dist = None

        # 计算 max_step：优先用路网约束下的有效距离
        if road_start is not None and self._road_end is not None and self._bfs_dist is not None and road_start in self._bfs_dist:
            # 有效距离=起点到最近路网距离+多源距离场距离+终点到最近路网距离
            eff_dist = (
                hex_distance(hex_start, road_start)
                + self._bfs_dist[road_start]
                + hex_distance(self._road_end, hex_end)
            )
            # 最大步数设定为有效距离的3倍
            self.max_step = max(1, int(eff_dist * 3))
        else:
            h_dist = hex_distance(hex_start, hex_end)
            self.max_step = max(1, int(h_dist * 3))

        # neighbor: 半径1六边形邻域
        self.neighbor = get_hex_neighborhood(
            self.multi_mapdata, hex_start[0], hex_start[1], hex_start[2], radius=1
        )
        initial_bfs_dist = self._effective_distance_to_goal(hex_start)
        self.initial_bfs_distance = max(1.0, float(initial_bfs_dist))

        self.traj_cnt += 1

        # 计算剩余距离 cube 偏移
        rem = hex_sub(hex_end, hex_start)  # (dq, dr, ds)

        self.state = {
            'remaining_distance': np.array(rem),       # cube 偏移
            'previous_remaining_distance': np.array(rem),
            'normalized_bfs_remaining': float(initial_bfs_dist) / self.initial_bfs_distance,
            'current_mode': self.selected_mode,
            'patch': self._build_patch(hex_start),
        }

        self.visited_points = 1
        self.on_road_points = int(self._is_on_selected_road(hex_start))
        self.match_ratio = self.on_road_points / self.visited_points

        return self.state

    def _read_hex_coords(self, row):
        """从 CSV 行中读取 hex cube 坐标。
        CSV 的 loczo/loczd 由三个分量独立 round 生成，可能违反 q+r+s=0。
        这里用 q/r 重新推导 s，与 hex_grid.pkl 保持一致。
        """
        locxo = int(row['locxo'])
        locyo = int(row['locyo'])
        locxd = int(row['locxd'])
        locyd = int(row['locyd'])
        # 确保q+r+s=0
        loczo = int(-(locxo + locyo))
        loczd = int(-(locxd + locyd))

        return (locxo, locyo, loczo), (locxd, locyd, loczd)

    def _build_hex_spatial_index(self):
        """为 hex_mapdata_raw 构建 kd-tree 空间索引。"""
        if self.hex_mapdata_raw is None:
            self._hex_kd_keys = []
            self._hex_kd_tree = None
            return

        from scipy.spatial import cKDTree
        keys = list(self.hex_mapdata_raw.keys())
        lons = np.array([self.hex_mapdata_raw[k]['lon'] for k in keys])
        lats = np.array([self.hex_mapdata_raw[k]['lat'] for k in keys])
        points = np.column_stack([lons, lats])
        self._hex_kd_keys = keys
        self._hex_kd_tree = cKDTree(points)

    def _wgs84_to_hex(self, lon, lat):
        """WGS84 经纬度 → 最近 hex cube 坐标（kd-tree 查找）。"""
        if self.hex_mapdata_raw is None:
            return (0.0, 0.0, 0.0)

        if not hasattr(self, '_hex_kd_tree') or self._hex_kd_tree is None:
            self._build_hex_spatial_index()

        if self._hex_kd_tree is None:
            return (0.0, 0.0, 0.0)

        dist, idx = self._hex_kd_tree.query([lon, lat])
        key = self._hex_kd_keys[idx]
        return tuple(float(v) for v in key)

    def _get_map_value(self, mode, q, r, s):
        """安全获取地图值，不存在返回 0。"""
        key = (int(round(q)), int(round(r)), int(round(s)))
        return self.mapdata[mode].get(key, 0)

    def _is_on_selected_road(self, pos):
        key = (int(round(pos[0])), int(round(pos[1])), int(round(pos[2])))
        return self.multi_mapdata.get(key, 0) != 0

    def set_mode_sampling_range(self, min_mode_count: int = 1, max_mode_count: int = 4):
        self.min_mode_count = max(1, min(int(min_mode_count), len(modelist)))
        self.max_mode_count = max(1, min(int(max_mode_count), len(modelist)))
        if self.min_mode_count > self.max_mode_count:
            self.min_mode_count = self.max_mode_count

    def calculate_reward(self, reward, prev_dist, curr_dist, neighbor, action,
                         prev_cube_dist=None, curr_cube_dist=None):
        """
        势能场奖励：R_dist = α·ΔD_cube + β·ΔD_eff

        ΔD_cube = D_cube(s_t, D) - D_cube(s_{t+1}, D)  (cube 距离变化)
        ΔD_eff = D_eff(s_t, D) - D_eff(s_{t+1}, D)    (有效路网距离变化)
        α = reward_alpha, β = reward_beta
        """
        if prev_cube_dist is None:
            prev_cube_dist = hex_distance(self.hex_start, self.hex_end)
        if curr_cube_dist is None:
            curr_cube_dist = hex_distance(self.hex_start, self.hex_end)

        delta_cube = prev_cube_dist - curr_cube_dist
        delta_net = prev_dist - curr_dist

        is_on_road = neighbor[ACTION_TO_HEX_IDX[action]] != 0

        r_dist_cube = self.reward_alpha * delta_cube
        r_dist_net = self.reward_beta * delta_net

        reward += r_dist_cube + r_dist_net

        if is_on_road:
            reward += 1.0
        else:
            reward -= self.offroad_penalty

        return reward

    def _adsorption_distance_to_C_D(self, pos):
        """
        计算吸附距离 D_net(p, D) = min_{v ∈ C_p} [d_cube(p, v) + d_net(v, C_D)]

        C_p = {v ∈ G_allowed | d_cube(p, v) ≤ r}
        d_net(v, C_D) 从多源 BFS 距离场直接查询
        """
        pos_key = (int(round(pos[0])), int(round(pos[1])), int(round(pos[2])))

        # 已在 BFS 距离场中，直接返回
        if self._bfs_dist is not None and pos_key in self._bfs_dist:
            return float(self._bfs_dist[pos_key])

        # 否则枚举 C_p（半径 r 内的路网点）
        best = float('inf')
        for ring_r in range(self.adsorption_radius + 1):
            for dq, dr, ds in _hex_ring_offsets(ring_r):
                cp_key = (pos_key[0] + dq, pos_key[1] + dr, pos_key[2] + ds)
                if self.multi_mapdata.get(cp_key, 0) == 0:
                    continue
                if self._bfs_dist is not None and cp_key in self._bfs_dist:
                    d_net_v = self._bfs_dist[cp_key]
                else:
                    continue
                candidate = ring_r + d_net_v
                if candidate < best:
                    best = candidate

        return float(best) if best != float('inf') else float(hex_distance(pos, self.hex_end))

    def _effective_distance_to_goal(self, pos):
        """
        计算 pos 到终点的有效距离（考虑路网约束和终点吸附）。

        D_eff = D_net(pos, D) + d_cube(v_D, hex_end)
        其中 v_D 是 C_D 中离 hex_end 最近的代表点（self._road_end）
        """
        d_net = self._adsorption_distance_to_C_D(pos)
        if self._road_end is not None:
            d_extra = hex_distance(self._road_end, self.hex_end)
        else:
            d_extra = hex_distance(pos, self.hex_end)
        return d_net + d_extra

    def get_episode_metadata(self):
        """Return episode-level metadata needed by HER/EpisodeBuffer."""
        return {
            'hex_start': tuple(self.hex_start),
            'hex_end': tuple(self.hex_end),
            'bfs_dist': self._bfs_dist,
            'road_end': self._road_end,
            'C_D': self._C_D,
        }

    def step(self, action: int):
        '''
        采取动作，计算奖励，更新状态
        '''
        if action < 0 or action >= len(HEX_DIRECTIONS):
            raise ValueError(f"Invalid PathEnv action {action}; expected 0-{len(HEX_DIRECTIONS) - 1}.")

        success = 0
        reward = -self.step_penalty
        done = False
        self.step_cnt += 1

        # 移动前的状态
        pre_move_pos = self.hex_start
        pre_move_cube_dist = hex_distance(pre_move_pos, self.hex_end)
        pre_move_bfs_dist = self._effective_distance_to_goal(pre_move_pos)

        # 更新绝对坐标
        self.hex_start = hex_add(self.hex_start, HEX_DIRECTIONS[action])

        self.visited_points += 1
        self.on_road_points += int(self._is_on_selected_road(self.hex_start))
        self.match_ratio = self.on_road_points / self.visited_points

        # 更新上一步剩余距离向量
        self.state['previous_remaining_distance'] = self.state['remaining_distance']

        # 更新剩余距离向量
        rem = hex_sub(self.hex_end, self.hex_start)
        self.state['remaining_distance'] = (rem[0], rem[1], rem[2])

        # 移动后的状态
        curr_bfs_dist = self._effective_distance_to_goal(self.hex_start)
        curr_cube_dist = hex_distance(self.hex_start, self.hex_end)

        self.state['normalized_bfs_remaining'] = (
            float(curr_bfs_dist) / self.initial_bfs_distance
        )

        # 计算奖励（势能场公式）
        reward = self.calculate_reward(
            reward, pre_move_bfs_dist, curr_bfs_dist,
            self.neighbor, action,
            prev_cube_dist=pre_move_cube_dist,
            curr_cube_dist=curr_cube_dist,
        )

        # 更新 neighbor
        self.neighbor = get_hex_neighborhood(
            self.multi_mapdata, *self.hex_start, radius=1
        )
        self.state['patch'] = self._build_patch(self.hex_start)

        # 判断 done
        if (
            self._is_on_selected_road(self.hex_start)
            and curr_cube_dist <= self.distance_threshold
        ):
            reward += 50.0 * self.match_ratio
            done = True
            success = 1
        elif self.step_cnt >= self.max_step:
            reward -= 20.0 * (1.0 - self.match_ratio)
            done = True

        return self.state, reward, done, success



class CounterfactualPathEvaluator:
    """Run frozen PathAgent rollouts and score counterfactual mode sets."""

    def __init__(
        self,
        model_path: str,
        hex_mapdata_raw: dict,
        mode_maps: dict,
        fov: int = 3,
        distance_threshold: float = 1.0,
        use_conv: bool = False,
    ):
        if hex_mapdata_raw is None:
            raise ValueError("CounterfactualPathEvaluator requires raw hex mapdata.")

        self.hex_mapdata_raw = hex_mapdata_raw
        self.mode_maps = mode_maps
        self.fov = fov
        self.distance_threshold = distance_threshold

        cfg = SACConfig()
        device = torch.device(cfg.device)
        path_agent = DiscreteSACAgent(
            vec_dim=11,
            hex_radius=self.fov,
            action_dim=6,
            cfg=cfg,
            use_gnn=use_conv,
            in_channels=6,
        )
        state_dict = torch.load(model_path, map_location=device)
        path_agent.actor.load_state_dict(state_dict)
        path_agent.actor.eval()
        self.path_agent = path_agent

    def _compute_quality(self, result):
        return float(
            2.0 * float(result["success"])
            + 3.0 * float(result["multi_match_rate"])
            + 1.0 * float(result["progress_score"])
            - 0.5 * float(result["normalized_steps"])
        )

    def evaluate(self, traj_one: pd.DataFrame, selected_modes):
        selected_modes = list(selected_modes)
        if len(selected_modes) == 0:
            return {
                "match_rates": [0.0 for _ in modelist],
                "multi_match_rate": 0.0,
                "success": 0,
                "steps": 0,
                "path_len": 0.0,
                "progress_score": 0.0,
                "normalized_steps": 1.0,
                "final_distance": float("inf"),
                "q": -0.5,
            }

        env = PathEnv(
            train_mode=False,
            selected_mode=np.array(selected_modes),
            mapdata=self.hex_mapdata_raw,
            traj=traj_one.reset_index(drop=True),
            FOV=self.fov,
            distance_threshold=self.distance_threshold,
        )

        state = env.reset()
        start_pos = tuple(env.hex_start)
        goal_pos = tuple(env.hex_end)
        initial_distance = max(float(hex_distance(start_pos, goal_pos)), 1.0)
        traj_points = [start_pos]
        steps = 0
        success = 0
        done = False

        while not done:
            state_vec = state_to_vector(state)
            action = self.path_agent.select_action(state_vec, evaluate=True)
            state, _, done, succ = env.step(int(action))
            traj_points.append(tuple(env.hex_start))
            steps += 1
            success = int(succ)

        final_distance = float(hex_distance(tuple(env.hex_start), goal_pos))
        progress_score = max(0.0, min(1.0, (initial_distance - final_distance) / initial_distance))
        normalized_steps = float(steps) / float(max(getattr(env, "max_step", steps), 1))
        normalized_steps = max(0.0, min(1.0, normalized_steps))

        selected_set = set(selected_modes)
        mode_scores = {m: 0.0 for m in modelist}
        total_points = float(max(len(traj_points), 1))
        selected_maps = {m: self.mode_maps[m] for m in selected_modes}
        for point in traj_points:
            q, r, s = int(round(point[0])), int(round(point[1])), int(round(point[2]))
            for mode in selected_modes:
                if selected_maps[mode].get((q, r, s), 0) != 0:
                    mode_scores[mode] += 1.0

        result = {
            "match_rates": [
                float(mode_scores[m] / total_points) if m in selected_set else 0.0
                for m in modelist
            ],
            "multi_match_rate": float(env.match_ratio),
            "success": success,
            "steps": steps,
            "path_len": float(steps),
            "progress_score": progress_score,
            "normalized_steps": normalized_steps,
            "final_distance": final_distance,
        }
        result["q"] = self._compute_quality(result)
        return result

    def counterfactual_eval(self, traj_one: pd.DataFrame, active_modes):
        active_modes = list(active_modes)
        base_eval = self.evaluate(traj_one, active_modes)
        q_base = float(base_eval["q"])
        q_without = [0.0 for _ in modelist]
        delta_q = [0.0 for _ in modelist]
        active_set = set(active_modes)

        for idx, mode in enumerate(modelist):
            if mode not in active_set:
                continue
            without_modes = [m for m in active_modes if m != mode]
            if len(without_modes) == 0:
                q_without[idx] = 0.0
                delta_q[idx] = q_base
                continue
            without_eval = self.evaluate(traj_one, without_modes)
            q_without[idx] = float(without_eval["q"])
            delta_q[idx] = float(q_base - q_without[idx])

        return base_eval, q_without, delta_q


class ModeEnv:
    """
    Counterfactual Evidence–Bayesian Fusion–RL Decision Framework

    逻辑是：

    1. 因果/反事实产生证据
    对每个候选模式做干预：

    观察去掉该模式后路径恢复质量变化：


    这就是“该模式是否必要”的证据。

    2. 贝叶斯推理融合证据
    维护模式信念：

    每获得一个反事实证据，就更新一次：

    3. RL 决定何时获取证据、剔除和停止
    RL 的动作不是直接预测模式，而是决定：
    检验哪个模式
    剔除哪个模式；
    是否停止并输出结果。

    剔除式 Mode 选择环境:
    1 episode = 1 OD, 3 步 step 逐步剔除 mode（4→3→2→1）。
    动作: 0=GSD, 1=GG, 2=TS, 3=TG（剔除哪个）。
    state 包含同 ID 前面段选出的 mode（序贯决策）。
    """
    def __init__(
        self,
        model_path: str,
        mapdata,
        traj: pd.DataFrame,
        train_mode: bool = True,
        fov: int = 3,
        distance_threshold: float = 1.0,
    ):
        """
        输入:
            model_path: 冻结 PathAgent 模型路径
            mapdata: hex 路网数据 {(q,r,s): {'code': int, ...}}
            traj: 轨迹 DataFrame，需含 ID/mode/distance_cells/time 等列
            train_mode: 是否为训练模式（影响 episode 采样方式）
            fov: PathEnv 观测半径
            distance_threshold: PathEnv 成功判定距离阈值
        输出: 无（初始化内部状态）
        功能: 初始化 ModeEnv，加载 PathAgent 评估器，预计算归一化参数，构建 ID block 索引
        """
        self.model_path = model_path
        self.traj = traj
        self.train_mode = train_mode
        self.fov = fov
        self.distance_threshold = distance_threshold

        self.hex_radius = HEX_RADIUS
        self.traj_cnt = 0
        self.current_row = None
        self.step_cnt = 0
        self.active_modes = list(modelist)
        self.last_eval = None
        self.max_mode_steps = 4
        self.stop_action = len(modelist)
        self.belief_alpha = 2.0
        self.belief_beta = 0.5
        self.belief = np.ones(len(modelist), dtype=np.float32) / len(modelist)
        self.q_base = 0.0
        self.q_without = [0.0 for _ in modelist]
        self.delta_q = [0.0 for _ in modelist]
        self.pred_mode = None

        # ID 序贯历史
        self._last_id = None
        self.current_id_history = []
        self.history_len = 3
        self.valid_modes = set(modelist)
        self._id_blocks = self._build_id_blocks()
        self._episode_order = []
        self._episode_order_pos = 0

        # 归一化统计量（预计算）
        self._precompute_norm_stats()

        # 加载并处理 hex mapdata
        first_key = next(iter(mapdata)) if mapdata else None
        if isinstance(first_key, tuple) and len(first_key) == 3:
            first_val = mapdata[first_key]
            if isinstance(first_val, dict) and 'code' in first_val:
                self.hex_mapdata_raw = mapdata
                code_dict = {k: int(v['code']) for k, v in mapdata.items()}
                self.mapdata = code_to_mode_matrices(code_dict)
            elif isinstance(first_val, (int, float, np.integer, np.floating)):
                self.hex_mapdata_raw = None
                self.mapdata = code_to_mode_matrices(mapdata)
            else:
                self.hex_mapdata_raw = None
                self.mapdata = mapdata
        else:
            raise ValueError("ModeEnv requires hex mapdata. Use load_hex_mapdata_raw().")

        self.mode_maps = self.mapdata

        self.evaluator = CounterfactualPathEvaluator(
            model_path=self.model_path,
            hex_mapdata_raw=self.hex_mapdata_raw,
            mode_maps=self.mapdata,
            fov=self.fov,
            distance_threshold=self.distance_threshold,
        )

    def _build_id_blocks(self):
        """
        输入: 无（读取 self.traj, self.valid_modes）
        输出: List[List[int]]，每个子列表是同一 ID 下的行索引集合（block）
        功能: 按 ID 列对轨迹分组，过滤掉 mode 不合法的行，用于保证同一 ID 的轨迹段在训练中按序出现
        """
        blocks_by_id = {}
        if "ID" not in self.traj.columns:
            for idx in range(len(self.traj)):
                row = self.traj.iloc[idx]
                if "mode" not in self.traj.columns or str(row.get("mode", "")).strip() in self.valid_modes:
                    blocks_by_id[f"row_{idx}"] = [idx]
            return list(blocks_by_id.values())

        for idx in range(len(self.traj)):
            row = self.traj.iloc[idx]
            if "mode" in self.traj.columns and str(row.get("mode", "")).strip() not in self.valid_modes:
                continue
            traj_id = str(row.get("ID", "")).strip()
            if traj_id == "":
                traj_id = f"row_{idx}"
            if traj_id not in blocks_by_id:
                blocks_by_id[traj_id] = []
            blocks_by_id[traj_id].append(idx)
        return [block for block in blocks_by_id.values() if block]

    def _reshuffle_episode_order(self):
        """
        输入: 无（读取 self._id_blocks）
        输出: 无（修改 self._episode_order, self._episode_order_pos）
        功能: 随机打乱 ID block 顺序，展开为平坦的行索引序列，一轮遍历完所有 block
        """
        if not self._id_blocks:
            self._episode_order = []
            self._episode_order_pos = 0
            return

        block_indices = np.arange(len(self._id_blocks))
        np.random.shuffle(block_indices)
        self._episode_order = []
        for block_idx in block_indices:
            self._episode_order.extend(self._id_blocks[int(block_idx)])
        self._episode_order_pos = 0

    def _next_episode_index(self):
        """
        输入: 无（读取 self._episode_order, self._episode_order_pos）
        输出: int，当前 episode 应使用的行索引
        功能: 按序返回下一个 episode 的行索引；若当前轮次用完则 reshuffle
        """
        if not self._episode_order or self._episode_order_pos >= len(self._episode_order):
            self._reshuffle_episode_order()
        if not self._episode_order:
            raise ValueError("ModeEnv has no valid training episodes after mode filtering.")

        idx = self._episode_order[self._episode_order_pos]
        self._episode_order_pos += 1
        self.traj_cnt += 1
        return idx

    def reset_episode_order(self):
        """
        输入: 无
        输出: 无
        功能: 重置 episode 采样状态（顺序指针、计数器、ID 历史），用于测试时从头开始
        """
        self._episode_order = []
        self._episode_order_pos = 0
        self.traj_cnt = 0
        self._last_id = None
        self.current_id_history = []

    def _precompute_norm_stats(self):
        """预计算轨迹特征的归一化参数（max / mean+std）。"""
        if 'distance_cells' in self.traj.columns:
            self._dist_max = max(float(self.traj['distance_cells'].max()), 1.0)
        else:
            self._dist_max = 1.0
        if 'time' in self.traj.columns:
            self._time_max = max(float(self.traj['time'].max()), 1.0)
        else:
            self._time_max = 1.0
        if 'velocity' in self.traj.columns:
            v_col = self.traj['velocity']
            self._vel_max = max(float(v_col.max()), 1e-6)
        else:
            self._vel_max = 1.0

    def _norm_distance(self, val):
        """
        输入: val (float) — 原始栅格距离
        输出: float, [0, 1] 归一化距离
        功能: 除以全局最大距离，clip 到 1
        """
        return min(float(val) / self._dist_max, 1.0)

    def _norm_time(self, val):
        """
        输入: val (float) — 原始时间
        输出: float, [0, 1] 归一化时间
        功能: 除以全局最大时间，clip 到 1
        """
        return min(float(val) / self._time_max, 1.0)

    def _norm_velocity(self, val):
        """
        输入: val (float) — 原始速度
        输出: float, [0, 1] 归一化速度
        功能: 除以全局最大速度，clip 到 1
        """
        return min(float(val) / self._vel_max, 1.0)

    def _initial_belief(self):
        """
        输入: 无（读取 self.current_id_history）
        输出: np.ndarray (4,) — 初始信念概率分布
        功能: 基于当前 ID 历史中已选出的 mode 构造先验信念，历史中出现的 mode 权重 +0.5
        """
        prior = np.ones(len(modelist), dtype=np.float32)
        for mode in self.current_id_history:
            if mode in modelist:
                prior[modelist.index(mode)] += 0.5
        return prior / max(float(prior.sum()), 1e-8)

    def _update_belief(self):
        """
        输入: 无（读取 self.delta_q, self.belief, self.active_modes, self.belief_alpha, self.belief_beta）
        输出: 无（修改 self.belief）
        功能: 用反事实证据 delta_q 更新贝叶斯信念。logits = alpha * delta_q + beta * log(belief)，只保留 active mode
        """
        active_mask = np.array([1.0 if m in self.active_modes else 0.0 for m in modelist], dtype=np.float32)
        if active_mask.sum() <= 0:
            self.belief = np.ones(len(modelist), dtype=np.float32) / len(modelist)
            return

        old_logit = np.log(np.maximum(self.belief.astype(np.float32), 1e-6))
        delta = np.array(self.delta_q, dtype=np.float32)
        logits = self.belief_alpha * delta + self.belief_beta * old_logit
        logits = np.where(active_mask > 0.0, logits, -1e9)
        logits = logits - float(np.max(logits))
        probs = np.exp(logits) * active_mask
        denom = float(probs.sum())
        if denom <= 1e-8:
            probs = active_mask / float(active_mask.sum())
        else:
            probs = probs / denom
        self.belief = probs.astype(np.float32)
        belief_str = "  ".join(f"{m}:{self.belief[i]:.3f}" for i, m in enumerate(modelist))
        delta_str = "  ".join(f"{m}:{delta[i]:.3f}" for i, m in enumerate(modelist))
        print(f"  [belief更新] delta_q=[{delta_str}]")
        print(f"  [belief更新] belief=[{belief_str}]")

    def _predict_mode(self):
        """
        输入: 无（读取 self.belief, self.active_modes）
        输出: str — 当前信念最高的 mode 名称
        功能: 在 active_modes 中取 belief 最大的 mode 作为预测结果
        """
        active_indices = [i for i, m in enumerate(modelist) if m in self.active_modes]
        if not active_indices:
            return modelist[int(np.argmax(self.belief))]
        best_idx = max(active_indices, key=lambda i: float(self.belief[i]))
        return modelist[best_idx]

    def _refresh_counterfactual_state(self, update_belief=True):
        """
        输入:
            update_belief: bool — 是否同步更新 belief
        输出: 无（修改 self.last_eval, self.q_base, self.q_without, self.delta_q, self.belief, self.pred_mode）
        功能: 对当前 active_modes 执行反事实评估，计算每个 mode 的 leave-one-out delta_q，可选更新 belief
        """
        base_eval, q_without, delta_q = self.evaluator.counterfactual_eval(
            self.current_row,
            self.active_modes,
        )
        self.last_eval = base_eval
        self.q_base = float(base_eval["q"])
        self.q_without = [float(v) for v in q_without]
        self.delta_q = [float(v) for v in delta_q]
        print(f"  [反事实评估] active={self.active_modes}  q_base={self.q_base:.3f}  "
              f"success={base_eval['success']}  multi_match={base_eval['multi_match_rate']:.3f}  "
              f"steps={base_eval['steps']}")
        mr_str = "  ".join(f"{m}:{base_eval['match_rates'][i]:.3f}" for i, m in enumerate(modelist))
        print(f"  [反事实评估] match_rates=[{mr_str}]")
        qw_str = "  ".join(f"{m}:{self.q_without[i]:.3f}" for i, m in enumerate(modelist))
        dq_str = "  ".join(f"{m}:{self.delta_q[i]:.3f}" for i, m in enumerate(modelist))
        print(f"  [反事实评估] q_without=[{qw_str}]")
        print(f"  [反事实评估] delta_q  =[{dq_str}]")
        if update_belief:
            self._update_belief()
        self.pred_mode = self._predict_mode()
        belief_str = "  ".join(f"{m}:{self.belief[i]:.3f}" for i, m in enumerate(modelist))
        print(f"  [预测结果] pred_mode={self.pred_mode}  belief=[{belief_str}]")

    def _stop_allowed(self):
        """
        输入: 无
        输出: int — 1 允许 stop，0 不允许
        功能: 判断当前步骤是否允许执行 stop 动作（当前恒返回 1）
        """
        return 1

    def _build_state(self):
        """
        输入: 无（读取 self.active_modes, self.belief, self.delta_q, self.q_base, self.q_without, self.last_eval, self.current_row, self.current_id_history, self.pred_mode）
        输出: dict — DQN 可消费的 state 字典，包含 active_mode_mask/belief/delta_q/q_base/q_without/stop_allowed/match_rates 等 18 个字段
        功能: 将内部状态打包为标准 state 字典，供 mode_state_to_vector 展平为向量
        """
        active_mask = [1 if m in self.active_modes else 0 for m in modelist]
        return {
            "active_mode_mask": active_mask,
            "belief": self.belief.astype(np.float32).tolist(),
            "delta_q": [float(v) for v in self.delta_q],
            "q_base": float(self.q_base),
            "q_without": [float(v) for v in self.q_without],
            "stop_allowed": int(self._stop_allowed()),
            "match_rates": self.last_eval["match_rates"] if self.last_eval else [0.0 for _ in modelist],
            "multi_match_rate": float(self.last_eval["multi_match_rate"]) if self.last_eval else 0.0,
            "success": int(self.last_eval["success"]) if self.last_eval else 0,
            "steps": int(self.last_eval["steps"]) if self.last_eval else 0,
            "progress_score": float(self.last_eval["progress_score"]) if self.last_eval else 0.0,
            "normalized_steps": float(self.last_eval["normalized_steps"]) if self.last_eval else 0.0,
            "distance_cells": self._norm_distance(self.current_row['distance_cells'].iat[0]),
            "time": self._norm_time(self.current_row['time'].iat[0]),
            "velocity": self._norm_velocity(self.current_row['velocity'].iat[0]),
            "remaining_count": len(self.active_modes),
            "prev_modes": self.current_id_history.copy(),
            "pred_mode": self.pred_mode,
        }

    def _finish_episode(self):
        """
        输入: 无（读取 self.belief, self.active_modes, self.current_id_history, self.history_len）
        输出: 无（修改 self.pred_mode, self.current_id_history）
        功能: episode 结束时确定最终预测 mode，追加到 ID 历史（最多保留 history_len 条）
        """
        self.pred_mode = self._predict_mode()
        true_mode = str(self.current_row['mode'].iat[0]).strip() if 'mode' in self.current_row.columns else "?"
        correct = int(self.pred_mode == true_mode) if true_mode in modelist else -1
        mark = "OK" if correct == 1 else ("MISS" if correct == 0 else "NO_LABEL")
        print(f"  [Episode结束] pred={self.pred_mode}  true={true_mode}  {mark}  "
              f"history={self.current_id_history}")
        if self.pred_mode in modelist:
            self.current_id_history.append(self.pred_mode)
            if len(self.current_id_history) > self.history_len:
                self.current_id_history = self.current_id_history[-self.history_len:]

    def invalid_action_mask(self):
        """
        输入: 无（读取 self.active_modes）
        输出: np.ndarray (5,) bool — True 表示该动作不可选
        功能: 标记无效动作：已不在 active 中的 mode 不可剔除；仅剩 1 个 mode 时不可再剔除
        """
        mask = np.zeros(len(modelist) + 1, dtype=bool)
        for idx, mode in enumerate(modelist):
            if mode not in self.active_modes or len(self.active_modes) <= 1:
                mask[idx] = True
        if not self._stop_allowed():
            mask[self.stop_action] = True
        return mask

    def _run_PathMode(self, selected_modes):
        """
        输入: selected_modes (list[str]) — 要评估的 mode 组合
        输出: tuple(match_rates, multi_match_rate, success, steps, path_len)
        功能: 委托 evaluator.evaluate 对当前轨迹在指定 mode 组合下做 PathAgent rollout
        """
        result = self.evaluator.evaluate(self.current_row, selected_modes)
        return (
            result["match_rates"],
            result["multi_match_rate"],
            result["success"],
            result["steps"],
            result["path_len"],
        )

    def _stop_reward(self):
        """
        输入: 无（读取 self.belief, self.q_base, self.active_modes）
        输出: float, clipped to [-3.0, 3.0]
        功能: 计算 stop 动作的奖励 = q_base + 0.5*confidence - 0.2*entropy - sparsity_penalty
        """
        belief = np.maximum(self.belief.astype(np.float32), 1e-8)
        entropy = -float(np.sum(belief * np.log(belief)))
        confidence = float(np.max(belief))
        sparsity_penalty = 0.1 * float(max(len(self.active_modes) - 1, 0))
        reward = float(self.q_base) + 0.5 * confidence - 0.2 * entropy - sparsity_penalty
        return float(np.clip(reward, -3.0, 3.0))

    def reset(self):
        """
        输入: 无（读取 self.traj, self.train_mode）
        输出: dict — 初始 state 字典（4 mode 全 active，已完成首轮反事实评估）
        功能: 采样一条轨迹，重置 active_modes 为全部 4 个 mode，初始化 belief，执行反事实评估得到 delta_q，构建初始 state
        """
        if self.train_mode:
            idx = self._next_episode_index()
        else:
            idx = self.traj_cnt % len(self.traj)
            self.traj_cnt += 1
        self.current_row = self.traj.iloc[[idx]].copy()
        self.step_cnt = 0
        self.active_modes = list(modelist)
        self.q_base = 0.0
        self.q_without = [0.0 for _ in modelist]
        self.delta_q = [0.0 for _ in modelist]
        self.pred_mode = None

        cur_id = str(self.current_row['ID'].iat[0]).strip()
        if cur_id != self._last_id:
            self.current_id_history = []
            self._last_id = cur_id

        self.belief = self._initial_belief()
        print(f"\n{'='*80}")
        print(f"[Episode] ID={cur_id}  prev_history={self.current_id_history}  "
              f"dist={self.current_row['distance_cells'].iat[0]:.1f}  "
              f"velocity={self.current_row.get('velocity', pd.Series([0])).iat[0]:.2f}")
        self._refresh_counterfactual_state(update_belief=True)
        self.state = self._build_state()
        return self.state

    def step(self, action):
        """
        输入:
            action: int — 0-3 剔除对应 mode，4(stop_action) 提前停止
        输出: tuple(state: dict, reward: float, done: bool, success: int)
        功能: 执行动作。stop → 计算 stop_reward 并结束；剔除 → 移除 mode、重新反事实评估、计算 reward = -delta_q + 0.1；达 max_mode_steps 时自动结束
        """
        self.step_cnt += 1

        if action == self.stop_action:
            reward = self._stop_reward()
            reward -= 0.05
            belief_str = "  ".join(f"{m}:{self.belief[i]:.3f}" for i, m in enumerate(modelist))
            print(f"  [Step {self.step_cnt}] >>> STOP  reward={reward:.3f}  pred={self.pred_mode}  belief=[{belief_str}]")
            self._finish_episode()
            self.state = self._build_state()
            return self.state, float(reward), True, int(self.state["success"])

        invalid = (
            action < 0
            or action >= len(modelist)
            or modelist[action] not in self.active_modes
            or len(self.active_modes) <= 1
        )
        if invalid:
            print(f"  [Step {self.step_cnt}] >>> INVALID action={action}  active={self.active_modes}")
            reward = -2.0 - 0.05
            done = self.step_cnt >= self.max_mode_steps
            if done:
                self._finish_episode()
            self.state = self._build_state()
            return self.state, float(reward), done, int(self.state["success"])

        removed_mode = modelist[action]
        delta_removed = float(self.delta_q[action])
        self.active_modes.remove(removed_mode)
        print(f"  [Step {self.step_cnt}] >>> REMOVE {removed_mode}  delta_q={delta_removed:.3f}  "
              f"remaining={self.active_modes}")
        self._refresh_counterfactual_state(update_belief=True)

        reward = -delta_removed + 0.1
        reward -= 0.05
        reward = float(np.clip(reward, -3.0, 3.0))

        done = self.step_cnt >= self.max_mode_steps
        if done:
            print(f"  [Step {self.step_cnt}] --- 达到 max_mode_steps，自动结束 ---")
            reward += self._stop_reward()
            reward = float(np.clip(reward, -3.0, 3.0))
            self._finish_episode()

        self.state = self._build_state()
        return self.state, float(reward), done, int(self.state["success"])


if __name__ == "__main__":

    TEST_ENV = 'Path'  # 'Path' or 'Mode'

    if TEST_ENV == 'Path':
        print("Loading hex mapdata...")
        hex_mapdata_raw = load_hex_mapdata_raw('data/hex_grid.pkl')
        traj = pd.read_csv('data\\artificial_od_all.csv')

        pathenv = PathEnv(train_mode=True, mapdata=hex_mapdata_raw, traj=traj,
                          FOV=3, distance_threshold=1.0)

        # ====== 基础信息 ======
        print("====== PathEnv 环境检测 ======")
        print(f"  FOV: {pathenv.FOV}")
        print(f"  train_mode: {pathenv.train_mode}")
        print(f"  轨迹数量: {len(pathenv.traj)}")

        # ====== 单次 reset 检测 ======
        state = pathenv.reset()
        print(f"\n====== Reset 检测 ======")
        print(f"  hex_start: {pathenv.hex_start}")
        print(f"  hex_end:   {pathenv.hex_end}")
        print(f"  hex_dist:  {hex_distance(pathenv.hex_start, pathenv.hex_end)}")
        print(f"  selected_mode: {pathenv.selected_mode}")
        print(f"  max_step: {pathenv.max_step}")

        # 检查 state 各字段
        print(f"\n  state keys: {list(state.keys())}")
        for k in ['remaining_distance', 'previous_remaining_distance']:
            v = state[k]
            print(f"  {k}: shape={np.array(v).shape}, value={v}")
        for k in ['current_mode', 'normalized_bfs_remaining']:
            print(f"  {k}: {state[k]}")

        patch = np.array(state['patch'])
        n_cells = 3 * pathenv.FOV**2 + 3 * pathenv.FOV + 1
        print(f"  patch: shape={patch.shape}, expected 6×{n_cells}={6*n_cells}")
        assert patch.shape == (6 * n_cells,), \
            f"patch shape mismatch: {patch.shape} != ({6 * n_cells},)"
        print(f"  patch[0:{n_cells}] (multi) 非零数: {np.count_nonzero(patch[:n_cells])}")
        for i, m in enumerate(['GSD', 'GG', 'TS', 'TG'], 1):
            ch = patch[i*n_cells:(i+1)*n_cells]
            nonzero = np.count_nonzero(ch)
            active = "✓" if m in pathenv.selected_mode else "✗(zero)"
            print(f"  patch[{i}*{n_cells}:] ({m}): 非零数={nonzero}, active={active}")
        bfs_patch = patch[5*n_cells:6*n_cells]
        print(f"  patch[5*{n_cells}:] (BFS gradient): "
              f"min={bfs_patch.min():.1f}, max={bfs_patch.max():.1f}")

        # neighbor 检查
        print(f"\n  neighbor (radius=1): {pathenv.neighbor}")
        print(f"  neighbor shape: {pathenv.neighbor.shape}")
        assert len(pathenv.neighbor) == 7, f"neighbor should have 7 cells, got {len(pathenv.neighbor)}"

        # ====== 动作映射检测 ======
        from utils.hex_utils import ACTION_TO_HEX_IDX, HEX_DIRECTIONS
        action_names = ['北', '西北', '西南', '南', '东南', '东北']
        print(f"\n====== 动作方向检测 ======")
        for a in range(6):
            idx = ACTION_TO_HEX_IDX[a]
            dq, dr, ds = HEX_DIRECTIONS[a]
            nbr_val = pathenv.neighbor[idx]
            print(f"  action {a} ({action_names[a]}): "
                  f"offset=({dq:+d},{dr:+d},{ds:+d}), "
                  f"neighbor[{idx}]={'on_road' if nbr_val != 0 else 'off_road'}")

        # ====== 随机 rollout ======
        print(f"\n====== 随机 Rollout (3 episodes) ======")
        for ep in range(3):
            state = pathenv.reset()
            total_reward = 0.0
            step = 0
            done = False
            traj = [pathenv.hex_start]
            while not done:
                action = np.random.randint(0, 6)
                state, r, done, succ = pathenv.step(action)
                total_reward += r
                step += 1
                traj.append(pathenv.hex_start)
                if done:
                    break
            final_dist = hex_distance(pathenv.hex_start, pathenv.hex_end)
            print(f"  Ep {ep+1}: steps={step}, reward={total_reward:.1f}, "
                  f"success={succ}, final_dist={final_dist}, "
                  f"mode={pathenv.selected_mode}, match={pathenv.match_ratio:.3f}")

        print(f"\n====== 环境检测完成 ======")

    elif TEST_ENV == 'Mode':
        print("Loading hex mapdata...")
        hex_mapdata_raw = load_hex_mapdata_raw('data/hex_grid.pkl')
        traj = pd.read_csv('data/artificial_od_all.csv')

        if "velocity" not in traj.columns:
            traj["velocity"] = traj["distance_km"] / traj["time"].replace(0, np.nan)
            traj["velocity"] = traj["velocity"].fillna(0.0)

        modeenv = ModeEnv(model_path='PathModel/PathModel.pth',
                          mapdata=hex_mapdata_raw,
                          traj=traj,
                          train_mode=True,
                          )

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"env_test_log_{timestamp}.txt"

        with open(log_filename, 'w', encoding='utf-8') as log_file:
            log_file.write("开始测试环境...\n")

            for episode in range(10):
                log_file.write(f"\n===== Episode {episode + 1} =====\n")
                state = modeenv.reset()
                log_file.write(f"初始previous: {modeenv.state['previous']}\n")
                log_file.write(f"初始current: {modeenv.state['current']}\n")

                step = 0
                done = False
                total_reward = 0

                while not done:
                    action = np.random.randint(0, 15)
                    state, r, done, succ, _ = modeenv.step(action)
                    total_reward += r
                    step += 1

                    log_file.write(f"step={step}, action={action}, reward={r}\n")

                    if done:
                        log_file.write(f"Episode结束!\n")
                        break

    else:
        print("无效的测试环境配置，请选择 'Path' 或 'Mode'。")
