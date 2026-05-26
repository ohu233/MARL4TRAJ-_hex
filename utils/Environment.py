import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.SoftActorCritic import DiscreteSACAgent, SACConfig
from utils.tools import mapdata_to_modelmatrix, state_to_vector, calculate_match_rate
from utils.hex_utils import (
    HEX_DIRECTIONS, ACTION_TO_HEX_IDX,
    hex_distance, hex_is_valid, hex_add, hex_sub,
    get_hex_neighborhood, load_hex_mapdata, load_hex_mapdata_raw,
    code_to_mode_matrices,
    HEX_RADIUS,
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
                 selected_mode: np.ndarray = None,
                 train_mode: bool = True,
                 curriculum_mode: bool = True,
                 mapdata: dict = None,
                 traj: pd.DataFrame = None,
                 FOV: int = 3,
                 distance_threshold: float = 1.0,
                 ):

        self.selected_mode = selected_mode
        self.train_mode = train_mode
        self.curriculum_mode = curriculum_mode
        self.traj = traj
        self.traj_cnt = 0
        self.FOV = FOV
        self.distance_threshold = distance_threshold

        # 加载 hex 地图数据
        if mapdata is not None:
            first_key = next(iter(mapdata)) if mapdata else None
            if isinstance(first_key, tuple) and len(first_key) == 3:
                first_val = mapdata[first_key]
                if isinstance(first_val, dict) and 'code' in first_val:
                    # 原始 pkl 格式: {(q,r,s): {'lon','lat','code'}}
                    self.hex_mapdata_raw = mapdata
                    code_dict = {k: int(v['code']) for k, v in mapdata.items()}
                    self.mapdata = code_to_mode_matrices(code_dict)
                elif isinstance(first_val, (int, float, np.integer, np.floating)):
                    # code-only dict: {(q,r,s) → code}
                    self.hex_mapdata_raw = None
                    self.mapdata = code_to_mode_matrices(mapdata)
                else:
                    # 已经是 per-mode dict
                    self.hex_mapdata_raw = None
                    self.mapdata = mapdata
            else:
                raise ValueError(
                    "PathEnv now requires hex mapdata. "
                    "Use load_hex_mapdata_raw() to load data/hex_grid.pkl"
                )
        else:
            self.hex_mapdata_raw = None
            self.mapdata = None
        self.node_memory = set()
        self.curriculum_stage = 0
        self.min_mode_count = 1
        self.max_mode_count = len(modelist)
        if self.selected_mode is None:
            self.selected_mode = np.array(modelist)

    def _patch_or_zero(self, mode, q, r, s):
        """未选中的 mode 返回全零 FOV，避免噪声干扰。"""
        if mode in self.selected_mode:
            return get_hex_neighborhood(self.mapdata[mode], q, r, s, radius=self.FOV)
        else:
            n_cells = 3 * self.FOV**2 + 3 * self.FOV + 1
            return np.zeros(n_cells, dtype=np.float32)

    def reset(self):

        self.step_cnt = 0

        # 计算当前轨迹索引
        current_traj_idx = self.traj_cnt % len(self.traj)

        # 训练模式下随机选择 mode 组合（必然包含真实mode）
        if self.train_mode:
            max_modes = self.max_mode_count if self.curriculum_mode else len(modelist)
            max_modes = max(1, min(max_modes, len(modelist)))
            min_modes = max(1, min(self.min_mode_count, max_modes))
            num_modes = np.random.randint(min_modes, max_modes + 1)
            real_mode = str(self.traj.loc[current_traj_idx, 'mode']).strip()
            if real_mode in modelist:
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

        self.hex_start = hex_start
        self.hex_end = hex_end

        # 构建 multi_mapdata（dict 合并）
        self.multi_mapdata = {}
        for mode in self.selected_mode:
            mode_dict = self.mapdata[mode]
            for cube in mode_dict:
                self.multi_mapdata[cube] = 1

        # neighbor: 半径1六边形邻域
        self.neighbor = get_hex_neighborhood(
            self.multi_mapdata, hex_start[0], hex_start[1], hex_start[2], radius=1
        )

        h_dist = hex_distance(hex_start, hex_end)
        self.max_step = max(1, int(h_dist * 3))

        self.traj_cnt += 1

        # 引入 Node Memory
        self.node_memory = dict()

        # 计算剩余距离 cube 偏移
        rem = hex_sub(hex_end, hex_start)  # (dq, dr, ds)

        self.state = {
            'current_position': np.array([0, 0, 0]),  # cube 偏移
            'remaining_distance': np.array(rem),       # cube 偏移
            'previous_remaining_distance': np.array(rem),
            'total_distance': np.array(rem),           # 总偏移（定值）
            'current_mode': self.selected_mode,
            'patch': (
                get_hex_neighborhood(self.multi_mapdata, *hex_start, radius=self.FOV).tolist() +
                self._patch_or_zero('TG', *hex_start).tolist() +
                self._patch_or_zero('GG', *hex_start).tolist() +
                self._patch_or_zero('GSD', *hex_start).tolist() +
                self._patch_or_zero('TS', *hex_start).tolist()
            ),
            'visit_count': 0,
            'candidate_modes': set(),
        }

        start_modes = {
            mode for mode in self.selected_mode
            if self._get_map_value(mode, *hex_start) == 1
        }
        self.candidate_modes = start_modes.copy()
        self.state['candidate_modes'] = self.candidate_modes
        self.min_trans_count = 0

        return self.state

    def _read_hex_coords(self, row):
        """从 CSV 行中读取 hex cube 坐标。
        CSV 的 loczo/loczd 由三个分量独立 round 生成，可能违反 q+r+s=0。
        这里用 q/r 重新推导 s，与 hex_grid.pkl 保持一致。
        """
        locxo = float(row['locxo'])
        locyo = float(row['locyo'])
        locxd = float(row['locxd'])
        locyd = float(row['locyd'])
        loczo = -(locxo + locyo)
        loczd = -(locxd + locyd)

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

    def split_traj_by_distance(self, num_stages: int = 4):
        if self.traj is None or len(self.traj) == 0:
            return [self.traj]

        required_cols = {'locxo', 'locyo', 'loczo', 'locxd', 'locyd', 'loczd'}

        df = self.traj.copy().reset_index(drop=True)

        if required_cols.issubset(df.columns):
            df['_dist'] = df.apply(
                lambda r: hex_distance(
                    (r['locxo'], r['locyo'], r['loczo']),
                    (r['locxd'], r['locyd'], r['loczd']),
                ), axis=1
            )
        else:
            raise ValueError(f"Trajectory dataframe missing required columns: {required_cols}")

        bins = min(num_stages, max(1, int(df['_dist'].nunique())))
        if bins == 1:
            return [df.drop(columns=['_dist']).reset_index(drop=True)]

        sid = pd.qcut(df['_dist'], q=bins, labels=False, duplicates='drop')
        df['_sid'] = sid.astype(int)

        stage_trajs = []
        for stage_id in sorted(df['_sid'].unique().tolist()):
            stage_df = df[df['_sid'] == stage_id].drop(columns=['_dist', '_sid']).reset_index(drop=True)
            stage_trajs.append(stage_df)

        return stage_trajs

    def set_curriculum_stage(self, stage_idx: int, traj_subset: pd.DataFrame = None, max_mode_count: int = 4):
        self.curriculum_stage = int(stage_idx)
        self.max_mode_count = max(1, min(int(max_mode_count), len(modelist)))
        if traj_subset is not None:
            self.traj = traj_subset.reset_index(drop=True)
            self.traj_cnt = 0

    def set_mode_sampling_range(self, min_mode_count: int = 1, max_mode_count: int = 4):
        self.min_mode_count = max(1, min(int(min_mode_count), len(modelist)))
        self.max_mode_count = max(1, min(int(max_mode_count), len(modelist)))
        if self.min_mode_count > self.max_mode_count:
            self.min_mode_count = self.max_mode_count

    def calculate_reward(self, reward, prev_dist, curr_dist, neighbor, action):
        '''
        改进后的奖励函数（hex 版本）。
        neighbor: 半径1六边形邻域 [center, dir0(北), dir1(西北), dir2(西南), dir3(南), dir4(东南), dir5(东北)]
        '''
        is_on_road = neighbor[ACTION_TO_HEX_IDX[action]] != 0
        dist_change = prev_dist - curr_dist

        # 接近目标就给正奖励，远离则惩罚
        if is_on_road:
            reward += 2
            if dist_change > 0:
                reward += 2
            else:
                reward -= 2
        else:
            reward -= 2
            if dist_change > 0:
                reward += 1
            else:
                reward -= 3

        # 步数惩罚：鼓励尽快到达
        reward -= 0.3

        return reward

    def step(self, action: int):
        '''
        采取动作，计算奖励，更新状态
        '''
        success = 0
        reward = 0.0
        done = False
        self.step_cnt += 1

        # 计算移动前的距离（hex distance）
        prev_dist = hex_distance(
            (0, 0, 0),
            tuple(self.state['remaining_distance'])
        )

        # 更新位置偏移（cube coords）
        dq, dr, ds = HEX_DIRECTIONS[action]
        self.state['current_position'] = (
            self.state['current_position'][0] + dq,
            self.state['current_position'][1] + dr,
            self.state['current_position'][2] + ds,
        )

        # 更新节点记忆
        pos_key = tuple(self.state['current_position'])
        if pos_key in self.node_memory:
            self.node_memory[pos_key] += 1
        else:
            self.node_memory[pos_key] = 1

        visit_count = self.node_memory.get(pos_key)
        self.state['visit_count'] = visit_count
        # reward -= min(visit_count * 2, 2)

        # 更新绝对坐标
        self.hex_start = hex_add(self.hex_start, HEX_DIRECTIONS[action])

        curr_active_modes = {
            mode for mode in self.selected_mode
            if self._get_map_value(mode, *self.hex_start) == 1
        }
        new_candidate = self.candidate_modes & curr_active_modes
        if (self.candidate_modes or curr_active_modes) and len(new_candidate) == 0:
            self.min_trans_count += 1
            # reward -= 1
            self.candidate_modes = curr_active_modes.copy()
        else:
            self.candidate_modes = new_candidate
        self.state['candidate_modes'] = self.candidate_modes

        # 更新上一步剩余距离向量
        self.state['previous_remaining_distance'] = self.state['remaining_distance']

        # 更新剩余距离向量
        rem = hex_sub(self.hex_end, self.hex_start)
        self.state['remaining_distance'] = (rem[0], rem[1], rem[2])

        # 计算移动后的距离
        curr_dist = hex_distance((0, 0, 0), rem)

        # 计算奖励
        reward = self.calculate_reward(reward, prev_dist, curr_dist, self.neighbor, action)

        # 更新 neighbor
        self.neighbor = get_hex_neighborhood(
            self.multi_mapdata, *self.hex_start, radius=1
        )
        self.state['patch'] = (
            get_hex_neighborhood(self.multi_mapdata, *self.hex_start, radius=self.FOV).tolist() +
            self._patch_or_zero('TG', *self.hex_start).tolist() +
            self._patch_or_zero('GG', *self.hex_start).tolist() +
            self._patch_or_zero('GSD', *self.hex_start).tolist() +
            self._patch_or_zero('TS', *self.hex_start).tolist()
        )

        # 判断 done
        if curr_dist <= self.distance_threshold:
            done = True
            success = 1
            reward += 50
        elif self.step_cnt >= self.max_step:
            done = True
            reward -= 20
        else:
            done = False

        return self.state, reward, done, success



class ModeEnv:
    # TODO:重写:输入为同一ID的一批数据
    """
    Train: 选择 mode 组合（4bit），调用已训练 PathAgent 回放路径，输出匹配指标与奖励
    Test:  同样流程，但关闭扰动，使用评估动作
    """
    def __init__(
        self,
        model_path: str,
        mapdata,
        traj: pd.DataFrame,
        train_mode: bool = True,
        fov: int = 3,
        distance_threshold: float = 1.0,
        use_conv: bool = False,
    ):
        self.model_path = model_path
        self.traj = traj
        self.train_mode = train_mode
        self.fov = fov
        self.distance_threshold = distance_threshold
        self.use_conv = use_conv

        self.no_change_patience = 5
        self.no_change_streak = 0

        self.hex_radius = HEX_RADIUS
        self.traj_cnt = 0
        self.current_row = None

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
        self.mode_speed_stats = self._build_mode_speed_stats()

        cfg = SACConfig()
        device = torch.device(cfg.device)

        path_agent = DiscreteSACAgent(vec_dim=20, hex_radius=self.fov, action_dim=6,
                                       cfg=cfg, use_gnn=True, in_channels=5)
        state_dict = torch.load(self.model_path, map_location=device)
        path_agent.actor.load_state_dict(state_dict)
        path_agent.actor.eval()
        self.path_agent = path_agent

    def _mask_to_modes(self, mask):
        return [modelist[i] for i, v in enumerate(mask) if int(v) == 1]

    def _default_mode_mask(self):
        return [1, 1, 1, 1]

    def _infer_init_mode_mask(self, idx: int):
        """
        初始化 mode 状态（hex 版本）。
        """
        if len(self.traj) <= 1 or idx <= 0:
            return self._default_mode_mask()

        if "ID" not in self.traj.columns:
            return self._default_mode_mask()

        cur_row = self.traj.iloc[idx]
        prev_row = self.traj.iloc[idx - 1]

        cur_id = str(cur_row.get("ID", "")).strip()
        prev_id = str(prev_row.get("ID", "")).strip()
        if cur_id == "" or prev_id == "" or cur_id != prev_id:
            return self._default_mode_mask()

        # 读取前一个终点的 cube 坐标
        try:
            if all(c in prev_row.index for c in ['locxd', 'locyd', 'loczd']):
                q = int(round(float(prev_row["locxd"])))
                r = int(round(float(prev_row["locyd"])))
                s = int(round(float(prev_row["loczd"])))
        except Exception:
            return self._default_mode_mask()

        key = (q, r, s)
        mask = []
        for m in modelist:
            mask.append(1 if self.mode_maps[m].get(key, 0) != 0 else 0)

        if int(np.sum(mask)) == 0:
            return self._default_mode_mask()

        return mask

    def _build_mode_speed_stats(self):
        """构建各 mode 的速度统计（占位）。"""
        return {m: {'mean': 60.0, 'std': 20.0} for m in modelist}

    def _speed_deviation_reward(self, cur_mask, velocity):
        """速度偏差奖励（占位）。"""
        return 0.0

    def _run_PathMode(self, selected_modes):
        traj_one = self.current_row.reset_index(drop=True)

        env = PathEnv(
            train_mode=False,
            selected_mode=np.array(selected_modes),
            mapdata=self.mapdata,
            traj=traj_one,
            FOV=self.fov,
            distance_threshold=self.distance_threshold,
        )
        env.hex_mapdata_raw = getattr(self, 'hex_mapdata_raw', None)

        s = env.reset()
        traj_points = [(env.hex_start[0], env.hex_start[1], env.hex_start[2])]
        steps = 0
        trans_times = 0
        success = 0
        done = False

        while not done:
            s_vec = state_to_vector(s)
            a = self.path_agent.select_action(s_vec, evaluate=True)
            s, _, done, succ = env.step(int(a))
            traj_points.append((
                env.hex_start[0], env.hex_start[1], env.hex_start[2],
            ))
            steps += 1
            success = int(succ)

        path_len = float(steps)

        # trans_times 计算
        trans_times = 0
        candidate_modes = None
        NO_MODE = "__NO_MODE__"

        _mode_maps = {m: self.mapdata[m] for m in selected_modes}

        for p in traj_points:
            if p is None or len(p) < 3:
                continue
            q, r, s = int(round(p[0])), int(round(p[1])), int(round(p[2]))
            key = (q, r, s)
            curr_modes = {m for m in selected_modes if _mode_maps[m].get(key, 0) != 0}

            if not curr_modes:
                curr_modes = {NO_MODE}

            if candidate_modes is None:
                candidate_modes = set(curr_modes)
                continue

            candidate_modes &= curr_modes
            if len(candidate_modes) == 0:
                trans_times += 1
                candidate_modes = set(curr_modes)

        multi_match_rate = float(calculate_match_rate(
            [(p[0], p[1], p[2]) for p in traj_points],
            env.multi_mapdata,
        ))

        selected_set = set(selected_modes)
        mode_scores = {m: 0.0 for m in modelist}
        total_points = float(max(len(traj_points), 1))

        if len(traj_points) > 0 and len(selected_modes) > 0:
            _mode_maps = {m: self.mapdata[m] for m in selected_modes}
            for p in traj_points:
                if p is None or len(p) < 3:
                    continue
                q, r, s = int(round(p[0])), int(round(p[1])), int(round(p[2]))
                for m in selected_modes:
                    if _mode_maps[m].get((q, r, s), 0) != 0:
                        mode_scores[m] += 1.0

        match_rate = []
        for m in modelist:
            if m in selected_set:
                match_rate.append(float(mode_scores[m] / total_points))
            else:
                match_rate.append(0.0)

        return match_rate, multi_match_rate, success, steps, path_len, trans_times

    def reset(self):
        idx = self.traj_cnt % len(self.traj)
        self.current_row = self.traj.iloc[[idx]].copy()
        self.traj_cnt += 1
        self.step_cnt = 0
        self.finish = False
        self.no_change_streak = 0

        init_mode_mask = self._infer_init_mode_mask(idx)

        self.state = {
            "previous": {
                "mode": [1, 1, 1, 1],
                "match_rate": [0.0, 0.0, 0.0, 0.0],
                "multi_match_rate": 0.0,
                "success": 0,
                "steps": 0,
                "path_len": 0.0,
                "time": 0,
                "distance": 0,
                "velocity": 0,
                "trans_times": 0,
            },
            "current": {
                "mode": init_mode_mask,
                "match_rate": [0.0, 0.0, 0.0, 0.0],
                "multi_match_rate": 0.0,
                "success": 0,
                "steps": 0,
                "path_len": 0.0,
                "time": 0,
                "distance": 0,
                "velocity": 0,
                "trans_times": 0,
            }
        }

        return self.state

    def step(self, action):
        self.step_cnt += 1
        reward = 0.0

        self.state["previous"] = copy.deepcopy(self.state["current"])

        time = float(self.current_row['time'].iat[0])
        distance = float(self.current_row['distance_km'].iat[0])
        velocity = float(self.current_row['velocity'].iat[0])

        prev_mask = np.asarray(self.state["current"]["mode"], dtype=np.int64)
        if int(prev_mask.sum()) == 0:
            prev_mask[np.random.randint(0, 4)] = 1

        # action: 0~14 → 4-bit mask
        mask_int = int(action) + 1
        cur_mask = np.array([
            (mask_int >> 3) & 1,
            (mask_int >> 2) & 1,
            (mask_int >> 1) & 1,
            mask_int & 1,
        ], dtype=np.int64)
        selected_modes = self._mask_to_modes(cur_mask)

        changed = not np.array_equal(cur_mask, prev_mask)
        if changed:
            self.no_change_streak = 0
        else:
            self.no_change_streak += 1

        match_rate, multi_match_rate, success, steps, path_len, trans_times = self._run_PathMode(selected_modes)

        reward += 0.2 * success
        reward += multi_match_rate if multi_match_rate >= 0.6 else -1

        for i in range(len(cur_mask)):
            if cur_mask[i] == 1 and match_rate[i] == 0:
                reward -= 1

        reward += max(match_rate)
        reward -= min(0.5 * cur_mask.sum(), 5)
        reward -= min(0.5 * trans_times, 5)
        reward += self._speed_deviation_reward(cur_mask, velocity)

        self.state["current"] = {
            "mode": cur_mask.tolist(),
            "match_rate": match_rate,
            "multi_match_rate": multi_match_rate,
            "success": int(success),
            "steps": int(steps),
            "path_len": float(path_len),
            "time": time,
            "distance": distance,
            "velocity": velocity,
            "trans_times": trans_times,
        }

        if self.no_change_streak >= self.no_change_patience:
            done = True
            self.finish = True
        elif hasattr(self, 'max_mode_steps') and self.step_cnt >= self.max_mode_steps:
            done = True
            reward -= 100
        else:
            done = False

        success = int(success)

        return self.state, float(reward), done, success, multi_match_rate


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
        print(f"  curriculum_mode: {pathenv.curriculum_mode}")
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
        for k in ['current_position', 'remaining_distance', 'total_distance']:
            v = state[k]
            print(f"  {k}: shape={np.array(v).shape}, value={v}")
        for k in ['current_mode', 'candidate_modes', 'visit_count']:
            print(f"  {k}: {state[k]}")

        patch = np.array(state['patch'])
        n_cells = 3 * pathenv.FOV**2 + 3 * pathenv.FOV + 1
        print(f"  patch: shape={patch.shape}, expected 5×{n_cells}={5*n_cells}")
        assert patch.shape == (5 * n_cells,), \
            f"patch shape mismatch: {patch.shape} != ({5 * n_cells},)"
        print(f"  patch[0:{n_cells}] (multi) 非零数: {np.count_nonzero(patch[:n_cells])}")
        for i, m in enumerate(['TG', 'GG', 'GSD', 'TS'], 1):
            ch = patch[i*n_cells:(i+1)*n_cells]
            nonzero = np.count_nonzero(ch)
            active = "✓" if m in pathenv.selected_mode else "✗(zero)"
            print(f"  patch[{i}*{n_cells}:] ({m}): 非零数={nonzero}, active={active}")

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
                  f"mode={pathenv.selected_mode}, trans={pathenv.min_trans_count}")

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
