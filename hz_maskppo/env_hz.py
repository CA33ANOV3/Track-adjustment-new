# env_hz.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
import random
import re
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from route_planner import Candidate


DEFAULT_TRACKS = [f"{i}G" for i in range(14, 26)] + ["XIXG", "XXG"]


def norm(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s


def as_date(x):
    if isinstance(x, dt.date):
        return x
    return pd.to_datetime(x).date()


def to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    s = norm(x)
    return s in {"是", "Y", "y", "True", "true", "1", "需要"}


def line_dir(line_text: str) -> str:
    s = norm(line_text)
    if "上行" in s:
        return "up"
    if "下行" in s:
        return "down"
    return "unk"


def line_side(line_text: str) -> str:
    s = norm(line_text)
    if "昆明端" in s:
        return "KM"
    if "上海端" in s:
        return "SH"
    return "UNK"


class HZMaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg,
        episodes: Dict[dt.date, pd.DataFrame],
        risk_day: Optional[Dict[dt.date, pd.DataFrame]],
        planner,
        risk,
        train_mode: bool = True,
        active_dates: Optional[List[dt.date]] = None,
    ):
        super().__init__()
        self.cfg = cfg

        self.episodes = {as_date(k): v.copy() for k, v in episodes.items()}
        self.risk_day = {as_date(k): v.copy() for k, v in (risk_day or {}).items()}

        self.planner = planner
        self.risk = risk
        self.train_mode = train_mode

        self.tracks = list(getattr(cfg, "assign_tracks", DEFAULT_TRACKS))
        self.track_set = set(self.tracks)
        self.sewage_tracks = set(getattr(cfg, "sewage_tracks", ("17G", "18G", "21G", "22G")))
        self.max_actions = int(getattr(cfg, "max_actions", 128))

        # 关键改进1：动作语义固定（推荐按股道）
        # - by_track: action i 恒表示选择 tracks[i]（稳定语义，避免“第0动作总是最优”泄露）
        # - by_candidate: 兼容旧模式（不推荐）
        self.action_mode = str(getattr(cfg, "action_mode", "by_track")).strip().lower()
        if self.action_mode not in {"by_track", "by_candidate"}:
            self.action_mode = "by_track"

        # 关键改进2：默认开启全轨兜底，提升可行动作分支
        self.allow_all_tracks_fallback = bool(getattr(cfg, "allow_all_tracks_fallback", True))

        # invalid动作回退策略
        self.invalid_fallback_policy = str(getattr(cfg, "invalid_fallback_policy", "best")).strip().lower()
        if self.invalid_fallback_policy not in {"best", "first", "random"}:
            self.invalid_fallback_policy = "best"

        # 约束（秒）
        self.min_dwell = int(getattr(cfg, "min_dwell", 120))
        self.aa_same_track = int(getattr(cfg, "aa_same_track", 180))
        self.ff_same_track = int(getattr(cfg, "ff_same_track", 180))
        self.fa_same_track = int(getattr(cfg, "fa_same_track", 120))

        # 可选咽喉约束
        self.use_side_constraints = bool(getattr(cfg, "use_side_constraints", False))
        self.ff_down_side = int(getattr(cfg, "ff_down_side", 240))
        self.aa_up_side = int(getattr(cfg, "aa_up_side", 240))
        self.ff_up_side = int(getattr(cfg, "ff_up_side", 180))
        self.aa_down_side = int(getattr(cfg, "aa_down_side", 180))
        self.fa_side = int(getattr(cfg, "fa_side", 360))
        self.af_side = int(getattr(cfg, "af_side", 60))

        # 奖励（主目标：总发车晚点最小）
        self.fail_penalty = float(getattr(cfg, "fail_penalty", 50000.0))
        self.invalid_penalty = float(getattr(cfg, "invalid_penalty", 1000.0))
        self.completion_bonus = float(getattr(cfg, "completion_bonus", 200.0))
        self.w_dep = float(getattr(cfg, "w_dep", 1.0))
        self.w_change = float(getattr(cfg, "w_change", 0.0))
        self.w_reverse = float(getattr(cfg, "w_reverse", 0.0))
        self.w_risk = float(getattr(cfg, "w_risk", 0.0))
        self.w_hot = float(getattr(cfg, "w_hot", 0.0))
        self.w_load = float(getattr(cfg, "w_load", 0.0))  # 新增：负载惩罚（可选）
        self.hard_hotspot_mask = bool(getattr(cfg, "hard_hotspot_mask", False))

        # 候选排序代价（用于“同轨道内挑最佳方案”及invalid回退）
        self.topk_change_sec = float(getattr(cfg, "topk_change_sec", 90.0))
        self.topk_reverse_sec = float(getattr(cfg, "topk_reverse_sec", 180.0))
        self.topk_risk_coef = float(getattr(cfg, "topk_risk_coef", 0.0))
        self.topk_hot_sec = float(getattr(cfg, "topk_hot_sec", 0.0))
        self.topk_track_load_sec = float(getattr(cfg, "topk_track_load_sec", 0.0))
        self.topk_track_load_win = int(getattr(cfg, "topk_track_load_win", 1800))  # ±30分钟

        self.refit_risk_in_env = bool(getattr(cfg, "refit_risk_in_env", False))

        if active_dates is None:
            self.dates = sorted(self.episodes.keys())
        else:
            tmp = [as_date(d) for d in active_dates]
            for d in tmp:
                if d not in self.episodes:
                    raise ValueError(f"active_dates 中日期不在 episodes: {d}")
            self.dates = sorted(tmp)

        if len(self.dates) == 0:
            raise ValueError("active_dates为空，环境无可用样本")

        self._eval_idx = 0

        # 动作空间
        if self.action_mode == "by_track":
            self.action_space = spaces.Discrete(len(self.tracks))
        else:
            self.action_space = spaces.Discrete(self.max_actions)
        self.action_dim = int(self.action_space.n)

        # 观测空间
        obs_dim = 12 + 3 * len(self.tracks)  # 任务特征12 + 每股道(最早可用/risk/hot)
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(obs_dim,), dtype=np.float32)

        self.current_date = None
        self.tasks = None
        self.ptr = 0

        self.scheduled: List[dict] = []
        self.day_schedule: List[dict] = []

        # preview缓存（关键改进3：防止step里重复重算导致动作映射不稳定）
        self.cur_candidates: List[Optional[Candidate]] = [None] * self.action_dim
        self.preview: Dict[int, dict] = {}
        self._mask_cache = np.zeros(self.action_dim, dtype=bool)
        self._preview_ptr = -1
        self._preview_sched_n = -1

        # KPI
        self.sum_pri = 0.0
        self.sum_sec = 0.0
        self.sum_dep = 0.0
        self.sum_change = 0
        self.sum_reverse = 0

        # 诊断指标（关键改进4：判断动作是否“真有选择”）
        self.decision_steps = 0
        self.sum_feasible_actions = 0
        self.nontrivial_steps = 0
        self.action_hist = np.zeros(self.action_dim, dtype=np.int64)

    # ---------- reset / obs ----------
    def _pick_date(self, options=None):
        if options and "date" in options:
            d = as_date(options["date"])
            if d not in self.dates:
                raise ValueError(f"指定date不在active_dates中: {d}")
            return d
        if self.train_mode:
            return random.choice(self.dates)
        d = self.dates[self._eval_idx % len(self.dates)]
        self._eval_idx += 1
        return d

    def _update_risk_for_day(self, d: dt.date):
        if not self.refit_risk_in_env:
            return
        if self.risk is None or (not hasattr(self.risk, "fit")) or len(self.risk_day) == 0:
            return

        use_rolling = bool(getattr(self.cfg, "use_rolling_risk", False))
        k_days = int(getattr(self.cfg, "rolling_k_days", 7))
        all_days = sorted(self.risk_day.keys())

        if use_rolling:
            prev_days = [x for x in all_days if x < d][-k_days:]
            rec_list = [self.risk_day[x] for x in prev_days if len(self.risk_day.get(x, [])) > 0]
            if len(rec_list) == 0:
                rec_list = [v for _, v in self.risk_day.items() if len(v) > 0]
        else:
            rec_list = [v for _, v in self.risk_day.items() if len(v) > 0]

        if len(rec_list) == 0:
            return
        self.risk.fit(pd.concat(rec_list, ignore_index=True))

    def _invalidate_preview(self):
        self.cur_candidates = [None] * self.action_dim
        self.preview = {}
        self._mask_cache = np.zeros(self.action_dim, dtype=bool)
        self._preview_ptr = -1
        self._preview_sched_n = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        d = self._pick_date(options)
        self.current_date = d
        self.tasks = self.episodes[d].copy().reset_index(drop=True)

        for c in ["plan_arr_sec", "plan_dep_sec", "init_delay_sec"]:
            if c not in self.tasks.columns:
                self.tasks[c] = 0
            self.tasks[c] = pd.to_numeric(self.tasks[c], errors="coerce").fillna(0).astype(int)

        # 保留原计划发车秒（导出对照）
        self.tasks["plan_dep_raw_sec"] = self.tasks["plan_dep_sec"]

        # 跨日修复：计划发车秒小于计划到达秒 -> 次日发车
        cross = self.tasks["plan_dep_sec"] < self.tasks["plan_arr_sec"]
        self.tasks.loc[cross, "plan_dep_sec"] = self.tasks.loc[cross, "plan_dep_sec"] + 86400
        self.tasks["cross_day_dep"] = cross.astype(int)

        # 按注入晚点后的到达时刻排序
        self.tasks["arr0_sec"] = self.tasks["plan_arr_sec"] + self.tasks["init_delay_sec"]
        self.tasks = self.tasks.sort_values(["arr0_sec", "plan_dep_sec"], kind="stable").reset_index(drop=True)

        self._update_risk_for_day(d)

        self.ptr = 0
        self.scheduled = []
        self.day_schedule = []
        self._invalidate_preview()

        self.sum_pri = 0.0
        self.sum_sec = 0.0
        self.sum_dep = 0.0
        self.sum_change = 0
        self.sum_reverse = 0

        self.decision_steps = 0
        self.sum_feasible_actions = 0
        self.nontrivial_steps = 0
        self.action_hist[:] = 0

        return self._get_obs(), {"date": str(d)}

    def _risk_phi(self, track: str, hour: int) -> float:
        if self.risk is None or (not hasattr(self.risk, "phi")):
            return 0.0
        try:
            v = float(self.risk.phi(track, hour))
            return v if np.isfinite(v) else 0.0
        except Exception:
            return 0.0

    def _risk_hot(self, track: str, hour: int) -> int:
        if self.risk is None or (not hasattr(self.risk, "hot")):
            return 0
        try:
            return int(self.risk.hot(track, hour))
        except Exception:
            return 0

    def _need_water(self, task: dict) -> bool:
        if "need_water" in task:
            return to_bool(task["need_water"])
        return to_bool(task.get("是否上水作业", False))

    def _need_sewage(self, task: dict) -> bool:
        if "need_sewage" in task:
            return to_bool(task["need_sewage"])
        return to_bool(task.get("是否吸污作业", False))

    def _get_obs(self):
        if self.ptr >= len(self.tasks):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        t = self.tasks.iloc[self.ptr].to_dict()

        plan_arr = int(t.get("plan_arr_sec", 0))
        plan_dep = int(t.get("plan_dep_sec", 0))
        init_delay = int(t.get("init_delay_sec", 0))
        ttype = norm(t.get("列车类型", ""))

        if "delay_cls" in t:
            cls = int(t["delay_cls"])
        else:
            mins = init_delay / 60.0
            cls = 0 if mins <= 5 else (1 if mins <= 30 else 2)

        progress = self.ptr / max(len(self.tasks), 1)
        cur_clock = (plan_arr + init_delay) / 86400.0
        hour = int((plan_arr + init_delay) // 3600) % 24

        feat = [
            progress,
            cur_clock,
            plan_arr / 86400.0,
            plan_dep / 86400.0,
            init_delay / 3600.0,
            1.0 if ttype == "过路" else 0.0,
            1.0 if ttype == "立折" else 0.0,
            1.0 if self._need_water(t) else 0.0,
            1.0 if self._need_sewage(t) else 0.0,
            1.0 if cls == 0 else 0.0,
            1.0 if cls == 1 else 0.0,
            1.0 if cls == 2 else 0.0,
        ]

        next_free, risk_vec, hot_vec = [], [], []
        for g in self.tracks:
            dep_times = [
                int(x.get("dep", x.get("act_dep", 0)))
                for x in self.scheduled
                if norm(x.get("track", x.get("act_track", ""))) == g
            ]
            nf = max(dep_times) if dep_times else 0
            next_free.append(nf / 86400.0)
            risk_vec.append(self._risk_phi(g, hour))
            hot_vec.append(float(self._risk_hot(g, hour)))

        return np.array(feat + next_free + risk_vec + hot_vec, dtype=np.float32)

    # ---------- feasibility ----------
    def _facility_ok(self, task: dict, track: str) -> bool:
        if self._need_water(task) and (track in {"XIXG", "XXG"}):
            return False
        if self._need_sewage(task) and (track not in self.sewage_tracks):
            return False
        return True

    def _aa_req(self, arr_dir: str) -> int:
        if arr_dir == "up":
            return self.aa_up_side
        if arr_dir == "down":
            return self.aa_down_side
        return max(self.aa_up_side, self.aa_down_side)

    def _ff_req(self, dep_dir: str) -> int:
        if dep_dir == "up":
            return self.ff_up_side
        if dep_dir == "down":
            return self.ff_down_side
        return max(self.ff_up_side, self.ff_down_side)

    def _compute_times(self, task: dict, track: str) -> Optional[Tuple[int, int, int, int, int]]:
        plan_arr = int(task.get("plan_arr_sec", 0))
        plan_dep = int(task.get("plan_dep_sec", 0))
        init_delay = int(task.get("init_delay_sec", 0))

        # 兜底：若没经过reset修复，这里再修一次跨日
        if plan_dep < plan_arr:
            plan_dep += 86400

        arr0 = max(plan_arr + init_delay, plan_arr)
        arr = arr0

        # 可选同侧咽喉约束
        if self.use_side_constraints:
            arr_side = line_side(task.get("进站线路", ""))
            arr_dir = line_dir(task.get("进站线路", ""))
            if arr_side != "UNK":
                for r in self.scheduled:
                    r_arr = int(r.get("arr", r.get("act_arr", 0)))
                    r_dep = int(r.get("dep", r.get("act_dep", 0)))
                    r_arr_side = norm(r.get("arr_side", "UNK"))
                    r_dep_side = norm(r.get("dep_side", "UNK"))

                    if r_arr_side == arr_side:
                        arr = max(arr, r_arr + self._aa_req(arr_dir))
                    if r_dep_side == arr_side:
                        arr = max(arr, r_dep + self.fa_side)

        # 同股道 到到 / 发到
        for r in self.scheduled:
            r_track = norm(r.get("track", r.get("act_track", "")))
            if r_track != track:
                continue
            r_arr = int(r.get("arr", r.get("act_arr", 0)))
            r_dep = int(r.get("dep", r.get("act_dep", 0)))
            arr = max(arr, r_arr + self.aa_same_track)
            arr = max(arr, r_dep + self.fa_same_track)

        # 发车下界：不早发 + 最短停站
        dep = max(plan_dep, arr + self.min_dwell)

        # 可选同侧 发发 / 到发
        if self.use_side_constraints:
            dep_side = line_side(task.get("出站线路", ""))
            dep_dir = line_dir(task.get("出站线路", ""))
            if dep_side != "UNK":
                for r in self.scheduled:
                    r_arr = int(r.get("arr", r.get("act_arr", 0)))
                    r_dep = int(r.get("dep", r.get("act_dep", 0)))
                    r_arr_side = norm(r.get("arr_side", "UNK"))
                    r_dep_side = norm(r.get("dep_side", "UNK"))

                    if r_dep_side == dep_side:
                        dep = max(dep, r_dep + self._ff_req(dep_dir))
                    if r_arr_side == dep_side:
                        dep = max(dep, r_arr + self.af_side)

        # 同股道 发发
        for r in self.scheduled:
            r_track = norm(r.get("track", r.get("act_track", "")))
            if r_track != track:
                continue
            r_dep = int(r.get("dep", r.get("act_dep", 0)))
            dep = max(dep, r_dep + self.ff_same_track)

        if dep > 172800:
            return None

        dep_delay = max(0, dep - plan_dep)
        pri = max(0, arr0 - plan_arr)
        sec = max(0, dep_delay - pri)
        return arr, dep, pri, sec, dep_delay

    def _track_load(self, track: str, t_sec: int) -> int:
        """统计同股道在 [t-win, t+win] 时间窗内有重叠的已排车数量。"""
        if self.topk_track_load_sec <= 0:
            return 0
        win = max(int(self.topk_track_load_win), 0)
        if win <= 0:
            return 0

        left, right = t_sec - win, t_sec + win
        cnt = 0
        for r in self.scheduled:
            r_track = norm(r.get("track", r.get("act_track", "")))
            if r_track != track:
                continue
            r_arr = int(r.get("arr", r.get("act_arr", 0)))
            r_dep = int(r.get("dep", r.get("act_dep", 0)))
            if not (r_dep < left or r_arr > right):
                cnt += 1
        return cnt

    def _rank_score(self, dep_delay: int, change: int, reverse: int, risk: float, hot: int, load: int) -> float:
        return float(
            dep_delay
            + self.topk_change_sec * change
            + self.topk_reverse_sec * reverse
            + self.topk_risk_coef * risk
            + self.topk_hot_sec * hot
            + self.topk_track_load_sec * load
        )

    def _prepare_action_preview(self):
        # 缓存命中：同一个状态不重复重算
        if self._preview_ptr == self.ptr and self._preview_sched_n == len(self.scheduled):
            return

        self.cur_candidates = [None] * self.action_dim
        self.preview = {}
        self._mask_cache[:] = False

        if self.ptr >= len(self.tasks):
            self._preview_ptr = self.ptr
            self._preview_sched_n = len(self.scheduled)
            return

        task = self.tasks.iloc[self.ptr].to_dict()

        # 1) 原始候选
        try:
            all_cands = self.planner.build_candidates(task)
        except Exception:
            all_cands = []

        # 保底原股道
        plan_track = norm(task.get("接入股道", ""))
        if plan_track and all(norm(c.track) != plan_track for c in all_cands):
            all_cands.append(Candidate("保底原股道", plan_track, plan_track, reverse_cnt=0))

        # 全轨兜底（默认开启）
        if self.allow_all_tracks_fallback:
            for g in self.tracks:
                if all(norm(c.track) != g for c in all_cands):
                    all_cands.append(Candidate("全轨备选", g, g, reverse_cnt=0))

        # 2) 去重 + 过滤轨道
        uniq = []
        seen = set()
        for c in all_cands:
            key = (norm(c.scheme), norm(c.track), norm(c.stage1_end), int(getattr(c, "reverse_cnt", 0)))
            if key in seen:
                continue
            seen.add(key)
            if norm(c.track) in self.track_set:
                uniq.append(c)

        if len(uniq) == 0:
            self._preview_ptr = self.ptr
            self._preview_sched_n = len(self.scheduled)
            return

        # 3) 计算每个候选的时刻与代价
        scored: List[Tuple[float, int, int, int, int, float, Candidate, dict]] = []
        for c in uniq:
            tr = norm(c.track)
            if not self._facility_ok(task, tr):
                continue

            tm = self._compute_times(task, tr)
            if tm is None:
                continue

            arr, dep, pri, sec, dep_delay = tm
            hour = int(arr // 3600) % 24

            risk_v = float(self._risk_phi(tr, hour))
            hot_v = int(self._risk_hot(tr, hour))
            change_v = int(norm(task.get("接入股道", "")) != tr)
            reverse_v = int(getattr(c, "reverse_cnt", 0))
            load_v = int(self._track_load(tr, arr))
            rank_score = self._rank_score(dep_delay, change_v, reverse_v, risk_v, hot_v, load_v)

            detail = {
                "arr": int(arr),
                "dep": int(dep),
                "pri": int(pri),
                "sec": int(sec),
                "dep_delay": int(dep_delay),
                "risk": float(risk_v),
                "hot": int(hot_v),
                "change": int(change_v),
                "reverse": int(reverse_v),
                "load": int(load_v),
                "track": tr,
                "scheme": norm(c.scheme),
                "stage1_end": norm(c.stage1_end),
                "rank_score": float(rank_score),
            }
            scored.append((rank_score, int(dep_delay), load_v, change_v, reverse_v, risk_v, c, detail))

        if len(scored) == 0:
            self._preview_ptr = self.ptr
            self._preview_sched_n = len(self.scheduled)
            return

        # 4) 映射动作索引
        if self.action_mode == "by_track":
            # 每个股道仅保留“该股道最优方案”，动作语义稳定
            best_by_track: Dict[str, Tuple[Tuple, Candidate, dict]] = {}
            for item in scored:
                rank_score, dep_delay, load_v, change_v, reverse_v, risk_v, cand, detail = item
                tr = detail["track"]
                key = (
                    rank_score, dep_delay, load_v, change_v, reverse_v, risk_v,
                    detail["scheme"], detail["stage1_end"]
                )
                if tr not in best_by_track or key < best_by_track[tr][0]:
                    best_by_track[tr] = (key, cand, detail)

            for i, tr in enumerate(self.tracks):
                if tr in best_by_track:
                    self.cur_candidates[i] = best_by_track[tr][1]
                    self.preview[i] = best_by_track[tr][2]
                    self._mask_cache[i] = True

        else:
            # 兼容旧模式：不按rank_score直接排序，避免“动作0≈最优”泄露
            scored.sort(key=lambda x: (
                x[7]["track"], x[7]["scheme"], x[7]["stage1_end"],
                x[3], x[4], x[2], x[1], x[0], x[5]
            ))
            top = scored[: self.action_dim]
            for i, item in enumerate(top):
                self.cur_candidates[i] = item[6]
                self.preview[i] = item[7]
                self._mask_cache[i] = True

        self._preview_ptr = self.ptr
        self._preview_sched_n = len(self.scheduled)

    # ---------- mask ----------
    def action_masks(self):
        self._prepare_action_preview()
        mask = self._mask_cache.copy()

        if self.hard_hotspot_mask:
            feasible = np.where(mask)[0].tolist()
            non_hot = [i for i in feasible if self.preview[i]["hot"] == 0]
            # 仅当仍有可行动作时才硬过滤，避免把mask清空
            if len(non_hot) > 0:
                mask[:] = False
                mask[non_hot] = True

        return mask.astype(bool)

    def _fallback_action(self, feasible: List[int]) -> int:
        if len(feasible) == 0:
            return 0

        if self.invalid_fallback_policy == "random":
            return int(random.choice(feasible))
        if self.invalid_fallback_policy == "first":
            return int(feasible[0])

        # best
        return int(min(feasible, key=lambda i: self.preview[i]["rank_score"]))

    # ---------- step ----------
    def step(self, action):
        if self.ptr >= len(self.tasks):
            return self._get_obs(), 0.0, True, False, {}

        done = False
        trunc = False
        info = {}

        mask = self.action_masks()
        feasible = np.where(mask)[0].tolist()

        if len(feasible) == 0:
            reward = -self.fail_penalty
            done = True
            info["reason"] = "no_feasible_action"
            info["episode_kpi"] = self._kpi()
            return self._get_obs(), float(reward), done, trunc, info

        try:
            action = int(action)
        except Exception:
            action = -1

        reward = 0.0
        if (action < 0) or (action >= self.action_dim) or (not mask[action]):
            reward -= self.invalid_penalty
            action = self._fallback_action(feasible)

        p = self.preview[action]
        task = self.tasks.iloc[self.ptr].to_dict()

        # 诊断计数
        self.decision_steps += 1
        self.sum_feasible_actions += len(feasible)
        if len(feasible) > 1:
            self.nontrivial_steps += 1
        if 0 <= action < self.action_dim:
            self.action_hist[action] += 1

        # 主奖励
        reward -= self.w_dep * (p["dep_delay"] / 60.0)
        reward -= self.w_change * p["change"]
        reward -= self.w_reverse * p["reverse"]
        reward -= self.w_risk * p["risk"]
        reward -= self.w_hot * p["hot"]
        reward -= self.w_load * p["load"]

        rec = {
            "date": self.current_date,
            "train": norm(task.get("列车", "")),
            "type": norm(task.get("列车类型", "")),
            "plan_track": norm(task.get("接入股道", "")),
            "plan_arr": int(task.get("plan_arr_sec", 0)),
            "plan_dep": int(task.get("plan_dep_raw_sec", task.get("plan_dep_sec", 0))),
            "plan_dep_adj": int(task.get("plan_dep_sec", 0)),
            "cross_day_dep": int(task.get("cross_day_dep", 0)),

            # 内部字段
            "track": p["track"],
            "arr": int(p["arr"]),
            "dep": int(p["dep"]),

            # 导出别名
            "act_track": p["track"],
            "act_arr": int(p["arr"]),
            "act_dep": int(p["dep"]),

            "scheme": p["scheme"],
            "stage1_end": p["stage1_end"],
            "pri_delay": int(p["pri"]),
            "sec_delay": int(p["sec"]),
            "dep_delay": int(p["dep_delay"]),
            "load": int(p["load"]),
            "action_idx": int(action),

            "arr_side": line_side(task.get("进站线路", "")),
            "dep_side": line_side(task.get("出站线路", "")),
            "arr_dir": line_dir(task.get("进站线路", "")),
            "dep_dir": line_dir(task.get("出站线路", "")),
        }

        self.scheduled.append(rec)
        self.day_schedule.append(rec)

        self.sum_pri += p["pri"]
        self.sum_sec += p["sec"]
        self.sum_dep += p["dep_delay"]
        self.sum_change += p["change"]
        self.sum_reverse += p["reverse"]

        self.ptr += 1
        self._invalidate_preview()  # 状态已变，清缓存

        if self.ptr >= len(self.tasks):
            done = True
            reward += self.completion_bonus
            info["episode_kpi"] = self._kpi()

        return self._get_obs(), float(reward), done, trunc, info

    # ---------- output ----------
    def _kpi(self):
        n = max(len(self.tasks), 1)
        steps = max(self.decision_steps, 1)
        return {
            "date": str(self.current_date),
            "trains": int(len(self.tasks)),
            "sum_dep_delay_sec": float(self.sum_dep),
            "avg_dep_delay_sec": float(self.sum_dep / n),
            "sum_pri_sec": float(self.sum_pri),
            "sum_sec_sec": float(self.sum_sec),
            "track_change_cnt": int(self.sum_change),
            "reverse_cnt": int(self.sum_reverse),

            # 诊断信息（判断策略是否有控制力）
            "avg_feasible_actions": float(self.sum_feasible_actions / steps),
            "nontrivial_ratio": float(self.nontrivial_steps / steps),
            "unique_actions_used": int((self.action_hist > 0).sum()),
            "action_mode": self.action_mode,
            "action_dim": int(self.action_dim),
        }

    def export_day_schedule(self, path: str):
        cols = [
            "date", "train", "type",
            "plan_track", "track", "act_track",
            "scheme", "stage1_end",
            "plan_arr", "plan_dep", "plan_dep_adj", "cross_day_dep",
            "arr", "dep", "act_arr", "act_dep",
            "pri_delay", "sec_delay", "dep_delay",
            "arr_side", "dep_side", "arr_dir", "dep_dir",
            "load", "action_idx",
        ]
        if len(self.day_schedule) == 0:
            pd.DataFrame([], columns=cols).to_excel(path, index=False)
        else:
            df = pd.DataFrame(self.day_schedule)
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            df[cols].to_excel(path, index=False)