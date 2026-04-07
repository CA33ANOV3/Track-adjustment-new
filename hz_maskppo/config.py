# config.py
from dataclasses import dataclass
from typing import Tuple


_DEFAULT_TRACKS: Tuple[str, ...] = tuple([f"{i}G" for i in range(14, 26)] + ["XIXG", "XXG"])


@dataclass
class CFG:
    # ===== 输入文件 =====
    base_file: str = r"G:\hjh\gudao\沪杭长场基本图.xls"
    delay_file: str = r"G:\hjh\gudao\7.1-7.24杭州东沪杭长场.xls"

    up_fwd_file: str = r"G:\hjh\gudao\上行过路车.xlsx"
    down_fwd_file: str = r"G:\hjh\gudao\下行过路车.xlsx"
    up_rev_file: str = r"G:\hjh\gudao\上行过路车反.xlsx"
    down_rev_file: str = r"G:\hjh\gudao\下行过路车反.xlsx"
    cut_file: str = r"G:\hjh\gudao\切割正线.xlsx"

    station_name: str = "杭州东"
    keep_train_types: Tuple[str, ...] = ("过路", "立折")

    # ===== 数据切分 =====
    train_md: Tuple[str, ...] = tuple()
    test_md: Tuple[str, ...] = ("07-03",)

    # ===== 股道 =====
    assign_tracks: Tuple[str, ...] = _DEFAULT_TRACKS
    sewage_tracks: Tuple[str, ...] = ("17G", "18G", "21G", "22G")

    # ===== 动作（适配新版 env_hz）=====
    action_mode: str = "by_track"          # "by_track"(推荐) / "by_candidate"
    max_actions: int = 128                 # by_track 下会自动改成 len(assign_tracks)
    allow_all_tracks_fallback: bool = True
    invalid_fallback_policy: str = "best"  # "best" / "first" / "random"

    # ===== 约束开关 =====
    use_side_constraints: bool = False     # 先False，避免动作空间塌缩
    hard_hotspot_mask: bool = False        # 先False，避免mask过硬

    # ===== 间隔/停站约束（秒）=====
    min_dwell: int = 120

    ff_down_side: int = 240
    aa_up_side: int = 240
    ff_up_side: int = 180
    aa_down_side: int = 180
    fa_side: int = 360
    af_side: int = 60

    ff_same_track: int = 180
    aa_same_track: int = 180
    fa_same_track: int = 120

    # ===== 奖励（阶段A：先压总晚点）=====
    fail_penalty: float = 50000.0
    invalid_penalty: float = 1000.0
    completion_bonus: float = 200.0

    w_dep: float = 1.0
    w_change: float = 0.0
    w_reverse: float = 0.0
    w_risk: float = 0.0
    w_hot: float = 0.0
    w_load: float = 0.02                  # 新增：轻度负载惩罚

    # ===== 候选排序代价（秒）=====
    # 注意：这是“筛候选/回退best”的代价，不是直接奖励
    topk_change_sec: float = 90.0
    topk_reverse_sec: float = 180.0
    topk_risk_coef: float = 0.0
    topk_hot_sec: float = 0.0

    # 新增：股道负载惩罚（缓解单股道拥挤）
    topk_track_load_sec: float = 30.0
    topk_track_load_win: int = 1800       # ±30分钟

    # ===== 风险画像参数 =====
    risk_threshold_sec: float = 300.0
    cvar_alpha: float = 0.9
    hotspot_quantile: float = 0.8
    eb_kappa: float = 20.0
    risk_w1: float = 0.25
    risk_w2: float = 0.35
    risk_w3: float = 0.25
    risk_w4: float = 0.15

    refit_risk_in_env: bool = False
    use_rolling_risk: bool = False
    rolling_k_days: int = 7

    # ===== 业务开关 =====
    enable_through_down_turnback: bool = True

    # ===== 切分 =====
    split_mode: str = "all_minus_test"    # manual / all_minus_test / last_n_test / ratio
    test_days_n: int = 3                  # 当未指定test_md时生效
    train_ratio: float = 0.8              # ratio模式生效
    split_seed: int = 42

    # ===== PPO参数 =====
    total_timesteps: int = 800000
    learning_rate: float = 2e-4
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.99
    seed: int = 42
    device: str = "auto"

    # ===== 阶段B参考（续训时手动改）=====
    phase2_timesteps: int = 200000
    phase2_w_change: float = 0.2
    phase2_w_reverse: float = 1.0

    def __post_init__(self):
        # 1) 轨道清洗
        tracks = tuple(str(x).strip() for x in self.assign_tracks if str(x).strip())
        if len(tracks) == 0:
            tracks = _DEFAULT_TRACKS
        self.assign_tracks = tracks

        sewage = tuple(str(x).strip() for x in self.sewage_tracks if str(x).strip())
        sewage = tuple(x for x in sewage if x in self.assign_tracks)
        if len(sewage) == 0:
            fallback = ("17G", "18G", "21G", "22G")
            sewage = tuple(x for x in fallback if x in self.assign_tracks)
        self.sewage_tracks = sewage

        # 2) 动作模式与回退策略兜底
        if self.action_mode not in {"by_track", "by_candidate"}:
            self.action_mode = "by_track"

        if self.invalid_fallback_policy not in {"best", "first", "random"}:
            self.invalid_fallback_policy = "best"

        # by_track 模式下动作维度固定为轨道数
        if self.action_mode == "by_track":
            self.max_actions = len(self.assign_tracks)
        else:
            self.max_actions = max(int(self.max_actions), 1)

        # 3) 数值防御
        self.min_dwell = max(int(self.min_dwell), 0)
        self.ff_same_track = max(int(self.ff_same_track), 0)
        self.aa_same_track = max(int(self.aa_same_track), 0)
        self.fa_same_track = max(int(self.fa_same_track), 0)

        self.topk_track_load_win = max(int(self.topk_track_load_win), 0)

        self.fail_penalty = float(max(self.fail_penalty, 0.0))
        self.invalid_penalty = float(max(self.invalid_penalty, 0.0))
        self.completion_bonus = float(self.completion_bonus)

        self.w_dep = float(self.w_dep)
        self.w_change = float(self.w_change)
        self.w_reverse = float(self.w_reverse)
        self.w_risk = float(self.w_risk)
        self.w_hot = float(self.w_hot)
        self.w_load = float(self.w_load)


# 兼容不同 train 脚本加载方式
cfg = CFG()
Config = CFG
