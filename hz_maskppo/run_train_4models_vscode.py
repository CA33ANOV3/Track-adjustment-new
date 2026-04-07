# -*- coding: utf-8 -*-
from __future__ import annotations

"""
run_train_4models_vscode.py
四模型流程：
1) 每个模型独立生成 train/eval 场景注入包
2) 注入包转 RL data_pack
3) 每模型多seed训练（支持按模型配置 warm-start / lr_override）
   并且支持冻结模型复用checkpoint（可跳过base/exp1训练）
4) 交叉评估输出 mean/std/95%CI（支持评估子集）
5) 专长验收：expk@expk 相对 base 的提升率与CI（支持目标子集）
"""

import json
import os
import pickle
import random
import re
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import get_schedule_fn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CFG
from env_hz import HZMaskEnv
from risk_profile import RiskProfile
from route_planner import RoutePlanner


# =========================
# 路径与开关
# =========================
PYTHON = sys.executable

SCENE02 = ROOT / "scene_02_generate_plan.py"
SCENE03 = ROOT / "scene_03_inject_plan.py"
LIB_JSON = ROOT / "artifacts" / "scene_lib" / "scene_library.json"

OUT_ROOT = ROOT / "artifacts" / "four_models"
PLAN_ROOT = OUT_ROOT / "plans"
INJECT_ROOT = OUT_ROOT / "injected"
RLPACK_ROOT = OUT_ROOT / "rlpacks"
CKPT_ROOT = OUT_ROOT / "checkpoints"
REPORT_ROOT = OUT_ROOT / "reports"

SUMMARY_JSON = REPORT_ROOT / "run_summary.json"
SUMMARY_XLSX = REPORT_ROOT / "run_summary.xlsx"

ENABLE_TRAIN = True
ENABLE_CROSS_EVAL = True
USE_BASE_INIT_FOR_EXPERTS = False  # 全局开关（按模型 init_from_base 可单独控制）

# =========================
# 提速与范围控制
# =========================
FREEZE_MODELS = {"base", "exp1"}      # 这些模型不训练，直接复用已有checkpoint
AUTO_TRAIN_IF_MISSING = False         # frozen模型缺checkpoint时，是否自动回退训练

# 评估子集（可进一步提速）
EVAL_MODEL_KEYS = ["base", "exp2", "exp3"]
EVAL_SET_KEYS = ["exp2", "exp3"]

# 专长验收目标（默认聚焦未稳定模型）
SPECIALIZATION_TARGETS = ["exp2", "exp3"]

# =========================
# 场景生成参数
# =========================
PROFILE = "moderate_aug"
QUOTA_JITTER = 0.20
DELAY_NOISE_STD = 0.10
MAX_TIME_SHIFT_MIN = 6
COMBINE_MODE = "sum"

N_DAYS_TRAIN = 60
N_DAYS_EVAL = 30
START_DATE_TRAIN = "2024-08-01"
START_DATE_EVAL = "2024-11-01"
DATE_GAP_DAYS = 45  # 不同模型日期段错开，避免重叠

# =========================
# 训练参数
# =========================
TRAIN_SEEDS = [42, 52, 62, 72, 82]
BASE_STEPS = 1_500_000
EXPERT_STEPS = 1_500_000

# 评估参数
EVAL_DETERMINISTIC = True
EVAL_BASE_SEED = 12345

# 显著性相关
N_BOOT = 3000
CI_ALPHA = 0.05  # 95%CI
SPECIALIZATION_MIN_IMPROVE = 0.05  # 5%

# =========================
# 四模型配置
# 支持字段：
# - init_from_base: bool      是否按seed从base warm-start
# - lr_override: float|None   该模型训练学习率覆盖
# =========================
MODEL_SPECS = OrderedDict({
    "base": {
        "scene_mix": "单列车短时晚点:4,混合型晚点场景:4,大面积晚点干扰:2",
        "intensity_low": 0.95,
        "intensity_high": 1.35,
        "overlap_rate": 0.35,
        "plan_seed": 101,
        "steps": BASE_STEPS,
        "init_from_base": False,
        "lr_override": None,
    },
    "exp1": {
        "scene_mix": "单列车短时晚点:8,混合型晚点场景:2,大面积晚点干扰:0",
        "intensity_low": 0.80,
        "intensity_high": 1.10,
        "overlap_rate": 0.10,
        "plan_seed": 201,
        "steps": EXPERT_STEPS,
        "init_from_base": False,
        "lr_override": None,
    },
    "exp2": {
        "scene_mix": "单列车短时晚点:4,混合型晚点场景:5,大面积晚点干扰:1",
        "intensity_low": 0.90,
        "intensity_high": 1.20,
        "overlap_rate": 0.20,
        "plan_seed": 301,
        "steps": EXPERT_STEPS,
        "init_from_base": False,  # 按稳态补丁：先关闭
        "lr_override": 2e-4,
    },
    "exp3": {
        "scene_mix": "单列车短时晚点:2,混合型晚点场景:3,大面积晚点干扰:5",
        "intensity_low": 1.15,
        "intensity_high": 1.70,
        "overlap_rate": 0.45,
        "plan_seed": 401,
        "steps": EXPERT_STEPS,
        "init_from_base": True,   # 按稳态补丁：打开warm-start
        "lr_override": 1e-4,
    },
})


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rel(p: Optional[Path]) -> Optional[str]:
    if p is None:
        return None
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def seed_hash(*parts: Any) -> int:
    s = "|".join(str(x) for x in parts)
    return 1000 + sum(ord(c) for c in s)


def date_add(s: str, days: int) -> str:
    dt0 = datetime.strptime(s, "%Y-%m-%d")
    return (dt0 + timedelta(days=days)).strftime("%Y-%m-%d")


def sort_date_key(x) -> Tuple[int, Any]:
    ts = pd.to_datetime(str(x), errors="coerce")
    if pd.isna(ts):
        return (1, str(x))
    return (0, ts.to_pydatetime())


def run_cmd(name: str, cmd: List[str], cwd: Path = ROOT):
    print("\n" + "=" * 80)
    print(f"[RUN] {name}")
    print("[CMD]", " ".join([f'"{c}"' if " " in str(c) else str(c) for c in cmd]))
    t0 = time.time()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    print(f"[DONE] {name} 用时 {time.time() - t0:.2f}s")


def _fmtf(x: Any, digits: int = 3) -> str:
    try:
        xv = float(x)
        if np.isfinite(xv):
            return f"{xv:.{digits}f}"
    except Exception:
        pass
    return "nan"


def _tb_root() -> Path:
    return OUT_ROOT / "tb"


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def expected_ckpt(model_key: str, seed: int) -> Path:
    return CKPT_ROOT / model_key / f"{model_key}_s{int(seed)}.zip"


# =========================
# 基本图读取（复用 data_prepare 逻辑）
# =========================
def read_excel_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="xlrd")
    except Exception:
        return pd.read_excel(path)


def find_col(df: pd.DataFrame, keys, required=True):
    cols = [str(c).strip() for c in df.columns]
    for k in keys:
        for i, c in enumerate(cols):
            if c == k or (k in c):
                return df.columns[i]
    if required:
        raise KeyError(f"未找到列 {keys}，现有列: {list(df.columns)}")
    return None


def norm_train_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    s = s.replace("次", "").replace(" ", "")
    if re.search(r"[A-Za-z]", s):
        s = s.upper()
    return s


def hms_to_sec(h, m, s):
    h = int(float(h)) if pd.notna(h) else 0
    m = int(float(m)) if pd.notna(m) else 0
    s = int(float(s)) if pd.notna(s) else 0
    return h * 3600 + m * 60 + s


def parse_clock_to_sec(x):
    if pd.isna(x):
        return None
    if isinstance(x, pd.Timestamp):
        return int(x.hour) * 3600 + int(x.minute) * 60 + int(x.second)

    if isinstance(x, (int, np.integer, float, np.floating)):
        v = float(x)
        if np.isnan(v):
            return None
        if 0 <= v < 1.2:  # excel 小数天
            return int(round(v * 86400))
        iv = int(round(v))
        if 0 <= iv <= 235959:  # HHMMSS
            s = f"{iv:06d}"
            hh, mm, ss = int(s[:2]), int(s[2:4]), int(s[4:])
            if hh < 24 and mm < 60 and ss < 60:
                return hh * 3600 + mm * 60 + ss
        if 0 <= iv < 172800:  # 秒
            return iv
        return None

    s = str(x).strip()
    if not s:
        return None

    s = s.replace("：", ":").replace("时", ":").replace("分", ":").replace("秒", "")
    m = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3) or 0)
        if hh < 24 and mm < 60 and ss < 60:
            return hh * 3600 + mm * 60 + ss

    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return None
    return int(t.hour) * 3600 + int(t.minute) * 60 + int(t.second)


def load_base_std(cfg: CFG) -> pd.DataFrame:
    base_path = Path(cfg.base_file)
    if not base_path.exists():
        raise FileNotFoundError(f"base_file不存在: {base_path}")

    base = read_excel_any(str(base_path)).copy()

    c_train = find_col(base, ["列车", "车次", "客票车次"])
    c_type = find_col(base, ["列车类型"])
    c_track = find_col(base, ["接入股道", "股道"])
    c_in = find_col(base, ["进站线路"])
    c_out = find_col(base, ["出站线路"])

    c_water = find_col(base, ["是否上水作业", "上水作业"], required=False)
    c_sewage = find_col(base, ["是否吸污作业", "吸污作业"], required=False)

    c_arr_h = find_col(base, ["到达时间(时)"], required=False)
    c_arr_m = find_col(base, ["到达时间(分)"], required=False)
    c_arr_s = find_col(base, ["到达时间(秒)"], required=False)

    c_dep_h = find_col(base, ["出发时间(时)"], required=False)
    c_dep_m = find_col(base, ["出发时间(分)"], required=False)
    c_dep_s = find_col(base, ["出发时间(秒)"], required=False)

    if c_arr_h is not None and c_arr_m is not None and c_arr_s is not None:
        arr_sec = base.apply(lambda r: hms_to_sec(r[c_arr_h], r[c_arr_m], r[c_arr_s]), axis=1)
    else:
        c_arr_t = find_col(base, ["到达时刻", "到达时间", "计划到达"], required=True)
        arr_sec = base[c_arr_t].map(lambda x: parse_clock_to_sec(x) if parse_clock_to_sec(x) is not None else 0)

    if c_dep_h is not None and c_dep_m is not None and c_dep_s is not None:
        dep_sec = base.apply(lambda r: hms_to_sec(r[c_dep_h], r[c_dep_m], r[c_dep_s]), axis=1)
    else:
        c_dep_t = find_col(base, ["出发时刻", "出发时间", "计划出发"], required=True)
        dep_sec = base[c_dep_t].map(lambda x: parse_clock_to_sec(x) if parse_clock_to_sec(x) is not None else 0)

    base_std = pd.DataFrame({
        "列车": base[c_train].map(norm_train_id),
        "列车类型": base[c_type].astype(str).str.strip(),
        "接入股道": base[c_track].map(norm_train_id),
        "进站线路": base[c_in].map(norm_train_id),
        "出站线路": base[c_out].map(norm_train_id),
        "是否上水作业": base[c_water].values if c_water is not None else 0,
        "是否吸污作业": base[c_sewage].values if c_sewage is not None else 0,
        "plan_arr_sec": pd.to_numeric(arr_sec, errors="coerce").fillna(0).astype(int),
        "plan_dep_sec": pd.to_numeric(dep_sec, errors="coerce").fillna(0).astype(int),
    })

    keep_types = set(getattr(cfg, "keep_train_types", tuple()))
    if len(keep_types) > 0:
        base_std = base_std[base_std["列车类型"].isin(keep_types)].copy()

    base_std = base_std.reset_index(drop=True)
    if len(base_std) == 0:
        raise ValueError("基本图过滤后为空，请检查 keep_train_types 或列名映射")

    return base_std


# =========================
# 注入包 -> 训练包(data_pack.pkl) 转换
# =========================
def _load_delay_map_from_injected(inj_obj: dict) -> "OrderedDict[date, Dict[str, float]]":
    raw = inj_obj.get("date_to_train_delay_sec")
    if not isinstance(raw, dict):
        raw = inj_obj.get("date_to_delay")
    if not isinstance(raw, dict):
        raise ValueError("injected_pack中未找到 date_to_train_delay_sec/date_to_delay")

    out = OrderedDict()
    for dk, dv in sorted(raw.items(), key=lambda kv: sort_date_key(kv[0])):
        ts = pd.to_datetime(str(dk), errors="coerce")
        if pd.isna(ts):
            continue
        d = ts.date()
        mp: Dict[str, float] = {}
        if isinstance(dv, dict):
            for tid, sec in dv.items():
                t = norm_train_id(tid)
                try:
                    s = float(sec)
                except Exception:
                    continue
                if not np.isfinite(s) or s <= 0:
                    continue
                old = mp.get(t, 0.0)
                if s > old:
                    mp[t] = s
        out[d] = mp
    return out


def convert_injected_to_data_pack(
    injected_pack_path: Path,
    out_data_pack_path: Path,
    base_std: pd.DataFrame,
    split_name: str = "train",
) -> Dict[str, Any]:
    with open(injected_pack_path, "rb") as f:
        inj = pickle.load(f)

    day_map = _load_delay_map_from_injected(inj)
    if len(day_map) == 0:
        raise ValueError(f"注入包无有效按天延误: {injected_pack_path}")

    episodes = OrderedDict()
    risk_day = OrderedDict()
    delayed_ratio_list = []

    base_df = base_std.copy()
    base_df["列车"] = base_df["列车"].map(norm_train_id)

    for d, dmap in day_map.items():
        day_df = base_df.copy()
        init_delay = day_df["列车"].map(lambda x: int(round(float(dmap.get(x, 0.0)))))
        day_df["date"] = d
        day_df["init_delay_sec"] = init_delay.astype(int)
        day_df["hist_sec_delay_sec"] = 0
        day_df["obs_arr_sec"] = (day_df["plan_arr_sec"] + day_df["init_delay_sec"]).astype(int)
        day_df["obs_dep_sec"] = (day_df["plan_dep_sec"] + day_df["init_delay_sec"]).astype(int)
        day_df["obs_dep_delay_sec"] = day_df["init_delay_sec"].astype(int)
        day_df["arr0_sec"] = (day_df["plan_arr_sec"] + day_df["init_delay_sec"]).astype(int)
        day_df = day_df.sort_values(["arr0_sec", "plan_dep_sec"], kind="stable").reset_index(drop=True)

        episodes[d] = day_df

        risk_day[d] = pd.DataFrame({
            "track": day_df["接入股道"].map(norm_train_id),
            "hour": (day_df["arr0_sec"] // 3600).astype(int) % 24,
            "sec_delay": day_df["init_delay_sec"].astype(float),  # 用注入晚点作为风险样本
            "fail": 0,
        })

        delayed_ratio_list.append(float((day_df["init_delay_sec"] >= 60).mean()))

    all_days = sorted(episodes.keys())

    if split_name == "train":
        train_days = all_days
        test_days = []
    else:
        train_days = all_days
        test_days = all_days

    out_obj = {
        "episodes": episodes,
        "risk_day": risk_day,
        "train_days": train_days,
        "test_days": test_days,
        "all_days": all_days,
    }

    ensure_parent(out_data_pack_path)
    with open(out_data_pack_path, "wb") as f:
        pickle.dump(out_obj, f)

    stats = {
        "days": int(len(all_days)),
        "trains_per_day": int(len(base_df)),
        "avg_delayed_ratio_ge_60s": float(np.mean(delayed_ratio_list)) if delayed_ratio_list else 0.0,
    }
    return stats


def read_manifest_stats(manifest_xlsx: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not manifest_xlsx.exists():
        return out
    try:
        df = pd.read_excel(manifest_xlsx, sheet_name="scene_manifest")
        out["scene_type_counts"] = {str(k): int(v) for k, v in df["scene_type"].value_counts().to_dict().items()}

        if "events" in df.columns:
            s = pd.to_numeric(
                df.loc[df["scene_type"] == "单列车短时晚点", "events"],
                errors="coerce",
            ).dropna().astype(int)
        else:
            s = pd.Series([], dtype=int)

        if len(s) > 0:
            out["single_events_dist"] = {int(k): int(v) for k, v in s.value_counts().sort_index().to_dict().items()}
            out["single_1_3_ratio"] = float(s.between(1, 3).mean())
        else:
            out["single_events_dist"] = {}
            out["single_1_3_ratio"] = None
    except Exception as e:
        out["error"] = str(e)
    return out


def generate_scene_and_rlpack(
    cfg: CFG,
    base_std: pd.DataFrame,
    model_key: str,
    split: str,
    scene_mix: str,
    intensity_low: float,
    intensity_high: float,
    overlap_rate: float,
    n_days: int,
    start_date: str,
    seed: int,
) -> Dict[str, Any]:
    plan_path = PLAN_ROOT / model_key / f"plan_{model_key}_{split}.json"
    injected_pack_path = INJECT_ROOT / model_key / f"injected_{model_key}_{split}.pkl"
    manifest_path = INJECT_ROOT / model_key / f"manifest_{model_key}_{split}.xlsx"
    rl_pack_path = RLPACK_ROOT / model_key / f"data_pack_{model_key}_{split}.pkl"

    ensure_parent(plan_path)
    ensure_parent(injected_pack_path)
    ensure_parent(manifest_path)
    ensure_parent(rl_pack_path)

    cmd2 = [
        PYTHON, str(SCENE02),
        "--library-json", str(LIB_JSON),
        "--out-plan", str(plan_path),
        "--profile", PROFILE,
        "--scene-mix", scene_mix,
        "--intensity-low", str(float(intensity_low)),
        "--intensity-high", str(float(intensity_high)),
        "--n-days", str(n_days),
        "--start-date", start_date,
        "--quota-jitter", str(QUOTA_JITTER),
        "--delay-noise-std", str(DELAY_NOISE_STD),
        "--max-time-shift-min", str(MAX_TIME_SHIFT_MIN),
        "--overlap-rate", str(float(overlap_rate)),
        "--seed", str(seed),
    ]
    run_cmd(f"scene_02 [{model_key}/{split}]", cmd2)

    cmd3 = [
        PYTHON, str(SCENE03),
        "--plan-json", str(plan_path),
        "--base-file", str(cfg.base_file),
        "--out-pack", str(injected_pack_path),
        "--out-manifest", str(manifest_path),
        "--combine-mode", COMBINE_MODE,
        "--seed", str(seed),
    ]
    run_cmd(f"scene_03 [{model_key}/{split}]", cmd3)

    conv_stats = convert_injected_to_data_pack(
        injected_pack_path=injected_pack_path,
        out_data_pack_path=rl_pack_path,
        base_std=base_std,
        split_name=split,
    )
    man_stats = read_manifest_stats(manifest_path)

    return {
        "plan_json": str(plan_path),
        "injected_pack": str(injected_pack_path),
        "manifest_xlsx": str(manifest_path),
        "data_pack": str(rl_pack_path),
        "convert_stats": conv_stats,
        "manifest_stats": man_stats,
    }


# =========================
# 训练与评估（改进版）
# =========================
def _fit_risk_profile(cfg: CFG, risk_day: Dict[date, pd.DataFrame], days: List[date]) -> RiskProfile:
    tracks = list(getattr(cfg, "assign_tracks", []))
    risk = RiskProfile(cfg, tracks=tracks)

    rec_list = [risk_day[d] for d in days if d in risk_day and len(risk_day[d]) > 0]
    if len(rec_list) > 0:
        risk_df = pd.concat(rec_list, ignore_index=True)
    else:
        risk_df = pd.DataFrame(columns=["track", "hour", "sec_delay", "fail"])
    risk.fit(risk_df)
    return risk


def _safe_action_to_int(action: Any) -> int:
    try:
        arr = np.asarray(action).reshape(-1)
        return int(arr[0])
    except Exception:
        try:
            return int(action)
        except Exception:
            return -1


def _check_mask_once(env: HZMaskEnv, seed: int) -> Dict[str, Any]:
    env.reset(seed=seed)
    mask = np.asarray(env.action_masks(), dtype=bool)
    ok_shape = (mask.ndim == 1 and mask.shape[0] == int(env.action_space.n))
    feasible_cnt = int(mask.sum()) if ok_shape else -1
    return {
        "ok_shape": ok_shape,
        "feasible_cnt": feasible_cnt,
        "action_dim": int(env.action_space.n),
    }


def train_one_model(
    cfg: CFG,
    data_pack_path: Path,
    out_prefix: Path,
    total_steps: int,
    seed: int,
    init_model_zip: Optional[Path] = None,
    lr_override: Optional[float] = None,
    tb_log_name: Optional[str] = None,
) -> Path:
    _set_global_seed(int(seed))

    with open(data_pack_path, "rb") as f:
        pack = pickle.load(f)

    episodes = pack["episodes"]
    risk_day = pack["risk_day"]

    train_days = sorted(pack.get("train_days", []))
    if len(train_days) == 0:
        train_days = sorted(episodes.keys())

    train_episodes = {d: episodes[d] for d in train_days if d in episodes}
    train_risk_day = {d: risk_day[d] for d in train_days if d in risk_day}

    if len(train_episodes) == 0:
        raise ValueError(f"训练集为空: {data_pack_path}")

    risk = _fit_risk_profile(cfg, train_risk_day, train_days)
    planner = RoutePlanner(cfg)
    env = HZMaskEnv(
        cfg=cfg,
        episodes=train_episodes,
        risk_day=train_risk_day,
        planner=planner,
        risk=risk,
        train_mode=True,
        active_dates=train_days,
    )

    sanity = _check_mask_once(env, seed=seed)
    if not sanity["ok_shape"]:
        raise RuntimeError(
            f"env.action_masks() 形状异常: got={sanity}, expected action_dim={env.action_space.n}"
        )
    if sanity["feasible_cnt"] <= 0:
        raise RuntimeError(f"首状态无可行动作，请检查候选生成/约束配置: {sanity}")

    lr_value = float(lr_override if lr_override is not None else cfg.learning_rate)
    use_init = init_model_zip is not None and Path(init_model_zip).exists()

    if use_init:
        model = MaskablePPO.load(str(init_model_zip), env=env, device=getattr(cfg, "device", "auto"))

        model_n = getattr(getattr(model, "action_space", None), "n", None)
        env_n = int(env.action_space.n)
        if model_n is not None and int(model_n) != env_n:
            raise ValueError(
                f"warm-start动作维度不一致: model_n={model_n}, env_n={env_n}. "
                f"请删除旧checkpoint后重训。"
            )

        try:
            model.learning_rate = lr_value
            model.lr_schedule = get_schedule_fn(lr_value)
        except Exception:
            pass

        try:
            model.set_random_seed(int(seed))
        except Exception:
            pass

        reset_num_timesteps = False
    else:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=lr_value,
            n_steps=int(cfg.n_steps),
            batch_size=int(cfg.batch_size),
            gamma=float(cfg.gamma),
            gae_lambda=float(getattr(cfg, "gae_lambda", 0.95)),
            clip_range=float(getattr(cfg, "clip_range", 0.2)),
            ent_coef=float(getattr(cfg, "ent_coef", 0.0)),
            vf_coef=float(getattr(cfg, "vf_coef", 0.5)),
            max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5)),
            seed=int(seed),
            device=getattr(cfg, "device", "auto"),
            tensorboard_log=str(_tb_root().resolve()),
        )
        reset_num_timesteps = True

    if tb_log_name is None:
        tb_log_name = f"{out_prefix.name}_s{seed}"

    model.learn(
        total_timesteps=int(total_steps),
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=tb_log_name,
        progress_bar=False,
    )

    ensure_parent(out_prefix)
    model.save(str(out_prefix))
    return out_prefix.with_suffix(".zip")


def evaluate_model_on_pack(
    cfg: CFG,
    model_zip: Path,
    data_pack_path: Path,
    deterministic: bool = True,
    eval_seed: int = 12345,
) -> Dict[str, Any]:
    with open(data_pack_path, "rb") as f:
        pack = pickle.load(f)

    episodes = pack["episodes"]
    risk_day = pack["risk_day"]
    days = sorted(pack.get("all_days", list(episodes.keys())))

    if len(days) == 0:
        return {
            "days": 0,
            "avg_sum_dep_delay_sec": np.nan,
            "avg_sum_dep_delay_min": np.nan,
            "std_sum_dep_delay_min": np.nan,
            "avg_train_dep_delay_sec": np.nan,
            "avg_episode_reward": np.nan,
            "avg_feasible_actions": np.nan,
            "avg_nontrivial_ratio": np.nan,
            "avg_unique_actions_used": np.nan,
            "daily_sum_dep_delay_min": [],
        }

    risk = _fit_risk_profile(cfg, risk_day, days)
    planner = RoutePlanner(cfg)
    env = HZMaskEnv(
        cfg=cfg,
        episodes={d: episodes[d] for d in days},
        risk_day={},  # 评估时不在环境内重拟合
        planner=planner,
        risk=risk,
        train_mode=False,
        active_dates=days,
    )

    model = MaskablePPO.load(str(model_zip), env=env, device=getattr(cfg, "device", "auto"))

    model_n = getattr(getattr(model, "action_space", None), "n", None)
    env_n = int(env.action_space.n)
    if model_n is not None and int(model_n) != env_n:
        raise ValueError(
            f"评估动作维度不一致: model_n={model_n}, env_n={env_n}. "
            f"请确认action_mode/config一致。"
        )

    rows = []
    for i, d in enumerate(days):
        obs, _ = env.reset(seed=int(eval_seed + i), options={"date": d})
        done = False
        ep_reward = 0.0
        info = {}

        while not done:
            mask = np.asarray(env.action_masks(), dtype=bool)
            if mask.ndim != 1 or mask.shape[0] != env.action_space.n:
                raise RuntimeError(
                    f"env.action_masks() 异常: shape={mask.shape}, expected=({env.action_space.n},)"
                )

            if mask.any():
                action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
                action_i = _safe_action_to_int(action)
            else:
                action_i = 0

            obs, reward, term, trunc, info = env.step(action_i)
            ep_reward += float(reward)
            done = bool(term or trunc)

        kpi = info.get("episode_kpi", {})
        rows.append({
            "date": str(d),
            "trains": int(kpi.get("trains", 0)),
            "sum_dep_delay_sec": float(kpi.get("sum_dep_delay_sec", 0.0)),
            "avg_dep_delay_sec": float(kpi.get("avg_dep_delay_sec", 0.0)),
            "episode_reward": float(ep_reward),
            "avg_feasible_actions": float(kpi.get("avg_feasible_actions", np.nan)),
            "nontrivial_ratio": float(kpi.get("nontrivial_ratio", np.nan)),
            "unique_actions_used": float(kpi.get("unique_actions_used", np.nan)),
            "action_mode": kpi.get("action_mode", None),
            "action_dim": kpi.get("action_dim", None),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return {
            "days": 0,
            "avg_sum_dep_delay_sec": np.nan,
            "avg_sum_dep_delay_min": np.nan,
            "std_sum_dep_delay_min": np.nan,
            "avg_train_dep_delay_sec": np.nan,
            "avg_episode_reward": np.nan,
            "avg_feasible_actions": np.nan,
            "avg_nontrivial_ratio": np.nan,
            "avg_unique_actions_used": np.nan,
            "daily_sum_dep_delay_min": [],
        }

    sum_dep_sec = pd.to_numeric(df["sum_dep_delay_sec"], errors="coerce")
    daily_sum_dep_delay_min = (sum_dep_sec / 60.0).tolist()

    return {
        "days": int(len(df)),
        "avg_sum_dep_delay_sec": float(sum_dep_sec.mean()),
        "avg_sum_dep_delay_min": float((sum_dep_sec / 60.0).mean()),
        "std_sum_dep_delay_min": float((sum_dep_sec / 60.0).std(ddof=1)) if len(df) > 1 else 0.0,
        "avg_train_dep_delay_sec": float(pd.to_numeric(df["avg_dep_delay_sec"], errors="coerce").mean()),
        "avg_episode_reward": float(pd.to_numeric(df["episode_reward"], errors="coerce").mean()),
        "avg_feasible_actions": float(pd.to_numeric(df["avg_feasible_actions"], errors="coerce").mean()),
        "avg_nontrivial_ratio": float(pd.to_numeric(df["nontrivial_ratio"], errors="coerce").mean()),
        "avg_unique_actions_used": float(pd.to_numeric(df["unique_actions_used"], errors="coerce").mean()),
        "daily_sum_dep_delay_min": daily_sum_dep_delay_min,
    }


# =========================
# 统计工具
# =========================
def bootstrap_ci_mean(values, n_boot=N_BOOT, alpha=CI_ALPHA, seed=2026) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def summarize_values(values, ci_seed=2026) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}

    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    lo, hi = bootstrap_ci_mean(arr, seed=ci_seed)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": std,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def collect_ok_models_by_seed(train_block: Dict[str, Any]) -> "OrderedDict[int, Path]":
    out: "OrderedDict[int, Path]" = OrderedDict()
    for seed, rec in train_block.get("seeds", {}).items():
        if rec.get("ok", False) and rec.get("model_zip"):
            out[int(seed)] = Path(rec["model_zip"])
    return out


def to_jsonable(x):
    if isinstance(x, Path):
        return str(x)

    if isinstance(x, (datetime, date, pd.Timestamp)):
        return x.isoformat()

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        xv = float(x)
        return None if not np.isfinite(xv) else xv

    if isinstance(x, (np.bool_,)):
        return bool(x)

    if isinstance(x, float):
        return None if not np.isfinite(x) else x

    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]

    return x


def precheck(cfg: CFG):
    if not SCENE02.exists():
        raise FileNotFoundError(f"缺少脚本: {SCENE02}")
    if not SCENE03.exists():
        raise FileNotFoundError(f"缺少脚本: {SCENE03}")
    if not LIB_JSON.exists():
        raise FileNotFoundError(f"缺少场景库: {LIB_JSON}")

    if not Path(cfg.base_file).exists():
        raise FileNotFoundError(f"base_file不存在: {cfg.base_file}")

    for p in [PLAN_ROOT, INJECT_ROOT, RLPACK_ROOT, CKPT_ROOT, REPORT_ROOT]:
        ensure_dir(p)


def main():
    print("[INFO] run_train_4models_vscode 启动（改进版）")
    print(f"[INFO] ROOT: {ROOT}")

    cfg = CFG()
    precheck(cfg)

    all_keys = list(MODEL_SPECS.keys())
    freeze_keys = [k for k in all_keys if k in set(FREEZE_MODELS)]
    eval_model_keys = [k for k in EVAL_MODEL_KEYS if k in MODEL_SPECS]
    eval_set_keys = [k for k in EVAL_SET_KEYS if k in MODEL_SPECS]
    spec_targets = [k for k in SPECIALIZATION_TARGETS if k in MODEL_SPECS and k != "base"]

    if len(eval_model_keys) == 0:
        raise ValueError("EVAL_MODEL_KEYS 过滤后为空")
    if len(eval_set_keys) == 0:
        raise ValueError("EVAL_SET_KEYS 过滤后为空")
    if len(spec_targets) > 0 and "base" not in eval_model_keys:
        print("[WARN] specialization 需要 base 参与评估；当前 EVAL_MODEL_KEYS 不含 base，可能导致 n_pairs=0")

    base_std = load_base_std(cfg)
    print(f"[INFO] 基本图样本数: {len(base_std)}")

    run_meta: Dict[str, Any] = OrderedDict()
    run_meta["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_meta["root"] = str(ROOT)
    run_meta["config"] = {
        "ENABLE_TRAIN": ENABLE_TRAIN,
        "ENABLE_CROSS_EVAL": ENABLE_CROSS_EVAL,
        "USE_BASE_INIT_FOR_EXPERTS": USE_BASE_INIT_FOR_EXPERTS,
        "FREEZE_MODELS": freeze_keys,
        "AUTO_TRAIN_IF_MISSING": AUTO_TRAIN_IF_MISSING,
        "EVAL_MODEL_KEYS": eval_model_keys,
        "EVAL_SET_KEYS": eval_set_keys,
        "SPECIALIZATION_TARGETS": spec_targets,
        "PROFILE": PROFILE,
        "QUOTA_JITTER": QUOTA_JITTER,
        "DELAY_NOISE_STD": DELAY_NOISE_STD,
        "MAX_TIME_SHIFT_MIN": MAX_TIME_SHIFT_MIN,
        "COMBINE_MODE": COMBINE_MODE,
        "N_DAYS_TRAIN": N_DAYS_TRAIN,
        "N_DAYS_EVAL": N_DAYS_EVAL,
        "START_DATE_TRAIN": START_DATE_TRAIN,
        "START_DATE_EVAL": START_DATE_EVAL,
        "DATE_GAP_DAYS": DATE_GAP_DAYS,
        "TRAIN_SEEDS": TRAIN_SEEDS,
        "BASE_STEPS": BASE_STEPS,
        "EXPERT_STEPS": EXPERT_STEPS,
        "EVAL_DETERMINISTIC": EVAL_DETERMINISTIC,
        "EVAL_BASE_SEED": EVAL_BASE_SEED,
        "SPECIALIZATION_MIN_IMPROVE": SPECIALIZATION_MIN_IMPROVE,
        "MODEL_SPECS": MODEL_SPECS,
        "CFG_learning_rate": getattr(cfg, "learning_rate", None),
        "CFG_n_steps": getattr(cfg, "n_steps", None),
        "CFG_batch_size": getattr(cfg, "batch_size", None),
        "CFG_ent_coef": getattr(cfg, "ent_coef", None),
        "CFG_clip_range": getattr(cfg, "clip_range", None),
        "CFG_action_mode": getattr(cfg, "action_mode", None),
        "CFG_max_actions": getattr(cfg, "max_actions", None),
        "CFG_assign_tracks_n": len(getattr(cfg, "assign_tracks", [])),
    }
    run_meta["models"] = OrderedDict()

    # 1) 生成四套 train/eval 注入包 + RL训练包（保持全量生成）
    for idx, (model_key, spec) in enumerate(MODEL_SPECS.items()):
        print("\n" + "#" * 80)
        print(
            f"[MODEL] {model_key} | mix={spec['scene_mix']} | "
            f"intensity={spec['intensity_low']}~{spec['intensity_high']} | overlap={spec['overlap_rate']}"
        )

        d_off = idx * DATE_GAP_DAYS
        train_start = date_add(START_DATE_TRAIN, d_off)
        eval_start = date_add(START_DATE_EVAL, d_off)

        train_art = generate_scene_and_rlpack(
            cfg=cfg,
            base_std=base_std,
            model_key=model_key,
            split="train",
            scene_mix=spec["scene_mix"],
            intensity_low=float(spec["intensity_low"]),
            intensity_high=float(spec["intensity_high"]),
            overlap_rate=float(spec["overlap_rate"]),
            n_days=N_DAYS_TRAIN,
            start_date=train_start,
            seed=int(spec["plan_seed"]),
        )
        eval_art = generate_scene_and_rlpack(
            cfg=cfg,
            base_std=base_std,
            model_key=model_key,
            split="eval",
            scene_mix=spec["scene_mix"],
            intensity_low=float(spec["intensity_low"]),
            intensity_high=float(spec["intensity_high"]),
            overlap_rate=float(spec["overlap_rate"]),
            n_days=N_DAYS_EVAL,
            start_date=eval_start,
            seed=int(spec["plan_seed"]) + 1000,
        )

        run_meta["models"][model_key] = {
            "scene_mix": spec["scene_mix"],
            "intensity_low": spec["intensity_low"],
            "intensity_high": spec["intensity_high"],
            "overlap_rate": spec["overlap_rate"],
            "steps": spec["steps"],
            "init_from_base": bool(spec.get("init_from_base", False)),
            "lr_override": spec.get("lr_override", None),
            "train": train_art,
            "eval": eval_art,
        }

        print(f"[INFO] {model_key}/train convert_stats: {train_art['convert_stats']}")
        print(f"[INFO] {model_key}/train manifest_stats: {train_art['manifest_stats']}")

    # 2) 训练：每个模型多seed（支持冻结复用）
    train_result: Dict[str, Any] = OrderedDict()
    base_seed_zips: Dict[int, Path] = {}

    if ENABLE_TRAIN:
        for model_key in MODEL_SPECS.keys():
            spec = MODEL_SPECS[model_key]
            rec = run_meta["models"][model_key]
            train_pack = Path(rec["train"]["data_pack"])

            train_result[model_key] = {
                "steps": int(spec["steps"]),
                "seeds": OrderedDict(),
            }

            for seed in TRAIN_SEEDS:
                seed = int(seed)
                t0 = time.time()

                ok = True
                err = ""
                model_zip: Optional[Path] = None
                init_zip: Optional[Path] = None
                lr_override = spec.get("lr_override", None)

                frozen = (model_key in set(FREEZE_MODELS))

                # 1) frozen 模型直接复用checkpoint
                if frozen:
                    p = expected_ckpt(model_key, seed)
                    if p.exists():
                        model_zip = p
                        print(f"[SKIP-TRAIN] 复用 {model_key}[s{seed}] -> {rel(p)}")
                    elif AUTO_TRAIN_IF_MISSING:
                        frozen = False
                        print(f"[WARN] 缺少checkpoint，回退训练 {model_key}[s{seed}]")
                    else:
                        ok = False
                        err = f"冻结模型缺少checkpoint: {p}"
                        print(f"[ERR] {err}")

                # 2) 非frozen，或frozen缺失且允许回退时执行训练
                if (not frozen) and ok:
                    out_prefix = CKPT_ROOT / model_key / f"{model_key}_s{seed}"

                    # init策略：全局开关 or 按模型开关
                    if model_key != "base":
                        if USE_BASE_INIT_FOR_EXPERTS or bool(spec.get("init_from_base", False)):
                            init_zip = base_seed_zips.get(seed, None)
                            if init_zip is None:
                                print(f"[WARN] {model_key}[s{seed}] 请求base warm-start但未找到对应base checkpoint，改为从零训练")

                    print("\n" + "-" * 80)
                    print(
                        f"[TRAIN] {model_key} | seed={seed} | steps={spec['steps']} | "
                        f"init={rel(init_zip) if init_zip else None} | "
                        f"lr_override={lr_override}"
                    )

                    try:
                        model_zip = train_one_model(
                            cfg=cfg,
                            data_pack_path=train_pack,
                            out_prefix=out_prefix,
                            total_steps=int(spec["steps"]),
                            seed=seed,
                            init_model_zip=init_zip,
                            lr_override=lr_override,
                            tb_log_name=f"{model_key}_s{seed}",
                        )
                    except Exception as e:
                        ok = False
                        err = str(e)
                        print(f"[ERR] 训练失败 {model_key}[seed={seed}]: {e}")

                train_result[model_key]["seeds"][str(seed)] = {
                    "ok": bool(ok),
                    "error": err,
                    "model_zip": str(model_zip) if model_zip else None,
                    "train_pack": str(train_pack),
                    "init_zip": str(init_zip) if init_zip else None,
                    "lr_override": lr_override,
                    "reused": bool((model_key in set(FREEZE_MODELS)) and model_zip is not None),
                    "elapsed_sec": round(time.time() - t0, 3),
                }

                # base用于给专家warm-start（无论是训练得到还是复用得到）
                if model_key == "base" and ok and model_zip is not None:
                    base_seed_zips[seed] = Path(model_zip)
    else:
        print("[WARN] ENABLE_TRAIN=False，跳过训练")

    run_meta["training"] = train_result

    # 3) 交叉评估矩阵（seed级统计）
    eval_detail_rows: List[Dict[str, Any]] = []
    eval_stats_rows: List[Dict[str, Any]] = []

    # seed_metric_map[model][eval_set][seed] = avg_sum_dep_delay_min
    seed_metric_map: Dict[str, Dict[str, Dict[int, float]]] = OrderedDict()
    for mkey in MODEL_SPECS.keys():
        seed_metric_map[mkey] = OrderedDict()
        for ekey in MODEL_SPECS.keys():
            seed_metric_map[mkey][ekey] = OrderedDict()

    if ENABLE_CROSS_EVAL and ENABLE_TRAIN:
        print("\n" + "=" * 80)
        print("[EVAL] 开始交叉评估（多seed）")

        for mkey in eval_model_keys:
            seed_models = collect_ok_models_by_seed(train_result.get(mkey, {}))

            for ekey in eval_set_keys:
                eval_pack = Path(run_meta["models"][ekey]["eval"]["data_pack"])
                vals = []

                for seed, model_zip in seed_models.items():
                    try:
                        metric = evaluate_model_on_pack(
                            cfg=cfg,
                            model_zip=model_zip,
                            data_pack_path=eval_pack,
                            deterministic=EVAL_DETERMINISTIC,
                            eval_seed=int(EVAL_BASE_SEED + seed_hash(mkey, ekey, seed) % 100000),
                        )
                        v = float(metric.get("avg_sum_dep_delay_min", np.nan))
                        if np.isfinite(v):
                            vals.append(v)
                            seed_metric_map[mkey][ekey][int(seed)] = v

                        metric_clean = dict(metric)
                        metric_clean.pop("daily_sum_dep_delay_min", None)

                        row = {"model": mkey, "train_seed": int(seed), "eval_set": ekey}
                        row.update(metric_clean)
                        eval_detail_rows.append(row)

                        print(f"[EVAL] model={mkey}[s{seed}] on {ekey}: avg_sum_dep_delay_min={_fmtf(v)}")
                    except Exception as e:
                        print(f"[WARN] 评估失败 model={mkey}[s{seed}], eval_set={ekey}: {e}")
                        eval_detail_rows.append({
                            "model": mkey,
                            "train_seed": int(seed),
                            "eval_set": ekey,
                            "days": 0,
                            "avg_sum_dep_delay_sec": np.nan,
                            "avg_sum_dep_delay_min": np.nan,
                            "std_sum_dep_delay_min": np.nan,
                            "avg_train_dep_delay_sec": np.nan,
                            "avg_episode_reward": np.nan,
                            "avg_feasible_actions": np.nan,
                            "avg_nontrivial_ratio": np.nan,
                            "avg_unique_actions_used": np.nan,
                            "error": str(e),
                        })

                st = summarize_values(vals, ci_seed=seed_hash("eval", mkey, ekey))
                eval_stats_rows.append({
                    "model": mkey,
                    "eval_set": ekey,
                    **st,
                })

                print(
                    f"[EVAL-STAT] model={mkey} on {ekey}: "
                    f"mean={_fmtf(st['mean'])}, std={_fmtf(st['std'])}, n={st['n']}, "
                    f"ci95=({_fmtf(st['ci95_low'])},{_fmtf(st['ci95_high'])})"
                )

    run_meta["cross_eval_detail"] = eval_detail_rows
    run_meta["cross_eval_stats"] = eval_stats_rows

    # 4) 专长验收（expk 在 expk 数据上相对 base 的提升）
    specialization_rows: List[Dict[str, Any]] = []
    for k in spec_targets:
        base_map = seed_metric_map.get("base", {}).get(k, {})
        exp_map = seed_metric_map.get(k, {}).get(k, {})

        common_seeds = sorted(set(base_map.keys()).intersection(set(exp_map.keys())))
        rels = []
        for s in common_seeds:
            b = float(base_map[s])
            e = float(exp_map[s])
            if np.isfinite(b) and np.isfinite(e) and b > 1e-9:
                rels.append((b - e) / b)

        st = summarize_values(rels, ci_seed=seed_hash("specialization", k))
        improve_seed_ratio = float(np.mean(np.asarray(rels) > 0)) if len(rels) > 0 else np.nan

        pass_flag = bool(
            st["n"] >= 2
            and np.isfinite(st["mean"])
            and np.isfinite(st["ci95_low"])
            and (st["mean"] > SPECIALIZATION_MIN_IMPROVE)
            and (st["ci95_low"] > 0.0)
        )

        specialization_rows.append({
            "dataset": k,
            "n_pairs": int(st["n"]),
            "improve_mean": float(st["mean"]) if np.isfinite(st["mean"]) else np.nan,
            "improve_std": float(st["std"]) if np.isfinite(st["std"]) else np.nan,
            "ci95_low": float(st["ci95_low"]) if np.isfinite(st["ci95_low"]) else np.nan,
            "ci95_high": float(st["ci95_high"]) if np.isfinite(st["ci95_high"]) else np.nan,
            "positive_seed_ratio": improve_seed_ratio,
            "pass": pass_flag,
        })

    run_meta["specialization_check"] = specialization_rows

    # 5) 导出摘要 JSON
    ensure_parent(SUMMARY_JSON)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(run_meta), f, ensure_ascii=False, indent=2)

    # 6) 导出 Excel
    pack_rows = []
    for mkey, mobj in run_meta["models"].items():
        for split in ["train", "eval"]:
            r = mobj[split]
            row = {
                "model_key": mkey,
                "split": split,
                "scene_mix": mobj["scene_mix"],
                "intensity_low": mobj["intensity_low"],
                "intensity_high": mobj["intensity_high"],
                "overlap_rate": mobj["overlap_rate"],
                "steps": mobj["steps"],
                "init_from_base": mobj.get("init_from_base", False),
                "lr_override": mobj.get("lr_override", None),
                "plan_json": r["plan_json"],
                "injected_pack": r["injected_pack"],
                "manifest_xlsx": r["manifest_xlsx"],
                "data_pack": r["data_pack"],
                "days": r["convert_stats"].get("days"),
                "trains_per_day": r["convert_stats"].get("trains_per_day"),
                "avg_delayed_ratio_ge_60s": r["convert_stats"].get("avg_delayed_ratio_ge_60s"),
                "scene_type_counts": json.dumps(r["manifest_stats"].get("scene_type_counts", {}), ensure_ascii=False),
                "single_events_dist": json.dumps(r["manifest_stats"].get("single_events_dist", {}), ensure_ascii=False),
                "single_1_3_ratio": r["manifest_stats"].get("single_1_3_ratio"),
            }
            pack_rows.append(row)
    pack_df = pd.DataFrame(pack_rows)

    train_rows = []
    for mkey, mobj in train_result.items():
        for seed, srec in mobj.get("seeds", {}).items():
            row = {"model_key": mkey, "seed": int(seed), "steps": mobj.get("steps")}
            row.update(srec)
            train_rows.append(row)
    train_df = pd.DataFrame(train_rows)

    eval_detail_df = pd.DataFrame(eval_detail_rows)
    eval_stats_df = pd.DataFrame(eval_stats_rows)
    specialization_df = pd.DataFrame(specialization_rows)

    with pd.ExcelWriter(SUMMARY_XLSX) as writer:
        pack_df.to_excel(writer, sheet_name="packs", index=False)
        train_df.to_excel(writer, sheet_name="training", index=False)

        if len(eval_detail_df) > 0:
            eval_detail_df.to_excel(writer, sheet_name="cross_eval_detail", index=False)

        if len(eval_stats_df) > 0:
            eval_stats_df.to_excel(writer, sheet_name="cross_eval_stats", index=False)
            try:
                piv_mean = eval_stats_df.pivot(index="model", columns="eval_set", values="mean")
                piv_mean.to_excel(writer, sheet_name="cross_eval_matrix_mean")
            except Exception:
                pass

            try:
                piv_std = eval_stats_df.pivot(index="model", columns="eval_set", values="std")
                piv_std.to_excel(writer, sheet_name="cross_eval_matrix_std")
            except Exception:
                pass

        if len(specialization_df) > 0:
            specialization_df.to_excel(writer, sheet_name="specialization", index=False)

    print("\n" + "=" * 80)
    if len(specialization_df) > 0 and "pass" in specialization_df.columns:
        n_pass = int((specialization_df["pass"] == True).sum())
        print(f"[CHECK] 专长验收通过: {n_pass}/{len(spec_targets)}")
    else:
        print("[CHECK] 未生成 specialization 结果")
    print("[ALL DONE] 4模型流程完成（改进版）")
    print(f"- 输出目录: {rel(OUT_ROOT)}")
    print(f"- 摘要JSON: {rel(SUMMARY_JSON)}")
    print(f"- 摘要XLSX: {rel(SUMMARY_XLSX)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FAILED] 4模型流程失败：", e)
        traceback.print_exc()
        raise
