
# -*- coding: utf-8 -*-
"""
scene_02_generate_plan.py
从标准化场景库生成注入计划

改进点：
1) 新增 --exp-profile=none/base/exp1/exp2/exp3（覆盖 scene-mix/intensity/overlap）
2) 场景类型归一化增强（容忍空格、别名、孤立后缀）
3) scene-mix 解析增强（中文/英文分隔符）
4) 输出 requested/effective 配置，便于复现实验
"""

import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict

import numpy as np

ROOT = Path(__file__).resolve().parent

SCENE_SINGLE = "单列车短时晚点"
SCENE_MIXED = "混合型晚点场景"
SCENE_LARGE = "大面积晚点干扰"

SCENE_SET = {SCENE_SINGLE, SCENE_MIXED, SCENE_LARGE}


def _normalize_type_key(x: str) -> str:
    """场景类型字符串归一化：去全角空格、去'孤立'后缀、去内部空白、lower"""
    s = str(x or "")
    s = s.replace("\u3000", " ").strip()
    s = re.sub(r"\s*[（(]?\s*孤立\s*[)）]?\s*$", "", s)
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "", s)  # 兼容“大面积晚点干 扰”
    return s.lower()


DEFAULTS = {
    "library_json": "artifacts/scene_lib/scene_library.json",
    "out_plan": "artifacts/scene_lib/injection_plan_moderate.json",
    "profile": "moderate_aug",
    "exp_profile": "none",
    "scene_mix": "单列车短时晚点:4,混合型晚点场景:4,大面积晚点干扰:2",
    "intensity_low": 0.95,
    "intensity_high": 1.35,
    "n_days": 10,
    "start_date": "2024-08-01",
    "quota_jitter": 0.20,
    "delay_noise_std": 0.10,
    "max_time_shift_min": 6,
    "overlap_rate": 0.35,
    "seed": 42,
}

# 兼容别名（会走 _normalize_type_key）
TYPE_ALIAS_RAW = {
    "single": SCENE_SINGLE,
    "single_train": SCENE_SINGLE,
    "singletrain": SCENE_SINGLE,
    "单列车短时晚点": SCENE_SINGLE,

    "mixed": SCENE_MIXED,
    "multi": SCENE_MIXED,
    "multi_train": SCENE_MIXED,
    "multitrain": SCENE_MIXED,
    "多列车中时晚点": SCENE_MIXED,
    "混合型晚点场景": SCENE_MIXED,

    "large": SCENE_LARGE,
    "large_area": SCENE_LARGE,
    "大面积晚点干扰": SCENE_LARGE,
    "大面积晚点干扰(孤立)": SCENE_LARGE,
    "大面积晚点干扰（孤立）": SCENE_LARGE,
}

TYPE_ALIAS = {}
for _k, _v in TYPE_ALIAS_RAW.items():
    TYPE_ALIAS[_normalize_type_key(_k)] = _v
for _st in [SCENE_SINGLE, SCENE_MIXED, SCENE_LARGE]:
    TYPE_ALIAS[_normalize_type_key(_st)] = _st

# 固定实验档位（用于拉开数据分布）
EXP_PROFILE_PRESETS = {
    "none": None,
    "base": {
        "scene_mix": "单列车短时晚点:4,混合型晚点场景:4,大面积晚点干扰:2",
        "intensity_low": 0.95,
        "intensity_high": 1.35,
        "overlap_rate": 0.35,
        "profile_name": "base_aug",
    },
    "exp1": {
        "scene_mix": "单列车短时晚点:8,混合型晚点场景:2,大面积晚点干扰:0",
        "intensity_low": 0.80,
        "intensity_high": 1.10,
        "overlap_rate": 0.10,
        "profile_name": "exp1_light_single",
    },
    "exp2": {
        "scene_mix": "单列车短时晚点:2,混合型晚点场景:7,大面积晚点干扰:1",
        "intensity_low": 1.00,
        "intensity_high": 1.40,
        "overlap_rate": 0.35,
        "profile_name": "exp2_mixed_medium",
    },
    "exp3": {
        "scene_mix": "单列车短时晚点:1,混合型晚点场景:2,大面积晚点干扰:7",
        "intensity_low": 1.30,
        "intensity_high": 2.00,
        "overlap_rate": 0.60,
        "profile_name": "exp3_large_heavy",
    },
}

TYPE_INTENSITY_BIAS = {
    SCENE_SINGLE: 0.95,
    SCENE_MIXED: 1.05,
    SCENE_LARGE: 1.15,
}

TRAIN_KEYS = ["train_id", "train", "train_no", "train_code", "车次", "列车", "车次号"]
DELAY_SEC_KEYS = ["delay_sec", "delay_seconds", "delay_s", "delay", "晚点秒", "晚点(秒)"]
DELAY_MIN_KEYS = ["delay_min", "delay_minutes", "晚点分钟", "晚点(分)"]


def to_abs(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return float(v)
    except Exception:
        pass
    return float(default)


def pyify(x):
    if isinstance(x, dict):
        return {k: pyify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [pyify(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def canonical_scene_type(x: str) -> str:
    key = _normalize_type_key(x)
    return TYPE_ALIAS.get(key, SCENE_MIXED)


def extract_train_id(d: dict) -> str:
    for k in TRAIN_KEYS:
        if k in d:
            v = str(d.get(k)).strip()
            if v and v.lower() not in {"nan", "none", "null"}:
                return v
    return ""


def extract_delay_sec(d: dict) -> float:
    for k in DELAY_SEC_KEYS:
        if k in d:
            v = safe_float(d.get(k), 0.0)
            if v > 0:
                return v
    for k in DELAY_MIN_KEYS:
        if k in d:
            v = safe_float(d.get(k), 0.0) * 60.0
            if v > 0:
                return v
    return 0.0


def normalize_event_list(raw_list) -> list:
    agg = OrderedDict()
    for e in raw_list or []:
        if not isinstance(e, dict):
            continue
        tid = str(extract_train_id(e)).strip()
        dsec = safe_float(extract_delay_sec(e), 0.0)
        if not tid or dsec <= 0:
            continue
        if dsec > agg.get(tid, 0.0):
            agg[tid] = dsec
    return [{"train_id": k, "delay_sec": round(float(v), 3)} for k, v in agg.items()]


def extract_events_any(template: dict) -> list:
    """
    只从事件相关键提取，不扫描整模板，
    避免把 stats/config 误识别为事件。
    """
    if not isinstance(template, dict):
        return []

    list_keys = [
        "events", "items", "records", "delays", "affected_trains",
        "injections", "event_list", "delay_list", "samples"
    ]
    map_keys = [
        "train_delay_map", "delay_map", "train_delays", "train2delay",
        "train_delay", "delay_by_train"
    ]

    # 1) 列表结构
    for k in list_keys:
        v = template.get(k)
        if isinstance(v, list):
            ev = normalize_event_list(v)
            if ev:
                return ev

    # 2) 映射结构 train->delay
    for k in map_keys:
        v = template.get(k)
        if isinstance(v, dict):
            raw = [{"train_id": kk, "delay_sec": vv} for kk, vv in v.items()]
            ev = normalize_event_list(raw)
            if ev:
                return ev

    # 3) 单条事件
    if any(k in template for k in TRAIN_KEYS):
        ev = normalize_event_list([template])
        if ev:
            return ev

    # 4) 常见嵌套容器
    for k in ["template", "payload", "scene", "data"]:
        v = template.get(k)
        if isinstance(v, dict):
            ev = extract_events_any(v)
            if ev:
                return ev
        elif isinstance(v, list):
            ev = normalize_event_list(v)
            if ev:
                return ev

    return []


def normalize_scene_types(scene_types: dict) -> OrderedDict:
    merged = OrderedDict()
    for st in [SCENE_SINGLE, SCENE_MIXED, SCENE_LARGE]:
        merged[st] = {"templates": [], "model": {}, "augment": {}}

    for raw_type, payload in (scene_types or {}).items():
        stype = canonical_scene_type(raw_type)

        templates = []
        model = {}
        augment = {}

        if isinstance(payload, dict):
            templates = payload.get("templates", []) or []
            if not isinstance(templates, list):
                templates = []
            model = payload.get("model", {}) if isinstance(payload.get("model"), dict) else {}
            augment = payload.get("augment", {}) if isinstance(payload.get("augment"), dict) else {}
        elif isinstance(payload, list):
            templates = payload
        else:
            templates = []

        if not merged[stype]["model"] and model:
            merged[stype]["model"] = model
        if not merged[stype]["augment"] and augment:
            merged[stype]["augment"] = augment

        merged[stype]["templates"].extend([t for t in templates if isinstance(t, dict)])

    return merged


def parse_scene_mix(scene_mix: str) -> OrderedDict:
    """
    支持:
      单列车短时晚点:4,混合型晚点场景:4,大面积晚点干扰:2
      单列车短时晚点：4；混合型晚点场景：4；大面积晚点干扰：2
    """
    out = OrderedDict()
    if not scene_mix:
        return out

    txt = str(scene_mix).strip()
    txt = re.sub(r"[，；;]+", ",", txt)
    chunks = [c.strip() for c in txt.split(",") if c.strip()]

    for c in chunks:
        if ":" in c:
            k, v = c.split(":", 1)
        elif "：" in c:
            k, v = c.split("：", 1)
        else:
            continue

        stype = canonical_scene_type(k.strip())
        try:
            n = int(round(float(v.strip())))
        except Exception:
            continue

        # 允许0（便于记录requested mix），后续effective mix会过滤<=0
        if n >= 0 and stype in SCENE_SET:
            out[stype] = n

    # 固定输出顺序
    ordered = OrderedDict()
    for st in [SCENE_SINGLE, SCENE_MIXED, SCENE_LARGE]:
        if st in out:
            ordered[st] = out[st]
    for st, n in out.items():
        if st not in ordered:
            ordered[st] = n
    return ordered


def auto_scene_mix(scene_types: OrderedDict) -> OrderedDict:
    has = {k for k, v in scene_types.items() if v.get("templates")}
    mix = OrderedDict()

    # 默认10份
    if SCENE_SINGLE in has:
        mix[SCENE_SINGLE] = 4
    if SCENE_MIXED in has:
        mix[SCENE_MIXED] = 4
    if SCENE_LARGE in has:
        mix[SCENE_LARGE] = 2

    if not mix:
        for k in has:
            mix[k] = 3

    # 稀有类型略增配额（按augment.rarity_weight）
    for k in list(mix.keys()):
        rarity = safe_float((scene_types[k].get("augment") or {}).get("rarity_weight"), 1.0)
        if rarity >= 2.0:
            mix[k] += 1

    return mix


def calc_stats_from_events(events: list) -> dict:
    arr = np.array([safe_float(e.get("delay_sec"), 0.0) for e in events], dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return {"affected_trains": 0, "total_delay_min": 0.0, "avg_delay_min": 0.0, "max_delay_min": 0.0, "severity_score": 0.0}

    affected = int(arr.size)
    total_min = float(arr.sum() / 60.0)
    avg_min = float(arr.mean() / 60.0)
    max_min = float(arr.max() / 60.0)
    sev = float(np.log1p(total_min) * (1.0 + 0.35 * np.log1p(max(affected, 1))) + 0.15 * np.log1p(max_min))

    return {
        "affected_trains": affected,
        "total_delay_min": total_min,
        "avg_delay_min": avg_min,
        "max_delay_min": max_min,
        "severity_score": sev,
    }


def template_weight(tpl: dict) -> float:
    """
    修复点:
    - 优先读取 total_delay_min（与scene_01一致）
    - total_delay_sec 仅作兼容回退
    """
    st = tpl.get("stats", {}) if isinstance(tpl.get("stats"), dict) else {}

    sev = safe_float(st.get("severity_score"), 0.0)
    aff = safe_float(st.get("affected_trains"), 0.0)

    total_min = safe_float(st.get("total_delay_min"), 0.0)
    if total_min <= 0:
        total_sec = safe_float(st.get("total_delay_sec"), 0.0)
        if total_sec > 0:
            total_min = total_sec / 60.0

    avg_min = safe_float(st.get("avg_delay_min"), 0.0)
    max_min = safe_float(st.get("max_delay_min"), 0.0)
    if max_min <= 0:
        max_sec = safe_float(st.get("max_delay_sec"), 0.0)
        if max_sec > 0:
            max_min = max_sec / 60.0

    # stats缺失时，从events估算
    if (sev <= 0) and isinstance(tpl.get("events"), list) and tpl["events"]:
        est = calc_stats_from_events(tpl["events"])
        sev = safe_float(est.get("severity_score"), 0.0)
        if aff <= 0:
            aff = safe_float(est.get("affected_trains"), 0.0)
        if total_min <= 0:
            total_min = safe_float(est.get("total_delay_min"), 0.0)
        if avg_min <= 0:
            avg_min = safe_float(est.get("avg_delay_min"), 0.0)
        if max_min <= 0:
            max_min = safe_float(est.get("max_delay_min"), 0.0)

    if sev <= 0:
        sev = float(
            np.log1p(max(total_min, 0.0)) * (1.0 + 0.35 * np.log1p(max(aff, 1.0)))
            + 0.15 * np.log1p(max(max_min, 0.0))
        )

    tpl_boost = safe_float(tpl.get("sample_weight", 1.0), 1.0)
    if tpl_boost <= 0:
        tpl_boost = 1.0

    w = (
        1.0
        + 0.65 * sev
        + 0.22 * np.log1p(max(aff, 0.0))
        + 0.03 * max(max_min, 0.0)
        + 0.008 * max(total_min, 0.0)
    )
    w *= tpl_boost

    return float(max(w, 1e-6))


def parse_args():
    ap = argparse.ArgumentParser("Generate injection plan")
    ap.add_argument("--library-json", type=str, default=DEFAULTS["library_json"])
    ap.add_argument("--out-plan", type=str, default=DEFAULTS["out_plan"])
    ap.add_argument("--profile", type=str, default=DEFAULTS["profile"])

    ap.add_argument(
        "--exp-profile",
        type=str,
        default=DEFAULTS["exp_profile"],
        choices=list(EXP_PROFILE_PRESETS.keys()),
        help="none/base/exp1/exp2/exp3；非none时将覆盖scene-mix/intensity/overlap"
    )

    ap.add_argument("--scene-mix", type=str, default=DEFAULTS["scene_mix"])
    ap.add_argument("--intensity-low", type=float, default=DEFAULTS["intensity_low"])
    ap.add_argument("--intensity-high", type=float, default=DEFAULTS["intensity_high"])
    ap.add_argument("--n-days", type=int, default=DEFAULTS["n_days"])
    ap.add_argument("--start-date", type=str, default=DEFAULTS["start_date"])
    ap.add_argument("--quota-jitter", type=float, default=DEFAULTS["quota_jitter"])
    ap.add_argument("--delay-noise-std", type=float, default=DEFAULTS["delay_noise_std"])
    ap.add_argument("--max-time-shift-min", type=int, default=DEFAULTS["max_time_shift_min"])
    ap.add_argument("--overlap-rate", type=float, default=DEFAULTS["overlap_rate"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    return ap.parse_args()


def apply_exp_profile(args):
    key = str(getattr(args, "exp_profile", "none") or "none").strip().lower()
    if key not in EXP_PROFILE_PRESETS:
        raise ValueError(f"未知exp-profile: {key}")

    preset = EXP_PROFILE_PRESETS.get(key)
    if preset is None:
        return None

    args.scene_mix = str(preset["scene_mix"])
    args.intensity_low = float(preset["intensity_low"])
    args.intensity_high = float(preset["intensity_high"])
    args.overlap_rate = float(preset["overlap_rate"])

    # profile未显式设置时，自动设置
    p = str(getattr(args, "profile", "") or "").strip()
    if (not p) or (p == DEFAULTS["profile"]):
        args.profile = str(preset.get("profile_name", key))
    return preset


def main():
    if len(sys.argv) == 1:
        print("[INFO] 未传命令行参数，使用默认配置（可直接在 VSCode 运行）。")

    args = parse_args()

    # 先应用实验档位（会覆盖 scene-mix/intensity/overlap）
    applied_profile = apply_exp_profile(args)
    if applied_profile is not None:
        print(
            f"[INFO] 已应用 exp-profile={args.exp_profile}: "
            f"scene_mix={args.scene_mix}, "
            f"intensity=[{args.intensity_low}, {args.intensity_high}], "
            f"overlap_rate={args.overlap_rate}"
        )

    # 参数安全裁剪
    args.n_days = max(1, int(args.n_days))
    args.quota_jitter = float(np.clip(args.quota_jitter, 0.0, 0.95))
    args.overlap_rate = float(np.clip(args.overlap_rate, 0.0, 1.0))
    args.delay_noise_std = float(max(0.0, args.delay_noise_std))
    args.max_time_shift_min = int(max(0, args.max_time_shift_min))
    if args.intensity_low > args.intensity_high:
        args.intensity_low, args.intensity_high = args.intensity_high, args.intensity_low

    lib_path = to_abs(args.library_json)
    if not lib_path.exists():
        raise FileNotFoundError(f"未找到场景库JSON: {lib_path}")

    with open(lib_path, "r", encoding="utf-8") as f:
        lib = json.load(f)

    if not isinstance(lib, dict):
        raise ValueError("library-json格式错误，应为dict结构。")

    scene_types_raw = lib.get("scene_types", {})
    if not isinstance(scene_types_raw, dict):
        raise ValueError("library-json缺少 scene_types(dict)。")

    scene_types = normalize_scene_types(scene_types_raw)

    # 不强制events覆盖；events空也保留模板（scene_03兜底合成）
    event_parse_stats = {}
    for stype in [SCENE_SINGLE, SCENE_MIXED, SCENE_LARGE]:
        raw_tpls = scene_types[stype].get("templates", []) or []
        clean_tpls = []

        total_n, with_events_n = 0, 0
        for i, tpl in enumerate(raw_tpls, start=1):
            if not isinstance(tpl, dict):
                continue
            total_n += 1

            ev = extract_events_any(tpl)
            if ev:
                with_events_n += 1

            t2 = dict(tpl)
            t2["scene_type"] = stype
            t2["template_id"] = str(tpl.get("template_id") or tpl.get("id") or f"{stype}_{i:04d}")
            t2["events"] = ev
            if not isinstance(t2.get("stats"), dict):
                t2["stats"] = {}

            clean_tpls.append(t2)

        scene_types[stype]["templates"] = clean_tpls
        event_parse_stats[stype] = {
            "template_total": int(total_n),
            "template_with_events": int(with_events_n),
            "event_coverage": round(with_events_n / total_n, 4) if total_n else 0.0
        }

    print(f"[INFO] 模板事件可解析统计: {event_parse_stats}")

    # 配额：requested -> effective
    requested_mix = parse_scene_mix(args.scene_mix)
    if requested_mix and sum(max(int(v), 0) for v in requested_mix.values()) <= 0:
        raise ValueError("scene_mix 全为0，无法生成注入计划。")

    if not requested_mix:
        requested_mix = auto_scene_mix(scene_types)

    available_types = {k for k, v in scene_types.items() if len(v.get("templates", []) or []) > 0}

    missing_types = [k for k, v in requested_mix.items() if int(v) > 0 and k not in available_types]
    if missing_types:
        print(f"[WARN] 以下类型在场景库中无可用模板，已自动跳过: {missing_types}")

    mix = OrderedDict((k, int(v)) for k, v in requested_mix.items() if k in available_types and int(v) > 0)

    if not mix:
        mix = OrderedDict((k, 3) for k in [SCENE_SINGLE, SCENE_MIXED, SCENE_LARGE] if k in available_types)

    if not mix:
        raise ValueError("scene_mix过滤后为空，且可用模板也为空，请检查scene_library。")

    print(f"[INFO] 请求场景配额 mix(requested): {dict(requested_mix)}")
    print(f"[INFO] 最终场景配额 mix(effective): {dict(mix)}")

    rng = np.random.default_rng(args.seed)
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    dates = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(args.n_days)]

    days = []
    per_type_total = {k: 0 for k in mix.keys()}
    per_day_count = {}

    for d in dates:
        injections = []

        for stype, base_quota in mix.items():
            tpls = scene_types[stype]["templates"]
            if not tpls:
                continue

            scale = float(rng.uniform(max(0.0, 1.0 - args.quota_jitter), 1.0 + args.quota_jitter))
            quota = int(round(base_quota * scale))
            if base_quota > 0 and quota <= 0:
                quota = 1

            weights = np.array([template_weight(t) for t in tpls], dtype=float)
            probs = weights / weights.sum() if np.any(weights > 0) else None
            idxs = rng.choice(len(tpls), size=quota, replace=True, p=probs)

            # 场景级别偏置：允许augment覆盖默认bias
            aug = scene_types[stype].get("augment", {}) or {}
            stype_bias = safe_float(aug.get("intensity_bias"), TYPE_INTENSITY_BIAS.get(stype, 1.0))
            stype_bias = float(np.clip(stype_bias, 0.6, 1.6))

            for _idx in idxs:
                tpl = tpls[int(_idx)]
                tid = str(tpl.get("template_id", f"{stype}_NA"))

                base_intensity = float(rng.uniform(args.intensity_low, args.intensity_high))
                intensity = float(np.clip(base_intensity * stype_bias, 0.50, 2.50))

                noise_std = float(args.delay_noise_std * rng.uniform(0.8, 1.2))
                time_shift = int(rng.integers(-args.max_time_shift_min, args.max_time_shift_min + 1))
                combine_mode = "sum" if float(rng.random()) < args.overlap_rate else "max"

                scene_id = f"S{d.replace('-', '')}_{len(injections) + 1:03d}"
                injections.append({
                    "scene_id": scene_id,
                    "scene_type": stype,
                    "template_id": tid,
                    "intensity": round(intensity, 3),
                    "combine_mode": combine_mode,
                    "jitter": {
                        "delay_noise_std": round(noise_std, 4),
                        "time_shift_min": time_shift
                    },
                    "events": tpl.get("events", []),  # 可为空，scene_03会兜底合成
                    "template_stats": tpl.get("stats", {}),
                })
                per_type_total[stype] += 1

        rng.shuffle(injections)
        days.append({"date": d, "injections": injections})
        per_day_count[d] = len(injections)

    total_tpl = sum(v["template_total"] for v in event_parse_stats.values())
    total_with_events = sum(v["template_with_events"] for v in event_parse_stats.values())

    plan = {
        "version": "2.4",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "profile": args.profile,
        "library_json": str(lib_path),
        "config": {
            "exp_profile": args.exp_profile,
            "exp_profile_applied": bool(applied_profile is not None),
            "scene_mix_requested": requested_mix,   # 保留0配额，便于复现实验设定
            "scene_mix": mix,                       # 兼容旧字段（effective）
            "scene_mix_effective": mix,
            "intensity_low": args.intensity_low,
            "intensity_high": args.intensity_high,
            "n_days": args.n_days,
            "start_date": args.start_date,
            "quota_jitter": args.quota_jitter,
            "delay_noise_std": args.delay_noise_std,
            "max_time_shift_min": args.max_time_shift_min,
            "overlap_rate": args.overlap_rate,
            "seed": args.seed,
        },
        "event_parse_stats": {
            "by_type": event_parse_stats,
            "total_templates": int(total_tpl),
            "templates_with_events": int(total_with_events),
            "overall_coverage": round(total_with_events / max(total_tpl, 1), 4),
        },
        "days": days,
        "summary": {
            "per_day_scene_count": per_day_count,
            "per_type_total": per_type_total,
            "missing_types_skipped": missing_types,
        },
    }

    out_path = to_abs(args.out_plan)
    ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pyify(plan), f, ensure_ascii=False, indent=2)

    sample = dict(list(per_day_count.items())[:3])
    print(f"[OK] 注入计划已生成: {display_path(out_path)}")
    print(f"[INFO] 每日场景数示例: {sample}")
    print(f"[INFO] 场景类型总量: {per_type_total}")


if __name__ == "__main__":
    main()
