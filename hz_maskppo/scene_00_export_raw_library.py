# -*- coding: utf-8 -*-
"""
scene_00_export_raw_library.py
把 delay_scene_dbscan_from_pack.xlsx 导出为 scene_library_raw.json
（供 scene_01_build_library.py 的 --input-json 使用）

不修改 scene_01/02/03。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

SCENE_SINGLE = "单列车短时晚点"
SCENE_MIXED = "混合型晚点场景"
SCENE_LARGE = "大面积晚点干扰"
SCENE_TYPES = [SCENE_SINGLE, SCENE_MIXED, SCENE_LARGE]

SCENE_ALIAS = {
    "single": SCENE_SINGLE,
    "single_train": SCENE_SINGLE,
    "单列车短时晚点": SCENE_SINGLE,

    "mixed": SCENE_MIXED,
    "multi": SCENE_MIXED,
    "multi_train": SCENE_MIXED,
    "多列车中时晚点": SCENE_MIXED,
    "混合型晚点场景": SCENE_MIXED,

    "large": SCENE_LARGE,
    "大面积晚点干扰": SCENE_LARGE,
    "大面积晚点干扰(孤立)": SCENE_LARGE,
}


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


def choose_col(columns, candidates):
    cols = list(columns)
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        lc = str(c).lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def canonical_scene_type(x: str) -> str:
    s = str(x or "").strip()
    s = re.sub(r"\s*[（(]孤立[)）]\s*$", "", s)
    if not s:
        return SCENE_MIXED
    s2 = SCENE_ALIAS.get(s.lower(), SCENE_ALIAS.get(s, s))
    if s2 not in SCENE_TYPES:
        return SCENE_MIXED
    return s2


def normalize_train_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    s = s.replace("次", "").replace(" ", "")
    if not s:
        return ""
    if s.lower() in {"nan", "none", "null"}:
        return ""
    if re.search(r"[A-Za-z]", s):
        s = s.upper()
    return s


def looks_like_train_token(x: str) -> bool:
    s = normalize_train_id(x)
    if not s or len(s) > 20:
        return False
    # 避免把 1/2/3 这类统计数字当车次
    if re.fullmatch(r"\d{1,2}", s):
        return False
    # 常见车次格式
    if re.fullmatch(r"[A-Z]{1,3}\d{1,6}[A-Z]?", s):
        return True
    if re.fullmatch(r"\d{3,6}", s):
        return True
    if re.fullmatch(r"[A-Z0-9]{2,12}", s) and re.search(r"[A-Z]", s) and re.search(r"\d", s):
        return True
    return False


def norm_date_str(x) -> str:
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return str(x) if x is not None else ""
    return ts.strftime("%Y-%m-%d")


def calc_stats_from_events(events: list) -> dict:
    arr = np.array([safe_float(e.get("delay_sec"), 0.0) for e in events], dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return {
            "affected_trains": 0,
            "total_delay_min": 0.0,
            "avg_delay_min": 0.0,
            "max_delay_min": 0.0,
            "severity_score": 0.0,
        }

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


def build_sample_map(df_samples: pd.DataFrame):
    if df_samples is None or df_samples.empty:
        return {}

    sid_col = choose_col(df_samples.columns, ["scenario_id", "scene_id", "场景ID"])
    if sid_col is None:
        return {}

    stype_col = choose_col(df_samples.columns, ["scene_type", "场景类型", "type", "label"])
    reason_col = choose_col(df_samples.columns, ["scene_reason", "reason", "判定原因"])
    sev_col = choose_col(df_samples.columns, ["severity_score", "severity", "score"])

    m = {}
    x = df_samples.copy()
    x = x.drop_duplicates(subset=[sid_col], keep="first")

    for _, r in x.iterrows():
        sid = str(r.get(sid_col, "")).strip()
        if not sid:
            continue
        m[sid] = {
            "scene_type": canonical_scene_type(r.get(stype_col, SCENE_MIXED) if stype_col else SCENE_MIXED),
            "scene_reason": str(r.get(reason_col, "")).strip() if reason_col else "",
            "severity_score": safe_float(r.get(sev_col, np.nan), np.nan) if sev_col else np.nan,
        }
    return m


def build_raw_from_xlsx(
    input_xlsx: Path,
    sheet_events: str,
    sheet_samples: str,
    min_delay_sec: float,
    max_delay_sec: float,
    min_events: int,
):
    df_events = pd.read_excel(input_xlsx, sheet_name=sheet_events)
    try:
        df_samples = pd.read_excel(input_xlsx, sheet_name=sheet_samples)
    except Exception:
        df_samples = pd.DataFrame()

    if df_events.empty:
        raise ValueError(f"{sheet_events} 为空，无法生成 raw。")

    sid_col = choose_col(df_events.columns, ["scenario_id", "scene_id", "场景ID"])
    train_col = choose_col(df_events.columns, ["train", "列车", "train_id", "车次"])
    delay_col = choose_col(df_events.columns, ["dep_delay_sec", "delay_sec", "obs_dep_delay_sec", "晚点秒", "延误秒"])
    date_col = choose_col(df_events.columns, ["date", "日期"])
    event_sec_col = choose_col(df_events.columns, ["event_sec", "time_sec", "actual_dep_sec", "事件时刻秒"])

    if sid_col is None or train_col is None or delay_col is None:
        raise ValueError(
            f"sheet={sheet_events} 缺少必要列。"
            f" 需要 scenario_id/train/dep_delay_sec（或同义列），当前列={list(df_events.columns)}"
        )

    sample_map = build_sample_map(df_samples)

    scene_types = OrderedDict((t, {"templates": [], "model": {}, "augment": {}}) for t in SCENE_TYPES)

    template_total = 0
    template_with_events = 0
    events_total = 0

    for sid, g in df_events.groupby(sid_col, sort=False):
        sid_str = str(sid).strip()
        if not sid_str:
            continue

        info = sample_map.get(sid_str, {})
        stype = canonical_scene_type(info.get("scene_type", SCENE_MIXED))
        reason = str(info.get("scene_reason", "")).strip()
        sev_given = safe_float(info.get("severity_score", np.nan), np.nan)

        agg = OrderedDict()
        for _, r in g.iterrows():
            tid = normalize_train_id(r.get(train_col, ""))
            if not looks_like_train_token(tid):
                continue
            ds = safe_float(r.get(delay_col, 0.0), 0.0)
            if ds < min_delay_sec:
                continue
            ds = float(np.clip(ds, min_delay_sec, max_delay_sec))
            if ds > agg.get(tid, 0.0):
                agg[tid] = ds

        events = [{"train_id": k, "delay_sec": round(v, 3)} for k, v in agg.items()]

        template_total += 1
        if len(events) < max(1, int(min_events)):
            continue

        template_with_events += 1
        events_total += len(events)

        stats = calc_stats_from_events(events)
        if np.isfinite(sev_given):
            stats["severity_score"] = float(sev_given)

        dmins = np.array([e["delay_sec"] for e in events], dtype=float) / 60.0
        low = float(np.percentile(dmins, 20)) if dmins.size else 0.0
        high = float(np.percentile(dmins, 90)) if dmins.size else 0.0
        if high < low:
            low, high = high, low

        if event_sec_col and event_sec_col in g.columns:
            tt = pd.to_numeric(g[event_sec_col], errors="coerce").dropna()
            if len(tt) > 0:
                start_sec = int(float(tt.min()))
                end_sec = int(float(tt.max()))
                duration_min = round(max(0.0, (end_sec - start_sec) / 60.0), 6)
            else:
                start_sec, end_sec, duration_min = None, None, 0.0
        else:
            start_sec, end_sec, duration_min = None, None, 0.0

        source_date = norm_date_str(g[date_col].iloc[0]) if (date_col and date_col in g.columns and len(g) > 0) else ""

        tpl = {
            "template_id": sid_str,
            "scene_type": stype,
            "source_scenario_id": sid_str,
            "source_date": source_date,
            "events": events,
            "stats": {
                "affected_trains": int(stats["affected_trains"]),
                "total_delay_min": round(float(stats["total_delay_min"]), 6),
                "avg_delay_min": round(float(stats["avg_delay_min"]), 6),
                "max_delay_min": round(float(stats["max_delay_min"]), 6),
                "severity_score": round(float(stats["severity_score"]), 6),
            },
            # 你要的“n辆车 + n~k分钟”提炼参数
            "template_pattern": {
                "n_trains": int(stats["affected_trains"]),
                "delay_min_low": round(low, 6),
                "delay_min_high": round(high, 6),
            },
            "time_window": {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_min": duration_min,
            },
            "scene_reason": reason,
            "sample_weight": round(float(np.clip(1.0 + 0.35 * stats["severity_score"], 0.5, 5.0)), 6),
        }

        scene_types[stype]["templates"].append(tpl)

    # 补 model/augment
    counts = {k: len(v["templates"]) for k, v in scene_types.items()}
    mx = max(max(counts.values()), 1)
    for st in SCENE_TYPES:
        n = counts[st]
        scene_types[st]["model"] = {"source": "from_delay_scene_dbscan_xlsx"}
        scene_types[st]["augment"] = {"rarity_weight": round(float(mx / max(n, 1)), 6)}

    raw_obj = {
        "version": "raw_v1",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_xlsx": str(input_xlsx),
        "sheets": {"events": sheet_events, "samples": sheet_samples},
        "scene_types": scene_types,
        "summary": {
            "template_total_seen": int(template_total),
            "template_written": int(sum(counts.values())),
            "template_with_events": int(template_with_events),
            "events_total": int(events_total),
            "counts_by_type": counts,
            "event_coverage": round(float(template_with_events / max(template_total, 1)), 6),
        },
    }
    return raw_obj


def parse_args():
    ap = argparse.ArgumentParser("Export scene_library_raw.json from scenario xlsx")
    ap.add_argument("--input-xlsx", type=str, default="artifacts/scenario_dbscan_pack/delay_scene_dbscan_from_pack.xlsx")
    ap.add_argument("--sheet-events", type=str, default="scenario_events")
    ap.add_argument("--sheet-samples", type=str, default="scenario_samples")
    ap.add_argument("--out-json", type=str, default="artifacts/scene_lib/scene_library_raw.json")
    ap.add_argument("--min-delay-sec", type=float, default=60.0)
    ap.add_argument("--max-delay-sec", type=float, default=6 * 3600.0)
    ap.add_argument("--min-events", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()

    in_xlsx = to_abs(args.input_xlsx)
    out_json = to_abs(args.out_json)

    if not in_xlsx.exists():
        raise FileNotFoundError(f"输入xlsx不存在: {in_xlsx}")

    raw_obj = build_raw_from_xlsx(
        input_xlsx=in_xlsx,
        sheet_events=args.sheet_events,
        sheet_samples=args.sheet_samples,
        min_delay_sec=float(max(1.0, args.min_delay_sec)),
        max_delay_sec=float(max(args.min_delay_sec, args.max_delay_sec)),
        min_events=int(max(1, args.min_events)),
    )

    ensure_parent(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pyify(raw_obj), f, ensure_ascii=False, indent=2)

    summ = raw_obj.get("summary", {})
    print(f"[OK] raw场景库已导出: {display_path(out_json)}")
    print(f"[INFO] 模板写入数: {summ.get('template_written', 0)}")
    print(f"[INFO] 事件覆盖率: {summ.get('event_coverage', 0.0):.2%}")
    print(f"[INFO] 各类型模板数: {summ.get('counts_by_type', {})}")


if __name__ == "__main__":
    main()
