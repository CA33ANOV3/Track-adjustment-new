# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


ROOT = Path(__file__).resolve().parent

SCENE_SINGLE = "单列车短时晚点"
SCENE_MIXED = "混合型晚点场景"
SCENE_LARGE = "大面积晚点干扰"

TYPE_ALIAS = {
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

DEFAULTS = {
    "input_json": "artifacts\scene_lib\scene_library_raw.json",
    "fallback_input_json": "artifacts/scene_lib/scene_library.json",
    "out_json": "artifacts/scene_lib/scene_library.json",
    "out_summary": "artifacts/scene_lib/scene_library_summary.xlsx",
    "dedup": "true",
    "seed": 42,

    # 第一类（允许1~3辆短时）
    "single_max_trains": 3,
    "single_avg_max_min": 12.0,
    "single_max_max_min": 20.0,
    "single_total_max_min": 40.0,

    # 第三类（大面积）
    "large_min_trains": 18,
    "large_min_total_min": 220.0,
    "large_min_max_min": 45.0,
}

# 基础关键词
TRAIN_KEY_RE = re.compile(r"(train|车次|列车|车号|车次号|train_id|trainno|train_no|checi)", re.I)
DELAY_KEY_RE = re.compile(r"(delay|晚点|延误|误点|延时)", re.I)
MAP_HINT_RE = re.compile(r"(map|dict|pair|train.*delay|delay.*train|车次.*(晚点|延误)|晚点.*车次|延误.*车次)", re.I)

# 关键过滤：这些“train相关键”是统计量，不是事件字段
TRAIN_KEY_BLOCK_RE = re.compile(
    r"(affected|impact|受影响|影响|count|num|number|total|sum|ratio|比例|占比|数量|总数|累计|avg|mean|max|min)",
    re.I
)

# 关键过滤：这些“delay相关键”是比例/参数，不是具体延误值
DELAY_KEY_BLOCK_RE = re.compile(
    r"(ratio|rate|prob|概率|占比|比例|std|方差|variance|阈值|threshold|coef|weight)",
    re.I
)

# 某些容器通常是配置/统计，不应解释成train->delay map
NON_EVENT_CONTAINER_RE = re.compile(
    r"(model|augment|config|meta|summary|stat|stats|feature|profile|参数|指标)",
    re.I
)

AFFECT_PATTERNS = [
    r"affected.*train", r"train.*affected", r"n[_\-\s]?affected",
    r"affected_trains", r"train_count", r"n[_\-\s]?trains",
    r"受影响.*车", r"影响.*车", r"受扰动.*车", r"列车数量", r"车次数量"
]
TOTAL_PATTERNS = [
    r"total.*delay", r"delay.*total", r"sum.*delay", r"累计.*(晚点|延误)",
    r"总.*(晚点|延误)", r"total_delay", r"delay_total"
]
AVG_PATTERNS = [
    r"avg.*delay", r"mean.*delay", r"average.*delay", r"平均.*(晚点|延误)",
    r"avg_delay", r"mean_delay"
]
MAX_PATTERNS = [
    r"max.*delay", r"peak.*delay", r"最大.*(晚点|延误)",
    r"worst.*delay", r"max_delay", r"peak_delay"
]


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


def parse_bool(x) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def parse_number(x):
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float, np.number)):
        if np.isfinite(x):
            return float(x)
        return None
    s = str(x).strip().replace(",", "")
    if not s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group())
    except Exception:
        return None


def canonical_scene_type(x: str) -> str:
    s = str(x or "").strip()
    s = re.sub(r"\s*[（(]孤立[)）]\s*$", "", s)
    if not s:
        return ""
    return TYPE_ALIAS.get(s.lower(), TYPE_ALIAS.get(s, s))


def normalize_train_id(x) -> str:
    s = str(x).strip()
    s = s.replace("次", "").replace(" ", "")
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    if re.search(r"[A-Za-z]", s):
        s = s.upper()
    return s


def looks_like_train_token(x) -> bool:
    """
    收紧车次识别，防止把统计数字当车次：
    - 允许：G1234, D12, 1234, K123A
    - 拒绝：5, 12, 1.5, 含明显分隔符
    """
    s = normalize_train_id(x)
    if not s:
        return False
    if len(s) > 20:
        return False
    if re.search(r"[/:.\-\s]", s):
        return False

    # 主流模式：字母+数字
    if re.fullmatch(r"[A-Z]{1,3}\d{1,6}[A-Z]?", s):
        return True
    # 纯数字车次通常 >= 3 位
    if re.fullmatch(r"\d{3,6}", s):
        return True
    # 兜底：2~12位，含字母且含数字
    if re.fullmatch(r"[A-Z0-9]{2,12}", s) and re.search(r"[A-Z]", s) and re.search(r"\d", s):
        return True
    return False


def is_train_key(k: str) -> bool:
    ks = str(k or "").strip()
    if not ks:
        return False
    if TRAIN_KEY_RE.search(ks) is None:
        return False
    if TRAIN_KEY_BLOCK_RE.search(ks) is not None:
        return False
    return True


def is_delay_key(k: str) -> bool:
    ks = str(k or "").strip()
    if not ks:
        return False
    if DELAY_KEY_RE.search(ks) is None:
        return False
    if DELAY_KEY_BLOCK_RE.search(ks) is not None:
        return False
    return True


def value_to_sec(v, key_hint: str = "") -> float:
    num = parse_number(v)
    if num is None or num <= 0:
        return 0.0

    k = str(key_hint or "").lower()

    # 明确单位：秒
    if any(tok in k for tok in ["sec", "second", "秒", "_s"]):
        return float(num)

    # 明确单位：分钟
    if any(tok in k for tok in ["min", "minute", "分钟", "分"]):
        return float(num * 60.0)

    # 单位不明：小值按分钟，大值按秒
    if num <= 120:
        return float(num * 60.0)
    return float(num)


def extract_train_from_event_dict(d: dict) -> str:
    for k, v in d.items():
        if is_train_key(k):
            tid = normalize_train_id(v)
            if looks_like_train_token(tid):
                return tid
    return ""


def extract_delay_from_event_dict(d: dict) -> float:
    best = 0.0
    for k, v in d.items():
        if is_delay_key(k):
            sec = value_to_sec(v, str(k))
            if sec > best:
                best = sec
    return best


def dict_as_train_delay_map(d: dict, key_hint: str = "") -> list:
    if not isinstance(d, dict) or not d:
        return []

    # 明显非事件容器不解释成 map
    if NON_EVENT_CONTAINER_RE.search(str(key_hint or "")):
        return []

    pairs = []
    bad = 0
    for kk, vv in d.items():
        tid = normalize_train_id(kk)
        if not looks_like_train_token(tid):
            bad += 1
            continue
        sec = value_to_sec(vv, key_hint)
        if sec <= 0:
            bad += 1
            continue
        pairs.append({"train_id": tid, "delay_sec": sec})

    if not pairs:
        return []

    # 常规可靠判据
    if len(pairs) >= 2 and bad <= len(pairs):
        return pairs

    # key_hint 有 map 特征时放宽
    if MAP_HINT_RE.search(str(key_hint or "")) and len(pairs) >= 1:
        return pairs

    return []


def extract_events_recursive(obj) -> list:
    raw_events = []

    def walk(x, parent_key="", depth=0):
        if depth > 14:
            return

        if isinstance(x, dict):
            # 直接event结构（需同时识别到 train + delay）
            tid = extract_train_from_event_dict(x)
            dsec = extract_delay_from_event_dict(x)
            if tid and dsec > 0:
                raw_events.append({"train_id": tid, "delay_sec": dsec})

            # 当前dict本身可能是 map
            m0 = dict_as_train_delay_map(x, parent_key)
            if m0:
                raw_events.extend(m0)

            # 子dict可能是 map；并继续递归
            for k, v in x.items():
                k_str = str(k)

                if isinstance(v, dict):
                    m1 = dict_as_train_delay_map(v, k_str)
                    if m1:
                        raw_events.extend(m1)

                # 对明显配置/统计容器减少递归误判
                if NON_EVENT_CONTAINER_RE.search(k_str):
                    continue

                walk(v, k_str, depth + 1)

        elif isinstance(x, list):
            for it in x:
                walk(it, parent_key, depth + 1)

    walk(obj)

    # 去重：同车次保留最大延误
    agg = {}
    for e in raw_events:
        tid = normalize_train_id(e.get("train_id", ""))
        sec = safe_float(e.get("delay_sec"), 0.0)
        if not tid or sec <= 0:
            continue
        sec = float(np.clip(sec, 1.0, 8 * 3600.0))
        if tid not in agg or sec > agg[tid]:
            agg[tid] = sec

    return [{"train_id": k, "delay_sec": round(v, 3)} for k, v in agg.items()]


def flatten_numeric(obj, prefix="", out=None, depth=0):
    if out is None:
        out = []
    if depth > 14:
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (int, float, np.number, str)):
                n = parse_number(v)
                if n is not None and np.isfinite(n):
                    out.append((kk, float(n)))
            flatten_numeric(v, kk, out, depth + 1)

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            kk = f"{prefix}[{i}]"
            if isinstance(v, (int, float, np.number, str)):
                n = parse_number(v)
                if n is not None and np.isfinite(n):
                    out.append((kk, float(n)))
            flatten_numeric(v, kk, out, depth + 1)

    return out


def pick_stat(flat_vals, patterns):
    cand = []
    for k, v in flat_vals:
        kl = k.lower()
        if any(re.search(p, kl, re.I) for p in patterns):
            if v > 0 and np.isfinite(v):
                cand.append((k, float(v)))
    if not cand:
        return None, None
    cand = [x for x in cand if x[1] < 1e7] or cand
    return max(cand, key=lambda x: x[1])


def to_minutes_by_key(key, val: float) -> float:
    if key is None:
        return 0.0
    k = str(key).lower()
    if any(tok in k for tok in ["sec", "second", "秒", "_s"]):
        return float(val / 60.0)
    if any(tok in k for tok in ["min", "minute", "分钟", "分"]):
        return float(val)
    if val > 300:
        return float(val / 60.0)
    return float(val)


def extract_aggregate_stats(template: dict) -> dict:
    flat = flatten_numeric(template)

    aff_k, aff_v = pick_stat(flat, AFFECT_PATTERNS)
    tot_k, tot_v = pick_stat(flat, TOTAL_PATTERNS)
    avg_k, avg_v = pick_stat(flat, AVG_PATTERNS)
    max_k, max_v = pick_stat(flat, MAX_PATTERNS)

    affected = int(round(aff_v)) if aff_v is not None else 0
    total_min = to_minutes_by_key(tot_k, tot_v) if tot_v is not None else 0.0
    avg_min = to_minutes_by_key(avg_k, avg_v) if avg_v is not None else 0.0
    max_min = to_minutes_by_key(max_k, max_v) if max_v is not None else 0.0

    return {
        "affected_trains": max(0, int(affected)),
        "total_delay_min": max(0.0, float(total_min)),
        "avg_delay_min": max(0.0, float(avg_min)),
        "max_delay_min": max(0.0, float(max_min)),
    }


def calc_stats(events: list, agg_stats: dict) -> dict:
    if events:
        arr = np.array([safe_float(e.get("delay_sec"), 0.0) for e in events], dtype=float)
        arr = arr[arr > 0]
        if arr.size > 0:
            arr = np.clip(arr, 1.0, 8 * 3600.0)
            affected = int(arr.size)
            total_min = float(arr.sum() / 60.0)
            avg_min = float(arr.mean() / 60.0)
            max_min = float(arr.max() / 60.0)
        else:
            affected = int(agg_stats.get("affected_trains", 0))
            total_min = float(agg_stats.get("total_delay_min", 0.0))
            avg_min = float(agg_stats.get("avg_delay_min", 0.0))
            max_min = float(agg_stats.get("max_delay_min", 0.0))
    else:
        affected = int(agg_stats.get("affected_trains", 0))
        total_min = float(agg_stats.get("total_delay_min", 0.0))
        avg_min = float(agg_stats.get("avg_delay_min", 0.0))
        max_min = float(agg_stats.get("max_delay_min", 0.0))

    # 互相补全
    if affected <= 0 and total_min > 0 and avg_min > 0:
        affected = int(max(1, round(total_min / max(avg_min, 1e-6))))
    if avg_min <= 0 and affected > 0 and total_min > 0:
        avg_min = total_min / affected
    if total_min <= 0 and affected > 0 and avg_min > 0:
        total_min = affected * avg_min
    if max_min <= 0 and avg_min > 0:
        max_min = avg_min

    severity = (
        np.log1p(max(total_min, 0.0)) * (1.0 + 0.35 * np.log1p(max(affected, 1)))
        + 0.15 * np.log1p(max(max_min, 0.0))
    )

    return {
        "affected_trains": int(max(0, affected)),
        "total_delay_min": round(float(max(0.0, total_min)), 6),
        "avg_delay_min": round(float(max(0.0, avg_min)), 6),
        "max_delay_min": round(float(max(0.0, max_min)), 6),
        "severity_score": round(float(max(0.0, severity)), 6),
    }


def has_signal(st: dict) -> bool:
    return (
        safe_float(st.get("affected_trains"), 0.0) > 0
        or safe_float(st.get("total_delay_min"), 0.0) > 0
        or safe_float(st.get("max_delay_min"), 0.0) > 0
    )


def is_short_local(st: dict, args) -> bool:
    n = int(safe_float(st.get("affected_trains"), 0))
    avg_m = safe_float(st.get("avg_delay_min"), 0.0)
    max_m = safe_float(st.get("max_delay_min"), 0.0)
    total_m = safe_float(st.get("total_delay_min"), 0.0)

    return (
        1 <= n <= int(args.single_max_trains)
        and avg_m <= float(args.single_avg_max_min)
        and max_m <= float(args.single_max_max_min)
        and total_m <= float(args.single_total_max_min)
    )


def is_large(st: dict, args) -> bool:
    n = int(safe_float(st.get("affected_trains"), 0))
    total_m = safe_float(st.get("total_delay_min"), 0.0)
    max_m = safe_float(st.get("max_delay_min"), 0.0)

    return (
        n >= int(args.large_min_trains)
        or total_m >= float(args.large_min_total_min)
        or max_m >= float(args.large_min_max_min)
    )


def two_cluster_labels(X: np.ndarray, seed=42) -> np.ndarray:
    if X.shape[0] == 0:
        return np.array([], dtype=int)

    if X.shape[0] < 6:
        score = X[:, 0] + 0.03 * X[:, 1] + 0.08 * X[:, 3]
        thr = float(np.median(score))
        return (score > thr).astype(int)

    Z = np.log1p(np.clip(X, 0, None))
    med = np.median(Z, axis=0)
    q1 = np.percentile(Z, 25, axis=0)
    q3 = np.percentile(Z, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    Z = (Z - med) / iqr

    if HAS_SKLEARN:
        km = KMeans(n_clusters=2, n_init=20, random_state=seed)
        return km.fit_predict(Z)

    score = 0.55 * Z[:, 0] + 0.35 * Z[:, 1] + 0.10 * Z[:, 3]
    thr = float(np.median(score))
    return (score > thr).astype(int)


def relabel_templates(processed: list, args) -> list:
    n = len(processed)
    out = [None] * n
    unresolved = []

    # 第一轮：硬规则（先锁定）
    for i, t in enumerate(processed):
        st = t.get("stats", {})
        if is_large(st, args):
            out[i] = SCENE_LARGE
            continue
        if is_short_local(st, args):
            out[i] = SCENE_SINGLE
            continue
        unresolved.append(i)

    # 第二轮：在“非大面积”剩余样本中做2类聚类（短时局部 vs 混合）
    idx_signal = [i for i in unresolved if has_signal(processed[i].get("stats", {}))]
    if idx_signal:
        X = []
        for i in idx_signal:
            st = processed[i]["stats"]
            X.append([
                safe_float(st.get("affected_trains"), 0.0),
                safe_float(st.get("total_delay_min"), 0.0),
                safe_float(st.get("avg_delay_min"), 0.0),
                safe_float(st.get("max_delay_min"), 0.0),
            ])
        X = np.array(X, dtype=float)
        lb = two_cluster_labels(X, seed=args.seed)

        # 找“更小扰动”的簇
        c_score = {}
        for c in [0, 1]:
            idx = np.where(lb == c)[0]
            if idx.size == 0:
                c_score[c] = 1e18
                continue
            med_aff = float(np.median(X[idx, 0]))
            med_total = float(np.median(X[idx, 1]))
            med_max = float(np.median(X[idx, 3]))
            c_score[c] = med_aff + 0.03 * med_total + 0.08 * med_max

        small_cluster = 0 if c_score[0] <= c_score[1] else 1

        for j, i in enumerate(idx_signal):
            st = processed[i]["stats"]
            n_aff = safe_float(st.get("affected_trains"), 0.0)
            avg_m = safe_float(st.get("avg_delay_min"), 0.0)
            max_m = safe_float(st.get("max_delay_min"), 0.0)

            if (
                lb[j] == small_cluster
                and 1 <= n_aff <= (args.single_max_trains + 1)
                and avg_m <= args.single_avg_max_min * 1.35
                and max_m <= args.single_max_max_min * 1.50
            ):
                out[i] = SCENE_SINGLE
            else:
                out[i] = SCENE_MIXED

    # 第三轮：无信号样本按来源兜底
    for i in unresolved:
        if out[i] is not None:
            continue
        src = canonical_scene_type(processed[i].get("source_scene_type", ""))

        if src == SCENE_SINGLE:
            out[i] = SCENE_SINGLE
        elif src == SCENE_MIXED:
            out[i] = SCENE_MIXED
        elif src == SCENE_LARGE:
            # 无信号的large不直接信，避免把0值全塞进大面积
            out[i] = SCENE_MIXED
        else:
            out[i] = SCENE_MIXED

    # 最后约束：第一类必须接近“1~3短时”
    for i, t in enumerate(processed):
        if out[i] != SCENE_SINGLE:
            continue
        st = t.get("stats", {})
        n_aff = safe_float(st.get("affected_trains"), 0.0)
        max_m = safe_float(st.get("max_delay_min"), 0.0)
        if n_aff > args.single_max_trains + 1 or max_m > args.single_max_max_min * 1.8:
            out[i] = SCENE_MIXED

    return out


def make_signature(tpl: dict):
    ev = tpl.get("events", [])
    if isinstance(ev, list) and ev:
        pairs = []
        for e in ev:
            if not isinstance(e, dict):
                continue
            tid = normalize_train_id(e.get("train_id", ""))
            ds = safe_float(e.get("delay_sec"), 0.0)
            if tid and ds > 0:
                pairs.append((tid, int(round(ds / 30.0))))
        if pairs:
            pairs.sort()
            return ("ev", tuple(pairs))

    st = tpl.get("stats", {}) if isinstance(tpl.get("stats"), dict) else {}
    return (
        "st",
        int(round(safe_float(st.get("affected_trains"), 0))),
        round(safe_float(st.get("total_delay_min"), 0.0), 2),
        round(safe_float(st.get("avg_delay_min"), 0.0), 2),
        round(safe_float(st.get("max_delay_min"), 0.0), 2),
    )


def dedup_templates(templates: list) -> list:
    keep = {}
    for t in templates:
        sig = make_signature(t)
        sev = safe_float((t.get("stats") or {}).get("severity_score"), 0.0)
        if sig not in keep:
            keep[sig] = t
        else:
            sev_old = safe_float((keep[sig].get("stats") or {}).get("severity_score"), 0.0)
            if sev > sev_old:
                keep[sig] = t
    return list(keep.values())


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


def collect_records(raw):
    records = []
    meta = defaultdict(lambda: {"model": {}, "augment": {}})
    counter = defaultdict(int)

    # 标准结构：scene_types
    if isinstance(raw, dict) and isinstance(raw.get("scene_types"), dict):
        for raw_type, payload in raw["scene_types"].items():
            src_type = canonical_scene_type(raw_type) or SCENE_MIXED

            tpls = []
            model = {}
            augment = {}
            if isinstance(payload, dict):
                tpls = payload.get("templates", []) if isinstance(payload.get("templates"), list) else []
                model = payload.get("model", {}) if isinstance(payload.get("model"), dict) else {}
                augment = payload.get("augment", {}) if isinstance(payload.get("augment"), dict) else {}
            elif isinstance(payload, list):
                tpls = payload

            if not meta[src_type]["model"] and model:
                meta[src_type]["model"] = model
            if not meta[src_type]["augment"] and augment:
                meta[src_type]["augment"] = augment

            for tpl in tpls:
                if not isinstance(tpl, dict):
                    continue
                counter[src_type] += 1
                tid = str(
                    tpl.get("template_id")
                    or tpl.get("id")
                    or f"{src_type}_{counter[src_type]:04d}"
                )
                records.append({
                    "template": tpl,
                    "source_scene_type": src_type,
                    "template_id": tid,
                })
        return records, meta

    # 其他结构兼容
    if isinstance(raw, dict):
        list_obj = None
        for k in ["templates", "scenes", "items", "records", "data"]:
            if isinstance(raw.get(k), list):
                list_obj = raw[k]
                break
        if list_obj is None:
            list_obj = []
    elif isinstance(raw, list):
        list_obj = raw
    else:
        list_obj = []

    for i, tpl in enumerate(list_obj, start=1):
        if not isinstance(tpl, dict):
            continue
        src = canonical_scene_type(tpl.get("scene_type", "")) or SCENE_MIXED
        tid = str(tpl.get("template_id") or tpl.get("id") or f"{src}_{i:04d}")
        records.append({
            "template": tpl,
            "source_scene_type": src,
            "template_id": tid,
        })

    return records, meta


def parse_args():
    ap = argparse.ArgumentParser("Build/normalize scene library with robust parsing and relabeling")
    ap.add_argument("--input-json", type=str, default=DEFAULTS["input_json"])
    ap.add_argument("--fallback-input-json", type=str, default=DEFAULTS["fallback_input_json"])
    ap.add_argument("--out-json", type=str, default=DEFAULTS["out_json"])
    ap.add_argument("--out-summary", type=str, default=DEFAULTS["out_summary"])
    ap.add_argument("--dedup", type=str, default=DEFAULTS["dedup"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    ap.add_argument("--single-max-trains", type=int, default=DEFAULTS["single_max_trains"])
    ap.add_argument("--single-avg-max-min", type=float, default=DEFAULTS["single_avg_max_min"])
    ap.add_argument("--single-max-max-min", type=float, default=DEFAULTS["single_max_max_min"])
    ap.add_argument("--single-total-max-min", type=float, default=DEFAULTS["single_total_max_min"])

    ap.add_argument("--large-min-trains", type=int, default=DEFAULTS["large_min_trains"])
    ap.add_argument("--large-min-total-min", type=float, default=DEFAULTS["large_min_total_min"])
    ap.add_argument("--large-min-max-min", type=float, default=DEFAULTS["large_min_max_min"])
    return ap.parse_args()


def main():
    if len(sys.argv) == 1:
        print("[INFO] 未传命令行参数，使用默认配置（可直接在 VSCode 运行）。")

    args = parse_args()
    np.random.seed(args.seed)

    in_primary = to_abs(args.input_json)
    in_fallback = to_abs(args.fallback_input_json)

    if in_primary.exists():
        in_path = in_primary
    elif in_fallback.exists():
        in_path = in_fallback
        print(f"[WARN] input-json不存在，改用fallback: {display_path(in_path)}")
    else:
        raise FileNotFoundError(f"输入JSON均不存在:\n- {in_primary}\n- {in_fallback}")

    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records, meta = collect_records(raw)
    if not records:
        raise ValueError("未在输入JSON中找到可用模板。")

    processed = []
    parsed_events_n = 0

    for rec in records:
        tpl = rec["template"]
        events = extract_events_recursive(tpl)
        if events:
            parsed_events_n += 1

        agg = extract_aggregate_stats(tpl)
        stats = calc_stats(events, agg)

        t2 = dict(tpl)
        t2["template_id"] = rec["template_id"]
        t2["source_scene_type"] = rec["source_scene_type"]
        t2["events"] = events
        t2["stats"] = stats
        processed.append(t2)

    # 重分类型（1~3短时 + 混合聚类）
    new_types = relabel_templates(processed, args)
    for t, nt in zip(processed, new_types):
        t["scene_type"] = nt

    # 按类型组装
    out_scene_types = OrderedDict()
    for st in [SCENE_SINGLE, SCENE_MIXED, SCENE_LARGE]:
        out_scene_types[st] = {
            "templates": [],
            "model": meta.get(st, {}).get("model", {}),
            "augment": meta.get(st, {}).get("augment", {}),
        }

    for t in processed:
        st = t.get("scene_type", SCENE_MIXED)
        if st not in out_scene_types:
            out_scene_types[st] = {"templates": [], "model": {}, "augment": {}}
        out_scene_types[st]["templates"].append(t)

    # 去重
    if parse_bool(args.dedup):
        for st in list(out_scene_types.keys()):
            out_scene_types[st]["templates"] = dedup_templates(out_scene_types[st]["templates"])

    # 按 severity 排序
    for st in list(out_scene_types.keys()):
        out_scene_types[st]["templates"].sort(
            key=lambda x: safe_float((x.get("stats") or {}).get("severity_score"), 0.0),
            reverse=True
        )

    out_json = to_abs(args.out_json)
    ensure_parent(out_json)

    total_tpl = sum(len(v["templates"]) for v in out_scene_types.values())
    with_events = sum(
        1
        for v in out_scene_types.values()
        for t in v["templates"]
        if isinstance(t.get("events"), list) and len(t["events"]) > 0
    )

    lib_out = {
        "version": "2.4",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_json": str(in_path),
        "build_info": {
            "dedup": parse_bool(args.dedup),
            "cluster_method": "hybrid(rule + 2cluster)",
            "single_rule": {
                "affected_trains": f"1~{args.single_max_trains}",
                "avg_delay_min_max": args.single_avg_max_min,
                "max_delay_min_max": args.single_max_max_min,
                "total_delay_min_max": args.single_total_max_min,
            },
            "large_rule": {
                "affected_trains_min": args.large_min_trains,
                "total_delay_min_min": args.large_min_total_min,
                "max_delay_min_min": args.large_min_max_min,
            },
            "event_parse_coverage": {
                "templates_total": total_tpl,
                "templates_with_events": with_events,
                "coverage": round(with_events / max(total_tpl, 1), 6),
            },
            "parser_guard": {
                "strict_train_token": True,
                "ignore_affected_train_key": True,
                "ignore_delay_ratio_key": True,
            }
        },
        "scene_types": out_scene_types,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pyify(lib_out), f, ensure_ascii=False, indent=2)

    # 汇总
    rows = []
    for st, payload in out_scene_types.items():
        for t in payload.get("templates", []):
            ss = t.get("stats", {}) if isinstance(t.get("stats"), dict) else {}
            rows.append({
                "scene_type": st,
                "template_id": t.get("template_id", ""),
                "affected_trains": int(safe_float(ss.get("affected_trains"), 0)),
                "total_delay_min": round(safe_float(ss.get("total_delay_min"), 0.0), 6),
                "avg_delay_min": round(safe_float(ss.get("avg_delay_min"), 0.0), 6),
                "max_delay_min": round(safe_float(ss.get("max_delay_min"), 0.0), 6),
                "severity_score": round(safe_float(ss.get("severity_score"), 0.0), 6),
            })

    df = pd.DataFrame(rows, columns=[
        "scene_type", "template_id", "affected_trains", "total_delay_min",
        "avg_delay_min", "max_delay_min", "severity_score"
    ])

    out_summary = to_abs(args.out_summary)
    ensure_parent(out_summary)
    df.to_excel(out_summary, index=False)

    type_counts = {
        st: len(payload.get("templates", []))
        for st, payload in out_scene_types.items()
        if len(payload.get("templates", [])) > 0
    }

    print(f"[OK] 场景库JSON: {display_path(out_json)}")
    print(f"[OK] 场景库汇总: {display_path(out_summary)}")
    print(f"[INFO] 类型数量: {len(type_counts)}")
    print(f"[INFO] 各类型模板数: {type_counts}")
    print(f"[INFO] 模板events可解析率: {with_events}/{max(total_tpl, 1)} = {with_events / max(total_tpl, 1):.2%}")


if __name__ == "__main__":
    main()
