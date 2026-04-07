# -*- coding: utf-8 -*-
"""
scene_03_inject_plan.py
读取 injection_plan，注入到按天车次延误矩阵，并输出 pack / manifest
"""

import argparse
import json
import pickle
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

SCENE_SINGLE = "单列车短时晚点"
SCENE_MIXED = "混合型晚点场景"
SCENE_LARGE = "大面积晚点干扰"

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

SYNTH_RULES = {
    SCENE_SINGLE: {"n_range": (1, 3), "delay_min_range": (3, 10)},
    SCENE_MIXED: {"n_range": (3, 10), "delay_min_range": (4, 18)},
    SCENE_LARGE: {"n_range": (20, 80), "delay_min_range": (6, 30)},
}

# 关键词
TRAIN_KEY_RE = re.compile(r"(train|车次|列车|车号|车次号|train_id|trainno|train_no|checi)", re.I)
DELAY_KEY_RE = re.compile(r"(delay|late|晚点|延误|误点|延时|delay_sec|delay_min|total_delay)", re.I)
MAP_HINT_RE = re.compile(
    r"(map|dict|pair|train.*delay|delay.*train|车次.*(晚点|延误)|晚点.*车次|延误.*车次)",
    re.I,
)
TRAIN_COL_RE = re.compile(r"(train|车次|列车|车号|车次号|train_id|trainno)", re.I)

# 防误识别过滤（关键）
TRAIN_KEY_BLOCK_RE = re.compile(
    r"(affected|impact|受影响|影响|count|num|number|total|sum|ratio|比例|占比|数量|总数|累计|avg|mean|max|min)",
    re.I
)
DELAY_KEY_BLOCK_RE = re.compile(
    r"(ratio|rate|prob|概率|占比|比例|std|方差|variance|阈值|threshold|coef|weight|noise)",
    re.I
)
NON_EVENT_CONTAINER_RE = re.compile(
    r"(model|augment|config|meta|summary|stat|stats|feature|profile|build_info|jitter|template_stats|参数|指标)",
    re.I
)

DEFAULTS = {
    "plan_json": "artifacts/scene_lib/injection_plan_moderate.json",
    "base_file": "沪杭长场基本图.xls",
    "out_pack": "artifacts/injected_pack.pkl",
    "out_manifest": "artifacts/injected_manifest.xlsx",
    "combine_mode": "sum",
    "seed": 42,

    "delay_noise_std": 0.10,
    "delay_threshold_sec": 60.0,
    "min_delay_sec": 30.0,
    "max_delay_sec": 6 * 3600.0,
    "max_total_delay_sec": 8 * 3600.0,

    # 单列车场景重整形参数
    "single_target_probs": "0.40,0.40,0.20",  # 1/2/3辆概率
    "single_expand_from_one_prob": 0.80,
    "single_delay_min_sec": 120.0,
    "single_delay_max_sec": 1200.0,

    # 新增：是否尊重scene_02里每条注入自带参数
    "respect_scene_combine_mode": 1,
    "respect_scene_noise_std": 1,
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


def parse_number(x):
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float, np.number)):
        v = float(x)
        if np.isfinite(v):
            return v
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


def parse_prob_vector(s: str, default_vec: np.ndarray) -> np.ndarray:
    try:
        vals = [float(x) for x in re.split(r"[,;/\s]+", str(s).strip()) if x.strip()]
        arr = np.array(vals, dtype=float)
        if arr.size != default_vec.size or np.any(arr < 0) or arr.sum() <= 0:
            return default_vec.copy()
        return arr / arr.sum()
    except Exception:
        return default_vec.copy()


def canonical_scene_type(x: str) -> str:
    s = str(x or "").strip()
    s = re.sub(r"\s*[（(]孤立[)）]\s*$", "", s)
    if not s:
        return SCENE_MIXED
    return SCENE_ALIAS.get(s.lower(), SCENE_ALIAS.get(s, s))


def normalize_train_id(x) -> str:
    s = str(x).strip()
    if not s:
        return ""
    s = s.replace("次", "").replace(" ", "")
    if s.lower() in {"nan", "none", "null"}:
        return ""
    if re.search(r"[A-Za-z]", s):
        s = s.upper()
    return s


def looks_like_train_token(x) -> bool:
    """
    收紧车次识别，避免把统计数字(5/12)当车次
    """
    s = normalize_train_id(x)
    if not s:
        return False
    if len(s) > 20:
        return False
    if re.search(r"[/:.\-\s]", s):
        return False

    if re.fullmatch(r"[A-Z]{1,3}\d{1,6}[A-Z]?", s):
        return True
    if re.fullmatch(r"\d{3,6}", s):
        return True
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

    if any(tok in k for tok in ["sec", "second", "秒", "_s"]):
        return float(num)
    if any(tok in k for tok in ["min", "minute", "分钟", "分", "_m"]):
        return float(num * 60.0)

    if num <= 45:
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
    if len(pairs) >= 2 and bad <= len(pairs):
        return pairs
    if MAP_HINT_RE.search(str(key_hint or "")):
        return pairs
    return []


def parse_text_event(s: str):
    txt = str(s).strip()
    if not txt:
        return None
    m = re.search(r"([A-Za-z]{0,3}\d{1,6}[A-Za-z]?)\D+(-?\d+(?:\.\d+)?)", txt)
    if not m:
        return None
    tid = normalize_train_id(m.group(1))
    if not looks_like_train_token(tid):
        return None
    sec = value_to_sec(m.group(2), txt.lower())
    if sec <= 0:
        return None
    return {"train_id": tid, "delay_sec": sec}


def normalize_event_list(events) -> list:
    agg = OrderedDict()
    if not isinstance(events, list):
        return []

    for item in events:
        parsed = []

        if isinstance(item, dict):
            tid = extract_train_from_event_dict(item)
            dsec = extract_delay_from_event_dict(item)
            if tid and dsec > 0:
                parsed.append({"train_id": tid, "delay_sec": dsec})
            else:
                parsed.extend(dict_as_train_delay_map(item, ""))

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            tid = normalize_train_id(item[0])
            dsec = value_to_sec(item[1], "")
            if looks_like_train_token(tid) and dsec > 0:
                parsed.append({"train_id": tid, "delay_sec": dsec})

        elif isinstance(item, str):
            pe = parse_text_event(item)
            if pe:
                parsed.append(pe)

        for e in parsed:
            tid = normalize_train_id(e.get("train_id", ""))
            ds = safe_float(e.get("delay_sec"), 0.0)
            if not tid or ds <= 0:
                continue
            ds = float(np.clip(ds, 1.0, 8 * 3600.0))
            old = agg.get(tid, 0.0)
            if ds > old:
                agg[tid] = ds

    return [{"train_id": k, "delay_sec": round(v, 6)} for k, v in agg.items()]


def extract_events_recursive(obj, root_key: str = "") -> list:
    raw = []

    def walk(x, parent_key="", depth=0):
        if depth > 14:
            return

        if isinstance(x, list):
            ev = normalize_event_list(x)
            if ev:
                raw.extend(ev)
            for it in x:
                walk(it, parent_key, depth + 1)

        elif isinstance(x, dict):
            mp = dict_as_train_delay_map(x, parent_key)
            if mp:
                raw.extend(mp)

            tid = extract_train_from_event_dict(x)
            dsec = extract_delay_from_event_dict(x)
            if tid and dsec > 0:
                raw.append({"train_id": tid, "delay_sec": dsec})

            for k, v in x.items():
                k_str = str(k)
                if NON_EVENT_CONTAINER_RE.search(k_str):
                    continue
                if isinstance(v, (list, dict)):
                    walk(v, k_str, depth + 1)

    walk(obj, root_key, 0)
    return normalize_event_list(raw)


def extract_events_from_injection(inj: dict) -> list:
    """
    关键修复：
    - 仅从事件相关字段提取
    - 不再在失败时回退解析整个inj（避免template_stats污染成伪事件）
    """
    if not isinstance(inj, dict):
        return []

    candidates = []

    direct_keys = [
        "events", "event_list", "delay_list", "items", "records", "samples",
        "train_delay_map", "delay_map", "train_delays", "train2delay", "delay_by_train",
    ]
    for k in direct_keys:
        v = inj.get(k)
        if isinstance(v, (list, dict)):
            candidates.append((k, v))

    # 常见嵌套模板容器
    for box in ["template", "payload", "scene", "template_obj", "template_data", "data"]:
        v = inj.get(box)
        if isinstance(v, dict):
            for k in direct_keys:
                vv = v.get(k)
                if isinstance(vv, (list, dict)):
                    candidates.append((f"{box}.{k}", vv))
            # 兼容老模板：把模板本体也尝试一次（递归里已对stats/config做过滤）
            candidates.append((box, v))
        elif isinstance(v, list):
            candidates.append((box, v))

    all_events = []
    for hint, c in candidates:
        ev = extract_events_recursive(c, root_key=hint)
        if ev:
            all_events.extend(ev)

    return normalize_event_list(all_events)


def normalize_date_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return s
    return ts.strftime("%Y-%m-%d")


def looks_like_date_key(k: str) -> bool:
    s = str(k).strip()
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", s):
        return True
    if re.match(r"^\d{4}/\d{1,2}/\d{1,2}$", s):
        return True
    return False


def sort_date_key(s: str):
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return (1, str(s))
    return (0, ts.to_pydatetime())


def pick_injection_list_from_day_obj(day_obj: dict) -> list:
    if not isinstance(day_obj, dict):
        return []
    for k in ["injections", "scenes", "items", "scene_list", "plan"]:
        v = day_obj.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if isinstance(v, dict):
            arr = []
            for vv in v.values():
                if isinstance(vv, list):
                    arr.extend([x for x in vv if isinstance(x, dict)])
            if arr:
                return arr
    return []


def collect_day_injections(plan_obj) -> OrderedDict:
    out = OrderedDict()

    def add(date_key, injs):
        d = normalize_date_str(date_key)
        if not d:
            return
        arr = [x for x in injs if isinstance(x, dict)]
        if not arr:
            return
        if d not in out:
            out[d] = []
        out[d].extend(arr)

    if isinstance(plan_obj, dict):
        if isinstance(plan_obj.get("days"), list):
            for day_obj in plan_obj["days"]:
                if not isinstance(day_obj, dict):
                    continue
                d = day_obj.get("date") or day_obj.get("day") or day_obj.get("day_date")
                injs = pick_injection_list_from_day_obj(day_obj)
                add(d, injs)

        if not out:
            for key in ["daily_plan", "plan", "date_to_injections", "date_plan"]:
                v = plan_obj.get(key)
                if isinstance(v, dict):
                    for d, obj in v.items():
                        if isinstance(obj, list):
                            add(d, obj)
                        elif isinstance(obj, dict):
                            injs = pick_injection_list_from_day_obj(obj)
                            if injs:
                                add(d, injs)
                elif isinstance(v, list):
                    for day_obj in v:
                        if not isinstance(day_obj, dict):
                            continue
                        d = day_obj.get("date") or day_obj.get("day") or day_obj.get("day_date")
                        injs = pick_injection_list_from_day_obj(day_obj)
                        add(d, injs)

        if not out:
            for k, v in plan_obj.items():
                if not looks_like_date_key(k):
                    continue
                if isinstance(v, list):
                    add(k, v)
                elif isinstance(v, dict):
                    injs = pick_injection_list_from_day_obj(v)
                    if injs:
                        add(k, injs)

        if not out and isinstance(plan_obj.get("injections"), list):
            for inj in plan_obj["injections"]:
                if not isinstance(inj, dict):
                    continue
                d = inj.get("date") or inj.get("day") or inj.get("day_date")
                if d:
                    add(d, [inj])

    elif isinstance(plan_obj, list):
        for obj in plan_obj:
            if not isinstance(obj, dict):
                continue
            d = obj.get("date") or obj.get("day") or obj.get("day_date")
            if d:
                injs = pick_injection_list_from_day_obj(obj)
                if injs:
                    add(d, injs)
                elif "scene_type" in obj or "template" in obj:
                    add(d, [obj])

    ordered = OrderedDict()
    for d in sorted(out.keys(), key=sort_date_key):
        if out[d]:
            ordered[d] = out[d]
    return ordered


def extract_scene_type(inj: dict) -> str:
    cands = [
        inj.get("scene_type"),
        inj.get("type"),
        inj.get("scene"),
        inj.get("category"),
        inj.get("scene_name"),
    ]
    tpl = inj.get("template")
    if isinstance(tpl, dict):
        cands.extend([tpl.get("scene_type"), tpl.get("type"), tpl.get("scene_name")])

    for c in cands:
        if c is None:
            continue
        st = canonical_scene_type(c)
        if st in SYNTH_RULES:
            return st
    return SCENE_MIXED


def extract_template_id(inj: dict, fallback: str) -> str:
    keys = ["template_id", "id", "scene_id", "name"]
    for k in keys:
        v = inj.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()

    tpl = inj.get("template")
    if isinstance(tpl, dict):
        for k in keys:
            v = tpl.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()

    return fallback


def extract_intensity(inj: dict, default=1.0) -> float:
    keys = ["intensity", "scale", "severity_scale", "amp", "ratio"]
    for k in keys:
        if k in inj:
            return float(np.clip(safe_float(inj.get(k), default), 0.4, 2.5))

    tpl = inj.get("template")
    if isinstance(tpl, dict):
        for k in keys:
            if k in tpl:
                return float(np.clip(safe_float(tpl.get(k), default), 0.4, 2.5))
    return float(default)


def extract_time_shift_min(inj: dict, default=0.0) -> float:
    keys = ["time_shift_min", "shift_min", "max_time_shift_min", "time_shift", "shift"]
    for k in keys:
        if k in inj:
            return safe_float(inj.get(k), default)

    j = inj.get("jitter")
    if isinstance(j, dict):
        for k in ["time_shift_min", "shift_min", "time_shift", "shift"]:
            if k in j:
                return safe_float(j.get(k), default)

    return float(default)


def extract_scene_noise_std(inj: dict, default=0.10) -> float:
    # 优先 jitter.delay_noise_std
    j = inj.get("jitter")
    if isinstance(j, dict):
        for k in ["delay_noise_std", "noise_std", "noise"]:
            if k in j:
                return float(np.clip(safe_float(j.get(k), default), 0.0, 1.0))

    for k in ["delay_noise_std", "noise_std", "noise"]:
        if k in inj:
            return float(np.clip(safe_float(inj.get(k), default), 0.0, 1.0))

    tpl = inj.get("template")
    if isinstance(tpl, dict):
        for k in ["delay_noise_std", "noise_std", "noise"]:
            if k in tpl:
                return float(np.clip(safe_float(tpl.get(k), default), 0.0, 1.0))

    return float(np.clip(default, 0.0, 1.0))


def extract_combine_mode(inj: dict, default="sum") -> str:
    for k in ["combine_mode", "merge_mode", "combine"]:
        v = str(inj.get(k, "")).strip().lower()
        if v in {"sum", "max"}:
            return v

    tpl = inj.get("template")
    if isinstance(tpl, dict):
        for k in ["combine_mode", "merge_mode", "combine"]:
            v = str(tpl.get(k, "")).strip().lower()
            if v in {"sum", "max"}:
                return v

    return "max" if str(default).lower() == "max" else "sum"


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def unique_natural(seq):
    seen = set()
    out = []
    for x in seq:
        xx = normalize_train_id(x)
        if not xx:
            continue
        if xx not in seen:
            seen.add(xx)
            out.append(xx)
    out.sort(key=natural_key)
    return out


def extract_train_ids_from_series(sr: pd.Series) -> list:
    ids = []
    for v in sr.dropna().astype(str).tolist():
        tid = normalize_train_id(v)
        if looks_like_train_token(tid):
            ids.append(tid)
    return ids


def extract_train_ids_from_df(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []

    ids = []
    cols = list(df.columns)

    # 优先列名命中
    candidate_cols = [c for c in cols if TRAIN_COL_RE.search(str(c))]
    if candidate_cols:
        for c in candidate_cols:
            ids.extend(extract_train_ids_from_series(df[c]))
        return ids

    # 兜底：内容模式识别
    for c in cols:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        sample = s.head(1500)
        toks = [normalize_train_id(x) for x in sample]
        hits = [t for t in toks if looks_like_train_token(t)]
        ratio = len(hits) / max(len(sample), 1)
        if len(set(hits)) >= 20 and ratio >= 0.50:
            ids.extend(hits)

    return ids


def read_train_ids_from_excel(path: Path) -> list:
    ids = []
    xls = pd.ExcelFile(path)
    for sh in xls.sheet_names:
        try:
            df = xls.parse(sh, dtype=str)
        except Exception:
            continue
        ids.extend(extract_train_ids_from_df(df))
    return unique_natural(ids)


def build_train_master(base_path: Path, day_map: OrderedDict) -> list:
    ids = []

    if base_path.exists():
        ext = base_path.suffix.lower()
        try:
            if ext in {".xls", ".xlsx", ".xlsm"}:
                ids = read_train_ids_from_excel(base_path)

            elif ext == ".csv":
                df = None
                try:
                    df = pd.read_csv(base_path, dtype=str, encoding="utf-8")
                except Exception:
                    try:
                        df = pd.read_csv(base_path, dtype=str, encoding="gbk")
                    except Exception:
                        df = None
                if df is not None:
                    ids = unique_natural(extract_train_ids_from_df(df))

            elif ext == ".pkl":
                with open(base_path, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, pd.DataFrame):
                    ids = unique_natural(extract_train_ids_from_df(obj))
                elif isinstance(obj, dict):
                    if isinstance(obj.get("train_ids"), list):
                        ids = unique_natural(obj.get("train_ids"))

            else:
                try:
                    df = pd.read_excel(base_path, dtype=str)
                    ids = unique_natural(extract_train_ids_from_df(df))
                except Exception:
                    pass

        except Exception as e:
            print(f"[WARN] 读取base-file失败，尝试回退plan车次池: {e}")

    if ids:
        print(f"[INFO] 车次池来源: base-file, 数量={len(ids)}")
        return ids

    # 回退1：从plan事件提取
    tmp = []
    for _, injs in day_map.items():
        for inj in injs:
            for e in extract_events_from_injection(inj):
                tid = normalize_train_id(e.get("train_id", ""))
                if tid:
                    tmp.append(tid)
    ids = unique_natural(tmp)
    if ids:
        print(f"[WARN] base-file未提取到车次，回退到plan事件车次池，数量={len(ids)}")
        return ids

    # 回退2：伪车次池
    ids = [f"T{i:04d}" for i in range(1, 501)]
    print("[WARN] 未找到有效车次池，使用默认伪车次池500个。")
    return ids


def sample_train_ids(train_pool: list, n: int, rng: np.random.Generator) -> list:
    if n <= 0:
        return []
    if not train_pool:
        return [f"T{i:05d}" for i in range(1, n + 1)]
    n = int(max(1, n))
    if n <= len(train_pool):
        idx = rng.choice(len(train_pool), size=n, replace=False)
        return [train_pool[int(i)] for i in idx]
    idx = rng.choice(len(train_pool), size=n, replace=True)
    return [train_pool[int(i)] for i in idx]


def synthesize_events(scene_type: str, train_pool: list, intensity: float, rng: np.random.Generator) -> list:
    st = canonical_scene_type(scene_type)
    if st not in SYNTH_RULES:
        st = SCENE_MIXED

    rule = SYNTH_RULES[st]
    n_low, n_high = rule["n_range"]
    d_low, d_high = rule["delay_min_range"]

    n = int(rng.integers(n_low, n_high + 1))
    if st != SCENE_SINGLE:
        n = int(round(n * np.clip(0.90 + 0.45 * (intensity - 1.0), 0.6, 1.5)))
        n = max(1, n)

    if train_pool:
        n = min(n, max(1, len(train_pool)))

    tids = sample_train_ids(train_pool, n, rng)

    mins = rng.uniform(d_low, d_high, size=n)
    if st == SCENE_LARGE:
        mins = mins * rng.lognormal(mean=0.0, sigma=0.22, size=n)
    elif st == SCENE_MIXED:
        mins = mins * rng.lognormal(mean=0.0, sigma=0.15, size=n)
    else:
        mins = mins * rng.uniform(0.90, 1.10, size=n)

    sec = np.clip(mins * 60.0, 60.0, 4 * 3600.0)
    return [{"train_id": tid, "delay_sec": float(s)} for tid, s in zip(tids, sec)]


def reshape_single_events(
    events: list,
    train_pool: list,
    rng: np.random.Generator,
    target_probs: np.ndarray,
    expand_from_one_prob: float,
    clip_sec: tuple,
) -> list:
    ev = normalize_event_list(events)

    lo, hi = clip_sec
    tmp = OrderedDict()
    for e in ev:
        tid = normalize_train_id(e.get("train_id", ""))
        ds = float(np.clip(safe_float(e.get("delay_sec"), 0.0), lo, hi))
        if tid and ds > 0:
            old = tmp.get(tid, 0.0)
            if ds > old:
                tmp[tid] = ds
    ev = [{"train_id": k, "delay_sec": v} for k, v in tmp.items()]

    n0 = len(ev)
    if n0 <= 0:
        target = int(rng.choice([1, 2, 3], p=target_probs))
    elif n0 == 1:
        if float(rng.random()) < expand_from_one_prob:
            target = int(rng.choice([2, 3], p=[0.75, 0.25]))
        else:
            target = 1
    elif n0 == 2:
        target = 2 if float(rng.random()) < 0.75 else 3
    else:
        target = 3

    used = {e["train_id"] for e in ev}
    cand = [t for t in train_pool if t not in used]

    base = float(np.median([e["delay_sec"] for e in ev])) if ev else float(rng.uniform(240, 600))

    guard = 0
    while len(ev) < target and guard < 1000:
        guard += 1

        if cand:
            j = int(rng.integers(0, len(cand)))
            tid = cand.pop(j)
        elif train_pool:
            pool_unused = [t for t in train_pool if t not in used]
            if pool_unused:
                tid = pool_unused[int(rng.integers(0, len(pool_unused)))]
            else:
                tid = f"T{10000 + len(ev)}"
        else:
            tid = f"T{10000 + len(ev)}"

        ds = float(np.clip(base * float(rng.uniform(0.6, 1.0)), lo, hi))
        ev.append({"train_id": tid, "delay_sec": ds})
        used.add(tid)

    if len(ev) > target:
        ev = sorted(ev, key=lambda x: x["delay_sec"], reverse=True)[:target]

    return normalize_event_list(ev)


def apply_intensity_and_noise(
    events: list,
    intensity: float,
    noise_std: float,
    rng: np.random.Generator,
    min_delay_sec: float,
    max_delay_sec: float,
) -> list:
    out = []
    scale = float(np.clip(intensity, 0.4, 2.5))
    for e in normalize_event_list(events):
        tid = normalize_train_id(e.get("train_id", ""))
        ds = safe_float(e.get("delay_sec"), 0.0)
        if not tid or ds <= 0:
            continue
        noise = float(np.exp(rng.normal(0, noise_std))) if noise_std > 0 else 1.0
        ds2 = ds * scale * noise
        ds2 = float(np.clip(ds2, min_delay_sec, max_delay_sec))
        out.append({"train_id": tid, "delay_sec": ds2})
    return normalize_event_list(out)


def remap_unknown_train_ids(events: list, train_pool: list, rng: np.random.Generator) -> list:
    if not train_pool:
        return normalize_event_list(events)

    train_set = set(train_pool)
    used = set()
    out = []

    for e in normalize_event_list(events):
        tid = normalize_train_id(e.get("train_id", ""))
        ds = safe_float(e.get("delay_sec"), 0.0)
        if ds <= 0:
            continue

        if tid not in train_set:
            cand = [t for t in train_pool if t not in used]
            if not cand:
                cand = train_pool
            tid = cand[int(rng.integers(0, len(cand)))]

        used.add(tid)
        out.append({"train_id": tid, "delay_sec": ds})

    return normalize_event_list(out)


def inject_from_plan(day_map: OrderedDict, train_master: list, args):
    rng = np.random.default_rng(args.seed)

    date_to_train_delay = OrderedDict()
    scene_rows = []
    event_rows = []

    synth_scene_count = 0
    synth_event_count = 0
    single_events_dist = []

    for date, injections in day_map.items():
        day_delay = {}

        for sidx, inj in enumerate(injections, start=1):
            scene_type = extract_scene_type(inj)
            template_id = extract_template_id(inj, f"{scene_type}_{sidx:03d}")
            intensity = extract_intensity(inj, 1.0)
            time_shift_min = extract_time_shift_min(inj, 0.0)

            if args.respect_scene_combine_mode:
                scene_combine_mode = extract_combine_mode(inj, args.combine_mode)
            else:
                scene_combine_mode = args.combine_mode

            if args.respect_scene_noise_std:
                noise_std_scene = extract_scene_noise_std(inj, args.delay_noise_std)
            else:
                noise_std_scene = args.delay_noise_std

            events = extract_events_from_injection(inj)
            source = "template"

            if not events:
                events = synthesize_events(scene_type, train_master, intensity, rng)
                source = "synthetic"
                synth_scene_count += 1
                synth_event_count += len(events)

            # 第一类重整形为1~3辆，避免全1辆
            if scene_type == SCENE_SINGLE:
                before_n = len(events)
                events = reshape_single_events(
                    events=events,
                    train_pool=train_master,
                    rng=rng,
                    target_probs=args.single_target_probs,
                    expand_from_one_prob=args.single_expand_from_one_prob,
                    clip_sec=(args.single_delay_min_sec, args.single_delay_max_sec),
                )
                after_n = len(events)
                if source == "template" and after_n != before_n:
                    source = "template+reshape"
                elif source == "synthetic":
                    source = "synthetic+reshape"

            events = apply_intensity_and_noise(
                events=events,
                intensity=intensity,
                noise_std=noise_std_scene,
                rng=rng,
                min_delay_sec=args.min_delay_sec,
                max_delay_sec=args.max_delay_sec,
            )
            events = remap_unknown_train_ids(events, train_master, rng)

            for e in events:
                tid = e["train_id"]
                ds = safe_float(e["delay_sec"], 0.0)
                old = day_delay.get(tid, 0.0)

                if scene_combine_mode == "max":
                    new = max(old, ds)
                else:
                    new = old + ds

                new = float(np.clip(new, 0.0, args.max_total_delay_sec))
                day_delay[tid] = new

                event_rows.append({
                    "date": date,
                    "scene_idx": sidx,
                    "scene_type": scene_type,
                    "template_id": template_id,
                    "train_id": tid,
                    "delay_sec": round(ds, 6),
                    "delay_min": round(ds / 60.0, 6),
                    "intensity": round(float(intensity), 6),
                    "scene_noise_std": round(float(noise_std_scene), 6),
                    "combine_mode": scene_combine_mode,
                    "time_shift_min": round(float(time_shift_min), 3),
                    "event_source": source,
                })

            arr = np.array([safe_float(e["delay_sec"], 0.0) for e in events], dtype=float)
            arr = arr[arr > 0]
            total_m = float(arr.sum() / 60.0) if arr.size else 0.0
            avg_m = float(arr.mean() / 60.0) if arr.size else 0.0
            max_m = float(arr.max() / 60.0) if arr.size else 0.0

            scene_rows.append({
                "date": date,
                "scene_idx": sidx,
                "scene_type": scene_type,
                "template_id": template_id,
                "events": int(arr.size),
                "total_delay_min": round(total_m, 6),
                "avg_delay_min": round(avg_m, 6),
                "max_delay_min": round(max_m, 6),
                "intensity": round(float(intensity), 6),
                "scene_noise_std": round(float(noise_std_scene), 6),
                "combine_mode": scene_combine_mode,
                "time_shift_min": round(float(time_shift_min), 3),
                "event_source": source,
            })

            if scene_type == SCENE_SINGLE:
                single_events_dist.append(int(arr.size))

        day_delay = {k: round(v, 6) for k, v in day_delay.items() if v > 0}
        date_to_train_delay[date] = day_delay

    scene_df = pd.DataFrame(scene_rows, columns=[
        "date", "scene_idx", "scene_type", "template_id", "events",
        "total_delay_min", "avg_delay_min", "max_delay_min",
        "intensity", "scene_noise_std", "combine_mode", "time_shift_min", "event_source"
    ])
    event_df = pd.DataFrame(event_rows, columns=[
        "date", "scene_idx", "scene_type", "template_id", "train_id",
        "delay_sec", "delay_min", "intensity", "scene_noise_std", "combine_mode",
        "time_shift_min", "event_source"
    ])

    return date_to_train_delay, scene_df, event_df, synth_scene_count, synth_event_count, single_events_dist


def build_day_summary(date_to_train_delay: OrderedDict, train_master: list, threshold_sec: float) -> pd.DataFrame:
    rows = []
    trains_n = int(len(train_master))
    for date, day_delay in date_to_train_delay.items():
        vec = np.array([safe_float(day_delay.get(tid, 0.0), 0.0) for tid in train_master], dtype=float)
        mask = vec >= threshold_sec
        delayed = vec[mask] / 60.0

        rows.append({
            "date": date,
            "trains": trains_n,
            "delayed_trains(>=60s)": int(mask.sum()),
            "delay_ratio": round(float(mask.sum()) / max(trains_n, 1), 6),
            "avg_delay_min": round(float(delayed.mean()) if delayed.size else 0.0, 6),
            "p90_delay_min": round(float(np.percentile(delayed, 90)) if delayed.size else 0.0, 6),
            "max_delay_min": round(float(delayed.max()) if delayed.size else 0.0, 6),
        })

    return pd.DataFrame(rows, columns=[
        "date", "trains", "delayed_trains(>=60s)", "delay_ratio",
        "avg_delay_min", "p90_delay_min", "max_delay_min"
    ])


def build_delay_matrix(date_to_train_delay: OrderedDict, train_master: list) -> pd.DataFrame:
    dates = list(date_to_train_delay.keys())
    mat = pd.DataFrame(0.0, index=dates, columns=train_master, dtype=float)
    for d in dates:
        dd = date_to_train_delay.get(d, {})
        for tid, ds in dd.items():
            if tid in mat.columns:
                mat.at[d, tid] = float(ds)
    return mat


def parse_args():
    ap = argparse.ArgumentParser("Inject plan to train delays and export pack/manifest")
    ap.add_argument("--plan-json", type=str, default=DEFAULTS["plan_json"])
    ap.add_argument("--base-file", type=str, default=DEFAULTS["base_file"])
    ap.add_argument("--out-pack", type=str, default=DEFAULTS["out_pack"])
    ap.add_argument("--out-manifest", type=str, default=DEFAULTS["out_manifest"])
    ap.add_argument("--combine-mode", type=str, default=DEFAULTS["combine_mode"], choices=["sum", "max"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    ap.add_argument("--delay-noise-std", type=float, default=DEFAULTS["delay_noise_std"])
    ap.add_argument("--delay-threshold-sec", type=float, default=DEFAULTS["delay_threshold_sec"])
    ap.add_argument("--min-delay-sec", type=float, default=DEFAULTS["min_delay_sec"])
    ap.add_argument("--max-delay-sec", type=float, default=DEFAULTS["max_delay_sec"])
    ap.add_argument("--max-total-delay-sec", type=float, default=DEFAULTS["max_total_delay_sec"])

    ap.add_argument("--single-target-probs", type=str, default=DEFAULTS["single_target_probs"])
    ap.add_argument("--single-expand-from-one-prob", type=float, default=DEFAULTS["single_expand_from_one_prob"])
    ap.add_argument("--single-delay-min-sec", type=float, default=DEFAULTS["single_delay_min_sec"])
    ap.add_argument("--single-delay-max-sec", type=float, default=DEFAULTS["single_delay_max_sec"])

    ap.add_argument("--respect-scene-combine-mode", type=int, default=DEFAULTS["respect_scene_combine_mode"])
    ap.add_argument("--respect-scene-noise-std", type=int, default=DEFAULTS["respect_scene_noise_std"])
    return ap.parse_args()


def main():
    if len(sys.argv) == 1:
        print("[INFO] 未传命令行参数，使用默认配置（可直接在 VSCode 运行）。")

    args = parse_args()

    args.single_target_probs = parse_prob_vector(
        args.single_target_probs, np.array([0.40, 0.40, 0.20], dtype=float)
    )
    args.single_expand_from_one_prob = float(np.clip(args.single_expand_from_one_prob, 0.0, 1.0))
    args.single_delay_min_sec = float(max(1.0, args.single_delay_min_sec))
    args.single_delay_max_sec = float(max(args.single_delay_min_sec, args.single_delay_max_sec))

    args.min_delay_sec = float(max(1.0, args.min_delay_sec))
    args.max_delay_sec = float(max(args.min_delay_sec, args.max_delay_sec))
    args.max_total_delay_sec = float(max(args.max_delay_sec, args.max_total_delay_sec))
    args.delay_noise_std = float(max(0.0, args.delay_noise_std))

    args.respect_scene_combine_mode = bool(int(args.respect_scene_combine_mode))
    args.respect_scene_noise_std = bool(int(args.respect_scene_noise_std))

    plan_path = to_abs(args.plan_json)
    base_path = to_abs(args.base_file)
    out_pack = to_abs(args.out_pack)
    out_manifest = to_abs(args.out_manifest)

    if not plan_path.exists():
        raise FileNotFoundError(f"plan-json不存在: {plan_path}")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan_obj = json.load(f)

    day_map = collect_day_injections(plan_obj)
    if not day_map:
        raise ValueError("注入计划中未解析到按天注入数据（day->injections）。")

    train_master = build_train_master(base_path, day_map)

    (
        date_to_train_delay,
        scene_df,
        event_df,
        synth_scene_count,
        synth_event_count,
        single_events_dist,
    ) = inject_from_plan(day_map, train_master, args)

    day_df = build_day_summary(
        date_to_train_delay=date_to_train_delay,
        train_master=train_master,
        threshold_sec=args.delay_threshold_sec,
    )
    delay_matrix_df = build_delay_matrix(date_to_train_delay, train_master)

    pack = {
        "version": "2.1",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_plan_json": str(plan_path),
        "source_base_file": str(base_path),
        "combine_mode_default": args.combine_mode,
        "seed": int(args.seed),

        "train_ids": train_master,
        "dates": list(date_to_train_delay.keys()),
        "date_to_train_delay_sec": date_to_train_delay,
        "date_to_delay": date_to_train_delay,  # 兼容旧键
        "delay_matrix_sec": delay_matrix_df,

        "scene_manifest": scene_df,
        "event_manifest": event_df,
        "day_summary": day_df,

        "meta": {
            "delay_threshold_sec": float(args.delay_threshold_sec),
            "synth_scene_count": int(synth_scene_count),
            "synth_event_count": int(synth_event_count),
            "single_target_probs_1_2_3": [float(x) for x in args.single_target_probs.tolist()],
            "single_expand_from_one_prob": float(args.single_expand_from_one_prob),
            "respect_scene_combine_mode": bool(args.respect_scene_combine_mode),
            "respect_scene_noise_std": bool(args.respect_scene_noise_std),
        },
    }

    ensure_parent(out_pack)
    ensure_parent(out_manifest)

    with open(out_pack, "wb") as f:
        pickle.dump(pack, f)

    with pd.ExcelWriter(out_manifest) as writer:
        day_df.to_excel(writer, sheet_name="day_summary", index=False)
        scene_df.to_excel(writer, sheet_name="scene_manifest", index=False)
        event_df.to_excel(writer, sheet_name="event_manifest", index=False)

    print(f"[OK] 注入包: {display_path(out_pack)}")
    print(f"[OK] 注入清单: {display_path(out_manifest)}")
    print(f"[INFO] 生成测试天数: {len(date_to_train_delay)}")
    print(f"[INFO] 合成场景数(模板events为空触发): {synth_scene_count}")
    print(f"[INFO] 合成事件总数: {synth_event_count}")

    if single_events_dist:
        vc = pd.Series(single_events_dist).value_counts().sort_index().to_dict()
        vc = {int(k): int(v) for k, v in vc.items()}
        print(f"[INFO] 单列车短时晚点 events分布(场景内事件数): {vc}")


if __name__ == "__main__":
    main()
