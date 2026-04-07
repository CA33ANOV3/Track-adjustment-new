
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SCENE_TYPES = ["单列车短时晚点", "混合型晚点场景", "大面积晚点干扰"]


def q(x, p, default=0.0):
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, p)) if arr.size else float(default)


def peak_count(times_sec: np.ndarray, win_sec=900) -> int:
    t = np.sort(np.asarray(times_sec, dtype=float))
    if len(t) == 0:
        return 0
    j, best = 0, 0
    for i in range(len(t)):
        while t[i] - t[j] > win_sec:
            j += 1
        best = max(best, i - j + 1)
    return int(best)


def sget(df: pd.DataFrame, col: str, default):
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def norm_train(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    s = s.replace("次", "").replace(" ", "")
    if re.search(r"[A-Za-z]", s):
        s = s.upper()
    return s


def as_date(x):
    return pd.to_datetime(x).date()


def pick_days(pack: dict, scope: str):
    eps = {}
    for k, v in pack.get("episodes", {}).items():
        try:
            d = as_date(k)
        except Exception:
            continue
        eps[d] = v

    if scope == "all":
        return sorted(eps.keys()), eps

    if scope == "train":
        ds = [as_date(d) for d in pack.get("train_days", [])]
        ds = [d for d in ds if d in eps]
        return sorted(ds), eps

    if scope == "test":
        ds = [as_date(d) for d in pack.get("test_days", [])]
        ds = [d for d in ds if d in eps]
        return sorted(ds), eps

    raise ValueError(f"scope只支持 all/train/test，当前={scope}")


def flatten_pack(pack, scope="all") -> pd.DataFrame:
    days, episodes = pick_days(pack, scope=scope)
    rows = []

    for d in days:
        g = episodes[d].copy().reset_index(drop=True)
        if len(g) == 0:
            continue

        train = sget(g, "列车", "").map(norm_train)
        ttype = sget(g, "列车类型", "").astype(str).str.strip()
        track = sget(g, "接入股道", "").astype(str).str.strip()

        plan_arr = pd.to_numeric(sget(g, "plan_arr_sec", 0), errors="coerce").fillna(0).astype(float)
        plan_dep = pd.to_numeric(sget(g, "plan_dep_sec", 0), errors="coerce").fillna(0).astype(float)

        pri_sec = pd.to_numeric(sget(g, "init_delay_sec", 0), errors="coerce").fillna(0).clip(lower=0).astype(float)
        sec_sec = pd.to_numeric(sget(g, "hist_sec_delay_sec", 0), errors="coerce").fillna(0).clip(lower=0).astype(float)

        if "obs_dep_delay_sec" in g.columns:
            dep_delay = pd.to_numeric(g["obs_dep_delay_sec"], errors="coerce")
        else:
            dep_delay = pd.Series(np.nan, index=g.index)

        dep_delay = dep_delay.fillna(pri_sec + sec_sec).clip(lower=0).astype(float)

        # 跨日修复
        plan_dep_adj = plan_dep.copy()
        cross = plan_dep_adj < plan_arr
        plan_dep_adj.loc[cross] = plan_dep_adj.loc[cross] + 86400.0

        # 事件时刻：实际发车时刻（计划发车+发晚）
        event_sec = plan_dep_adj + dep_delay

        df = pd.DataFrame({
            "date": d,
            "train": train,
            "type": ttype,
            "track": track,
            "plan_arr_sec": plan_arr,
            "plan_dep_sec": plan_dep_adj,
            "pri_sec": pri_sec,
            "sec_sec": sec_sec,
            "dep_delay_sec": dep_delay,
            "event_sec": event_sec,
        })
        df["dep_delay_min"] = df["dep_delay_sec"] / 60.0
        df["hour"] = (df["event_sec"] // 3600).astype(int) % 24
        df["delayed"] = (df["dep_delay_sec"] >= 60).astype(int)
        rows.append(df)

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "date", "train", "type", "track", "plan_arr_sec", "plan_dep_sec",
            "pri_sec", "sec_sec", "dep_delay_sec", "event_sec",
            "dep_delay_min", "hour", "delayed"
        ])
    return pd.concat(rows, ignore_index=True)


def split_scenarios(delay_df: pd.DataFrame, gap_min=20.0, max_span_min=180.0, min_events=1):
    """
    改进点：
    1) 不仅看相邻gap，还限制单场景最大跨度 max_span_min，防止“链式串联”成超大场景
    2) 可选 min_events 过滤太小碎片
    """
    if len(delay_df) == 0:
        x = delay_df.copy()
        x["scenario_id"] = pd.Series(dtype=str)
        return x

    gap_sec = int(max(1, round(gap_min * 60)))
    max_span_sec = int(max(1, round(max_span_min * 60)))

    out = []
    for d, g in delay_df.sort_values(["date", "event_sec"]).groupby("date", sort=True):
        g = g.sort_values("event_sec").copy()
        g["scenario_id"] = None

        t = g["event_sec"].to_numpy(dtype=float)
        idx = np.arange(len(g), dtype=int)

        sid = 0
        i = 0
        while i < len(g):
            sid += 1
            start_t = t[i]
            prev_t = t[i]
            cur_pos = [i]

            j = i + 1
            while j < len(g):
                if (t[j] - prev_t > gap_sec) or (t[j] - start_t > max_span_sec):
                    break
                cur_pos.append(j)
                prev_t = t[j]
                j += 1

            g.loc[g.index[cur_pos], "scenario_id"] = f"{d}_S{sid:03d}"
            i = j

        out.append(g)

    out_df = pd.concat(out, ignore_index=True)

    if int(min_events) > 1 and len(out_df) > 0:
        c = out_df.groupby("scenario_id")["train"].count()
        keep = set(c[c >= int(min_events)].index.tolist())
        out_df = out_df[out_df["scenario_id"].isin(keep)].copy().reset_index(drop=True)

    return out_df


def extract_features(scen_rows: pd.DataFrame, day_total: dict):
    feats = []
    if len(scen_rows) == 0:
        return pd.DataFrame(columns=[
            "scenario_id", "date", "n_trains", "duration_min", "total_delay_min",
            "mean_delay_min", "median_delay_min", "p90_delay_min", "p95_delay_min", "max_delay_min",
            "peak15_cnt", "peak30_cnt", "impacted_ratio",
            "unique_tracks", "top1_track_share",
            "ratio_0_5", "ratio_5_15", "ratio_15_30", "ratio_30p", "ratio_60p",
        ])

    for sid, g in scen_rows.groupby("scenario_id"):
        d = g["date"].iloc[0]
        delays = g["dep_delay_min"].to_numpy(dtype=float)
        times = g["event_sec"].to_numpy(dtype=float)
        n = len(g)

        t0 = float(np.min(times))
        t1 = float(np.max(times))
        duration = 0.0 if n <= 1 else (t1 - t0) / 60.0

        trk = g["track"].fillna("").astype(str).str.strip()
        trk = trk[trk != ""]
        unique_tracks = int(trk.nunique()) if len(trk) else 0
        top1_track_share = float(trk.value_counts(normalize=True).iloc[0]) if len(trk) else 0.0

        r0_5 = float(np.mean((delays > 0) & (delays <= 5))) if n else 0.0
        r5_15 = float(np.mean((delays > 5) & (delays <= 15))) if n else 0.0
        r15_30 = float(np.mean((delays > 15) & (delays <= 30))) if n else 0.0
        r30p = float(np.mean(delays > 30)) if n else 0.0
        r60p = float(np.mean(delays > 60)) if n else 0.0

        feats.append({
            "scenario_id": sid,
            "date": d,
            "n_trains": int(n),
            "duration_min": float(duration),
            "total_delay_min": float(np.sum(delays)) if n else 0.0,
            "mean_delay_min": float(np.mean(delays)) if n else 0.0,
            "median_delay_min": q(delays, 50),
            "p90_delay_min": q(delays, 90),
            "p95_delay_min": q(delays, 95),
            "max_delay_min": float(np.max(delays)) if n else 0.0,
            "peak15_cnt": peak_count(times, win_sec=900),
            "peak30_cnt": peak_count(times, win_sec=1800),
            "impacted_ratio": float(n / max(day_total.get(d, n), 1)),
            "unique_tracks": unique_tracks,
            "top1_track_share": top1_track_share,
            "ratio_0_5": r0_5,
            "ratio_5_15": r5_15,
            "ratio_15_30": r15_30,
            "ratio_30p": r30p,
            "ratio_60p": r60p,
        })
    return pd.DataFrame(feats)


@dataclass
class RuleCfg:
    # 单列车短时晚点
    single_max_n: int = 3
    single_max_impacted_ratio: float = 0.10
    single_max_p90_delay: float = 15.0
    single_max_max_delay: float = 25.0
    single_max_total_delay: float = 45.0
    single_max_duration: float = 60.0

    # 大面积晚点（允许“强规模”或“中规模+强严重”）
    large_min_n: int = 12
    large_min_impacted_ratio: float = 0.20
    large_min_peak15: int = 8
    large_soft_n: int = 6
    large_min_total_delay: float = 300.0
    large_min_ratio30p: float = 0.35
    large_min_duration: float = 180.0
    large_min_max_delay: float = 60.0


def build_rule_cfg(args) -> RuleCfg:
    return RuleCfg(
        single_max_n=int(args.single_max_n),
        single_max_impacted_ratio=float(args.single_max_impacted_ratio),
        single_max_p90_delay=float(args.single_max_p90_delay),
        single_max_max_delay=float(args.single_max_max_delay),
        single_max_total_delay=float(args.single_max_total_delay),
        single_max_duration=float(args.single_max_duration),
        large_min_n=int(args.large_min_n),
        large_min_impacted_ratio=float(args.large_min_impacted_ratio),
        large_min_peak15=int(args.large_min_peak15),
        large_soft_n=int(args.large_soft_n),
        large_min_total_delay=float(args.large_min_total_delay),
        large_min_ratio30p=float(args.large_min_ratio30p),
        large_min_duration=float(args.large_min_duration),
        large_min_max_delay=float(args.large_min_max_delay),
    )


def adapt_rule_cfg(scen_feat: pd.DataFrame, cfg: RuleCfg, enabled: bool) -> RuleCfg:
    """
    可选自适应阈值（默认关）：
    用样本分位数微调“大面积阈值”，适配不同站场规模。
    """
    if (not enabled) or len(scen_feat) < 15:
        return cfg

    c = RuleCfg(**asdict(cfg))
    c.large_min_n = int(np.clip(round(q(scen_feat["n_trains"], 75)), 8, 20))
    c.large_min_impacted_ratio = float(np.clip(q(scen_feat["impacted_ratio"], 75), 0.12, 0.35))
    c.large_min_peak15 = int(np.clip(round(q(scen_feat["peak15_cnt"], 75)), 5, 12))
    c.large_min_total_delay = float(np.clip(q(scen_feat["total_delay_min"], 75), 180, 900))
    c.large_min_ratio30p = float(np.clip(q(scen_feat["ratio_30p"], 75), 0.20, 0.60))
    c.large_min_duration = float(np.clip(q(scen_feat["duration_min"], 75), 90, 360))
    c.large_min_max_delay = float(np.clip(q(scen_feat["max_delay_min"], 75), 40, 120))
    c.large_soft_n = int(max(4, min(c.large_soft_n, c.large_min_n)))
    return c


def classify_scene_row(r: pd.Series, cfg: RuleCfg) -> Tuple[str, str, float]:
    n = float(r["n_trains"])
    imp = float(r["impacted_ratio"])
    p90 = float(r["p90_delay_min"])
    mxd = float(r["max_delay_min"])
    tot = float(r["total_delay_min"])
    dur = float(r["duration_min"])
    peak = float(r["peak15_cnt"])
    r30 = float(r["ratio_30p"])

    # 1) 单列车短时晚点
    is_single = (
        (n <= cfg.single_max_n)
        and (imp <= cfg.single_max_impacted_ratio)
        and (p90 <= cfg.single_max_p90_delay)
        and (mxd <= cfg.single_max_max_delay)
        and (tot <= cfg.single_max_total_delay)
        and (dur <= cfg.single_max_duration)
    )
    if is_single:
        reason = (
            f"single: n={n:.0f}<= {cfg.single_max_n}, imp={imp:.3f}, "
            f"p90={p90:.1f}, max={mxd:.1f}, total={tot:.1f}, dur={dur:.1f}"
        )
        score = min(1.0, (p90 / max(cfg.single_max_p90_delay, 1e-6)) * 0.4 + (n / max(cfg.single_max_n, 1e-6)) * 0.6)
        return "单列车短时晚点", reason, float(score)

    # 2) 大面积晚点（强规模 or 中规模+强严重）
    hard_scale = (n >= cfg.large_min_n) or (imp >= cfg.large_min_impacted_ratio) or (peak >= cfg.large_min_peak15)
    soft_severe = (
        (n >= cfg.large_soft_n and tot >= cfg.large_min_total_delay)
        or (n >= cfg.large_soft_n and r30 >= cfg.large_min_ratio30p)
        or (n >= cfg.large_soft_n and dur >= cfg.large_min_duration)
        or (n >= cfg.large_soft_n and mxd >= cfg.large_min_max_delay)
    )

    if hard_scale or soft_severe:
        triggers = []
        if n >= cfg.large_min_n:
            triggers.append(f"n={n:.0f}>={cfg.large_min_n}")
        if imp >= cfg.large_min_impacted_ratio:
            triggers.append(f"imp={imp:.3f}>={cfg.large_min_impacted_ratio:.3f}")
        if peak >= cfg.large_min_peak15:
            triggers.append(f"peak15={peak:.0f}>={cfg.large_min_peak15}")
        if n >= cfg.large_soft_n and tot >= cfg.large_min_total_delay:
            triggers.append(f"total={tot:.1f}>={cfg.large_min_total_delay}")
        if n >= cfg.large_soft_n and r30 >= cfg.large_min_ratio30p:
            triggers.append(f"ratio30p={r30:.3f}>={cfg.large_min_ratio30p:.3f}")
        if n >= cfg.large_soft_n and dur >= cfg.large_min_duration:
            triggers.append(f"dur={dur:.1f}>={cfg.large_min_duration}")
        if n >= cfg.large_soft_n and mxd >= cfg.large_min_max_delay:
            triggers.append(f"max={mxd:.1f}>={cfg.large_min_max_delay}")

        reason = "large: " + "; ".join(triggers)
        score = min(
            1.0,
            0.30 * min(1.0, n / max(cfg.large_min_n, 1))
            + 0.25 * min(1.0, imp / max(cfg.large_min_impacted_ratio, 1e-6))
            + 0.20 * min(1.0, peak / max(cfg.large_min_peak15, 1))
            + 0.25 * min(1.0, tot / max(cfg.large_min_total_delay, 1e-6))
        )
        return "大面积晚点干扰", reason, float(score)

    # 3) 其余 = 混合型
    reason = f"mixed: n={n:.0f}, imp={imp:.3f}, p90={p90:.1f}, total={tot:.1f}, dur={dur:.1f}"
    score = min(
        1.0,
        0.35 * min(1.0, n / max(cfg.large_min_n, 1))
        + 0.35 * min(1.0, p90 / max(cfg.large_min_max_delay, 1))
        + 0.30 * min(1.0, tot / max(cfg.large_min_total_delay, 1))
    )
    return "混合型晚点场景", reason, float(score)


def attach_scene_labels(scen_feat: pd.DataFrame, cfg: RuleCfg) -> pd.DataFrame:
    if len(scen_feat) == 0:
        z = scen_feat.copy()
        z["scene_type"] = pd.Series(dtype=str)
        z["scene_reason"] = pd.Series(dtype=str)
        z["severity_score"] = pd.Series(dtype=float)
        return z

    out = scen_feat.copy()
    labels, reasons, scores = [], [], []
    for _, r in out.iterrows():
        typ, rsn, sc = classify_scene_row(r, cfg)
        labels.append(typ)
        reasons.append(rsn)
        scores.append(sc)

    out["scene_type"] = labels
    out["scene_reason"] = reasons
    out["severity_score"] = scores
    return out


def auto_eps(Xs, min_samples=6, qtile=0.90):
    k = max(2, min(int(min_samples), len(Xs)))
    nbrs = NearestNeighbors(n_neighbors=k).fit(Xs)
    dists, _ = nbrs.kneighbors(Xs)
    kth = np.sort(dists[:, -1])
    eps = float(np.quantile(kth, qtile))
    if eps <= 1e-8:
        eps = float(np.mean(kth) + 1e-3)
    return eps


def run_dbscan_diag(scen_feat: pd.DataFrame, min_samples: int, eps: float | None, eps_quantile: float, use_dbscan: bool):
    if (not use_dbscan) or len(scen_feat) < 2:
        z = scen_feat.copy()
        z["cluster"] = np.nan
        cluster_summary = pd.DataFrame(columns=[
            "cluster", "scenario_cnt", "scene_type_mode",
            "n_trains_med", "duration_med", "p90_delay_med", "total_delay_med", "impacted_ratio_med"
        ])
        cluster_mix = pd.DataFrame(columns=["cluster"] + SCENE_TYPES)
        return z, cluster_summary, cluster_mix, np.nan, np.nan

    z = scen_feat.copy()
    feat_cols = [
        "n_trains", "duration_min", "total_delay_min",
        "p90_delay_min", "max_delay_min",
        "peak15_cnt", "impacted_ratio", "ratio_30p",
        "unique_tracks", "top1_track_share",
    ]
    X = z[feat_cols].fillna(0.0).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    ms = max(2, min(int(min_samples), len(z)))
    eps_used = float(eps) if eps is not None else auto_eps(Xs, min_samples=ms, qtile=float(eps_quantile))
    labels = DBSCAN(eps=eps_used, min_samples=ms).fit_predict(Xs)

    z["cluster"] = labels

    cluster_summary = z.groupby("cluster").agg(
        scenario_cnt=("scenario_id", "count"),
        scene_type_mode=("scene_type", lambda s: s.value_counts().index[0] if len(s) else "未知"),
        n_trains_med=("n_trains", "median"),
        duration_med=("duration_min", "median"),
        p90_delay_med=("p90_delay_min", "median"),
        total_delay_med=("total_delay_min", "median"),
        impacted_ratio_med=("impacted_ratio", "median"),
    ).reset_index().sort_values("scenario_cnt", ascending=False)

    mix = z.pivot_table(index="cluster", columns="scene_type", values="scenario_id", aggfunc="count", fill_value=0)
    for t in SCENE_TYPES:
        if t not in mix.columns:
            mix[t] = 0
    mix = mix[SCENE_TYPES].reset_index()

    return z, cluster_summary, mix, eps_used, ms


def make_scene_summary(scen_feat: pd.DataFrame) -> pd.DataFrame:
    if len(scen_feat) == 0:
        return pd.DataFrame(columns=[
            "scene_type", "scenario_cnt", "scenario_ratio",
            "trains_sum", "n_trains_med", "p90_delay_med", "total_delay_med"
        ])

    ss = scen_feat.groupby("scene_type").agg(
        scenario_cnt=("scenario_id", "count"),
        trains_sum=("n_trains", "sum"),
        n_trains_med=("n_trains", "median"),
        p90_delay_med=("p90_delay_min", "median"),
        total_delay_med=("total_delay_min", "median"),
    )
    ss = ss.reindex(SCENE_TYPES, fill_value=0).reset_index()
    total = max(float(ss["scenario_cnt"].sum()), 1.0)
    ss["scenario_ratio"] = ss["scenario_cnt"] / total
    return ss


def recommended_scene_mix(scene_summary: pd.DataFrame) -> str:
    # 输出 10份配比，给 scene_02 --scene-mix 直接用
    default_mix = "单列车短时晚点:4,混合型晚点场景:4,大面积晚点干扰:2"
    if len(scene_summary) == 0:
        return default_mix

    cnt = {t: 0.0 for t in SCENE_TYPES}
    for _, r in scene_summary.iterrows():
        t = str(r["scene_type"])
        if t in cnt:
            cnt[t] = float(r["scenario_cnt"])

    total = sum(cnt.values())
    if total <= 1e-9:
        return default_mix

    raw = [cnt[t] / total * 10.0 for t in SCENE_TYPES]
    base = [int(np.floor(x)) for x in raw]
    rem = [raw[i] - base[i] for i in range(3)]

    need = 10 - sum(base)
    order = np.argsort(rem)[::-1].tolist()
    i = 0
    while need > 0:
        base[order[i % 3]] += 1
        need -= 1
        i += 1

    return f"{SCENE_TYPES[0]}:{base[0]},{SCENE_TYPES[1]}:{base[1]},{SCENE_TYPES[2]}:{base[2]}"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-file", type=str, default="artifacts/data_pack.pkl")
    ap.add_argument("--scope", type=str, default="all", choices=["all", "train", "test"])
    ap.add_argument("--out-dir", type=str, default="artifacts/scenario_dbscan_pack")

    ap.add_argument("--min-delay-sec", type=float, default=60.0)
    ap.add_argument("--gap-min", type=float, default=20.0)
    ap.add_argument("--max-span-min", type=float, default=180.0)
    ap.add_argument("--min-scenario-events", type=int, default=1)

    # DBSCAN仅诊断
    ap.add_argument("--use-dbscan", type=int, default=1)
    ap.add_argument("--min-samples", type=int, default=6)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--eps-quantile", type=float, default=0.90)

    # 规则阈值（可按业务调）
    ap.add_argument("--adaptive-thresholds", type=int, default=0)

    ap.add_argument("--single-max-n", type=int, default=3)
    ap.add_argument("--single-max-impacted-ratio", type=float, default=0.10)
    ap.add_argument("--single-max-p90-delay", type=float, default=15.0)
    ap.add_argument("--single-max-max-delay", type=float, default=25.0)
    ap.add_argument("--single-max-total-delay", type=float, default=45.0)
    ap.add_argument("--single-max-duration", type=float, default=60.0)

    ap.add_argument("--large-min-n", type=int, default=12)
    ap.add_argument("--large-min-impacted-ratio", type=float, default=0.20)
    ap.add_argument("--large-min-peak15", type=int, default=8)
    ap.add_argument("--large-soft-n", type=int, default=6)
    ap.add_argument("--large-min-total-delay", type=float, default=300.0)
    ap.add_argument("--large-min-ratio30p", type=float, default=0.35)
    ap.add_argument("--large-min-duration", type=float, default=180.0)
    ap.add_argument("--large-min-max-delay", type=float, default=60.0)

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.pack_file, "rb") as f:
        pack = pickle.load(f)

    raw = flatten_pack(pack, scope=args.scope)

    # ===== 全量统计 =====
    overall = pd.DataFrame([{
        "scope": args.scope,
        "total_trains": int(len(raw)),
        "delayed_trains(>=60s)": int(raw["delayed"].sum()) if len(raw) else 0,
        "delayed_ratio": float(raw["delayed"].mean()) if len(raw) else 0.0,
        "sum_pri_sec": float(raw["pri_sec"].sum()) if len(raw) else 0.0,
        "sum_sec_sec": float(raw["sec_sec"].sum()) if len(raw) else 0.0,
        "sum_dep_sec": float(raw["dep_delay_sec"].sum()) if len(raw) else 0.0,
        "avg_delay_min_all": float(raw["dep_delay_min"].mean()) if len(raw) else 0.0,
        "avg_delay_min_delayed": float(raw.loc[raw["delayed"] == 1, "dep_delay_min"].mean()) if (len(raw) and raw["delayed"].sum() > 0) else 0.0,
        "p90_delay_min": q(raw["dep_delay_min"], 90),
        "p95_delay_min": q(raw["dep_delay_min"], 95),
        "max_delay_min": float(raw["dep_delay_min"].max()) if len(raw) else 0.0,
    }])

    bins = [-1e-9, 0, 5, 15, 30, np.inf]
    labels = ["0分钟(正点)", "0-5分钟", "5-15分钟", "15-30分钟", "30分钟以上"]
    tmp = raw.copy()
    if len(tmp) > 0:
        tmp["delay_level"] = pd.cut(tmp["dep_delay_min"], bins=bins, labels=labels)
        delay_level = tmp.groupby("delay_level", dropna=False, observed=False).agg(
            trains=("train", "count")
        ).reset_index()
        delay_level["ratio"] = delay_level["trains"] / max(len(raw), 1)
    else:
        delay_level = pd.DataFrame({"delay_level": labels, "trains": 0, "ratio": 0.0})

    if len(raw) > 0:
        by_day = raw.groupby("date").agg(
            trains=("train", "count"),
            delayed_trains=("delayed", "sum"),
            delayed_ratio=("delayed", "mean"),
            sum_pri_sec=("pri_sec", "sum"),
            sum_sec_sec=("sec_sec", "sum"),
            sum_dep_sec=("dep_delay_sec", "sum"),
            avg_delay_min=("dep_delay_min", "mean"),
            p90_delay_min=("dep_delay_min", lambda x: q(x, 90)),
            max_delay_min=("dep_delay_min", "max"),
        ).reset_index().sort_values("date")

        by_hour = raw.groupby("hour").agg(
            trains=("train", "count"),
            delayed_trains=("delayed", "sum"),
            delayed_ratio=("delayed", "mean"),
            avg_delay_min=("dep_delay_min", "mean"),
        ).reset_index().sort_values("hour")
    else:
        by_day = pd.DataFrame(columns=["date", "trains", "delayed_trains", "delayed_ratio", "sum_pri_sec", "sum_sec_sec", "sum_dep_sec", "avg_delay_min", "p90_delay_min", "max_delay_min"])
        by_hour = pd.DataFrame(columns=["hour", "trains", "delayed_trains", "delayed_ratio", "avg_delay_min"])

    # ===== 场景切分 + 规则分类 =====
    delay_only = raw[raw["dep_delay_sec"] >= float(args.min_delay_sec)].copy()
    scen_rows = split_scenarios(
        delay_only,
        gap_min=float(args.gap_min),
        max_span_min=float(args.max_span_min),
        min_events=int(args.min_scenario_events),
    )

    day_total = raw.groupby("date")["train"].count().to_dict()
    scen_feat = extract_features(scen_rows, day_total=day_total)

    rule_cfg = build_rule_cfg(args)
    rule_cfg = adapt_rule_cfg(scen_feat, rule_cfg, enabled=bool(int(args.adaptive_thresholds)))
    scen_feat = attach_scene_labels(scen_feat, rule_cfg)

    # DBSCAN仅做诊断，不参与scene_type判定
    scen_feat, cluster_summary, cluster_scene_mix, eps_used, ms_used = run_dbscan_diag(
        scen_feat=scen_feat,
        min_samples=int(args.min_samples),
        eps=args.eps,
        eps_quantile=float(args.eps_quantile),
        use_dbscan=bool(int(args.use_dbscan)),
    )

    scene_summary = make_scene_summary(scen_feat)
    mix_hint = recommended_scene_mix(scene_summary)

    # ===== 场景区间 =====
    if len(scen_rows) > 0:
        spans = scen_rows.groupby("scenario_id").agg(
            date=("date", "first"),
            start_sec=("event_sec", "min"),
            end_sec=("event_sec", "max"),
            events=("train", "count"),
        ).reset_index()
        if len(scen_feat) > 0:
            spans = spans.merge(
                scen_feat[["scenario_id", "cluster", "scene_type", "scene_reason", "severity_score"]],
                on="scenario_id",
                how="left",
            )
        else:
            spans["cluster"] = np.nan
            spans["scene_type"] = "未知"
            spans["scene_reason"] = ""
            spans["severity_score"] = np.nan
    else:
        spans = pd.DataFrame(columns=["scenario_id", "date", "start_sec", "end_sec", "events", "cluster", "scene_type", "scene_reason", "severity_score"])

    # 列车级标注
    if len(scen_rows) > 0:
        task_scene = raw.merge(
            scen_rows[["date", "train", "scenario_id"]].drop_duplicates(),
            on=["date", "train"],
            how="left",
        )
    else:
        task_scene = raw.copy()
        task_scene["scenario_id"] = np.nan

    if len(scen_feat) > 0:
        task_scene = task_scene.merge(
            scen_feat[["scenario_id", "scene_type", "cluster", "severity_score"]],
            on="scenario_id",
            how="left",
        )
    else:
        task_scene["scene_type"] = np.nan
        task_scene["cluster"] = np.nan
        task_scene["severity_score"] = np.nan

    task_scene["scene_type"] = task_scene["scene_type"].fillna("非晚点/背景")

    params_obj = {
        "scope": args.scope,
        "pack_file": args.pack_file,
        "min_delay_sec": float(args.min_delay_sec),
        "gap_min": float(args.gap_min),
        "max_span_min": float(args.max_span_min),
        "min_scenario_events": int(args.min_scenario_events),
        "use_dbscan": int(args.use_dbscan),
        "dbscan_eps": float(eps_used) if pd.notna(eps_used) else np.nan,
        "dbscan_min_samples": float(ms_used) if pd.notna(ms_used) else np.nan,
        "eps_quantile": float(args.eps_quantile),
        "adaptive_thresholds": int(args.adaptive_thresholds),
        "recommended_scene_mix_10parts": mix_hint,
    }
    params_obj.update(asdict(rule_cfg))
    params = pd.DataFrame([params_obj])

    # ===== 输出 =====
    out_xlsx = os.path.join(args.out_dir, "delay_scene_dbscan_from_pack.xlsx")
    with pd.ExcelWriter(out_xlsx) as w:
        overall.to_excel(w, index=False, sheet_name="overall")
        delay_level.to_excel(w, index=False, sheet_name="delay_level")
        by_day.to_excel(w, index=False, sheet_name="by_day")
        by_hour.to_excel(w, index=False, sheet_name="by_hour")

        scen_rows.to_excel(w, index=False, sheet_name="scenario_events")
        scen_feat.to_excel(w, index=False, sheet_name="scenario_samples")
        cluster_summary.to_excel(w, index=False, sheet_name="cluster_summary")
        cluster_scene_mix.to_excel(w, index=False, sheet_name="cluster_scene_mix")
        scene_summary.to_excel(w, index=False, sheet_name="scene_summary")
        spans.to_excel(w, index=False, sheet_name="scenario_spans")
        task_scene.to_excel(w, index=False, sheet_name="task_scene")
        params.to_excel(w, index=False, sheet_name="params")

    interval_json = os.path.join(args.out_dir, "scene_intervals.json")
    intervals = {}
    for d, g in spans.sort_values(["date", "start_sec"]).groupby("date", sort=True):
        arr = []
        for _, r in g.iterrows():
            arr.append({
                "scenario_id": str(r["scenario_id"]),
                "start_sec": int(float(r["start_sec"])),
                "end_sec": int(float(r["end_sec"])),
                "scene_type": str(r.get("scene_type", "未知")),
            })
        intervals[str(d)] = arr

    with open(interval_json, "w", encoding="utf-8") as f:
        json.dump(intervals, f, ensure_ascii=False, indent=2)

    print(f"[OK] Excel输出: {out_xlsx}")
    print(f"[OK] 区间输出: {interval_json}")
    print("\n=== overall ===")
    print(overall.to_string(index=False))
    print("\n=== scene_summary ===")
    if len(scene_summary):
        print(scene_summary.to_string(index=False))
    else:
        print("无晚点场景")
    print(f"\n[HINT] 推荐 scene-mix(10份): {mix_hint}")


if __name__ == "__main__":
    main()
