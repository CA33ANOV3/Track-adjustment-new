# data_prepare.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import pickle
import argparse
import numpy as np
import pandas as pd

from config import CFG

ROOT = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(ROOT, "artifacts")


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


def norm_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s


def hms_to_sec(h, m, s):
    h = int(float(h)) if pd.notna(h) else 0
    m = int(float(m)) if pd.notna(h) else 0
    s = int(float(s)) if pd.notna(s) else 0
    return h * 3600 + m * 60 + s


def parse_clock_to_sec(x):
    """把各种时刻格式解析为当天秒数"""
    if pd.isna(x):
        return None

    if isinstance(x, pd.Timestamp):
        return int(x.hour) * 3600 + int(x.minute) * 60 + int(x.second)

    if isinstance(x, (int, np.integer, float, np.floating)):
        v = float(x)
        if np.isnan(v):
            return None

        # Excel 时间小数（0~1）
        if 0 <= v < 1.2:
            return int(round(v * 86400))

        iv = int(round(v))

        # 可能是 HHMMSS 形式（如 92315）
        if 0 <= iv <= 235959:
            s = f"{iv:06d}"
            hh, mm, ss = int(s[:2]), int(s[2:4]), int(s[4:])
            if hh < 24 and mm < 60 and ss < 60:
                return hh * 3600 + mm * 60 + ss

        # 可能本身就是秒
        if 0 <= iv < 172800:
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


def delay_to_min(x):
    """从“晚点xx分”文本里抽取分钟数，支持负号/小数"""
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else 0.0


def align_to_plan(actual_sec, plan_sec):
    """把实际秒对齐到离计划最近的一天（-1/0/+1天）"""
    if actual_sec is None:
        return None
    cands = [actual_sec - 86400, actual_sec, actual_sec + 86400]
    return int(min(cands, key=lambda z: abs(z - plan_sec)))


def md_str(d):
    return f"{d.month:02d}-{d.day:02d}"


def _to_md_set(x):
    if x is None:
        return set()
    if isinstance(x, str):
        x = [i.strip() for i in x.split(",")]
    return set([str(i).strip() for i in x if str(i).strip()])


def resolve_day_split(all_days, cfg):
    """
    split_mode:
      - manual         : 使用 cfg.train_md / cfg.test_md
      - all_minus_test : 训练=全部-测试（推荐）
      - last_n_test    : 最后N天测试
      - ratio          : 随机比例切分
    """
    all_days = sorted([pd.to_datetime(d).date() for d in all_days])
    if len(all_days) < 2:
        raise ValueError(f"有效日期不足2天，无法切分。all_days={all_days}")

    mode = str(getattr(cfg, "split_mode", "all_minus_test")).strip().lower()
    md_train = _to_md_set(getattr(cfg, "train_md", ()))
    md_test = _to_md_set(getattr(cfg, "test_md", ()))

    if mode in {"manual", "md_list"}:
        train_days = [d for d in all_days if md_str(d) in md_train]
        test_days = [d for d in all_days if md_str(d) in md_test]

    elif mode in {"all_minus_test", "auto"}:
        if len(md_test) > 0:
            test_days = [d for d in all_days if md_str(d) in md_test]
            if len(test_days) == 0:
                raise ValueError(
                    f"test_md={sorted(md_test)} 在数据中无匹配日期。"
                    f"可用日期={ [md_str(d) for d in all_days] }"
                )
        else:
            n_test = int(getattr(cfg, "test_days_n", 1))
            n_test = max(1, min(n_test, len(all_days) - 1))
            test_days = all_days[-n_test:]

        test_set = set(test_days)
        train_days = [d for d in all_days if d not in test_set]

    elif mode == "last_n_test":
        n_test = int(getattr(cfg, "test_days_n", 1))
        n_test = max(1, min(n_test, len(all_days) - 1))
        test_days = all_days[-n_test:]
        train_days = all_days[:-n_test]

    elif mode == "ratio":
        ratio = float(getattr(cfg, "train_ratio", 0.8))
        seed = int(getattr(cfg, "split_seed", 42))
        rng = np.random.RandomState(seed)

        idx = np.arange(len(all_days))
        rng.shuffle(idx)

        n_train = int(round(len(all_days) * ratio))
        n_train = max(1, min(n_train, len(all_days) - 1))
        train_idx = set(idx[:n_train].tolist())

        train_days = sorted([all_days[i] for i in range(len(all_days)) if i in train_idx])
        test_days = sorted([all_days[i] for i in range(len(all_days)) if i not in train_idx])

    else:
        raise ValueError(f"未知 split_mode={mode}")

    overlap = set(train_days) & set(test_days)
    if overlap:
        raise ValueError(f"切分错误：train/test 有重叠 {sorted(overlap)}")
    if len(train_days) == 0:
        raise ValueError("训练集为空，请检查 split_mode / train_md / test_md")
    if len(test_days) == 0:
        raise ValueError("测试集为空，请检查 split_mode / test_md")

    return sorted(train_days), sorted(test_days), all_days


def apply_cli_overrides(cfg, args):
    if args is None:
        return cfg

    if args.split_mode is not None:
        setattr(cfg, "split_mode", args.split_mode)

    if args.train_md is not None:
        md = tuple([x.strip() for x in args.train_md.split(",") if x.strip()])
        setattr(cfg, "train_md", md)

    if args.test_md is not None:
        md = tuple([x.strip() for x in args.test_md.split(",") if x.strip()])
        setattr(cfg, "test_md", md)

    if args.test_days_n is not None:
        setattr(cfg, "test_days_n", int(args.test_days_n))

    if args.train_ratio is not None:
        setattr(cfg, "train_ratio", float(args.train_ratio))

    if args.split_seed is not None:
        setattr(cfg, "split_seed", int(args.split_seed))

    return cfg


def build_data(args=None):
    cfg = CFG()
    cfg = apply_cli_overrides(cfg, args)

    os.makedirs(ART_DIR, exist_ok=True)

    # ===== 1) 基本图 =====
    base = read_excel_any(cfg.base_file).copy()

    c_train = find_col(base, ["列车", "车次", "客票车次"])
    c_type = find_col(base, ["列车类型"])
    c_track = find_col(base, ["接入股道", "股道"])
    c_in = find_col(base, ["进站线路"])
    c_out = find_col(base, ["出站线路"])

    c_water = find_col(base, ["是否上水作业", "上水作业"], required=False)
    c_sewage = find_col(base, ["是否吸污作业", "吸污作业"], required=False)

    # 到发时刻（优先时分秒列）
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
        "列车": base[c_train].map(norm_id),
        "列车类型": base[c_type].astype(str).str.strip(),
        "接入股道": base[c_track].map(norm_id),
        "进站线路": base[c_in].map(norm_id),
        "出站线路": base[c_out].map(norm_id),
        "是否上水作业": base[c_water].values if c_water is not None else 0,
        "是否吸污作业": base[c_sewage].values if c_sewage is not None else 0,
        "plan_arr_sec": pd.to_numeric(arr_sec, errors="coerce").fillna(0).astype(int),
        "plan_dep_sec": pd.to_numeric(dep_sec, errors="coerce").fillna(0).astype(int),
    })

    base_std = base_std[base_std["列车类型"].isin(cfg.keep_train_types)].copy()
    if len(base_std) == 0:
        raise ValueError("基本图过滤后为空，请检查 keep_train_types / 列名匹配")

    base_train_set = set(base_std["列车"].unique())

    # ===== 2) 延误日志 =====
    delay = read_excel_any(cfg.delay_file).copy()

    c_tid = find_col(delay, ["客票车次", "车次", "列车"])
    c_date = find_col(delay, ["始发日期", "日期", "运行日", "运行日期"])
    c_station = find_col(delay, ["行车站", "站名", "车站"], required=False)

    c_arr_t = find_col(delay, ["到达时刻", "到达时间"], required=False)
    c_dep_t = find_col(delay, ["出发时刻", "出发时间"], required=False)
    c_arr_late = find_col(delay, ["到达晚点"], required=False)
    c_dep_late = find_col(delay, ["出发晚点"], required=False)

    delay["客票车次"] = delay[c_tid].map(norm_id)
    delay["始发日期"] = pd.to_datetime(delay[c_date], errors="coerce").dt.date
    delay = delay[delay["始发日期"].notna()].copy()

    if c_station is not None:
        delay = delay[delay[c_station].astype(str).str.contains(cfg.station_name, na=False)].copy()

    before = len(delay)
    delay = delay[delay["客票车次"].isin(base_train_set)].copy()
    print(f"[INFO] 已删除基本图不存在车次的日志行数: {before - len(delay)}")

    # ===== 3) 每日日程 =====
    episodes = {}
    risk_day = {}

    all_days = sorted(delay["始发日期"].unique())
    if len(all_days) == 0:
        raise ValueError("延误日志无有效日期或无匹配车次")

    for d in all_days:
        daylog = delay[delay["始发日期"] == d].copy()
        day_group = {k: v.copy() for k, v in daylog.groupby("客票车次")}
        rows = []

        for _, p in base_std.iterrows():
            tid = p["列车"]
            plan_arr = int(p["plan_arr_sec"])
            plan_dep = int(p["plan_dep_sec"])

            sub = day_group.get(tid, None)

            best_arr, best_dep = plan_arr, plan_dep
            best_score = 10**18

            if sub is not None and len(sub) > 0:
                for _, r in sub.iterrows():
                    arr = parse_clock_to_sec(r[c_arr_t]) if c_arr_t is not None else None
                    dep = parse_clock_to_sec(r[c_dep_t]) if c_dep_t is not None else None

                    if arr is None and c_arr_late is not None:
                        arr = plan_arr + int(round(60 * delay_to_min(r[c_arr_late])))
                    if dep is None and c_dep_late is not None:
                        dep = plan_dep + int(round(60 * delay_to_min(r[c_dep_late])))

                    if arr is None:
                        arr = plan_arr
                    if dep is None:
                        dep = plan_dep

                    arr = align_to_plan(arr, plan_arr)
                    dep = align_to_plan(dep, plan_dep)

                    score = abs(arr - plan_arr) + abs(dep - plan_dep)
                    if score < best_score:
                        best_score = score
                        best_arr, best_dep = arr, dep

            pri = max(0, best_arr - plan_arr)
            dep_delay = max(0, best_dep - plan_dep)
            sec = max(0, dep_delay - pri)

            rr = p.to_dict()
            rr.update({
                "date": d,
                "obs_arr_sec": int(best_arr),          # 新增：观测到达秒
                "obs_dep_sec": int(best_dep),          # 新增：观测发车秒
                "init_delay_sec": int(pri),            # 注入一次晚点
                "hist_sec_delay_sec": int(sec),        # 历史二次晚点
                "obs_dep_delay_sec": int(dep_delay),   # 观测发车晚点
            })
            rows.append(rr)

        day_df = pd.DataFrame(rows)
        day_df["arr0_sec"] = day_df["plan_arr_sec"] + day_df["init_delay_sec"]
        day_df = day_df.sort_values(["arr0_sec", "plan_dep_sec"], kind="stable").reset_index(drop=True)

        episodes[d] = day_df

        # 风险画像训练表：默认用二次延误
        hour_src = pd.to_numeric(day_df.get("obs_arr_sec", day_df["arr0_sec"]), errors="coerce").fillna(day_df["arr0_sec"]).astype(int)
        risk_day[d] = pd.DataFrame({
            "track": day_df["接入股道"].map(norm_id),
            "hour": (hour_src // 3600).astype(int) % 24,
            "sec_delay": day_df["hist_sec_delay_sec"].astype(float),
            "fail": 0
        })

    # ===== 4) 切分 =====
    train_days, test_days, all_days_sorted = resolve_day_split(episodes.keys(), cfg)

    out = {
        "episodes": episodes,
        "risk_day": risk_day,
        "train_days": sorted(train_days),
        "test_days": sorted(test_days),
        "all_days": sorted(all_days_sorted),
    }

    # 输出路径
    out_path = None
    if args is not None and getattr(args, "out_file", None):
        out_path = args.out_file
        if not os.path.isabs(out_path):
            out_path = os.path.join(ROOT, out_path)
    else:
        out_path = os.path.join(ART_DIR, "data_pack.pkl")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    # 日志打印
    print(f"[INFO] split_mode = {getattr(cfg, 'split_mode', 'all_minus_test')}")
    print("[INFO] all_days   =", [md_str(d) for d in all_days_sorted])
    print("[INFO] train_days =", [md_str(d) for d in train_days])
    print("[INFO] test_days  =", [md_str(d) for d in test_days])

    for d in sorted(episodes.keys()):
        day_df = episodes[d]
        dr = float((day_df["obs_dep_delay_sec"] >= 60).mean()) if len(day_df) > 0 else 0.0
        print(f"[INFO] {d}: {len(day_df)} trains, delayed_ratio(>=60s)={dr:.3f}")

    print(f"[OK] 数据已保存: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-mode", type=str, default=None,
                   help="manual / all_minus_test / last_n_test / ratio")
    p.add_argument("--train-md", type=str, default=None,
                   help="manual模式下训练日，逗号分隔，如 07-01,07-02")
    p.add_argument("--test-md", type=str, default=None,
                   help="测试日，逗号分隔，如 07-22,07-23,07-24")
    p.add_argument("--test-days-n", type=int, default=None,
                   help="last_n_test或all_minus_test(未给test_md)时生效")
    p.add_argument("--train-ratio", type=float, default=None,
                   help="ratio模式下训练比例")
    p.add_argument("--split-seed", type=int, default=None,
                   help="ratio模式随机种子")
    p.add_argument("--out-file", type=str, default=None,
                   help="输出pkl路径，默认 artifacts/data_pack.pkl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_data(args)
