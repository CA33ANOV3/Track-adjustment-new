# risk_profile.py
from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd


def norm(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s


def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cols = [str(c).strip() for c in df.columns]
    for n in names:
        for i, c in enumerate(cols):
            if c == n or (n in c):
                return df.columns[i]
    return None


def _cvar(arr: np.ndarray, alpha: float = 0.9) -> float:
    if arr is None or len(arr) == 0:
        return 0.0
    q = np.quantile(arr, alpha)
    tail = arr[arr >= q]
    return float(tail.mean()) if len(tail) > 0 else float(arr.mean())


def _minmax(x: pd.Series) -> pd.Series:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - mn) / (mx - mn)


class RiskProfile:
    def __init__(self, cfg, tracks: List[str]):
        self.cfg = cfg
        self.tracks = [norm(t) for t in tracks if norm(t)]
        self.hours = list(range(24))

        self.risk_threshold_sec = float(getattr(cfg, "risk_threshold_sec", 300))
        self.cvar_alpha = float(getattr(cfg, "cvar_alpha", 0.9))
        self.hotspot_quantile = float(getattr(cfg, "hotspot_quantile", 0.8))
        self.eb_kappa = float(getattr(cfg, "eb_kappa", 20.0))

        self.w1 = float(getattr(cfg, "risk_w1", 0.25))
        self.w2 = float(getattr(cfg, "risk_w2", 0.35))
        self.w3 = float(getattr(cfg, "risk_w3", 0.25))
        self.w4 = float(getattr(cfg, "risk_w4", 0.15))

        self.phi_tbl = pd.DataFrame(0.0, index=self.tracks, columns=self.hours)
        self.hot_tbl = pd.DataFrame(0, index=self.tracks, columns=self.hours)

    def reset(self):
        self.phi_tbl.loc[:, :] = 0.0
        self.hot_tbl.loc[:, :] = 0

    def fit(self, rec: pd.DataFrame):
        if rec is None or len(rec) == 0:
            self.reset()
            return

        df = rec.copy()
        c_track = _find_col(df, ["track", "股道", "接入股道"])
        c_hour = _find_col(df, ["hour", "时段", "小时"])
        c_sec = _find_col(df, ["sec_delay", "二次延误", "hist_sec_delay_sec"])
        c_fail = _find_col(df, ["fail", "失败", "is_fail"])

        if c_track is None or c_hour is None or c_sec is None:
            self.reset()
            return

        cols = [c_track, c_hour, c_sec] + ([c_fail] if c_fail is not None else [])
        df = df[cols].copy()
        df.columns = ["track", "hour", "sec_delay"] + (["fail"] if c_fail is not None else [])

        df["track"] = df["track"].map(norm)
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int) % 24
        df["sec_delay"] = pd.to_numeric(df["sec_delay"], errors="coerce").fillna(0).astype(float)
        if "fail" not in df.columns:
            df["fail"] = 0
        df["fail"] = pd.to_numeric(df["fail"], errors="coerce").fillna(0).clip(0, 1).astype(int)

        gp = float((df["sec_delay"] > self.risk_threshold_sec).mean())
        gmu = float(df["sec_delay"].mean())
        gc = _cvar(df["sec_delay"].to_numpy(), self.cvar_alpha)
        gf = float(df["fail"].mean())

        rows = []
        for (g, h), grp in df.groupby(["track", "hour"]):
            n = len(grp)
            if n == 0:
                continue

            p = float((grp["sec_delay"] > self.risk_threshold_sec).mean())
            mu = float(grp["sec_delay"].mean())
            cv = _cvar(grp["sec_delay"].to_numpy(), self.cvar_alpha)
            f = float(grp["fail"].mean())

            k = self.eb_kappa
            p_hat = (n * p + k * gp) / (n + k)
            mu_hat = (n * mu + k * gmu) / (n + k)
            c_hat = (n * cv + k * gc) / (n + k)
            f_hat = (n * f + k * gf) / (n + k)

            rows.append([norm(g), int(h), p_hat, mu_hat, c_hat, f_hat])

        stat = pd.DataFrame(rows, columns=["track", "hour", "p", "mu", "c", "f"])
        if len(stat) == 0:
            self.reset()
            return

        stat["pn"] = _minmax(stat["p"])
        stat["mun"] = _minmax(stat["mu"])
        stat["cn"] = _minmax(stat["c"])
        stat["fn"] = _minmax(stat["f"])

        stat["phi"] = (
            self.w1 * stat["pn"]
            + self.w2 * stat["mun"]
            + self.w3 * stat["cn"]
            + self.w4 * stat["fn"]
        )

        q = float(stat["phi"].quantile(self.hotspot_quantile))
        stat["hot"] = (stat["phi"] >= q).astype(int)

        self.reset()
        for _, r in stat.iterrows():
            g = norm(r["track"])
            h = int(r["hour"]) % 24
            if g in self.phi_tbl.index:
                self.phi_tbl.loc[g, h] = float(r["phi"])
                self.hot_tbl.loc[g, h] = int(r["hot"])

    def phi(self, track: str, hour: int) -> float:
        g = norm(track)
        h = int(hour) % 24
        if g not in self.phi_tbl.index:
            return 0.0
        return float(self.phi_tbl.loc[
    def hot(self, track: str, hour: int) -> int:
        g = norm(track)
        h = int(hour) % 24
        if g not in self.hot_tbl.index:
            return 0
        return int(self.hot_tbl.loc[g, h])
