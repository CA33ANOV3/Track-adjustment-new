# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import networkx as nx

__all__ = ["Candidate", "RoutePlanner"]


def read_excel_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="xlrd")
    except Exception:
        return pd.read_excel(path)


def norm(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s


def find_col(df: pd.DataFrame, keys: List[str], required=True):
    cols = [str(c).strip() for c in df.columns]
    for k in keys:
        for i, c in enumerate(cols):
            if c == k or (k in c):
                return df.columns[i]
    if required:
        raise KeyError(f"未找到列: {keys}, 现有列: {list(df.columns)}")
    return None


def line_dir(line_text: str) -> str:
    s = norm(line_text)
    if "上行" in s:
        return "up"
    if "下行" in s:
        return "down"
    return "unk"


@dataclass
class Candidate:
    scheme: str
    track: str
    stage1_end: str
    reverse_cnt: int


class RoutePlanner:
    def __init__(self, cfg):
        self.cfg = cfg

        self.graphs: Dict[str, nx.DiGraph] = {
            "UP_FWD": self._load_graph(cfg.up_fwd_file),
            "DOWN_FWD": self._load_graph(cfg.down_fwd_file),
            "UP_REV": self._load_graph(cfg.up_rev_file),
            "DOWN_REV": self._load_graph(cfg.down_rev_file),
        }
        self.cut: Dict[int, Tuple[str, str]] = self._load_cut(cfg.cut_file)

    def _load_graph(self, path: str) -> nx.DiGraph:
        df = read_excel_any(path)
        c_node = find_col(df, ["节点"])
        c_up = find_col(df, ["上游"])
        c_down = find_col(df, ["下游"])

        G = nx.DiGraph()
        for _, r in df.iterrows():
            node = norm(r[c_node])
            up = norm(r[c_up])
            down = norm(r[c_down])

            for n in [node, up, down]:
                if n:
                    G.add_node(n)

            if up and node:
                G.add_edge(up, node)
            if node and down:
                G.add_edge(node, down)
            if up and down:
                G.add_edge(up, down)

        return G

    def _load_cut(self, path: str) -> Dict[int, Tuple[str, str]]:
        df = read_excel_any(path)
        c_id = find_col(df, ["ID"])
        c_up = find_col(df, ["上游"])
        c_down = find_col(df, ["下游"])

        mp = {}
        for _, r in df.iterrows():
            try:
                cid = int(float(r[c_id]))
            except Exception:
                continue
            mp[cid] = (norm(r[c_up]), norm(r[c_down]))
        return mp

    def _has_path(self, gname: str, s: str, t: str) -> bool:
        G = self.graphs[gname]
        s, t = norm(s), norm(t)
        if not s or not t:
            return False
        if s not in G.nodes or t not in G.nodes:
            return False
        try:
            nx.shortest_path_length(G, s, t)
            return True
        except Exception:
            return False

    def _g_nodes(self, gname: str) -> List[str]:
        G = self.graphs[gname]
        return sorted([n for n in G.nodes if "G" in str(n)])

    def _dedupe(self, cands: List[Candidate]) -> List[Candidate]:
        seen = set()
        out = []
        for c in cands:
            key = (c.scheme, c.track, c.stage1_end, c.reverse_cnt)
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out

    def _direct_candidates(self, in_node: str, out_node: str, graph_name: str) -> List[Candidate]:
        cands = []
        for g in self._g_nodes(graph_name):
            if self._has_path(graph_name, in_node, g) and self._has_path(graph_name, g, out_node):
                cands.append(Candidate(scheme=f"直通-{graph_name}", track=g, stage1_end=g, reverse_cnt=0))
        return cands

    def _turnback_up(self, in_node: str, out_node: str) -> List[Candidate]:
        cands = []

        u3, v3 = self.cut.get(3, ("202", "204"))
        if self._has_path("UP_FWD", in_node, u3):
            for g in self._g_nodes("DOWN_REV"):
                if self._has_path("DOWN_REV", v3, g) and self._has_path("DOWN_FWD", g, out_node):
                    cands.append(Candidate("立折上行-方案1", g, g, reverse_cnt=1))

        u4, v4 = self.cut.get(4, ("210", "208"))
        for g in self._g_nodes("UP_REV"):
            if (
                self._has_path("UP_FWD", in_node, g)
                and self._has_path("UP_REV", g, u4)
                and self._has_path("DOWN_REV", v4, out_node)
            ):
                cands.append(Candidate("立折上行-方案2", g, g, reverse_cnt=0))

        return cands

    def _turnback_down(self, in_node: str, out_node: str) -> List[Candidate]:
        cands = []

        u1, v1 = self.cut.get(1, ("201", "203"))
        if self._has_path("DOWN_FWD", in_node, u1):
            for g in self._g_nodes("UP_REV"):
                if self._has_path("UP_REV", v1, g) and self._has_path("UP_FWD", g, out_node):
                    cands.append(Candidate("立折下行-方案1", g, g, reverse_cnt=1))

        u2, v2 = self.cut.get(2, ("215", "213"))
        for g in self._g_nodes("DOWN_REV"):
            if (
                self._has_path("DOWN_FWD", in_node, g)
                and self._has_path("DOWN_REV", g, u2)
                and self._has_path("UP_REV", v2, out_node)
            ):
                cands.append(Candidate("立折下行-方案2", g, g, reverse_cnt=0))

        return cands

    def build_candidates(self, task: dict) -> List[Candidate]:
        ttype = norm(task.get("列车类型", ""))
        in_node = norm(task.get("进站线路", ""))
        out_node = norm(task.get("出站线路", ""))
        in_dir = line_dir(in_node)

        enable_through_down_turnback = bool(getattr(self.cfg, "enable_through_down_turnback", True))
        cands: List[Candidate] = []

        if ttype == "立折":
            if in_dir == "up":
                cands = self._turnback_up(in_node, out_node)
            elif in_dir == "down":
                cands = self._turnback_down(in_node, out_node)
            else:
                cands = self._turnback_up(in_node, out_node) + self._turnback_down(in_node, out_node)

        elif ttype == "过路":
            if (in_dir == "down") and enable_through_down_turnback:
                cands = self._turnback_down(in_node, out_node)
            else:
                if ("上行" in in_node) or ("上行" in out_node):
                    cands = self._direct_candidates(in_node, out_node, "UP_FWD")
                elif ("下行" in in_node) or ("下行" in out_node):
                    cands = self._direct_candidates(in_node, out_node, "DOWN_FWD")
                else:
                    cands = (
                        self._direct_candidates(in_node, out_node, "UP_FWD")
                        + self._direct_candidates(in_node, out_node, "DOWN_FWD")
                    )

        return self._dedupe(cands)
