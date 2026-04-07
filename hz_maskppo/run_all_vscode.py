# -*- coding: utf-8 -*-
"""
run_all_vscode.py
一键串行运行：
1) scene_01_build_library.py
2) scene_02_generate_plan.py
3) scene_03_inject_plan.py

在 VSCode 直接点击 Run / F5 即可。
"""

import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ========= 你只需要改这里 =========
CFG = {
    "base_file": r"G:\hjh\gudao\沪杭长场基本图.xls",
    "n_days": 60,
    "start_date": "2024-08-01",

    # scene_02 参数（当 use_exp_profile_arg=False 时生效）
    "scene_mix": "单列车短时晚点:4,混合型晚点场景:4,大面积晚点干扰:2",
    "intensity_low": 0.95,
    "intensity_high": 1.35,
    "quota_jitter": 0.20,
    "delay_noise_std": 0.10,
    "max_time_shift_min": 6,
    "overlap_rate": 0.35,

    # 可选：如果你的 scene_02 已支持 --exp-profile，则可设 True
    "use_exp_profile_arg": False,
    "exp_profile": "none",  # none / base / exp1 / exp2 / exp3

    "seed": 42,
    "combine_mode": "sum",  # sum 更强；max 更保守
}

RUN_SWITCH = {
    "scene_01": True,
    "scene_02": True,
    "scene_03": True,
}
# =================================

ART = ROOT / "artifacts"
LIB_DIR = ART / "scene_lib"

RAW_LIBRARY_JSON = LIB_DIR / "scene_library_raw.json"
NORM_LIBRARY_JSON = LIB_DIR / "scene_library.json"
LIB_SUMMARY_XLSX = LIB_DIR / "scene_library_summary.xlsx"

PLAN_JSON = LIB_DIR / "injection_plan_moderate.json"

OUT_PACK = ART / "injected_pack.pkl"
OUT_MANIFEST = ART / "injected_manifest.xlsx"


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def run_module(module_name, args):
    print(f"\n{'=' * 72}")
    print(f"[RUN] {module_name}")
    print(f"[ARGS] {args}")

    t0 = time.time()

    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)  # 确保拿到最新代码

    if not hasattr(mod, "main"):
        raise AttributeError(f"{module_name}.py 未找到 main()")

    old_argv = sys.argv[:]
    sys.argv = [f"{module_name}.py"] + args
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    print(f"[DONE] {module_name} 用时 {time.time() - t0:.2f}s")


def check_exists(path, name):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} 未生成: {p}")
    print(f"[OK] {name}: {rel(p)}")


def build_scene02_args():
    args = [
        "--library-json", str(NORM_LIBRARY_JSON),
        "--out-plan", str(PLAN_JSON),
        "--profile", "moderate_aug",
        "--n-days", str(CFG["n_days"]),
        "--start-date", str(CFG["start_date"]),
        "--quota-jitter", str(CFG["quota_jitter"]),
        "--delay-noise-std", str(CFG["delay_noise_std"]),
        "--max-time-shift-min", str(CFG["max_time_shift_min"]),
        "--seed", str(CFG["seed"]),
    ]

    use_exp_profile = bool(CFG.get("use_exp_profile_arg", False))
    exp_profile = str(CFG.get("exp_profile", "none")).strip().lower()

    if use_exp_profile and exp_profile != "none":
        args += ["--exp-profile", exp_profile]
    else:
        args += [
            "--scene-mix", str(CFG["scene_mix"]),
            "--intensity-low", str(CFG["intensity_low"]),
            "--intensity-high", str(CFG["intensity_high"]),
            "--overlap-rate", str(CFG["overlap_rate"]),
        ]

    return args


def precheck():
    if not Path(CFG["base_file"]).exists():
        raise FileNotFoundError(f"base_file 不存在: {CFG['base_file']}")
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)


def main():
    print("[INFO] run_all_vscode 启动（无需命令行）")
    print(f"[INFO] ROOT: {ROOT}")
    precheck()

    if RUN_SWITCH.get("scene_01", True):
        args01 = [
            "--input-json", str(RAW_LIBRARY_JSON),
            # 关键修复：fallback 也指向 raw，避免把旧 norm 当回退源
            "--fallback-input-json", str(RAW_LIBRARY_JSON),
            "--out-json", str(NORM_LIBRARY_JSON),
            "--out-summary", str(LIB_SUMMARY_XLSX),
            "--dedup", "true",
        ]
        run_module("scene_01_build_library", args01)
        check_exists(NORM_LIBRARY_JSON, "标准化场景库JSON")
        check_exists(LIB_SUMMARY_XLSX, "场景库汇总")

    if RUN_SWITCH.get("scene_02", True):
        args02 = build_scene02_args()
        run_module("scene_02_generate_plan", args02)
        check_exists(PLAN_JSON, "注入计划JSON")

    if RUN_SWITCH.get("scene_03", True):
        args03 = [
            "--plan-json", str(PLAN_JSON),
            "--base-file", str(CFG["base_file"]),
            "--out-pack", str(OUT_PACK),
            "--out-manifest", str(OUT_MANIFEST),
            "--combine-mode", str(CFG["combine_mode"]),
            "--seed", str(CFG["seed"]),
        ]
        run_module("scene_03_inject_plan", args03)
        check_exists(OUT_PACK, "注入包PKL")
        check_exists(OUT_MANIFEST, "注入清单XLSX")

    print(f"\n{'=' * 72}")
    print("[ALL DONE] 全流程执行完成")
    print(f"- 场景库: {rel(NORM_LIBRARY_JSON)}")
    print(f"- 计划:   {rel(PLAN_JSON)}")
    print(f"- 注入包: {rel(OUT_PACK)}")
    print(f"- 清单:   {rel(OUT_MANIFEST)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FAILED] 流程执行失败：", e)
        traceback.print_exc()
        raise
