# test_maskppo.py
import os
import sys
import pickle
import pandas as pd

from sb3_contrib import MaskablePPO

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CFG
from route_planner import RoutePlanner
from risk_profile import RiskProfile
from env_hz import HZMaskEnv


def main():
    cfg = CFG()
    art_dir = os.path.join(ROOT, "artifacts")
    os.makedirs(art_dir, exist_ok=True)

    with open(os.path.join(art_dir, "data_pack.pkl"), "rb") as f:
        pack = pickle.load(f)

    episodes = pack["episodes"]
    risk_day = pack["risk_day"]
    train_days = sorted(pack["train_days"])
    test_days = sorted(pack["test_days"])

    if len(test_days) == 0:
        raise ValueError("test_days为空，请先运行 data_prepare.py")

    test_day = test_days[0]  # 按你设定：7/3

    tracks = list(getattr(cfg, "assign_tracks", []))
    risk = RiskProfile(cfg, tracks=tracks)
    rec_list = [risk_day[d] for d in train_days if d in risk_day and len(risk_day[d]) > 0]
    if len(rec_list) > 0:
        risk.fit(pd.concat(rec_list, ignore_index=True))
    else:
        risk.fit(pd.DataFrame(columns=["track", "hour", "sec_delay", "fail"]))

    planner = RoutePlanner(cfg)
    env = HZMaskEnv(
        cfg=cfg,
        episodes={test_day: episodes[test_day]},
        risk_day={},  # 防止测试日重拟合风险
        planner=planner,
        risk=risk,
        train_mode=False,
        active_dates=[test_day],
    )

    model = MaskablePPO.load(os.path.join(art_dir, "hz_maskppo_0701_0702"))

    obs, _ = env.reset(options={"date": test_day})
    done = False
    ep_reward = 0.0
    info = {}

    while not done:
        mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, reward, term, trunc, info = env.step(int(action))
        ep_reward += reward
        done = term or trunc

    kpi = info.get("episode_kpi", {})
    kpi["episode_reward"] = float(ep_reward)

    pd.DataFrame([kpi]).to_excel(os.path.join(art_dir, "test_kpi_0703.xlsx"), index=False)
    env.export_day_schedule(os.path.join(art_dir, "test_schedule_0703.xlsx"))

    print("[OK] 测试完成")
    print("[OK] KPI: artifacts/test_kpi_0703.xlsx")
    print("[OK] 排图: artifacts/test_schedule_0703.xlsx")


if __name__ == "__main__":
    main()
