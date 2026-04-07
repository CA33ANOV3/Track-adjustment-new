# train_maskppo.py
import os
import sys
import pickle
import random
import numpy as np
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
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    art_dir = os.path.join(ROOT, "artifacts")
    os.makedirs(art_dir, exist_ok=True)

    with open(os.path.join(art_dir, "data_pack.pkl"), "rb") as f:
        pack = pickle.load(f)

    episodes = pack["episodes"]
    risk_day = pack["risk_day"]
    train_days = sorted(pack["train_days"])

    train_episodes = {d: episodes[d] for d in train_days if d in episodes}
    train_risk_day = {d: risk_day[d] for d in train_days if d in risk_day}

    if len(train_episodes) == 0:
        raise ValueError("训练集为空，请先运行 data_prepare.py")

    tracks = list(getattr(cfg, "assign_tracks", []))
    risk = RiskProfile(cfg, tracks=tracks)

    rec_list = [train_risk_day[d] for d in train_days if d in train_risk_day and len(train_risk_day[d]) > 0]
    if len(rec_list) > 0:
        risk_rec = pd.concat(rec_list, ignore_index=True)
    else:
        risk_rec = pd.DataFrame(columns=["track", "hour", "sec_delay", "fail"])
    risk.fit(risk_rec)

    planner = RoutePlanner(cfg)
    env = HZMaskEnv(
        cfg=cfg,
        episodes=train_episodes,
        risk_day=train_risk_day,
        planner=planner,
        risk=risk,
        train_mode=True,
        active_dates=train_days,
    )

    # 方案A关键：直接依赖 env.action_masks()
    obs, _ = env.reset()
    m = env.action_masks()
    if m is None or len(m) != env.action_space.n:
        raise RuntimeError("env.action_masks() 异常，请检查 env_hz.py")

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        seed=cfg.seed,
        device="auto",
        tensorboard_log=os.path.join(art_dir, "tb"),
    )

    model.learn(total_timesteps=int(cfg.total_timesteps))

    save_path = os.path.join(art_dir, "hz_maskppo_0701_0702")
    model.save(save_path)
    print(f"[OK] 训练完成，模型已保存: {save_path}.zip")


if __name__ == "__main__":
    main()
