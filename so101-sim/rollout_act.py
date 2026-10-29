# -*- coding: utf-8 -*-
"""ACT checkpoint 仿真 rollout：加载 checkpoint，在 MuJoCo 里跑 N 条，统计成功率。

用法:
    wsl -d Debian -e bash -lc "cd /mnt/g/SO-ARM100/so101-sim && \
        ./.venv-wsl/bin/python -u rollout_act.py \
        --ckpt act_expert_out/checkpoints/0002500/pretrained_model --seeds 12"

判据与采集一致：step 返回 terminated 即成功（terminate_on_success=True）。
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import mujoco
import numpy as np
import torch
from lerobot.policies.act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from so101_nexus.envhub import make_env

W, H = 640, 480


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--home", default="hover", choices=["hover", "none"],
                    help="hover: 先用 IK 到物块上方 10cm 再交给策略（与采集分布一致）")
    ap.add_argument("--n-action", type=int, default=0,
                    help="覆盖 n_action_steps（0=用 checkpoint 自带值，50）")
    args = ap.parse_args()

    policy = ACTPolicy.from_pretrained(args.ckpt)
    if args.n_action:
        policy.config.n_action_steps = args.n_action
    policy.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
    print(f"policy loaded: chunk={policy.config.chunk_size} n_action={policy.config.n_action_steps}")

    envs = make_env(n_envs=1, env_id="MuJoCoPickLift-v1")
    env = envs["MuJoCoPickLift-v1"][0]
    u = env.envs[0].unwrapped
    lo, hi = np.asarray(u._target_low, float), np.asarray(u._target_high, float)
    expert = None
    if args.home == "hover":
        sys.path.insert(0, "/mnt/g/SO-ARM100/so101-sim")
        from expert_trial import PickExpert
        expert = PickExpert(env, open_val=float(hi[-1]), closed_val=float(lo[-1]))
    renderer = mujoco.Renderer(u.model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.24, 0.02, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 0.75, 90, -28

    def observe() -> dict:
        renderer.update_scene(u.data, camera=cam)
        img = renderer.render().astype(np.float32) / 255.0
        return {
            "observation.images.front": torch.from_numpy(img).permute(2, 0, 1)[None].cuda(),
            "observation.state": torch.from_numpy(
                np.asarray(u.data.qpos[:6], dtype=np.float32))[None].cuda(),
        }

    ok = 0
    lifts = []
    for seed in range(args.seeds):
        env.reset(seed=seed)
        if expert is not None:
            obj = u._get_target_pose()[:3]
            expert.goto(obj + [0, 0, 0.10], float(hi[-1]), u.data.qpos[:5].copy(), n=25)
        policy.reset()
        done = False
        best = 0.0
        for _ in range(args.max_steps):
            batch = pre(observe())
            with torch.no_grad():
                act = post(policy.select_action(batch))
            a = np.asarray(act.reshape(-1).float().cpu(), dtype=np.float64)
            _, _, term, trunc, info = env.step(a[None, :])
            best = max(best, float(np.asarray(info["lift_height"]).reshape(-1)[0]))
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                done = True
                break
        lifts.append(best)
        ok += int(done)
        print(f"seed {seed}: {'SUCCESS' if done else 'fail'} max_lift={best:.4f}", flush=True)

    print(f"ROLLOUT success {ok}/{args.seeds} = {100.0*ok/args.seeds:.1f}% "
          f"mean_lift={np.mean(lifts):.4f}")


if __name__ == "__main__":
    main()
