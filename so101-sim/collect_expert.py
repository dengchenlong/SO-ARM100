"""采集 SO-101 PickLift 专家示范（只落成功 episode）。

用法:
    wsl -d Debian -e bash -lc "cd /mnt/g/SO-ARM100/so101-sim && \
        ./.venv-wsl/bin/python collect_expert.py --target 25 --max-attempts 80"

产出:
    /mnt/g/SO-ARM100/so101-sim/datasets_expert    LeRobotDataset (fps=30)
        6 维关节 observation.state / 6 维关节 action / observation.images.front 480x640

要点（与 expert_trial.py 同源）:
    - 成功 = step 返回 terminated（terminate_on_success=True，成功后 info 已被重置）。
    - 采集只占 CPU，不占显存。
    - 失败的 attempt 整条丢弃，不落盘。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from so101_nexus.envhub import make_env

sys.path.insert(0, str(Path(__file__).parent))
from expert_trial import PickExpert  # noqa: E402

ROOT = Path("/mnt/g/SO-ARM100/so101-sim/datasets_expert")
W, H, FPS = 640, 480, 30
JOINT_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "observation.images.front": {"dtype": "video", "shape": (H, W, 3),
                                 "names": ["height", "width", "channels"]},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=25)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT))
    ap.add_argument("--repo-id", default="local/sim_picklift_expert")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    envs = make_env(n_envs=1, env_id="MuJoCoPickLift-v1")
    env = envs["MuJoCoPickLift-v1"][0]
    u = env.envs[0].unwrapped
    lo, hi = np.asarray(u._target_low, float), np.asarray(u._target_high, float)
    expert = PickExpert(env, open_val=float(hi[-1]), closed_val=float(lo[-1]))

    # --- 阶段 1：不渲染，快速筛出能成功的 seed ---
    good: list[int] = []
    t0 = time.time()
    for seed in range(args.max_attempts):
        log = expert.run(seed=seed)
        if log["success"]:
            good.append(seed)
            print(f"[dry] seed={seed} OK lift={log['max_lift']} ({len(good)}/{args.target})",
                  flush=True)
            if len(good) >= args.target:
                break
        else:
            print(f"[dry] seed={seed} fail ({log['grasp']})", flush=True)
    good = good[: args.target]
    print(f"PHASE1 done: {len(good)} good seeds in {time.time()-t0:.0f}s", flush=True)
    if not good:
        print("no successful seeds; abort")
        return

    # --- 阶段 2：对成功 seed 重放并录像落盘 ---
    renderer = mujoco.Renderer(u.model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.24, 0.02, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 0.75, 90, -28

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=FEATURES,
                               root=args.out, use_videos=True, robot_type="so101")
    print(f"dataset -> {args.out}", flush=True)

    orig_step = expert.step
    buf = {"frames": [], "states": [], "actions": []}

    def recording_step(q5, gripper):
        term, trunc, info = orig_step(q5, gripper)
        if term or trunc:
            return term, trunc, info
        renderer.update_scene(u.data, camera=cam)
        buf["frames"].append(renderer.render().copy())
        buf["states"].append(np.asarray(u.data.qpos[:6], dtype=np.float32).copy())
        buf["actions"].append(np.concatenate([np.asarray(q5, float), [gripper]]).astype(np.float32))
        return term, trunc, info

    expert.step = recording_step  # type: ignore[method-assign]

    done = 0
    for seed in good:
        buf["frames"].clear(); buf["states"].clear(); buf["actions"].clear()
        log = expert.run(seed=seed)
        if not log["success"]:
            print(f"[rec] seed={seed} failed on replay; skip", flush=True)
            continue
        n = len(buf["frames"])
        for i in range(n):
            ds.add_frame({
                "observation.state": buf["states"][i],
                "action": buf["actions"][i],
                "observation.images.front": buf["frames"][i],
                "task": "Pick up the red cube and lift it.",
            })
        ds.save_episode()
        done += 1
        print(f"[rec] seed={seed} ep={done}/{len(good)} frames={n} lift={log['max_lift']}",
              flush=True)

    ds.finalize()
    print(f"DONE episodes={done} elapsed={time.time()-t0:.0f}s")
    print("VERIFY", ds.num_episodes, ds.num_frames)


if __name__ == "__main__":
    main()
