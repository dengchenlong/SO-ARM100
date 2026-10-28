"""SO-101 PickLift 专家策略单条试跑（关节空间 + 自写阻尼最小二乘 IK）。

用法:
    wsl -d Debian -e bash -lc "cd /mnt/g/SO-ARM100/so101-sim && \
        ./.venv-wsl/bin/python expert_trial.py --seed 0"

产出:
    refs/expert_trial_seed0.png   关键帧拼图（悬停 / 下探 / 夹紧 / 抬升结束）
    控制台打印每阶段 tcp / obj / is_grasped / lift_height

注意（2026-10-27 实测结论）:
    - 环境自带 pd_ee_pose IK 只有 3 次迭代且 ee_orientation_weight=0.01，
      末端到不了桌面高度；本脚本改用 pd_joint_pos + 自写 IK。
    - 夹爪语义: ctrlrange [-0.17453, 1.74533]，低值=闭合、高值=张开。
    - TCP 比指尖高约 1.4cm，下探目标取 obj_z + 0.004。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from so101_nexus.envhub import make_env
from so101_nexus.kinematics import ee_ik_delta_q, rotvec_to_quat

Q_DOWN = rotvec_to_quat(np.array([np.pi, 0.0, 0.0]))
GRASP_OFFSETS = ((0.0, 0.0, 0.004), (0.004, 0.0, 0.002), (-0.004, 0.0, 0.002),
                 (0.0, 0.004, 0.002), (0.0, -0.004, 0.002))


class PickExpert:
    """关节空间专家：悬停 → 下探 → 慢速闭合 → 抬升。"""

    def __init__(self, env, open_val: float, closed_val: float):
        self.env = env
        self.u = env.envs[0].unwrapped if hasattr(env, "envs") else env.unwrapped
        self.m, self.d = self.u.model, self.u.data
        self.sid = self.u._tcp_site_id
        self.lo = np.asarray(self.u._target_low, float)
        self.hi = np.asarray(self.u._target_high, float)
        self.open = open_val
        self.closed = closed_val

    # --- 基础工具 -----------------------------------------------------
    def step(self, q5, gripper: float):
        a = np.concatenate([np.asarray(q5, float), [gripper]])
        _, _, term, trunc, info = self.env.step(a[None, :])
        return (bool(np.asarray(term).reshape(-1)[0]),
                bool(np.asarray(trunc).reshape(-1)[0]),
                {k: float(np.asarray(info[k]).reshape(-1)[0])
                 for k in ("is_grasped", "lift_height", "tcp_to_obj_dist", "success")})

    def solve_ik(self, target, seed, iters: int = 250, ow: float = 0.8) -> np.ndarray:
        q = np.asarray(seed, float)[:5].copy()
        for _ in range(iters):
            self.d.qpos[:5] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            jacp = np.zeros((3, self.m.nv))
            jacr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.sid)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, self.d.site_xmat[self.sid])
            dofs = np.arange(5)
            jac = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            dq = ee_ik_delta_q(jac, self.d.site_xpos[self.sid], quat,
                               np.asarray(target, float), Q_DOWN, orientation_weight=ow)
            q = np.clip(q + dq, self.lo[:5], self.hi[:5])
            if np.linalg.norm(dq) < 1e-5:
                break
        return q

    def goto(self, target, gripper: float, seed, n: int = 20):
        q = self.solve_ik(target, seed)
        start = self.d.qpos[:5].copy()
        term, trunc, info = self.step(start, gripper)
        for i in range(1, n + 1):
            term, trunc, info = self.step(start + (q - start) * i / n, gripper)
            if term or trunc:
                break
        return q, info, term

    def tcp(self) -> np.ndarray:
        return self.u._get_tcp_pose()

    # --- 主流程 -------------------------------------------------------
    def grasp(self, obj, seed_q, on_phase=None):
        q, _, _ = self.goto(obj + [0, 0, 0.10], self.open, seed_q, n=25)
        if on_phase:
            on_phase("hover")
        for i, (dx, dy, dz) in enumerate(GRASP_OFFSETS):
            qq = q if i == 0 else self.solve_ik(obj + [dx, dy, dz + 0.10], q)
            qq, info, _ = self.goto(obj + [dx, dy, dz], self.open, qq, n=40)
            for k in range(1, 41):  # 40 帧慢速闭合
                term, trunc, info = self.step(qq, self.open + (self.closed - self.open) * k / 40)
            if info["is_grasped"] > 0.5:
                if on_phase:
                    on_phase("grasp")
                return qq, info, "grasped"
            self.goto(obj + [0, 0, 0.10], self.open, qq, n=20)  # 张开回悬停再试下一个偏移
        return q, info, "no-grasp"

    def run(self, seed: int, lift_ramp: float = 0.0012, lift_steps: int = 320,
            on_phase=None) -> dict:
        self.env.reset(seed=seed)
        obj = self.u._get_target_pose()[:3]
        log = {"seed": seed, "obj": np.round(obj, 4).tolist()}
        q, info, tag = self.grasp(obj, self.d.qpos[:5].copy(), on_phase=on_phase)
        log["grasp"] = tag
        log["grip_qpos"] = round(float(self.d.qpos[5]), 4)
        if tag != "grasped":
            log.update(success=False, max_lift=0.0)
            return log

        best = 0.0
        base = q.copy()
        shot_taken = False
        for k in range(1, lift_steps + 1):
            cur = base.copy()
            cur[1] = base[1] - lift_ramp * k
            term, trunc, info = self.step(cur, self.closed)
            best = max(best, info["lift_height"])
            if on_phase and not shot_taken and best > 0.03:
                on_phase("lift")          # 成功会立刻 autoreset，所以中途取帧
                shot_taken = True
            if term or trunc:  # 成功 -> terminated（terminate_on_success=True）
                log.update(success=bool(term) or info["success"] > 0.5,
                           max_lift=round(best, 4), terminated=bool(term))
                return log
        log.update(success=False, max_lift=round(best, 4), terminated=False)
        return log


def render_montage(env, u, frames: list[np.ndarray], out: Path) -> None:
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    w, h = ims[0].size
    canvas = Image.new("RGB", (w * len(ims), h), "white")
    for i, im in enumerate(ims):
        canvas.paste(im, (i * w, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", action="store_true", help="导出关键帧拼图")
    args = ap.parse_args()

    envs = make_env(n_envs=1, env_id="MuJoCoPickLift-v1")
    env = envs["MuJoCoPickLift-v1"][0]
    u = env.envs[0].unwrapped
    lo, hi = np.asarray(u._target_low, float), np.asarray(u._target_high, float)
    expert = PickExpert(env, open_val=float(hi[-1]), closed_val=float(lo[-1]))

    renderer = None
    cam = None
    if args.frames:
        renderer = mujoco.Renderer(u.model, 360, 480)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.28, 0.05, 0.09]
        cam.distance, cam.azimuth, cam.elevation = 0.85, 120, -22

    shots: list[np.ndarray] = []

    def on_phase(name: str) -> None:  # noqa: ARG001
        if renderer is not None:
            renderer.update_scene(u.data, camera=cam)
            shots.append(renderer.render().copy())

    if renderer is not None:
        env.reset(seed=args.seed)
        on_phase("start")

    log = expert.run(seed=args.seed, on_phase=on_phase)
    print("RESULT", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in log.items()})

    if renderer is not None and shots:
        from PIL import Image
        ims = [Image.fromarray(f) for f in shots]
        w, h = ims[0].size
        cols = 2
        rows = (len(ims) + 1) // 2
        canvas = Image.new("RGB", (w * cols, h * rows), "white")
        for i, im in enumerate(ims):
            canvas.paste(im, ((i % cols) * w, (i // cols) * h))
        out = Path("refs") / f"expert_trial_seed{args.seed}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
        print("frames", len(shots), "->", out.resolve())


if __name__ == "__main__":
    main()
