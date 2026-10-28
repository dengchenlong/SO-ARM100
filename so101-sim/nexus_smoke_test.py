# -*- coding: utf-8 -*-
"""so101-nexus MuJoCo 环境冒烟测试:创建 PickLift 任务环境,随机动作跑 200 步。"""
import numpy as np

from so101_nexus.envhub import make_env

envs = make_env(n_envs=1)
env = envs["MuJoCoPickLift-v1"][0]
obs, info = env.reset(seed=0)

print("obs keys:", {k: np.asarray(v).shape for k, v in obs.items()})
print("action space:", env.action_space)

total = 0.0
for step in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total += float(np.asarray(reward).reshape(-1)[0])
    if terminated or truncated:
        obs, info = env.reset()

print(f"200 步随机动作完成, 累计 reward = {total:.4f}")
print("OK: so101-nexus 仿真跑通")
