"""Toy testbed self-checks (item 4): the env, the planted defects, and that defects actually hurt BC."""

import time

import numpy as np

from robocurate.curation.policy import MLPPolicy, train_bc
from toybed.demos import flatten, make_dataset, rollout_demo, success_rate
from toybed.env import PointReachEnv, expert_action, goal_distance_progress

EVAL = range(900000, 900100)  # toy env seeds, unrelated to protocol seeds


def test_expert_solves_env():
    env = PointReachEnv()
    ok = 0
    for s in range(50):
        obs, _ = env.reset(seed=s)
        for _ in range(env.max_steps):
            obs, _, term, trunc, info = env.step(expert_action(obs))
            if term or trunc:
                break
        ok += info["is_success"]
    assert ok == 50


def test_defects_are_executed_and_distinct():
    clean, _ = rollout_demo(3, "clean")
    paused_s, paused_a = rollout_demo(3, "pause")
    jit_s, jit_a = rollout_demo(3, "jitter")
    wrong_s, wrong_a = rollout_demo(3, "wrong")
    held = np.all(paused_a == 0, axis=1)
    assert held.sum() >= 10 and np.allclose(np.diff(paused_s[held][:, :2], axis=0), 0)  # pose really held
    assert len(paused_s) > len(clean)
    assert np.abs(np.diff(jit_a, axis=0)).mean() > 3 * np.abs(np.diff(rollout_demo(3)[1], axis=0)).mean()
    to_goal = wrong_s[:, 2:4] - wrong_s[:, 0:2]
    assert np.mean(np.sum(to_goal * wrong_a, axis=1) < 0) > 0.9  # actions point away from the observed goal
    p = goal_distance_progress(clean)
    assert p[0] == 0 and p[-1] > 0.85


def test_defects_hurt_bc_and_run_fast():
    t0 = time.time()
    view, labels = make_dataset()
    clean = [i for i, k in labels.items() if k == "clean"]
    s_all = success_rate(train_bc(MLPPolicy(4, 2), *flatten(view), steps=400, seed=0), EVAL)
    s_clean = success_rate(train_bc(MLPPolicy(4, 2), *flatten(view, clean), steps=400, seed=0), EVAL)
    assert s_clean >= s_all + 0.2, (s_clean, s_all)
    assert time.time() - t0 < 60
    order = [labels[i] for i in sorted(labels)]
    assert order != sorted(order)  # shuffled (D-021)
