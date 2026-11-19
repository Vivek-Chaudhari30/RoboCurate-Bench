# Task B readiness: facts only

Checked 2026-11-19 against the installed **lerobot 0.6.1** (paths relative to `site-packages/lerobot/`) and the
Hugging Face Hub. This page records facts, not decisions. "Not verified" means the check needs a package that is not
installed in the `rcb` env. Nothing was installed for this check.

| question | PushT + Diffusion | MetaWorld (scripted expert) | robomimic (mixed-human) |
|---|---|---|---|
| env in lerobot-eval | **yes**: `envs/configs.py:191` registers `pusht` (task `PushT-v0`, fps 10, episode_length 300, obs `pixels_agent_pos`) | **yes**: `envs/configs.py:454` registers `metaworld` (default task `metaworld-push-v2`, fps 80, episode_length 400), built by `envs/metaworld.py:286 create_metaworld_envs` | **no**: no `robomimic` env is registered in `envs/configs.py`. The registered types are aloha, pusht, libero, metaworld, robocasa, vlabench, isaaclab_arena, libero_plus, robotwin, robomme |
| required extra | `lerobot[pusht]`: gym-pusht >=0.1.5,<0.2.0 and pymunk >=6.6,<7 (METADATA:260-262) | `lerobot[metaworld]`: metaworld==3.0.0 (METADATA:269-271) | none in lerobot. The `robomimic` package is not a lerobot extra |
| installed here | no (`import gym_pusht` fails) | no (`import metaworld` fails) | no (`import robomimic` fails) |
| CPU reset + render works | not verified (package not installed) | not verified (package not installed) | n/a (no env) |
| scripted expert | none in gym-pusht's lerobot wrapper (lerobot/pusht is human teleop: 206 episodes, v3.0, fps 10) | **yes**: `envs/metaworld.py:24` imports `metaworld.policies`; `TASK_POLICY_MAPPING` (`:67-73`) maps each task to its scripted policy class and `MetaworldEnv` builds `self.expert_policy` (`:107`) | n/a in lerobot. robomimic provides machine-generated (`mg`) demos as hdf5 |
| Hub datasets (format) | `lerobot/pusht` @7628202a21 (v3.0, 206 eps, fps 10); `lerobot/pusht_image` @1b84c4882e; `lerobot/pusht_keypoints` @ace8c161a6 | `lerobot/metaworld_mt50` @b46a1c185a (v3.0, 2500 eps, 50 tasks, fps 80); `lerobot/metaworld_mt50_push_v2_image` @d105b817b0 | LeRobot format, community: `iantc104/robomimic_sim_square` @cc5144bfc4 (v2.1, 175 eps, fps 20); `iantc104/robomimic_sim_square_noisy` @0758336ee9 (v2.1, 5000 eps); `chaoqi-liu/robomimic_mt4_ph` @b8dea1c132 (v3.0, 800 eps, 4 tasks, proficient-human). Official hdf5 (not LeRobot format): `robomimic/robomimic_datasets` @74fa018461 has `mh` (multi-human, mixed quality) files for can, lift, square and transport, plus `ph`, `mg` and `paired` |
| mixed-quality human data in LeRobot format | no (single-operator teleop) | no (scripted) | not found under that label. The robomimic `mh` sets exist only as hdf5 in the official repo (a conversion would be needed). Whether the `iantc104` datasets derive from `mh` is not stated in their cards |

Notes:
- Diffusion Policy is registered as a policy (`policies/__init__.py:18` imports `DiffusionConfig`).
- LIBERO's extra (`hf-libero`) installs only on Linux (`METADATA:266`).
