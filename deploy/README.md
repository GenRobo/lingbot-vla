# Deploy

Websocket inference server + R1Pro robot driver for LingbotVLA.

## Quick start

```bash
# 1. Install (one-time, ~10–15 min on a fresh machine)
cd lingbot-vla
bash scripts/install_deploy.sh

# 2. Start the inference server (must run from this repo's root)
micromamba run -n lingbotvla-deploy --cwd $(pwd) python -m deploy.lingbot_vla_policy \
    --model_path output/r1pro_delta_right/checkpoints/global_step_4760/hf_ckpt \
    --port 8000 --use_length 25 --use_compile

# 3. (Optional) Smoke-test before plugging in the robot
micromamba run -n lingbotvla-deploy --cwd $(pwd) python -m deploy.r1pro_real.dry_run \
    --remote_host=localhost --remote_port=8000

# 4. Source ROS2 in the driver shell, then run the driver
source /opt/ros/<distro>/setup.bash
micromamba run -n lingbotvla-deploy --cwd $(pwd) python -m deploy.r1pro_real.main \
    --remote_host=localhost --remote_port=8000 \
    --instruction "pick up the apple and put it in the bowl"
```

Replace `<distro>` with your ROS2 distribution (e.g. `humble`, `iron`).
Replace `global_step_4760` with whichever checkpoint you trained.

## Architecture

Mirrors the openpi deployment pattern: one laptop runs **both** the
inference server and the robot driver. The robot itself just exposes its
factory Galaxea ROS2 topics — nothing on the robot is changed.

```
robot (factory ROS2)              laptop (everything)
─────────────────────             ────────────────────────────────
publishes:                        ┌─ inference server ────────────┐
  /hdas/feedback_arm_right        │  deploy.lingbot_vla_policy     │
  /hdas/feedback_gripper_right    │  loads ckpt, GPU inference     │
  /hdas/camera_*/.../compressed   │  serves ws://localhost:8000    │
subscribes:                       └────────────────┬───────────────┘
  /motion_target/target_joint_                     │ websocket
      state_arm_right                              ▼
  /motion_target/target_position_  ┌─ robot driver ────────────────┐
      gripper_right                │  deploy.r1pro_real.main        │
        ▲                          │  GalaxeaR1Pro (rclpy via GRA)  │
        │ ROS2 DDS over LAN        │  read sensors → infer →        │
        └──────────────────────────┤  publish joint commands @10 Hz │
                                   └────────────────────────────────┘
```

## Prerequisites

- A laptop with a CUDA 12.8 capable GPU (server) and on the robot's LAN.
- `micromamba` (or `mamba`/`conda`) installed.
- `GRID-Robot-API` checked out alongside this repo (default:
  `../GRID-Robot-API`). The install script uses **whatever branch is
  currently checked out** — verify before running.
- ROS2 (Humble or Iron) installed system-wide on the laptop. Only the
  driver needs ROS sourced; the server does not.
- A trained checkpoint under `output/<config>/checkpoints/global_step_*/hf_ckpt/`.

## What gets installed

`scripts/install_deploy.sh` creates a Python 3.12 env (default name:
`lingbotvla-deploy`) and installs:

- PyTorch 2.8.0 + CUDA 12.8
- transformers 4.51.3, lerobot 0.4.2, lingbotvla (editable)
- **flash-attn 2.8.3** — required: the Qwen2.5-VL backbone loads with
  `_attn_implementation=flash_attention_2` regardless of the server's
  `eager` override (different field name; the override doesn't take).
- `GRID-Robot-API` (editable) for the driver

If you already have the training env (`lingbotvla`) on the same machine,
you can shave the install time with:

```bash
CLONE_FROM=lingbotvla bash scripts/install_deploy.sh
```

This clones the existing env first, then runs the pip steps on top
(micromamba's `--clone` skips pip packages, so the pip phase is still
needed — but the conda layer is faster).

## Running the server

Always launch from the repo root:

```bash
cd lingbot-vla
micromamba run -n lingbotvla-deploy --cwd $(pwd) python -m deploy.lingbot_vla_policy \
    --model_path output/r1pro_delta_right/checkpoints/global_step_4760/hf_ckpt \
    --port 8000 \
    --use_length 25 \
    --num_denoising_step 10 \
    --use_compile
```

**Why repo root?** `LingbotVLAServer.reset()` resolves
`configs/robot_configs/<robo_name>.yaml` and the norm-stats JSON as
relative paths. Running from anywhere else and `reset()` will fail to
load.

**Wait for**: `INFO:websockets.server:server listening on 0.0.0.0:8000`.

**First call latency**: with `--use_compile`, the first `infer` call
triggers `torch.compile` and takes ~60s. Steady-state then drops to
~300 ms. Without compile: first call ~1.8s, steady-state ~1s. For a
10 Hz control loop, compile is recommended — run the dry-run first so
the compile pass is paid before the robot is in motion.

Key flags:
- `--model_path` — path to the `hf_ckpt/` dir from training.
- `--use_length` — server-side slice of the 50-action chunk before
  sending. Must be `>= --open_loop_horizon` on the driver. Default 25.
- `--num_denoising_step` — flow-matching denoising steps; 10 matches
  training.
- `--use_compile` — `torch.compile` the qwen+expert forward. Strongly
  recommended for any real-time use.

## Running the dry-run (recommended before the robot)

```bash
micromamba run -n lingbotvla-deploy --cwd $(pwd) python -m deploy.r1pro_real.dry_run \
    --remote_host=localhost --remote_port=8000 --num_calls 3
```

Sends 3 synthetic observations (zero state, zero images, a fixed prompt)
and prints the returned action chunk shapes + per-call timing. If this
prints `OK — server responded for all 3 calls`, the server is healthy.

## Running the robot driver

```bash
source /opt/ros/<distro>/setup.bash
micromamba run -n lingbotvla-deploy --cwd $(pwd) python -m deploy.r1pro_real.main \
    --remote_host=localhost --remote_port=8000 \
    --instruction "pick up the apple and put it in the bowl" \
    --open_loop_horizon 8
```

What it does:
1. Opens websocket to the server.
2. `client.reset(robo_name="r1pro_delta_right")` — server binds its
   FeatureTransform to `configs/robot_configs/r1pro_delta_right.yaml`.
3. `GalaxeaR1Pro()` — joins the ROS2 graph; subscribes to feedback/camera
   topics, prepares publishers for joint/gripper commands.
4. Homes the robot: `moveToNamedPose("home")` → torso to `TORSO_START`
   → both elbows bent → both grippers open. Matches the scene the
   policy was trained against (even though it only commands the right
   arm).
5. Verifies all three cameras produce non-zero frames.
6. 10 Hz loop: read obs → `client.infer(obs)` → execute the first
   `open_loop_horizon` actions of the returned chunk (absolute joint
   angles for the arm; binary 0/100 for the gripper, thresholded at 50)
   → repeat.
7. On Ctrl+C: return home and shut down.

Useful flags:
- `--step_through` — print each predicted action and wait for Enter
  before executing. Use this for the first run of a new policy.
- `--skip_homing` — skip the home init (only when the robot is already
  in a sensible pose).
- `--open_loop_horizon` — actions to execute per chunk. Must be
  `<= --use_length` on the server.
- `--control_frequency_hz` — defaults to 10 Hz to match the dataset.

## Wire format reference

What the driver sends per `infer` call:

| Key | Shape | dtype | Source |
|---|---|---|---|
| `observation.state.right_arm` | `(7,)` | float32 | `robot.right_arm.getJointAngles()` |
| `observation.state.right_gripper` | `(1,)` | float32 | `robot.right_arm.getGripperPosition()` |
| `observation.images.head` | `(H,W,3)` | uint8 | `robot.getImage("head_camera_left")` |
| `observation.images.left_wrist` | `(H,W,3)` | uint8 | `robot.getImage("wrist_left")` |
| `observation.images.right_wrist` | `(H,W,3)` | uint8 | `robot.getImage("wrist_right")` |
| `task` | str | — | `--instruction` |

What the server returns:

| Key | Shape | dtype | Semantics |
|---|---|---|---|
| `action.right_arm` | `(use_length, 7)` | float32 | Absolute joint angles. `FeatureTransform.unapply` already adds state back to the predicted delta. |
| `action.right_gripper` | `(use_length, 1)` | float32 | Raw 0..100 (binary in training data). Driver thresholds at 50. |
| `server_timing` | dict | — | `infer_ms`, `prev_total_ms`. |

## Deploying a different policy

The driver is hard-coded for `r1pro_delta_right` (right arm only). For a
different robot config:

1. Train (or have a checkpoint) under
   `output/<config_name>/checkpoints/global_step_*/hf_ckpt/`.
2. Make sure `configs/robot_configs/<config_name>.yaml` and
   `assets/norm_stats/<config_name>.json` exist.
3. Edit `deploy/r1pro_real/main.py` — change `ROBO_NAME`, the
   observation builder, and `_step()` to match the new schema (the
   wire keys and the robot commands you need to publish).
4. Restart the server with `--model_path` pointing at the new
   checkpoint.

## Troubleshooting

- **`Camera 'X' produced no data within 5s`** — robot ROS topics aren't
  reaching the driver. Check `ROS_DOMAIN_ID`, that the robot is up, and
  that `ros2 topic list` from the driver shell sees `/hdas/...`.
- **`open_loop_horizon=N exceeds the server's per-call action length`**
  — bump `--use_length` on the server, or lower `--open_loop_horizon`.
- **`keepalive ping timeout`** — should not happen with the current
  client (keepalive disabled); if you see it, you're hitting the older
  version of `deploy/websocket_client_policy.py`.
- **Server crashes in `FeatureTransform.apply` with a broadcast
  mismatch** — should not happen with the current server (we use the
  paired state's last dim for dummy actions); if you see it, you're
  hitting the older `org_features['states'][0]` codepath in
  `deploy/lingbot_vla_policy.py`.
- **First inference slow** — expected with `--use_compile` (~60s).
  Subsequent calls are ~300 ms.
