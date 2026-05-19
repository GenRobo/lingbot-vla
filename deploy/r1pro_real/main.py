"""R1Pro rollout driver for a LingbotVLA right-arm policy.

Sibling of `openpi/examples/r1pro_real/main.py`, adapted to the LingbotVLA
websocket protocol and to the `r1pro_delta_right` robot config (state =
right_arm[7] + right_gripper[1], action = right_arm[7] + right_gripper[1]).

Topology: this driver and the inference server both run on the laptop. The
driver imports `grid_robot_api.robot.mobile_manipulator.GalaxeaR1Pro`,
which joins the robot's ROS2 graph over the LAN and reads/publishes the
Galaxea factory topics. Nothing runs on the robot itself.

Inference: dict-of-numpy obs over msgpack-websocket. Server applies the
`r1pro_delta_right` FeatureTransform (selected by `client.reset(...)`),
runs the flow-matching policy, and returns absolute joint actions —
delta-to-absolute reconstruction is done server-side in
`FeatureTransform.unapply`.

Usage:
    python -m deploy.r1pro_real.main \\
        --remote_host=localhost --remote_port=8000 \\
        --instruction "pick up the apple"
"""

import dataclasses
import faulthandler
import logging
import time
from typing import Optional

import numpy as np
import tyro

from grid_robot_api.robot.mobile_manipulator import GalaxeaR1Pro

from deploy.websocket_client_policy import WebsocketClientPolicy

faulthandler.enable()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Match the dataset collection rate.
DEFAULT_CONTROL_FREQUENCY = 10  # Hz

# Training data uses the robot's raw 0..100 gripper space directly
# (0=closed, 100=open) and per-frame gripper actions are binary. Threshold
# at the midpoint to convert the policy's continuous prediction back to
# binary before commanding.
_GRIPPER_BINARY_THRESHOLD = 50.0

# Robot config name on the server side. Matches
# configs/robot_configs/r1pro_delta_right.yaml.
ROBO_NAME = "r1pro_delta_right"

# Initial pose. Mirrors the openpi r1pro_real script — torso forward, both
# elbows bent so the right arm has working room. Left arm is held in this
# pose throughout the rollout (the policy doesn't command it).
TORSO_START = [0.30, -0.50, -0.50, 0.0006]
ELBOW_JOINT_INDEX = 3
ELBOW_INIT_ANGLE = -1.85
TOPIC_WAIT_TIME = 2.0  # seconds — give ROS2 topics time to populate

# Camera names from the GalaxeaR1Pro driver, paired with the lingbot wire
# keys the server's FeatureTransform expects (origin_keys in
# configs/robot_configs/r1pro_delta_right.yaml).
HEAD_CAM = "head_camera_left"
LEFT_WRIST_CAM = "wrist_left"
RIGHT_WRIST_CAM = "wrist_right"
CAMERA_WIRE_KEYS = {
    HEAD_CAM: "observation.images.head",
    LEFT_WRIST_CAM: "observation.images.left_wrist",
    RIGHT_WRIST_CAM: "observation.images.right_wrist",
}

# Right-arm action layout. Server returns action.right_arm shape
# (use_length, 7) and action.right_gripper shape (use_length, 1).
_RIGHT_ARM_KEY = "action.right_arm"
_RIGHT_GRIPPER_KEY = "action.right_gripper"


@dataclasses.dataclass
class Args:
    # Language instruction. If None, prompt interactively at startup.
    instruction: Optional[str] = None

    # Inference server endpoint.
    remote_host: str = "localhost"
    remote_port: int = 8000

    # Number of actions from each predicted chunk to execute before
    # re-observing. Must be <= the server's --use_length.
    open_loop_horizon: int = 8

    # Control loop rate. Matches the dataset.
    control_frequency_hz: float = DEFAULT_CONTROL_FREQUENCY

    # If True, skip the home pose / torso init / elbow bend / gripper
    # open. Useful when the robot is already roughly in pose.
    skip_homing: bool = False

    # If True, print each predicted action and wait for Enter before
    # executing. Useful for the first dry runs of a new policy.
    step_through: bool = False


# ---------- robot helpers (mirrored from openpi) ----------------------------


def _get_camera_image(robot: GalaxeaR1Pro, name: str, timeout: float = 5.0) -> np.ndarray:
    """Return the latest camera frame, retrying for up to `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        img = robot.getImage(name)
        if img is not None:
            return img.data
        time.sleep(0.1)
    raise RuntimeError(f"Camera '{name}' produced no data within {timeout:.0f}s.")


def _action_to_robot_gripper(training_cmd: float) -> float:
    """Threshold a predicted gripper command into binary robot 0/100."""
    return 100.0 if training_cmd >= _GRIPPER_BINARY_THRESHOLD else 0.0


def _initialize_robot(robot: GalaxeaR1Pro) -> None:
    logger.info("Waiting %.1fs for ROS2 topics to populate...", TOPIC_WAIT_TIME)
    time.sleep(TOPIC_WAIT_TIME)

    robot.stop()
    robot.moveToNamedPose("home")
    time.sleep(1)

    robot.base.setJointAngles(TORSO_START)

    # Bend both elbows. Even though we only command the right arm, the
    # dataset was recorded with both arms in this configuration, so we
    # match the scene as closely as we can.
    left_angles = robot.left_arm.getJointAngles()
    right_angles = robot.right_arm.getJointAngles()
    left_angles[ELBOW_JOINT_INDEX] = ELBOW_INIT_ANGLE
    right_angles[ELBOW_JOINT_INDEX] = ELBOW_INIT_ANGLE
    robot.left_arm.setJointAngles(left_angles)
    robot.right_arm.setJointAngles(right_angles)
    time.sleep(1)

    robot.left_arm.setGripperPosition(100.0)
    robot.right_arm.setGripperPosition(100.0)
    logger.info("R1Pro initialized: home → torso=%s, elbows bent, grippers open.", TORSO_START)


def _verify_cameras(robot: GalaxeaR1Pro) -> None:
    """Sanity-check that all three cameras publish non-zero frames."""
    for cam in (HEAD_CAM, LEFT_WRIST_CAM, RIGHT_WRIST_CAM):
        img = _get_camera_image(robot, cam)
        if img.sum() == 0:
            raise RuntimeError(f"Camera {cam!r} returned an all-zero image.")
        logger.info("camera %s OK (shape=%s)", cam, img.shape)


# ---------- observation / action --------------------------------------------


def _extract_observation(robot: GalaxeaR1Pro) -> dict:
    """Collect the LingbotVLA wire-format obs dict for r1pro_delta_right.

    Keys match `origin_keys` in configs/robot_configs/r1pro_delta_right.yaml.
    The server's FeatureTransform maps these into the model's (arm.position,
    effector.position, camera_top, camera_wrist_*) feature space.
    """
    right_arm = np.asarray(robot.right_arm.getJointAngles(), dtype=np.float32)
    right_gripper = np.asarray(
        [robot.right_arm.getGripperPosition()], dtype=np.float32
    )
    obs: dict = {
        "observation.state.right_arm": right_arm,         # (7,)
        "observation.state.right_gripper": right_gripper, # (1,)
    }
    for cam, wire_key in CAMERA_WIRE_KEYS.items():
        obs[wire_key] = _get_camera_image(robot, cam)     # (H,W,3) uint8
    return obs


def _step(
    robot: GalaxeaR1Pro,
    arm_action: np.ndarray,
    gripper_action: float,
) -> None:
    """Execute one action: 7 absolute right-arm joint angles + gripper."""
    if arm_action.shape[-1] != 7:
        raise RuntimeError(f"Expected arm action dim == 7; got {arm_action.shape}")
    robot.right_arm.setJointAngles(arm_action.tolist())
    robot.right_arm.setGripperPosition(_action_to_robot_gripper(gripper_action))


# ---------- main loop -------------------------------------------------------


def main(args: Args):
    policy_client = WebsocketClientPolicy(args.remote_host, args.remote_port)

    # Push robot-config selection to the server. After this the server's
    # FeatureTransform is bound to r1pro_delta_right; subsequent infer
    # calls will run through it.
    logger.info("Resetting server feature transform to %s...", ROBO_NAME)
    policy_client.reset(robo_name=ROBO_NAME)

    logger.info("Initializing GalaxeaR1Pro...")
    robot = GalaxeaR1Pro()
    if not args.skip_homing:
        _initialize_robot(robot)
    _verify_cameras(robot)

    instruction = args.instruction or input("\nEnter instruction: ")
    step_period = 1.0 / args.control_frequency_hz
    logger.info(
        "Running rollout — frequency=%.1f Hz, open_loop_horizon=%d. Ctrl+C to stop.",
        args.control_frequency_hz,
        args.open_loop_horizon,
    )

    try:
        chunk_idx = 0
        while True:
            chunk_idx += 1
            obs = _extract_observation(robot)
            # LingbotVLA reads the language string from `task` (see
            # FeatureTransform.pad_and_concat — `"prompt": [item["task"]]`).
            obs["task"] = instruction

            t1 = time.time()
            response = policy_client.infer(obs)
            t2 = time.time()
            logger.info("[chunk %d] inference: %.3fs", chunk_idx, t2 - t1)
            if "server_timing" in response:
                logger.info("  server_timing: %s", response["server_timing"])

            if _RIGHT_ARM_KEY not in response or _RIGHT_GRIPPER_KEY not in response:
                raise RuntimeError(
                    f"Unexpected response keys: {list(response.keys())}"
                )
            arm_chunk = np.asarray(response[_RIGHT_ARM_KEY])     # (use_length, 7)
            grip_chunk = np.asarray(response[_RIGHT_GRIPPER_KEY])  # (use_length, 1)

            if arm_chunk.shape[0] != grip_chunk.shape[0]:
                raise RuntimeError(
                    f"Mismatched chunk lengths: arm {arm_chunk.shape}, "
                    f"gripper {grip_chunk.shape}"
                )
            if args.open_loop_horizon > arm_chunk.shape[0]:
                raise RuntimeError(
                    f"--open_loop_horizon={args.open_loop_horizon} exceeds the "
                    f"server's per-call action length ({arm_chunk.shape[0]}). "
                    "Increase the server's --use_length."
                )

            horizon = args.open_loop_horizon
            logger.info("[chunk %d] executing %d/%d actions",
                        chunk_idx, horizon, arm_chunk.shape[0])

            for i in range(horizon):
                arm_action = arm_chunk[i]
                grip_action = float(grip_chunk[i, 0])
                if args.step_through:
                    print(f"  [{i + 1}/{horizon}] action:")
                    print(f"    right_arm     : {[f'{a:.3f}' for a in arm_action]}")
                    print(f"    right_gripper : {grip_action:.3f} "
                          f"→ {_action_to_robot_gripper(grip_action):.0f}")
                    input("  Press Enter to execute (Ctrl+C to abort)...")
                start = time.time()
                _step(robot, arm_action, grip_action)
                elapsed = time.time() - start
                if elapsed < step_period and not args.step_through:
                    time.sleep(step_period - elapsed)

    except KeyboardInterrupt:
        logger.info("Returning to home pose...")
        try:
            robot.moveToNamedPose("home")
            time.sleep(1.0)
        except Exception as e:
            logger.warning("home pose move failed: %s", e)
    finally:
        robot.stop()
        robot.shutdown()


if __name__ == "__main__":
    main(tyro.cli(Args))
