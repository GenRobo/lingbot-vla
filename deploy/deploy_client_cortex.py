"""Step-by-step pi05 rollout via CortexClient on the R1Pro.

Same observe → infer → execute loop as the full-speed rollout, but
pauses for user confirmation before each action step.

Usage:
    CORTEX_API_KEY=<key> python test_pi05_real_step.py
"""

import logging
import os
import time

import numpy as np

from grid_cortex_client import CortexClient, ModelType
from grid_robot_api.robot.mobile_manipulator import GalaxeaR1Pro

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CORTEX_BASE_URL = "http://192.168.0.75:8000/"

TOPIC_WAIT_TIME = 2.0  # seconds

# R1Pro control frequency
R1PRO_CONTROL_FREQUENCY = 10  # Hz
_STEP_TIME = 1.0 / R1PRO_CONTROL_FREQUENCY

# Action vector layout (matches real pi05 output — total: 26)
_LEFT_ARM_SLICE = slice(0, 7)
_RIGHT_ARM_SLICE = slice(7, 14)
_TORSO_SLICE = slice(14, 18)
_CHASSIS_VEL_SLICE = slice(18, 24)
_LEFT_GRIPPER_IDX = 24
_RIGHT_GRIPPER_IDX = 25

OPEN_LOOP_HORIZON = 8   # actions to execute before re-observing

GRIPPER_CLOSE_STROKE = 10.0   # training-space value for fully closed
GRIPPER_OPEN_STROKE  = 90.0   # training-space value for fully open

TORSO_START = [0.30, -0.50, -0.50, 0.0006]


def _get_camera_image(robot: GalaxeaR1Pro, name: str, timeout: float = 5.0) -> np.ndarray:
    """Return the latest camera frame, retrying for up to ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        img = robot.getImage(name)
        if img is not None:
            return img.data
        time.sleep(0.1)
    raise RuntimeError(f"Camera '{name}' produced no data within {timeout:.0f}s.")


def _get_imu_vector(robot: GalaxeaR1Pro) -> np.ndarray:
    """Return chassis IMU as a flat 10-element vector."""
    imu = robot.getIMU("chassis")
    la = imu["linear_acceleration"]
    av = imu["angular_velocity"]
    ori = imu["orientation"]
    return np.array(
        [la["x"], la["y"], la["z"], av["x"], av["y"], av["z"],
         ori["x"], ori["y"], ori["z"], ori["w"]],
        dtype=np.float32,
    )


def _get_chassis_velocities(robot: GalaxeaR1Pro) -> np.ndarray:
    """Return chassis body velocities as [vx, vy, wz]."""
    with robot.base._chassis_js_lock:
        msg = robot.base._latest_chassis_js
    if msg is None or len(msg.velocity) < 3:
        return np.zeros(3, dtype=np.float32)
    return np.array(msg.velocity[:3], dtype=np.float32)


def _extract_observation(robot: GalaxeaR1Pro) -> dict:
    """Collect a full R1Pro observation snapshot for policy inference."""
    head_image = _get_camera_image(robot, "head_camera_left")
    left_wrist_image = _get_camera_image(robot, "wrist_left")
    right_wrist_image = _get_camera_image(robot, "wrist_right")

    left_arm = np.array(robot.left_arm.getJointAngles(), dtype=np.float32)
    right_arm = np.array(robot.right_arm.getJointAngles(), dtype=np.float32)
    torso = np.array(robot.base.getJointAngles(), dtype=np.float32)
    chassis_imu = _get_imu_vector(robot)
    chassis_vel = _get_chassis_velocities(robot)
    # Map robot [0, 100] → training [GRIPPER_CLOSE_STROKE, GRIPPER_OPEN_STROKE]
    _g_range = GRIPPER_OPEN_STROKE - GRIPPER_CLOSE_STROKE
    left_gripper  = np.array([robot.left_arm.getGripperPosition()  / 100.0 * _g_range + GRIPPER_CLOSE_STROKE], dtype=np.float32)
    right_gripper = np.array([robot.right_arm.getGripperPosition() / 100.0 * _g_range + GRIPPER_CLOSE_STROKE], dtype=np.float32)

    state = np.concatenate(
        [left_arm, right_arm, torso, chassis_imu, chassis_vel, left_gripper, right_gripper]
    )

    return {
        "head_rgb": head_image,
        "left_wrist_rgb": left_wrist_image,
        "right_wrist_rgb": right_wrist_image,
        "state": state,
    }


def _step(robot: GalaxeaR1Pro, action: np.ndarray) -> None:
    """Execute one 26-element action on the robot (arms + grippers)."""
    left_arm_angles = action[_LEFT_ARM_SLICE].tolist()
    right_arm_angles = action[_RIGHT_ARM_SLICE].tolist()
    left_gripper_cmd = float(action[_LEFT_GRIPPER_IDX])
    right_gripper_cmd = float(action[_RIGHT_GRIPPER_IDX])

    _threshold = (GRIPPER_CLOSE_STROKE + GRIPPER_OPEN_STROKE) / 2.0  # 50.0
    left_robot_grip  = 100.0 if left_gripper_cmd  <= _threshold else 0.0
    right_robot_grip = 100.0 if right_gripper_cmd <= _threshold else 0.0

    robot.setJointAngles({"left_arm": left_arm_angles, "right_arm": right_arm_angles})
    robot.left_arm.setGripperPosition(left_robot_grip)
    robot.right_arm.setGripperPosition(right_robot_grip)


def main():
    api_key = os.getenv("CORTEX_API_KEY")
    if api_key is None:
        raise EnvironmentError("CORTEX_API_KEY environment variable is not set.")
    policy_client = CortexClient(api_key=api_key, base_url=CORTEX_BASE_URL)

    print("Initializing GalaxeaR1Pro...")
    robot = GalaxeaR1Pro()

    print(f"Waiting {TOPIC_WAIT_TIME}s for ROS2 topics to populate...")
    time.sleep(TOPIC_WAIT_TIME)

    robot.stop()
    robot.moveToNamedPose("home")
    time.sleep(1)

    robot.base.setJointAngles(TORSO_START)

    # Set both arm joint 2 (3rd joint) to -1.65 rad from home
    left_angles = robot.left_arm.getJointAngles()
    right_angles = robot.right_arm.getJointAngles()
    left_angles[3] = -1.85
    right_angles[3] = -1.85
    robot.left_arm.setJointAngles(left_angles)
    robot.right_arm.setJointAngles(right_angles)
    time.sleep(1)

    robot.left_arm.setGripperPosition(100.0)
    robot.right_arm.setGripperPosition(100.0)

    logger.info("R1Pro initialized and moved to home pose (torso=%s).", TORSO_START)

    # Verify all cameras are producing data before proceeding
    print("\nChecking cameras...")
    for cam_name in ("head_camera_left", "wrist_left", "wrist_right"):
        img = _get_camera_image(robot, cam_name)
        if img.sum() == 0:
            raise RuntimeError(f"Camera '{cam_name}' returned an all-zero image.")
        print(f"  {cam_name}: OK (shape={img.shape})")

    instruction = input("\nEnter instruction: ")
    global_step = 0
    chunk_idx = 0

    try:
        while True:
            chunk_idx += 1
            # --- Observe ---
            print(f"\n{'='*50}")
            print(f"Chunk {chunk_idx} — Observing")
            obs = _extract_observation(robot)
            state = obs["state"]
            print(f"  left_arm      : {[f'{a:.3f}' for a in state[0:7]]}")
            print(f"  right_arm     : {[f'{a:.3f}' for a in state[7:14]]}")
            print(f"  torso         : {[f'{a:.3f}' for a in state[14:18]]}")
            print(f"  chassis_imu   : {[f'{a:.3f}' for a in state[18:28]]}")
            print(f"  chassis_vel   : {[f'{a:.3f}' for a in state[28:31]]}")
            print(f"  left_gripper  : {state[31]:.3f}")
            print(f"  right_gripper : {state[32]:.3f}")

            # --- Infer ---
            print(f"Chunk {chunk_idx} — Inferring...")
            t1 = time.time()
            action_response = policy_client.run(
                ModelType.PI05,
                # head_rgb=obs["head_rgb"],
                # left_wrist_rgb=obs["left_wrist_rgb"],
                # right_wrist_rgb=obs["right_wrist_rgb"],
                images = [obs["head_rgb"], obs["left_wrist_rgb"], obs["right_wrist_rgb"]],
                state=obs["state"],
                prompt=instruction,
            )
            t2 = time.time()
            logger.info("Inference time: %.3fs", t2 - t1)

            if "actions" not in action_response:
                raise RuntimeError(f"Unexpected response: {action_response.keys()}")
            action_chunk = action_response["actions"]
            print(f"  {len(action_chunk)} actions returned, executing first {OPEN_LOOP_HORIZON}")

            # --- Execute (first OPEN_LOOP_HORIZON actions only) ---
            for i, action in enumerate(action_chunk[:OPEN_LOOP_HORIZON]):
                global_step += 1
                print(f"\n  [{global_step}] action:")
                print(f"    left_arm      : {[f'{a:.3f}' for a in action[_LEFT_ARM_SLICE]]}")
                print(f"    right_arm     : {[f'{a:.3f}' for a in action[_RIGHT_ARM_SLICE]]}")
                print(f"    torso         : {[f'{a:.3f}' for a in action[_TORSO_SLICE]]}")
                print(f"    chassis_vel   : {[f'{a:.3f}' for a in action[_CHASSIS_VEL_SLICE]]}")
                print(f"    left_gripper  : {action[_LEFT_GRIPPER_IDX]:.3f}")
                print(f"    right_gripper : {action[_RIGHT_GRIPPER_IDX]:.3f}")
                input("  Press Enter to execute...")
                start_time = time.time()
                _step(robot, action)
                elapsed = time.time() - start_time
                if elapsed < _STEP_TIME:
                    time.sleep(_STEP_TIME - elapsed)

    except KeyboardInterrupt:
        print("\nReturning to home pose...")
        robot.moveToNamedPose("home")
        time.sleep(1.0)
    finally:
        robot.stop()
        robot.shutdown()


if __name__ == "__main__":
    main()