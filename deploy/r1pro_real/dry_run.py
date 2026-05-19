"""Smoke-test the LingbotVLA inference server without a robot.

Sends one synthetic observation (zero state, zero RGB, fixed prompt) for
the `r1pro_delta_right` schema and prints the returned action chunk
shape + a few statistics. Useful to verify that:

  1. The server is reachable on the given host/port.
  2. `reset(robo_name=...)` succeeds (FeatureTransform + norm-stats load).
  3. `infer(...)` runs end-to-end (model weights load + first
     `torch.compile` pass complete).

Run from the repo root in the same env as the server:
    python -m deploy.r1pro_real.dry_run --remote_host=localhost --remote_port=8000
"""

import argparse
import logging
import time

import numpy as np

from deploy.websocket_client_policy import WebsocketClientPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROBO_NAME = "r1pro_delta_right"
IMG_H, IMG_W = 480, 640  # any HxWx3 uint8 works; server resizes to 224x224


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote_host", default="localhost")
    parser.add_argument("--remote_port", type=int, default=8000)
    parser.add_argument(
        "--instruction",
        default="pick up the apple and put it in the bowl",
    )
    parser.add_argument(
        "--num_calls",
        type=int,
        default=3,
        help="Run this many infer calls. First will include the compile pass; "
        "later calls should be ~10-100x faster.",
    )
    args = parser.parse_args()

    client = WebsocketClientPolicy(args.remote_host, args.remote_port)
    logger.info("server metadata: %s", client.get_server_metadata())

    logger.info("reset(robo_name=%s) ...", ROBO_NAME)
    client.reset(robo_name=ROBO_NAME)

    zero_img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    obs_template = {
        "observation.state.right_arm": np.zeros(7, dtype=np.float32),
        "observation.state.right_gripper": np.zeros(1, dtype=np.float32),
        "observation.images.head": zero_img,
        "observation.images.left_wrist": zero_img,
        "observation.images.right_wrist": zero_img,
        "task": args.instruction,
    }

    for i in range(args.num_calls):
        # Fresh copy each call: msgpack-pack consumes the dict.
        obs = {k: (v.copy() if isinstance(v, np.ndarray) else v)
               for k, v in obs_template.items()}
        t0 = time.time()
        resp = client.infer(obs)
        dt = time.time() - t0
        if "action.right_arm" not in resp or "action.right_gripper" not in resp:
            raise RuntimeError(f"missing action keys; got {list(resp.keys())}")
        arm = np.asarray(resp["action.right_arm"])
        grip = np.asarray(resp["action.right_gripper"])
        logger.info(
            "[%d/%d] roundtrip=%.3fs  arm.shape=%s grip.shape=%s  "
            "arm[0]=%s grip[0]=%.3f  server_timing=%s",
            i + 1, args.num_calls, dt, arm.shape, grip.shape,
            np.round(arm[0], 3).tolist(), float(grip[0, 0]),
            resp.get("server_timing"),
        )

    logger.info("OK — server responded for all %d calls.", args.num_calls)


if __name__ == "__main__":
    main()
