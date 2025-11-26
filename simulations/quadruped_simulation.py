"""
Isaac Sim Spot 환경과 OmniVLA 추론을 연결하는 시뮬레이션 스크립트.

기존 `quadruped_example.py` 환경을 재사용하되, 키보드 입력 대신
OmniVLA 추론 스레드에서 공유 메모리를 통해 전달되는 명령을 사용한다.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 모듈 경로 설정: Isaac Sim submodule + OmniVLA inference
# ---------------------------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[1]
ISAACSIM_DIR = REPO_ROOT / "extern" / "isaacsim-spot-remotecontroldemo"

if str(ISAACSIM_DIR) not in sys.path:
    sys.path.append(str(ISAACSIM_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Isaac Sim 런타임 및 Spot 환경 로딩
from quadruped_example import DEFAULT_CONFIG, SpotSimulation, simulation_app  # type: ignore  # noqa: E402

# OmniVLA 추론 래퍼
from inference.sim_omnivla import (  # noqa: E402
    SharedMemoryChannels,
    SimOmniVLAConfig,
    SimOmniVLAController,
)


LOGGER = logging.getLogger("QuadrupedSim")
LOGGER.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Shared-memory 기반 Controller
# -----------------------------------------------------------------------------
class SharedMemoryCommandController:
    """OmniVLA 명령을 Isaac Sim 로봇 제어 명령으로 변환."""

    def __init__(self, shared_channels: SharedMemoryChannels, max_vx: float, max_vy: float, max_yaw: float):
        self.shared_channels = shared_channels
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_yaw = max_yaw

        self._command = np.zeros(3, dtype=np.float32)
        self._state_lock = threading.Lock()
        self._key_state = {"quit": False}

    def update(self):
        command = self.shared_channels.read_command(clear_event=False)
        command[0] = np.clip(command[0], -self.max_vx, self.max_vx)
        command[1] = np.clip(command[1], -self.max_vy, self.max_vy)
        command[2] = np.clip(command[2], -self.max_yaw, self.max_yaw)
        with self._state_lock:
            self._command = command.astype(np.float32)

    def get_command(self) -> np.ndarray:
        with self._state_lock:
            return self._command.copy()

    def is_quit_requested(self) -> bool:
        with self._state_lock:
            return bool(self._key_state.get("quit", False))


# -----------------------------------------------------------------------------
# Isaac Sim wrapper
# -----------------------------------------------------------------------------
class OmniVLASpotSimulation(SpotSimulation):
    def __init__(self, shared_channels: SharedMemoryChannels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_channels = shared_channels
        self._last_obs_publish = 0.0
        self.observation_period = 1.0 / 10.0  # 10 Hz

    def setup(self):
        super().setup()

        # 키보드/pygame 기반 컨트롤 → 공유 메모리 컨트롤러로 대체
        cfg = self.config
        self.controller = SharedMemoryCommandController(
            self.shared_channels,
            max_vx=cfg["max_vx"],
            max_vy=cfg["max_vy"],
            max_yaw=cfg["max_yaw"],
        )

        # Pygame 디스플레이 비활성화
        if self.display:
            self.display.stop()
            self.display = None

        # 기존 physics callback 대체
        if self.world.physics_callback_exists("physics_step"):
            self.world.remove_physics_callback("physics_step")
        self.world.add_physics_callback("physics_step", callback_fn=self._physics_step_with_bridge)

        LOGGER.info("OmniVLA Spot Simulation setup 완료")

    def _physics_step_with_bridge(self, step_size):
        super()._on_physics_step(step_size)

        if not self.physics_ready or self.shared_channels is None:
            return

        now = time.time()
        if now - self._last_obs_publish < self.observation_period:
            return
        self._last_obs_publish = now

        robot_pose, robot_quat = self.spot.robot.get_world_pose()
        robot_rpy = self.quaternion_to_rpy(robot_quat)
        ego_image = self.get_ego_camera_image()
        if ego_image is None:
            return

        robot_position = np.array([robot_pose[0], robot_pose[1], robot_pose[2]], dtype=np.float32)
        goal_position = np.array([self.goal_pos[0], self.goal_pos[1], robot_pose[2]], dtype=np.float32)
        goal_heading = math.atan2(goal_position[1] - robot_position[1], goal_position[0] - robot_position[0])

        self.shared_channels.write_observation(
            image=ego_image,
            robot_position=robot_position,
            robot_heading=float(robot_rpy[2]),
            goal_position=goal_position,
            goal_heading=float(goal_heading),
            timestamp=now,
        )


# -----------------------------------------------------------------------------
# 실행 스크립트
# -----------------------------------------------------------------------------
def main():
    ego_width, ego_height = DEFAULT_CONFIG["ego_camera_resolution"]
    shared_channels = SharedMemoryChannels((ego_width, ego_height))

    omnivla_cfg = SimOmniVLAConfig(
        goal_image_path=str(REPO_ROOT / "inference" / "frame211-56.597-ego.jpg"),
        lan_inst_prompt="navigate to the marked goal",
    )
    omnivla_controller = SimOmniVLAController(omnivla_cfg)
    stop_event = threading.Event()
    omnivla_thread = threading.Thread(
        target=omnivla_controller.serve_forever,
        args=(shared_channels, stop_event),
        daemon=True,
    )
    omnivla_thread.start()

    sim = OmniVLASpotSimulation(
        shared_channels=shared_channels,
        experiment_name="omnivla_sim",
        enable_csv_logging=False,
        enable_image_saving=False,
        object_type="none",
    )

    try:
        sim.setup()
        sim.run()
    finally:
        stop_event.set()
        omnivla_thread.join(timeout=2.0)
        sim.cleanup()
        shared_channels.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
