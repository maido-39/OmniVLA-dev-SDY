"""
Simulation-oriented OmniVLA inference utilities.

이 모듈은 Isaac Sim 기반 사족보행 시뮬레이션과 OmniVLA 추론 간의
공유 메모리 채널 및 추론 워커를 제공합니다.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


LOGGER = logging.getLogger("SimOmniVLA")
LOGGER.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Dynamically load run_omnivla-ARIL.py (file name contains a hyphen)
# ---------------------------------------------------------------------------
RUN_SCRIPT_PATH = Path(__file__).with_name("run_omnivla-ARIL.py")
SPEC = importlib.util.spec_from_file_location("run_omnivla_ARIL", RUN_SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"run_omnivla-ARIL.py 를 로드할 수 없습니다: {RUN_SCRIPT_PATH}")
RUN_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_MODULE)


# run_omnivla-ARIL 에 clip_angle 정의가 없을 경우 대비
if hasattr(RUN_MODULE, "clip_angle"):
    clip_angle_fn = RUN_MODULE.clip_angle
else:
    def clip_angle_fn(angle: float) -> float:
        """[-pi, pi] 범위로 wrapping."""
        return (angle + math.pi) % (2 * math.pi) - math.pi


# 기본 modality 설정 (필요 시 외부에서 덮어쓸 수 있음)
RUN_MODULE.pose_goal = False
RUN_MODULE.satellite = False
RUN_MODULE.image_goal = False
RUN_MODULE.lan_prompt = True


# ---------------------------------------------------------------------------
# Shared memory channel 정의
# ---------------------------------------------------------------------------
STATE_LENGTH = 10  # [robot_xyz(3), robot_heading, goal_xy(2), goal_heading, timestamp, seq, ready_flag]
CMD_LENGTH = 3     # [vx, vy, yaw]


@dataclass
class SimulationObservation:
    """Isaac Sim → OmniVLA 로 전달되는 관측치."""

    image: np.ndarray  # RGB uint8 (H, W, 3)
    robot_position: np.ndarray  # [x, y, z]
    robot_heading: float  # yaw (rad)
    goal_position: np.ndarray  # [x, y, z]
    goal_heading: float  # yaw (rad)
    timestamp: float


class SharedMemoryChannels:
    """Observation / Command 를 공유하기 위한 메모리 버퍼."""

    def __init__(self, image_resolution: Tuple[int, int]):
        width, height = image_resolution
        self.image_shape = (height, width, 3)
        image_bytes = int(np.prod(self.image_shape))

        self._image_shm = shared_memory.SharedMemory(create=True, size=image_bytes)
        self._image = np.ndarray(self.image_shape, dtype=np.uint8, buffer=self._image_shm.buf)

        self._state_shm = shared_memory.SharedMemory(create=True, size=STATE_LENGTH * 8)
        self._state = np.ndarray((STATE_LENGTH,), dtype=np.float64, buffer=self._state_shm.buf)
        self._state[:] = 0.0

        self._command_shm = shared_memory.SharedMemory(create=True, size=CMD_LENGTH * 8)
        self._command = np.ndarray((CMD_LENGTH,), dtype=np.float64, buffer=self._command_shm.buf)
        self._command[:] = 0.0

        self._obs_lock = threading.Lock()
        self._cmd_lock = threading.Lock()

        self._new_obs_event = threading.Event()
        self._new_cmd_event = threading.Event()

    # ---- Observation path -------------------------------------------------
    def write_observation(
        self,
        image: np.ndarray,
        robot_position: np.ndarray,
        robot_heading: float,
        goal_position: np.ndarray,
        goal_heading: float,
        timestamp: Optional[float] = None,
    ) -> None:
        if image.shape != self.image_shape:
            raise ValueError(f"이미지 shape 불일치: {image.shape} != {self.image_shape}")

        timestamp = timestamp if timestamp is not None else time.time()
        with self._obs_lock:
            np.copyto(self._image, image.astype(np.uint8, copy=False))

            self._state[0:3] = robot_position[:3]
            self._state[3] = float(robot_heading)
            self._state[4:6] = goal_position[:2]
            self._state[6] = float(goal_heading)
            self._state[7] = timestamp
            self._state[8] += 1  # seq
            self._state[9] = 1.0  # ready flag

        self._new_obs_event.set()

    def wait_for_observation(self, timeout: Optional[float] = None) -> bool:
        return self._new_obs_event.wait(timeout=timeout)

    def read_observation(self, clear_event: bool = True) -> Optional[SimulationObservation]:
        if self._state[9] == 0.0:
            return None

        with self._obs_lock:
            image_copy = np.copy(self._image)
            robot_position = self._state[0:3].copy()
            robot_heading = float(self._state[3])
            goal_position = np.array([self._state[4], self._state[5], 0.0], dtype=np.float64)
            goal_heading = float(self._state[6])
            timestamp = float(self._state[7])
            seq = self._state[8]

        if clear_event:
            self._new_obs_event.clear()

        LOGGER.debug(
            "공유 메모리 Observation 읽음 seq=%.0f time=%.3f pos=(%.2f, %.2f)",
            seq,
            timestamp,
            robot_position[0],
            robot_position[1],
        )

        return SimulationObservation(
            image=image_copy,
            robot_position=robot_position,
            robot_heading=robot_heading,
            goal_position=goal_position,
            goal_heading=goal_heading,
            timestamp=timestamp,
        )

    # ---- Command path -----------------------------------------------------
    def write_command(self, command: np.ndarray) -> None:
        if command.shape != (CMD_LENGTH,):
            raise ValueError("명령 shape 은 (3,) 이어야 합니다.")
        with self._cmd_lock:
            np.copyto(self._command, command.astype(np.float64, copy=False))
        self._new_cmd_event.set()

    def read_command(self, clear_event: bool = True) -> np.ndarray:
        with self._cmd_lock:
            cmd = self._command.astype(np.float64).copy()
        if clear_event:
            self._new_cmd_event.clear()
        return cmd

    def wait_for_command(self, timeout: Optional[float] = None) -> bool:
        return self._new_cmd_event.wait(timeout=timeout)

    # ---- Lifecycle --------------------------------------------------------
    def close(self) -> None:
        for shm in (self._image_shm, self._state_shm, self._command_shm):
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# OmniVLA inference wrapper
# ---------------------------------------------------------------------------


@dataclass
class SimOmniVLAConfig:
    goal_image_path: str
    lan_inst_prompt: str = "navigate to the goal"
    save_dir: str = "./inference"
    metric_waypoint_spacing: float = 0.1
    distance_threshold: float = 30.0
    linear_clip: Tuple[float, float] = (0.0, 0.5)
    angular_clip: Tuple[float, float] = (-1.0, 1.0)
    max_linear: float = 0.3
    max_yaw: float = 0.3


class SimOmniVLAController:
    """OmniVLA 추론 결과를 Isaac Sim 명령으로 변환하는 워커."""

    def __init__(self, config: SimOmniVLAConfig):
        self.config = config
        self.goal_image = Image.open(config.goal_image_path).convert("RGB")

        # 모델 초기화
        self.inference_cfg = RUN_MODULE.InferenceConfig()
        (
            self.vla,
            self.action_head,
            self.pose_projector,
            self.device_id,
            self.num_patches,
            self.action_tokenizer,
            self.processor,
        ) = RUN_MODULE.define_model(self.inference_cfg)

        RUN_MODULE.lan_prompt = bool(config.lan_inst_prompt)

        self.inference = RUN_MODULE.Inference(
            save_dir=config.save_dir,
            lan_inst_prompt=config.lan_inst_prompt,
            goal_utm=[0.0, 0.0],
            goal_compass=0.0,
            goal_image_PIL=self.goal_image,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )

        self.metric_waypoint_spacing = config.metric_waypoint_spacing

    # ---- Core inference ---------------------------------------------------
    def _prepare_goal_pose(self, observation: SimulationObservation) -> np.ndarray:
        cur_xy = observation.robot_position[:2]
        goal_xy = observation.goal_position[:2]
        delta_x, delta_y = self.inference.calculate_relative_position(
            cur_xy[0], cur_xy[1], goal_xy[0], goal_xy[1]
        )

        relative_x, relative_y = self.inference.rotate_to_local_frame(
            delta_x, delta_y, observation.robot_heading
        )

        radius = np.linalg.norm([relative_x, relative_y])
        if radius > self.config.distance_threshold:
            scale = self.config.distance_threshold / radius
            relative_x *= scale
            relative_y *= scale

        goal_pose_loc_norm = np.array(
            [
                relative_y / self.metric_waypoint_spacing,
                -relative_x / self.metric_waypoint_spacing,
                math.cos(observation.goal_heading - observation.robot_heading),
                math.sin(observation.goal_heading - observation.robot_heading),
            ],
            dtype=np.float32,
        )
        return goal_pose_loc_norm

    def _forward_waypoints(self, observation: SimulationObservation) -> np.ndarray:
        goal_pose_loc_norm = self._prepare_goal_pose(observation)
        current_image = Image.fromarray(observation.image.astype(np.uint8)).convert("RGB")

        lan_inst = self.inference.lan_inst_prompt if RUN_MODULE.lan_prompt else "xxxx"
        batch = self.inference.data_transformer_omnivla(
            current_image,
            lan_inst,
            self.goal_image,
            goal_pose_loc_norm,
            prompt_builder=RUN_MODULE.PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )

        actions, _ = self.inference.run_forward_pass(
            vla=self.vla.eval(),
            action_head=self.action_head.eval(),
            noisy_action_projector=None,
            pose_projector=self.pose_projector.eval(),
            batch=batch,
            action_tokenizer=self.action_tokenizer,
            device_id=self.device_id,
            use_l1_regression=self.inference_cfg.use_l1_regression,
            use_diffusion=self.inference_cfg.use_diffusion,
            use_film=self.inference_cfg.use_film,
            num_patches=self.num_patches,
        )

        return actions.float().cpu().numpy()

    def _waypoints_to_command(self, waypoints: np.ndarray) -> Tuple[float, float]:
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= self.metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        eps = 1e-8
        dt = 1 / 3
        if abs(dx) < eps and abs(dy) < eps:
            linear_vel = 0.0
            angular_vel = clip_angle_fn(math.atan2(hy, hx)) / dt
        elif abs(dx) < eps:
            linear_vel = 0.0
            angular_vel = math.copysign(math.pi / (2 * dt), dy)
        else:
            linear_vel = dx / dt
            angular_vel = math.atan(dy / dx) / dt

        linear_vel = float(
            np.clip(linear_vel, self.config.linear_clip[0], self.config.linear_clip[1])
        )
        angular_vel = float(
            np.clip(angular_vel, self.config.angular_clip[0], self.config.angular_clip[1])
        )

        maxv, maxw = self.config.max_linear, self.config.max_yaw
        if abs(linear_vel) <= maxv:
            if abs(angular_vel) <= maxw:
                return linear_vel, angular_vel
            rd = linear_vel / angular_vel
            return maxw * math.copysign(abs(rd), linear_vel), maxw * math.copysign(1.0, angular_vel)

        if abs(angular_vel) <= 0.001:
            return maxv * math.copysign(1.0, linear_vel), 0.0

        rd = linear_vel / angular_vel
        if abs(rd) >= maxv / maxw:
            linear_vel_limit = maxv * math.copysign(1.0, linear_vel)
            angular_vel_limit = maxv * math.copysign(1.0, angular_vel) / abs(rd)
        else:
            linear_vel_limit = maxw * math.copysign(abs(rd), linear_vel)
            angular_vel_limit = maxw * math.copysign(1.0, angular_vel)
        return linear_vel_limit, angular_vel_limit

    def compute_command(self, observation: SimulationObservation) -> np.ndarray:
        try:
            self.inference.goal_utm = observation.goal_position[:2]
            self.inference.goal_compass = observation.goal_heading
            waypoints = self._forward_waypoints(observation)
            linear_vel, angular_vel = self._waypoints_to_command(waypoints)
            command = np.array([linear_vel, 0.0, angular_vel], dtype=np.float32)
            LOGGER.debug(
                "OmniVLA 명령 생성 lin=%.3f ang=%.3f", linear_vel, angular_vel
            )
            return command
        except Exception as exc:  # pragma: no cover - 안전장치
            LOGGER.error("OmniVLA 추론 실패: %s", exc, exc_info=True)
            return np.zeros(3, dtype=np.float32)

    # ---- Worker loop ------------------------------------------------------
    def serve_forever(
        self,
        channels: SharedMemoryChannels,
        stop_event: threading.Event,
        poll_timeout: float = 0.5,
    ) -> None:
        LOGGER.info("OmniVLA 추론 워커 시작")
        while not stop_event.is_set():
            has_obs = channels.wait_for_observation(timeout=poll_timeout)
            if not has_obs:
                continue

            observation = channels.read_observation()
            if observation is None:
                continue

            command = self.compute_command(observation)
            channels.write_command(command)

        LOGGER.info("OmniVLA 추론 워커 종료")


__all__ = [
    "SimulationObservation",
    "SharedMemoryChannels",
    "SimOmniVLAConfig",
    "SimOmniVLAController",
]

