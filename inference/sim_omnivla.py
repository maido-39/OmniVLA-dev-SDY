"""
Simulation-oriented OmniVLA inference utilities.

이 모듈은 Isaac Sim 기반 사족보행 시뮬레이션과 OmniVLA 추론 간의
공유 메모리 채널 및 추론 워커를 제공합니다.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image


LOGGER = logging.getLogger("SimOmniVLA")
LOGGER.setLevel(logging.INFO)

# 별도 스레드에서 실행되므로 명시적 핸들러 설정 필요
# 기존 핸들러 제거 후 새로 추가 (중복 방지)
LOGGER.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(
    logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
)
LOGGER.addHandler(handler)
LOGGER.propagate = False  # 상위 로거로 전파 방지

# Isaac Sim이 stdout을 가로채는 경우를 대비해 stderr 핸들러도 추가
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.INFO)
stderr_handler.setFormatter(
    logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
)
LOGGER.addHandler(stderr_handler)


# ---------------------------------------------------------------------------
# Lazy loading for run_omnivla-ARIL.py (Isaac Sim과의 CUDA 충돌 방지)
# ---------------------------------------------------------------------------
RUN_MODULE = None
RUN_SCRIPT_PATH = Path(__file__).with_name("run_omnivla-ARIL.py")


def _load_run_module():
    """지연 로딩: 실제로 사용할 때만 모듈을 로드"""
    global RUN_MODULE
    if RUN_MODULE is not None:
        return RUN_MODULE
    
    LOGGER.info("  → run_omnivla-ARIL.py 모듈 로딩 중...")
    sys.stdout.flush()
    
    try:
        SPEC = importlib.util.spec_from_file_location("run_omnivla_ARIL", RUN_SCRIPT_PATH)
        if SPEC is None or SPEC.loader is None:
            raise ImportError(f"run_omnivla-ARIL.py 를 로드할 수 없습니다: {RUN_SCRIPT_PATH}")
        RUN_MODULE = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(RUN_MODULE)
        
        # 기본 modality 설정
        RUN_MODULE.pose_goal = False
        RUN_MODULE.satellite = False
        RUN_MODULE.image_goal = False
        RUN_MODULE.lan_prompt = True
        
        LOGGER.info("  → 모듈 로딩 완료")
        sys.stdout.flush()
        return RUN_MODULE
    except Exception as e:
        LOGGER.error(f"  → 모듈 로딩 실패: {e}", exc_info=True)
        sys.stdout.flush()
        raise


# run_omnivla-ARIL 에 clip_angle 정의가 없을 경우 대비
def clip_angle_fn(angle: float) -> float:
    """[-pi, pi] 범위로 wrapping."""
    if RUN_MODULE is not None and hasattr(RUN_MODULE, "clip_angle"):
        return RUN_MODULE.clip_angle(angle)
    return (angle + math.pi) % (2 * math.pi) - math.pi


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
    yaw_gain: float = 1.5  # 각속도 반응 강화를 위한 배율 (기본값 1.5)


class SimOmniVLAController:
    """OmniVLA 추론 결과를 Isaac Sim 명령으로 변환하는 워커."""

    def __init__(self, config: SimOmniVLAConfig, csv_logger=None):
        self.config = config
        self.csv_logger = csv_logger  # CSV 로거 참조
        LOGGER.info("  → Goal 이미지 로딩 중...")
        sys.stdout.flush()
        self.goal_image = Image.open(config.goal_image_path).convert("RGB")
        LOGGER.info("  → Goal 이미지 로딩 완료")
        sys.stdout.flush()

        # 모듈 지연 로딩 (Isaac Sim 초기화 후)
        run_module = _load_run_module()

        # 모델 초기화
        LOGGER.info("  → InferenceConfig 생성 중...")
        sys.stdout.flush()
        self.inference_cfg = run_module.InferenceConfig()
        
        # vla_path를 절대 경로로 변환 (상대 경로 문제 해결)
        if self.inference_cfg.vla_path.startswith("./"):
            # REPO_ROOT 기준으로 절대 경로 생성
            REPO_ROOT = Path(__file__).parent.parent
            vla_path_relative = self.inference_cfg.vla_path[2:]  # "./" 제거
            self.inference_cfg.vla_path = str(REPO_ROOT / vla_path_relative)
        
        LOGGER.info(f"  → 모델 경로: {self.inference_cfg.vla_path}")
        LOGGER.info("  → 모델 로딩 중... (GPU 메모리 할당 및 가중치 로딩, 시간이 걸릴 수 있습니다)")
        sys.stdout.flush()
        
        (
            self.vla,
            self.action_head,
            self.pose_projector,
            self.device_id,
            self.num_patches,
            self.action_tokenizer,
            self.processor,
        ) = run_module.define_model(self.inference_cfg)
        LOGGER.info("  → 모델 로딩 완료")
        sys.stdout.flush()

        run_module.lan_prompt = bool(config.lan_inst_prompt)
        LOGGER.info(f"  → Language prompt 활성화: {run_module.lan_prompt}")
        sys.stdout.flush()

        LOGGER.info("  → Inference 인스턴스 생성 중...")
        sys.stdout.flush()
        self.inference = run_module.Inference(
            save_dir=config.save_dir,
            lan_inst_prompt=config.lan_inst_prompt,
            goal_utm=[0.0, 0.0],
            goal_compass=0.0,
            goal_image_PIL=self.goal_image,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )
        LOGGER.info("  → Inference 인스턴스 생성 완료")
        sys.stdout.flush()

        self.metric_waypoint_spacing = config.metric_waypoint_spacing
        self.run_module = run_module  # 나중에 사용하기 위해 저장

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
        # 디버깅: 좌표계 변환 확인
        LOGGER.debug(
            "Goal pose: relative=(%.3f, %.3f), normalized=(%.3f, %.3f), "
            "robot_heading=%.3f, goal_heading=%.3f",
            relative_x, relative_y,
            goal_pose_loc_norm[0], goal_pose_loc_norm[1],
            observation.robot_heading, observation.goal_heading
        )
        return goal_pose_loc_norm

    def _forward_waypoints(self, observation: SimulationObservation) -> np.ndarray:
        goal_pose_loc_norm = self._prepare_goal_pose(observation)
        current_image = Image.fromarray(observation.image.astype(np.uint8)).convert("RGB")

        lan_inst = self.inference.lan_inst_prompt if self.run_module.lan_prompt else "xxxx"
        batch = self.inference.data_transformer_omnivla(
            current_image,
            lan_inst,
            self.goal_image,
            goal_pose_loc_norm,
            prompt_builder=self.run_module.PurePromptBuilder,
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
        # Yaw gain: 각속도 반응을 강화하기 위한 배율 (기본값 1.5)
        yaw_gain = getattr(self.config, 'yaw_gain', 1.5)
        
        if abs(dx) < eps and abs(dy) < eps:
            linear_vel = 0.0
            angular_vel = clip_angle_fn(math.atan2(hy, hx)) / dt * yaw_gain
        elif abs(dx) < eps:
            linear_vel = 0.0
            angular_vel = math.copysign(math.pi / (2 * dt), dy) * yaw_gain
        else:
            linear_vel = dx / dt
            angular_vel = math.atan(dy / dx) / dt * yaw_gain

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

    def compute_command(self, observation: SimulationObservation) -> Tuple[np.ndarray, np.ndarray]:
        """
        명령 계산 및 waypoints 반환.
        
        Returns:
            command: [linear_vel, 0.0, angular_vel]
            waypoints: 예측된 waypoints (batch, num_waypoints, 4)
        """
        try:
            self.inference.goal_utm = observation.goal_position[:2]
            self.inference.goal_compass = observation.goal_heading
            waypoints = self._forward_waypoints(observation)
            linear_vel, angular_vel = self._waypoints_to_command(waypoints)
            command = np.array([linear_vel, 0.0, angular_vel], dtype=np.float32)
            
            # 매 계산마다 cmd_vel 로깅
            waypoint_select = 4
            chosen_waypoint = waypoints[0][waypoint_select].copy()
            chosen_waypoint[:2] *= self.metric_waypoint_spacing
            
            # CSV 로깅
            if self.csv_logger:
                self.csv_logger.log(
                    timestamp=observation.timestamp,
                    robot_position=observation.robot_position,
                    robot_heading=observation.robot_heading,
                    goal_position=observation.goal_position,
                    goal_heading=observation.goal_heading,
                    linear_vel=linear_vel,
                    angular_vel=angular_vel,
                    waypoint_dx=chosen_waypoint[0],
                    waypoint_dy=chosen_waypoint[1],
                    waypoint_hx=chosen_waypoint[2],
                    waypoint_hy=chosen_waypoint[3],
                    cmd_vel_vx=command[0],
                    cmd_vel_vy=command[1],
                    cmd_vel_yaw=command[2],
                )
            
            # cmd_vel 로깅 (여러 방법으로 출력 시도)
            log_msg = (
                f"[OmniVLA cmd_vel] linear={linear_vel:.4f} m/s, angular={angular_vel:.4f} rad/s | "
                f"waypoint[{waypoint_select}]=(dx={chosen_waypoint[0]:.3f}, dy={chosen_waypoint[1]:.3f}, "
                f"hx={chosen_waypoint[2]:.3f}, hy={chosen_waypoint[3]:.3f}) | "
                f"robot_pos=({observation.robot_position[0]:.2f}, {observation.robot_position[1]:.2f}), "
                f"goal_pos=({observation.goal_position[0]:.2f}, {observation.goal_position[1]:.2f})"
            )
            LOGGER.info(log_msg)
            # Isaac Sim이 stdout을 가로채는 경우를 대비해 print도 사용
            print(log_msg, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            return command, waypoints
        except Exception as exc:  # pragma: no cover - 안전장치
            LOGGER.error("OmniVLA 추론 실패: %s", exc, exc_info=True)
            sys.stdout.flush()
            return np.zeros(3, dtype=np.float32), np.zeros((1, 8, 4), dtype=np.float32)

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

