"""
Isaac Sim Spot 환경과 OmniVLA 추론을 연결하는 시뮬레이션 스크립트.

기존 `quadruped_example.py` 환경을 재사용하되, 키보드 입력 대신
OmniVLA 추론 스레드에서 공유 메모리를 통해 전달되는 명령을 사용한다.
"""

from __future__ import annotations

import csv
import logging
import math
import signal
import sys
import threading
import time
from datetime import datetime
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

# Isaac Sim이 로깅을 가로채는 것을 방지하기 위해 명시적 핸들러 설정
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    LOGGER.addHandler(handler)
    LOGGER.propagate = False  # 상위 로거로 전파 방지


# -----------------------------------------------------------------------------
# CSV Logger for OmniVLA simulation data
# -----------------------------------------------------------------------------
class OmniVLACSVLogger:
    """OmniVLA 시뮬레이션 데이터를 CSV로 기록하는 클래스."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # CSV 헤더 작성
        header = [
            'timestamp',
            'robot_pos_x', 'robot_pos_y', 'robot_pos_z',
            'robot_heading',
            'goal_pos_x', 'goal_pos_y', 'goal_pos_z',
            'goal_heading',
            'linear_vel', 'angular_vel',
            'waypoint_dx', 'waypoint_dy', 'waypoint_hx', 'waypoint_hy',
            'cmd_vel_vx', 'cmd_vel_vy', 'cmd_vel_yaw',
        ]
        self.csv_writer.writerow(header)
        self.csv_file.flush()
        LOGGER.info(f"CSV 로거 초기화: {csv_path}")
    
    def log(self, timestamp, robot_position, robot_heading, goal_position, goal_heading,
            linear_vel, angular_vel, waypoint_dx, waypoint_dy, waypoint_hx, waypoint_hy,
            cmd_vel_vx=None, cmd_vel_vy=None, cmd_vel_yaw=None):
        """데이터를 CSV에 기록."""
        row = [
            timestamp,
            robot_position[0], robot_position[1], robot_position[2],
            robot_heading,
            goal_position[0], goal_position[1], goal_position[2],
            goal_heading,
            linear_vel, angular_vel,
            waypoint_dx, waypoint_dy, waypoint_hx, waypoint_hy,
            cmd_vel_vx if cmd_vel_vx is not None else 0.0,
            cmd_vel_vy if cmd_vel_vy is not None else 0.0,
            cmd_vel_yaw if cmd_vel_yaw is not None else 0.0,
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()
    
    def close(self):
        """CSV 파일 닫기."""
        if self.csv_file:
            self.csv_file.close()
            LOGGER.info(f"CSV 파일 닫힘: {self.csv_path}")


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
    def __init__(self, shared_channels: SharedMemoryChannels, shutdown_event=None, csv_logger=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_channels = shared_channels
        self._last_obs_publish = 0.0
        self.observation_period = 1.0 / 10.0  # 10 Hz
        self.shutdown_event = shutdown_event  # 종료 이벤트 참조
        self.csv_logger = csv_logger  # CSV 로거 인스턴스

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

        # Pygame 디스플레이 유지 (키보드 입력은 사용하지 않지만 시각화는 필요)
        # 기존 display는 super().setup()에서 이미 초기화됨
        # 키보드 입력 콜백은 None으로 설정 (OmniVLA가 제어하므로)
        if self.display:
            # 키보드 입력 콜백 제거 (OmniVLA가 제어)
            self.display.key_state_callback = None
            self.display.gate_transform_callback = None
            self.display.has_gate = False
            LOGGER.info("Pygame display 활성화 (시각화 전용)")

        # 기존 physics callback 대체
        if self.world.physics_callback_exists("physics_step"):
            self.world.remove_physics_callback("physics_step")
        self.world.add_physics_callback("physics_step", callback_fn=self._physics_step_with_bridge)

        LOGGER.info("OmniVLA Spot Simulation setup 완료")
        sys.stdout.flush()  # 로그 강제 출력

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

    def run(self):
        """
        Run main simulation loop with Pygame display updates.
        Overrides parent to add display frame updates.
        """
        if self.world is None or self.spot is None or self.controller is None:
            raise RuntimeError("Simulation must be setup first")

        self.logger.info("Starting OmniVLA simulation...")
        self.logger.info("Camera views displayed in Pygame window")
        sys.stdout.flush()

        # Initialize frame timing
        self.last_frame_time = time.time()

        # Main simulation loop
        while simulation_app.is_running():
            # Check for shutdown signal (Ctrl+C)
            if self.shutdown_event and self.shutdown_event.is_set():
                self.logger.info("Shutdown signal received - stopping simulation")
                break

            # Track frame time
            frame_start_time = time.time()

            # Check if quit requested
            if self.controller.is_quit_requested():
                self.logger.info("Quit requested - stopping simulation")
                break

            # Check if Pygame window was closed
            if self.display and not self.display.running:
                self.logger.info("Pygame window closed - quitting...")
                break

            # Step physics and rendering
            render_start_time = time.time()
            self.world.step(render=True)
            render_duration = (time.time() - render_start_time) * 1000
            if hasattr(self, 'render_times'):
                self.render_times.append(render_duration)
                if len(self.render_times) > 500:
                    self.render_times.pop(0)

            # Track overall frame time
            frame_duration = (time.time() - frame_start_time) * 1000
            if hasattr(self, 'frame_times'):
                self.frame_times.append(frame_duration)
                if len(self.frame_times) > 500:
                    self.frame_times.pop(0)

            # Update Pygame display with camera frames (every frame for smooth display)
            if self.display:
                ego_image = self.get_ego_camera_image()
                if ego_image is not None:
                    self.display.update_ego_frame(ego_image)

                top_image = self.get_top_camera_image()
                if top_image is not None:
                    self.display.update_top_frame(top_image)


# -----------------------------------------------------------------------------
# 실행 스크립트
# -----------------------------------------------------------------------------
def main():
    # 시그널 핸들러 설정 (Ctrl+C 안전 종료)
    stop_event = threading.Event()
    shutdown_requested = threading.Event()

    def signal_handler(signum, frame):
        """Ctrl+C 시그널 핸들러"""
        LOGGER.info("종료 신호 수신 (Ctrl+C) - 안전하게 종료 중...")
        sys.stdout.flush()
        shutdown_requested.set()
        stop_event.set()
        # 플래그만 설정하고, 실제 종료는 finally 블록에서 처리
        # (visualization이 실행되도록 보장하기 위함)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    LOGGER.info("[1/6] 공유 메모리 채널 초기화 중...")
    sys.stdout.flush()
    ego_width, ego_height = DEFAULT_CONFIG["ego_camera_resolution"]
    shared_channels = SharedMemoryChannels((ego_width, ego_height))
    LOGGER.info(f"[1/6] 완료: 공유 메모리 채널 ({ego_width}x{ego_height})")
    sys.stdout.flush()

    # 실험 디렉토리 생성
    experiment_name = "omnivla_sim"
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sim_data_dir = REPO_ROOT / "sim_data"
    sim_data_dir.mkdir(exist_ok=True)
    experiment_dir = sim_data_dir / f"{timestamp_str}-{experiment_name}"
    experiment_dir.mkdir(exist_ok=True)
    csv_path = experiment_dir / "data.csv"
    
    LOGGER.info(f"실험 디렉토리: {experiment_dir}")
    sys.stdout.flush()
    
    # CSV 로거 생성
    csv_logger = OmniVLACSVLogger(csv_path)
    
    LOGGER.info("[2/6] OmniVLA 모델 로딩 중... (시간이 걸릴 수 있습니다)")
    sys.stdout.flush()
    try:
        omnivla_cfg = SimOmniVLAConfig(
            goal_image_path=str(REPO_ROOT / "inference" / "frame211-56.597-ego.jpg"),
            lan_inst_prompt="move toward blue sphere",
        )
        LOGGER.info("[2/6] OmniVLA 설정 완료, 모델 인스턴스 생성 중...")
        sys.stdout.flush()
        omnivla_controller = SimOmniVLAController(omnivla_cfg, csv_logger=csv_logger)
        LOGGER.info("[2/6] 완료: OmniVLA 모델 로딩 완료")
        sys.stdout.flush()
    except Exception as e:
        LOGGER.error(f"[2/6] 실패: OmniVLA 모델 로딩 오류: {e}", exc_info=True)
        sys.stdout.flush()
        raise

    LOGGER.info("[3/6] OmniVLA 추론 스레드 시작 중...")
    sys.stdout.flush()
    omnivla_thread = threading.Thread(
        target=omnivla_controller.serve_forever,
        args=(shared_channels, stop_event),
        daemon=True,
    )
    omnivla_thread.start()
    LOGGER.info("[3/6] 완료: OmniVLA 추론 스레드 시작됨")
    sys.stdout.flush()

    LOGGER.info("[4/6] Isaac Sim 시뮬레이션 인스턴스 생성 중...")
    sys.stdout.flush()
    sim = OmniVLASpotSimulation(
        shared_channels=shared_channels,
        shutdown_event=shutdown_requested,
        csv_logger=csv_logger,
        experiment_name=experiment_name,
        enable_csv_logging=False,
        enable_image_saving=False,
        object_type="none",
    )
    LOGGER.info("[4/6] 완료: 시뮬레이션 인스턴스 생성됨")
    sys.stdout.flush()

    try:
        LOGGER.info("=" * 60)
        LOGGER.info("OmniVLA Isaac Sim 시뮬레이션 시작")
        LOGGER.info("=" * 60)
        sys.stdout.flush()

        LOGGER.info("[5/6] Isaac Sim 환경 설정 중... (로봇, 카메라, 물리 엔진 초기화)")
        sys.stdout.flush()
        sim.setup()
        LOGGER.info("[5/6] 완료: Isaac Sim 환경 설정 완료")
        sys.stdout.flush()

        LOGGER.info("[6/6] 시뮬레이션 루프 시작...")
        sys.stdout.flush()
        sim.run()
    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt 발생 - 종료 중...")
        sys.stdout.flush()
    except Exception as e:
        LOGGER.error(f"시뮬레이션 오류: {e}", exc_info=True)
        sys.stdout.flush()
    finally:
        LOGGER.info("시뮬레이션 종료 중...")
        sys.stdout.flush()

        # 안전한 종료 순서
        stop_event.set()
        shutdown_requested.set()

        # OmniVLA 스레드 종료 대기
        if omnivla_thread.is_alive():
            omnivla_thread.join(timeout=2.0)
            if omnivla_thread.is_alive():
                LOGGER.warning("OmniVLA 스레드가 제시간에 종료되지 않았습니다")

        # 시뮬레이션 정리
        try:
            sim.cleanup()
        except Exception as e:
            LOGGER.warning(f"Cleanup 중 오류: {e}")

        # CSV 파일 닫기
        try:
            csv_logger.close()
        except Exception as e:
            LOGGER.warning(f"CSV 로거 종료 중 오류: {e}")

        # 공유 메모리 정리
        try:
            shared_channels.close()
        except Exception as e:
            LOGGER.warning(f"공유 메모리 정리 중 오류: {e}")

        # Isaac Sim 종료
        try:
            simulation_app.close()
        except Exception as e:
            LOGGER.warning(f"SimulationApp 종료 중 오류: {e}")

        # Visualization 실행
        try:
            LOGGER.info("시각화 생성 중...")
            sys.stdout.flush()
            from simulations.visualize_simulation import visualize_simulation_results
            wall_size = DEFAULT_CONFIG.get("map_size", 10.0)
            visualize_simulation_results(
                csv_path=csv_path,
                wall_size=wall_size,
                save_path=experiment_dir
            )
            LOGGER.info(f"시각화 완료: {experiment_dir}")
            sys.stdout.flush()
        except Exception as e:
            LOGGER.warning(f"시각화 생성 중 오류: {e}", exc_info=True)
            sys.stdout.flush()

        LOGGER.info("모든 리소스 정리 완료")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
