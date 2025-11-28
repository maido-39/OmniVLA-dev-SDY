"""
Isaac Sim Hospital 환경과 OmniVLA 추론을 연결하는 시뮬레이션 스크립트.

Hospital USD 환경을 로드하고 Spot robot을 OmniVLA로 제어합니다.
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
# quadruped_example.py에서 simulation_app을 import하여 동일한 인스턴스 사용
from quadruped_example import simulation_app  # type: ignore  # noqa: E402

# SimulationApp 초기화 후에만 다른 모듈을 import할 수 있습니다.
import omni
from isaacsim.core.api import World
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from pxr import Gf, UsdGeom

# PygameDualCameraDisplay도 여기서 import (SimulationApp 이후)
from quadruped_example import PygameDualCameraDisplay  # type: ignore  # noqa: E402

# OmniVLA 추론 래퍼
from inference.sim_omnivla import (  # noqa: E402
    SharedMemoryChannels,
    SimOmniVLAConfig,
    SimOmniVLAController,
)

# -----------------------------------------------------------------------------
# 시뮬레이션 설정 (Configuration)
# -----------------------------------------------------------------------------
# 초기 언어 프롬프트 설정
INITIAL_PROMPT = "navigate to the reception desk"

# Robot 초기 설정
ROBOT_INITIAL_POSITION = [0.0, 0.0, 0.8]  # [x, y, z] in meters
ROBOT_INITIAL_HEADING = math.radians(135.0)  # Yaw angle in radians (135° = facing southeast)

# Hospital USD 파일 경로
HOSPITAL_USD_PATH = REPO_ROOT / "asset" / "Hospital_Flat.usd"

# Goal 위치 설정 (Hospital 환경에서 목표 지점)
GOAL_POSITION = [5.0, 0.0, 0.0]  # [x, y, z] in meters

LOGGER = logging.getLogger("HospitalQuadrupedSim")
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
# Hospital Spot Simulation
# -----------------------------------------------------------------------------
class HospitalSpotSimulation:
    """Hospital 환경에서 Spot robot을 OmniVLA로 제어하는 시뮬레이션 클래스."""
    
    def __init__(self, shared_channels: SharedMemoryChannels, shutdown_event=None, 
                 csv_logger=None):
        self.shared_channels = shared_channels
        self.shutdown_event = shutdown_event
        self.csv_logger = csv_logger
        
        # Isaac Sim components
        self.world = None
        self.stage = None
        self.spot = None
        self.controller = None
        
        # Simulation state
        self.physics_ready = False
        self.command_counter = 0
        self._last_obs_publish = 0.0
        self.observation_period = 1.0 / 10.0  # 10 Hz
        
        # Camera
        self.robot_camera_path = None
        self.camera_path = None  # Top camera
        self.render_products = {}
        self.rgb_annotators = {}
        self.camera_render_initialized = False
        
        # Display (Pygame)
        self.display = None
        
        # Goal position
        self.goal_pos = np.array(GOAL_POSITION)
        
        # Logger
        self.logger = LOGGER
    
    def initialize(self):
        """Initialize Isaac Sim world and stage."""
        # Create World: physics at 500Hz, rendering at 50Hz
        self.world = World(physics_dt=1.0/500.0, rendering_dt=10.0/500.0, stage_units_in_meters=1.0)
        # Get USD stage for scene manipulation
        self.stage = omni.usd.get_context().get_stage()
        self.logger.info("World initialized")
    
    def load_hospital_environment(self):
        """Load Hospital USD file."""
        if self.stage is None:
            raise RuntimeError("Stage must be initialized first")
        
        hospital_usd_path = str(HOSPITAL_USD_PATH.resolve())
        
        if not HOSPITAL_USD_PATH.exists():
            raise FileNotFoundError(f"Hospital USD file not found: {hospital_usd_path}")
        
        self.logger.info(f"Loading Hospital USD file: {hospital_usd_path}")
        
        # Open USD stage (replaces current stage)
        # Note: This will replace the default stage, so we need to load it as a reference instead
        # Using add_reference_to_stage to add it to the existing stage
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
            hospital_prim_path = "/World/Hospital"
            add_reference_to_stage(hospital_usd_path, hospital_prim_path)
            self.logger.info(f"Hospital environment loaded at: {hospital_prim_path}")
        except Exception as e:
            # Fallback: try opening the stage directly
            self.logger.warning(f"add_reference_to_stage failed: {e}, trying open_stage...")
            try:
                omni.usd.get_context().open_stage(hospital_usd_path)
                self.stage = omni.usd.get_context().get_stage()
                self.logger.info("Hospital USD stage opened directly")
            except Exception as e2:
                self.logger.error(f"Failed to load Hospital USD: {e2}")
                raise
    
    def setup_environment(self):
        """Setup environment: load Hospital USD and create cameras."""
        if self.world is None or self.stage is None:
            raise RuntimeError("World must be initialized first")
        
        # Load Hospital environment
        self.load_hospital_environment()
        
        # Add dome light for scene illumination (if not already in USD)
        try:
            light_prim = self.stage.GetPrimAtPath("/World/DomeLight")
            if not light_prim.IsValid():
                import omni.kit.commands
                omni.kit.commands.execute("CreatePrim", prim_path="/World/DomeLight", prim_type="DomeLight")
                light_prim = self.stage.GetPrimAtPath("/World/DomeLight")
                if light_prim.IsValid():
                    light_prim.GetAttribute("inputs:intensity").Set(600.0)
        except Exception as e:
            self.logger.warning(f"Failed to create dome light: {e}")
        
        # Create top-down camera
        self._create_top_camera()
        
        self.logger.info("Environment setup complete")
    
    def _create_top_camera(self):
        """Create top-down camera for visualization."""
        top_res = (1600, 1600)  # Square aspect ratio
        
        # Calculate quaternion for -90° rotation around Z axis
        rotation_z = -np.pi / 2.0
        half_angle = rotation_z / 2.0
        rotation_quat = (
            float(np.cos(half_angle)),  # w
            0.0,                        # x
            0.0,                        # y
            float(np.sin(half_angle))   # z
        )
        
        camera_path = self._create_camera(
            parent_path="/World",
            camera_name="TopCamera",
            translation=(0.0, 0.0, 20.0),  # 20m above origin
            rotation_quat=rotation_quat,  # -90° around Z axis
            focal_length=24.0,  # mm
            horizontal_aperture=36.0,  # mm
            resolution=top_res,
            clipping_range=(0.1, 100.0)
        )
        
        if camera_path:
            self.camera_path = camera_path
            self.logger.info(f"✓ Top camera created: {camera_path}")
        else:
            self.logger.warning("Failed to create top camera")
    
    def _create_camera(self, parent_path, camera_name, translation, rotation_quat, 
                       focal_length, horizontal_aperture, resolution, 
                       clipping_range=(0.01, 10000.0)):
        """General camera creation function."""
        try:
            # Create transform prim for camera
            camera_prim_path = f"{parent_path}/{camera_name}_Prim"
            camera_prim = UsdGeom.Xform.Define(self.stage, camera_prim_path)
            
            # Apply transformations
            camera_xform = UsdGeom.Xformable(camera_prim)
            camera_xform.ClearXformOpOrder()
            
            # Add translation
            translate_op = camera_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3f(*translation))
            
            # Add rotation
            rotate_op = camera_xform.AddOrientOp()
            rotate_op.Set(Gf.Quatf(*rotation_quat))
            
            # Create camera
            camera_path = f"{camera_prim_path}/{camera_name}"
            camera = UsdGeom.Camera.Define(self.stage, camera_path)
            
            # Calculate vertical aperture from resolution aspect ratio
            aspect_ratio = resolution[1] / resolution[0]  # height / width
            vertical_aperture = horizontal_aperture * aspect_ratio
            
            # Set camera intrinsic parameters
            camera.GetFocalLengthAttr().Set(focal_length)
            camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
            camera.GetVerticalApertureAttr().Set(vertical_aperture)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(*clipping_range))
            
            return camera_path
            
        except Exception as e:
            self.logger.error(f"Failed to create camera {camera_name}: {e}")
            return None
    
    def setup_robot(self):
        """Setup Spot robot at initial position with initial heading."""
        # Convert heading to quaternion
        half_yaw = ROBOT_INITIAL_HEADING / 2.0
        orientation = np.array([
            np.cos(half_yaw),  # w
            0.0,                # x
            0.0,                # y
            np.sin(half_yaw)   # z
        ])
        
        # Add Spot robot to scene
        self.spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            position=np.array([ROBOT_INITIAL_POSITION[0], ROBOT_INITIAL_POSITION[1], ROBOT_INITIAL_POSITION[2]]),
            orientation=orientation,
        )
        
        self.logger.info(f"Robot placed at [{ROBOT_INITIAL_POSITION[0]:.2f}, {ROBOT_INITIAL_POSITION[1]:.2f}, {ROBOT_INITIAL_POSITION[2]:.2f}] with heading {np.degrees(ROBOT_INITIAL_HEADING):.2f}°")
    
    def _setup_robot_camera(self):
        """Create ego-view camera attached to Spot robot body."""
        try:
            # Check if robot body exists
            body_path = "/World/Spot/body"
            body_prim = self.stage.GetPrimAtPath(body_path)
            if not body_prim.IsValid():
                self.logger.warning(f"Robot body prim not found at {body_path}")
                return
            
            # Get camera configuration
            ego_res = (640, 480)  # Ego camera resolution
            
            # Create ego camera
            camera_path = self._create_camera(
                parent_path=body_path,
                camera_name="EgoCamera",
                translation=(0.3, 0.0, 0.2),  # Position relative to robot body
                rotation_quat=(0.5, 0.5, -0.5, -0.5),  # RPY (90°, -90°, 0°)
                focal_length=18.0,  # mm
                horizontal_aperture=36.0,  # mm
                resolution=ego_res,
                clipping_range=(0.01, 10000.0)
            )
            
            if camera_path:
                self.robot_camera_path = camera_path
            else:
                self.logger.warning("Failed to create ego camera")
            
        except Exception as e:
            self.logger.error(f"Failed to setup robot camera: {e}")
    
    def _initialize_camera_render_products(self):
        """Initialize render products and annotators for cameras."""
        try:
            import omni.replicator.core as rep
            
            # Initialize render products dictionary and annotators
            self.render_products = {}
            self.rgb_annotators = {}
            
            # Setup ego camera render product
            if hasattr(self, 'robot_camera_path') and self.robot_camera_path:
                ego_res = (640, 480)
                self.render_products["ego"] = rep.create.render_product(
                    self.robot_camera_path, 
                    ego_res
                )
                self.rgb_annotators["ego"] = rep.AnnotatorRegistry.get_annotator("rgb")
                self.rgb_annotators["ego"].attach([self.render_products["ego"]])
                self.logger.info(f"✓ Ego camera render product initialized: {ego_res[0]}×{ego_res[1]}")
            
            # Setup top camera render product
            if hasattr(self, 'camera_path') and self.camera_path:
                top_res = (1600, 1600)
                self.render_products["top"] = rep.create.render_product(
                    self.camera_path, 
                    top_res
                )
                self.rgb_annotators["top"] = rep.AnnotatorRegistry.get_annotator("rgb")
                self.rgb_annotators["top"].attach([self.render_products["top"]])
                self.logger.info(f"✓ Top camera render product initialized: {top_res[0]}×{top_res[1]}")
            
            self.camera_render_initialized = True
            
        except ImportError as e:
            self.logger.warning(f"Replicator not available, camera capture disabled: {e}")
            self.camera_render_initialized = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize camera render products: {e}")
            self.camera_render_initialized = False
    
    def get_ego_camera_image(self):
        """Get current frame from ego camera."""
        if not self.camera_render_initialized:
            return None
        
        if "ego" not in self.rgb_annotators:
            return None
        
        try:
            rgb_data = self.rgb_annotators["ego"].get_data()
            if rgb_data is None:
                return None
            
            # Convert to numpy array
            image_array = np.asarray(rgb_data)
            
            # Check if we have valid image data
            if image_array is None or image_array.size == 0:
                return None
            
            # Handle RGBA to RGB conversion if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # Convert RGBA to RGB
                image_array = image_array[:, :, :3]
            
            # Ensure RGB format (not BGR)
            return image_array
                
        except Exception as e:
            # Silently return None on error (avoid spam during shutdown)
            return None
    
    def get_top_camera_image(self):
        """Get current frame from top camera."""
        if not self.camera_render_initialized:
            return None
        
        if "top" not in self.rgb_annotators:
            return None
        
        try:
            rgb_data = self.rgb_annotators["top"].get_data()
            if rgb_data is None:
                return None
            
            # Convert to numpy array
            image_array = np.asarray(rgb_data)
            
            # Check if we have valid image data
            if image_array is None or image_array.size == 0:
                return None
            
            # Handle RGBA to RGB conversion if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # Convert RGBA to RGB
                image_array = image_array[:, :, :3]
            
            # Ensure RGB format (not BGR)
            return image_array
            
        except Exception as e:
            # Silently return None on error (avoid spam during shutdown)
            return None
    
    @staticmethod
    def quaternion_to_rpy(q):
        """Convert quaternion [w, x, y, z] to Euler angles RPY [roll, pitch, yaw] in radians."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Roll (rotation around X-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (rotation around Y-axis)
        sinp = 2 * (w * y - z * x)
        pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
        
        # Yaw (rotation around Z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def _physics_step_with_bridge(self, step_size):
        """Physics step callback with OmniVLA bridge."""
        # Command update: update controller at 50Hz (every 10 physics steps)
        self.command_counter += 1
        if self.command_counter >= 10:
            self.command_counter = 0
            if self.controller:
                self.controller.update()
        
        # Robot control: apply commands to robot
        if self.physics_ready:
            # Robot is initialized, apply forward control with current command
            if self.controller:
                self.spot.forward(step_size, self.controller.get_command())
        else:
            # First physics step: initialize robot
            self.physics_ready = True
            self.spot.initialize()  # Initialize robot policy
            self.spot.post_reset()  # Post-reset setup
            self.spot.robot.set_joints_default_state(self.spot.default_pos)  # Set default joint positions
            self.logger.info("Spot initialized")

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
    
    def setup(self):
        """Complete simulation setup."""
        # Initialize world if not already done
        if self.world is None:
            self.initialize()
        
        # Setup environment (load Hospital USD, create cameras)
        self.setup_environment()
        
        # Setup robot at initial position
        self.setup_robot()
        
        # Create shared memory controller
        self.controller = SharedMemoryCommandController(
            self.shared_channels,
            max_vx=2.0,
            max_vy=2.0,
            max_yaw=2.0,
        )
        
        # Reset world (required before querying articulation properties)
        self.world.reset()
        
        # Create ego-view camera attached to robot base (after world reset, robot is fully initialized)
        self._setup_robot_camera()
        
        # Initialize camera render products (once, after cameras are set up)
        self._initialize_camera_render_products()
        
        # Initialize Pygame display (PygameDualCameraDisplay is already imported at top level)
        window_width = 1600
        window_height = 800
        ego_res = (640, 480)
        
        self.display = PygameDualCameraDisplay(
            window_size=(window_width, window_height),
            window_title="Hospital Spot Robot - Ego & Top View (OmniVLA Control)",
            key_state_callback=None,  # No keyboard control (OmniVLA controls)
            ego_camera_resolution=ego_res,
            gate_transform_callback=None
        )
        self.display.start()
        
        # Register physics callback
        self.world.add_physics_callback("physics_step", callback_fn=self._physics_step_with_bridge)
        
        self.logger.info("Hospital Spot Simulation setup 완료")
        sys.stdout.flush()
    
    def run(self):
        """Run main simulation loop."""
        if self.world is None or self.spot is None or self.controller is None:
            raise RuntimeError("Simulation must be setup first")
        
        self.logger.info("Starting Hospital OmniVLA simulation...")
        self.logger.info("Camera views displayed in Pygame window")
        sys.stdout.flush()
        
        # Main simulation loop
        while simulation_app.is_running():
            # Check for shutdown signal (Ctrl+C)
            if self.shutdown_event and self.shutdown_event.is_set():
                self.logger.info("Shutdown signal received - stopping simulation")
                break
            
            # Check if quit requested
            if self.controller.is_quit_requested():
                self.logger.info("Quit requested - stopping simulation")
                break
            
            # Check if Pygame window was closed
            if self.display and not self.display.running:
                self.logger.info("Pygame window closed - quitting...")
                break
            
            # Step physics and rendering
            self.world.step(render=True)
            
            # Update Pygame display with camera frames
            if self.display:
                ego_image = self.get_ego_camera_image()
                if ego_image is not None:
                    self.display.update_ego_frame(ego_image)
                
                top_image = self.get_top_camera_image()
                if top_image is not None:
                    self.display.update_top_frame(top_image)
    
    def cleanup(self):
        """Cleanup resources."""
        # Stop Pygame display
        if self.display:
            self.display.stop()
        
        # Cleanup camera render products and annotators
        if hasattr(self, 'rgb_annotators') and hasattr(self, 'render_products'):
            try:
                for camera_type, annotator in self.rgb_annotators.items():
                    if camera_type in self.render_products:
                        annotator.detach([self.render_products[camera_type]])
                self.logger.info("Camera annotators detached")
            except Exception as e:
                self.logger.debug(f"Error cleaning up camera annotators: {e}")
        
        # Remove physics callback
        if self.world and self.world.physics_callback_exists("physics_step"):
            self.world.remove_physics_callback("physics_step")
        
        self.logger.info("Cleanup complete")


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

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    LOGGER.info("[1/6] 공유 메모리 채널 초기화 중...")
    sys.stdout.flush()
    ego_width, ego_height = 640, 480
    shared_channels = SharedMemoryChannels((ego_width, ego_height))
    LOGGER.info(f"[1/6] 완료: 공유 메모리 채널 ({ego_width}x{ego_height})")
    sys.stdout.flush()

    # 실험 디렉토리 생성
    experiment_name = "hospital_omnivla_sim"
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
            lan_inst_prompt=INITIAL_PROMPT,
            yaw_gain=2.5,
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

    LOGGER.info("[4/6] Hospital Isaac Sim 시뮬레이션 인스턴스 생성 중...")
    sys.stdout.flush()
    sim = HospitalSpotSimulation(
        shared_channels=shared_channels,
        shutdown_event=shutdown_requested,
        csv_logger=csv_logger,
    )
    LOGGER.info("[4/6] 완료: 시뮬레이션 인스턴스 생성됨")
    sys.stdout.flush()

    try:
        LOGGER.info("=" * 60)
        LOGGER.info("Hospital OmniVLA Isaac Sim 시뮬레이션 시작")
        LOGGER.info("=" * 60)
        sys.stdout.flush()

        LOGGER.info("[5/6] Isaac Sim 환경 설정 중... (Hospital USD 로드, 로봇, 카메라 초기화)")
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

        LOGGER.info("모든 리소스 정리 완료")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

