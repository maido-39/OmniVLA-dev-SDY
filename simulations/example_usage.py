"""
SpotSimulation API 사용 예제

quadruped_example.py의 SpotSimulation 클래스를 다양한 방식으로 사용하는 예제입니다.
"""

import sys
import logging
from pathlib import Path

# 프로젝트 루트 경로 설정
FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[1]
ISAACSIM_DIR = REPO_ROOT / "extern" / "isaacsim-spot-remotecontroldemo"

# 모듈 경로에 추가
if str(ISAACSIM_DIR) not in sys.path:
    sys.path.insert(0, str(ISAACSIM_DIR))

from quadruped_example import SpotSimulation, simulation_app  # type: ignore


def example_1_basic():
    """예제 1: 기본 사용법"""
    print("=" * 60)
    print("예제 1: 기본 사용법")
    print("=" * 60)
    
    sim = SpotSimulation(
        experiment_name="example_basic",
        object_type="none",
        randomize=False
    )
    
    try:
        sim.setup()
        sim.run()
    finally:
        sim.cleanup()
        simulation_app.close()


def example_2_gate_navigation():
    """예제 2: Gate 네비게이션 태스크"""
    print("=" * 60)
    print("예제 2: Gate 네비게이션 태스크")
    print("=" * 60)
    
    sim = SpotSimulation(
        experiment_name="example_gate",
        object_type="gate",
        randomize=True,
        random_seed=42  # 재현성을 위한 고정 시드
    )
    
    try:
        sim.setup()
        sim.run()
    finally:
        sim.cleanup()
        simulation_app.close()


def example_3_box_pushing():
    """예제 3: 박스 푸싱 태스크 (여러 박스)"""
    print("=" * 60)
    print("예제 3: 박스 푸싱 태스크 (여러 박스)")
    print("=" * 60)
    
    sim = SpotSimulation(
        experiment_name="example_boxes",
        object_type="box",
        num_boxes=5,
        randomize=True,
        map_size=12.0,  # 더 큰 맵
        enable_csv_logging=True,
        enable_image_saving=True
    )
    
    try:
        sim.setup()
        sim.run()
    finally:
        sim.cleanup()
        simulation_app.close()


def example_4_performance_mode():
    """예제 4: 성능 최적화 모드 (로깅/저장 비활성화)"""
    print("=" * 60)
    print("예제 4: 성능 최적화 모드")
    print("=" * 60)
    
    sim = SpotSimulation(
        experiment_name="example_performance",
        object_type="none",
        randomize=False,
        enable_csv_logging=False,  # CSV 로깅 비활성화
        enable_image_saving=False,  # 이미지 저장 비활성화
        log_level=logging.WARNING  # 로깅 레벨 낮춤
    )
    
    try:
        sim.setup()
        sim.run()
    finally:
        sim.cleanup()
        simulation_app.close()


def example_5_custom_config():
    """예제 5: 커스텀 설정"""
    print("=" * 60)
    print("예제 5: 커스텀 설정")
    print("=" * 60)
    
    sim = SpotSimulation(
        experiment_name="example_custom",
        object_type="box",
        num_boxes=3,
        randomize=True,
        # 커스텀 설정
        map_size=15.0,
        max_vx=3.0,  # 더 빠른 최대 속도
        max_vy=3.0,
        max_yaw=3.0,
        # 컨트롤러 설정
        acc_vx=8.0,  # 더 빠른 가속도
        acc_vy=8.0,
        acc_yaw=15.0,
        # 카메라 설정
        ego_camera_resolution=[1280, 800],
        top_camera_resolution=[1024, 1024],
        top_camera_height=15.0
    )
    
    try:
        sim.setup()
        sim.run()
    finally:
        sim.cleanup()
        simulation_app.close()


def example_6_config_file():
    """예제 6: JSON 설정 파일 사용"""
    print("=" * 60)
    print("예제 6: JSON 설정 파일 사용")
    print("=" * 60)
    
    # 설정 파일 경로 (존재하는 경우)
    config_file = REPO_ROOT / "simulations" / "config.json"
    
    if config_file.exists():
        sim = SpotSimulation(
            config_file=str(config_file),
            experiment_name="example_config_file"
        )
    else:
        print(f"설정 파일을 찾을 수 없습니다: {config_file}")
        print("기본 설정으로 실행합니다.")
        sim = SpotSimulation(
            experiment_name="example_config_file",
            object_type="gate"
        )
    
    try:
        sim.setup()
        sim.run()
    finally:
        sim.cleanup()
        simulation_app.close()


def main():
    """메인 함수: 실행할 예제 선택"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SpotSimulation API 사용 예제",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
        help="실행할 예제 번호 (1-6)"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_basic,
        2: example_2_gate_navigation,
        3: example_3_box_pushing,
        4: example_4_performance_mode,
        5: example_5_custom_config,
        6: example_6_config_file
    }
    
    example_func = examples[args.example]
    example_func()


if __name__ == "__main__":
    main()

