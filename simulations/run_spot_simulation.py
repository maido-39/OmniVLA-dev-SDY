"""
Isaac Sim Spot 시뮬레이션 실행 스크립트

quadruped_example.py의 SpotSimulation 클래스를 API처럼 사용하여
시뮬레이션을 실행하는 간단한 래퍼 스크립트입니다.
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

# Isaac Sim 런타임 및 SpotSimulation 클래스 import
from quadruped_example import SpotSimulation, simulation_app  # type: ignore


def run_simulation(
    experiment_name: str = "test",
    object_type: str = "none",
    randomize: bool = True,
    enable_csv_logging: bool = True,
    enable_image_saving: bool = True,
    log_level: int = logging.INFO,
    **config_overrides
):
    """
    Spot 시뮬레이션을 실행하는 간단한 API 함수.
    
    Args:
        experiment_name: 실험 이름 (기본값: "test")
        object_type: 객체 타입 ("none", "box", "sphere", "gate")
        randomize: 랜덤화 활성화 여부 (기본값: True)
        enable_csv_logging: CSV 로깅 활성화 여부 (기본값: True)
        enable_image_saving: 이미지 저장 활성화 여부 (기본값: True)
        log_level: 로깅 레벨 (기본값: logging.INFO)
        **config_overrides: 추가 설정 오버라이드 (예: map_size=12.0, num_boxes=5)
    
    Examples:
        # 기본 실행
        run_simulation(experiment_name="my_experiment")
        
        # Gate 네비게이션 태스크
        run_simulation(
            experiment_name="gate_nav",
            object_type="gate",
            randomize=True
        )
        
        # 박스 푸싱 태스크 (성능 최적화)
        run_simulation(
            experiment_name="box_push",
            object_type="box",
            num_boxes=3,
            enable_csv_logging=False,
            enable_image_saving=False
        )
        
        # 커스텀 설정
        run_simulation(
            experiment_name="custom",
            map_size=15.0,
            max_vx=3.0,
            max_vy=3.0
        )
    """
    # 설정 오버라이드 준비
    config_overrides["object_type"] = object_type
    config_overrides["randomize"] = randomize
    
    # 시뮬레이션 인스턴스 생성
    sim = SpotSimulation(
        experiment_name=experiment_name,
        log_level=log_level,
        enable_csv_logging=enable_csv_logging,
        enable_image_saving=enable_image_saving,
        **config_overrides
    )
    
    try:
        # 시뮬레이션 설정 (환경, 로봇, 컨트롤러)
        sim.setup()
        
        # 시뮬레이션 실행
        sim.run()
        
    except KeyboardInterrupt:
        print("\n[INFO] 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"[ERROR] 시뮬레이션 오류: {e}")
        raise
    finally:
        # 리소스 정리
        sim.cleanup()
        simulation_app.close()
        print("[INFO] 시뮬레이션이 종료되었습니다.")


def main():
    """
    명령줄에서 실행할 때 사용하는 메인 함수.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Isaac Sim Spot 시뮬레이션 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예제:
  # 기본 실행
  python run_spot_simulation.py --experiment-name test
  
  # Gate 네비게이션
  python run_spot_simulation.py --experiment-name gate_nav --object-type gate
  
  # 박스 푸싱 (성능 최적화)
  python run_spot_simulation.py --experiment-name box_push --object-type box --no-csv --no-images
  
  # 커스텀 설정
  python run_spot_simulation.py --experiment-name custom --map-size 15.0 --max-vx 3.0
        """
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="test",
        help="실험 이름 (기본값: 'test')"
    )
    
    parser.add_argument(
        "--object-type",
        type=str,
        choices=["none", "box", "sphere", "gate"],
        default="none",
        help="객체 타입 (기본값: 'none')"
    )
    
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="랜덤화 비활성화"
    )
    
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="CSV 로깅 비활성화 (성능 향상)"
    )
    
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="이미지 저장 비활성화 (성능 향상)"
    )
    
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="로깅 레벨 (기본값: 'INFO')"
    )
    
    # 추가 설정 옵션들
    parser.add_argument("--map-size", type=float, help="맵 크기 (미터)")
    parser.add_argument("--num-boxes", type=int, help="박스 개수")
    parser.add_argument("--max-vx", type=float, help="최대 x 속도 (m/s)")
    parser.add_argument("--max-vy", type=float, help="최대 y 속도 (m/s)")
    parser.add_argument("--max-yaw", type=float, help="최대 yaw 속도 (rad/s)")
    parser.add_argument("--random-seed", type=int, help="랜덤 시드 (재현성)")
    
    args = parser.parse_args()
    
    # 로깅 레벨 변환
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = log_level_map[args.loglevel.upper()]
    
    # 설정 오버라이드 준비
    config_overrides = {}
    if args.map_size is not None:
        config_overrides["map_size"] = args.map_size
    if args.num_boxes is not None:
        config_overrides["num_boxes"] = args.num_boxes
    if args.max_vx is not None:
        config_overrides["max_vx"] = args.max_vx
    if args.max_vy is not None:
        config_overrides["max_vy"] = args.max_vy
    if args.max_yaw is not None:
        config_overrides["max_yaw"] = args.max_yaw
    if args.random_seed is not None:
        config_overrides["random_seed"] = args.random_seed
    
    # 시뮬레이션 실행
    run_simulation(
        experiment_name=args.experiment_name,
        object_type=args.object_type,
        randomize=not args.no_randomize,
        enable_csv_logging=not args.no_csv,
        enable_image_saving=not args.no_images,
        log_level=log_level,
        **config_overrides
    )


if __name__ == "__main__":
    main()

