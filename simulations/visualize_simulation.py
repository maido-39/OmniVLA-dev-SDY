"""
OmniVLA 시뮬레이션 결과 시각화 모듈

CSV 데이터를 읽어서 trajectory, cmd_vel, yaw 등을 시각화합니다.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np


def load_simulation_csv(csv_path: Path) -> dict:
    """
    시뮬레이션 CSV 파일을 로드합니다.
    
    Returns:
        dict: {
            'timestamps': array,
            'robot_positions': array (N, 2),
            'robot_headings': array (N,),
            'goal_positions': array (N, 2),
            'goal_headings': array (N,),
            'linear_vels': array (N,),
            'angular_vels': array (N,),
            'waypoints': array (N, 8, 4),  # (N, num_waypoints, 4)
            ...
        }
    """
    data = {
        'timestamps': [],
        'robot_pos_x': [],
        'robot_pos_y': [],
        'robot_heading': [],
        'goal_pos_x': [],
        'goal_pos_y': [],
        'goal_heading': [],
        'linear_vel': [],
        'angular_vel': [],
        'waypoint_dx': [],
        'waypoint_dy': [],
        'waypoint_hx': [],
        'waypoint_hy': [],
    }
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['timestamps'].append(float(row['timestamp']))
            data['robot_pos_x'].append(float(row['robot_pos_x']))
            data['robot_pos_y'].append(float(row['robot_pos_y']))
            data['robot_heading'].append(float(row['robot_heading']))
            data['goal_pos_x'].append(float(row['goal_pos_x']))
            data['goal_pos_y'].append(float(row['goal_pos_y']))
            data['goal_heading'].append(float(row['goal_heading']))
            data['linear_vel'].append(float(row['linear_vel']))
            data['angular_vel'].append(float(row['angular_vel']))
            data['waypoint_dx'].append(float(row['waypoint_dx']))
            data['waypoint_dy'].append(float(row['waypoint_dy']))
            data['waypoint_hx'].append(float(row['waypoint_hx']))
            data['waypoint_hy'].append(float(row['waypoint_hy']))
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    # Compute derived data
    data['robot_positions'] = np.column_stack([data['robot_pos_x'], data['robot_pos_y']])
    data['goal_positions'] = np.column_stack([data['goal_pos_x'], data['goal_pos_y']])
    data['times'] = data['timestamps'] - data['timestamps'][0]  # Relative time
    
    return data


def visualize_simulation_results(csv_path: Path, wall_size: float = 10.0, save_path: Optional[Path] = None):
    """
    시뮬레이션 결과를 시각화합니다.
    
    Args:
        csv_path: CSV 파일 경로
        wall_size: 월드 크기 (그래프 범위 계산용)
        save_path: 저장 경로 (None이면 표시만)
    """
    # Load data
    data = load_simulation_csv(csv_path)
    
    if len(data['timestamps']) == 0:
        print(f"CSV 파일에 데이터가 없습니다: {csv_path}")
        return
    
    # Calculate graph bounds (wall_size + 0.5m margin, square, equal scale)
    all_x = np.concatenate([data['robot_pos_x'], data['goal_pos_x']])
    all_y = np.concatenate([data['robot_pos_y'], data['goal_pos_y']])
    margin = 0.5
    graph_size = wall_size + margin * 2
    
    # Center on trajectory
    center_x = (all_x.min() + all_x.max()) / 2
    center_y = (all_y.min() + all_y.max()) / 2
    
    x_min = center_x - graph_size / 2
    x_max = center_x + graph_size / 2
    y_min = center_y - graph_size / 2
    y_max = center_y + graph_size / 2
    
    # ========== Figure 1: Trajectory and Commands ==========
    # 최소 1080px 해상도 (16인치 기준 약 68 DPI, 하지만 더 높은 DPI 사용)
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 16), dpi=100)
    fig1.suptitle('OmniVLA Simulation Results - Trajectory & Commands', fontsize=16, fontweight='bold')
    
    # 1.1: Trajectory plot
    ax = axes1[0, 0]
    ax.plot(data['robot_pos_x'], data['robot_pos_y'], 'b-', linewidth=2, label='Robot Trajectory', alpha=0.7)
    ax.scatter(data['robot_pos_x'][0], data['robot_pos_y'][0], c='green', s=200, marker='o', 
               label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax.scatter(data['goal_pos_x'][0], data['goal_pos_y'][0], c='red', s=200, marker='*', 
               label='Goal', zorder=5, edgecolors='black', linewidths=2)
    ax.scatter(data['robot_pos_x'][-1], data['robot_pos_y'][-1], c='orange', s=150, marker='s', 
               label='End', zorder=5, edgecolors='black', linewidths=2)
    
    # Draw heading arrows every N steps
    step = max(1, len(data['robot_pos_x']) // 20)
    for i in range(0, len(data['robot_pos_x']), step):
        x, y = data['robot_pos_x'][i], data['robot_pos_y'][i]
        heading = data['robot_heading'][i]
        dx = 0.3 * math.cos(heading)
        dy = 0.3 * math.sin(heading)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Robot Trajectory', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # 1.2: Linear velocity over time
    ax = axes1[0, 1]
    ax.plot(data['times'], data['linear_vel'], 'g-', linewidth=2, label='Linear Velocity')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Linear Velocity (m/s)', fontsize=12)
    ax.set_title('Linear Velocity Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # 1.3: Angular velocity (yaw rate) over time
    ax = axes1[1, 0]
    ax.plot(data['times'], data['angular_vel'], 'r-', linewidth=2, label='Angular Velocity')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax.set_title('Angular Velocity (Yaw Rate) Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # 1.4: Heading over time
    ax = axes1[1, 1]
    ax.plot(data['times'], np.degrees(data['robot_heading']), 'm-', linewidth=2, label='Robot Heading')
    ax.plot(data['times'], np.degrees(data['goal_heading']), 'c--', linewidth=2, label='Goal Heading', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Heading (degrees)', fontsize=12)
    ax.set_title('Heading Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        # 최소 1080px 해상도로 저장 (16인치 기준 약 68 DPI, 하지만 더 높은 DPI 사용)
        fig1.savefig(save_path / 'trajectory_and_commands.png', dpi=100, bbox_inches='tight')
        print(f"Figure 1 saved: {save_path / 'trajectory_and_commands.png'}")
    
    # Figure 1을 별도 창으로 표시 (저장 후)
    # Note: Agg backend에서는 show()가 작동하지 않으므로, 저장만 수행
    # 별도로 실행할 때는 interactive backend를 사용
    
    # ========== Figure 2: Analysis and Statistics ==========
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 16), dpi=100)
    fig2.suptitle('OmniVLA Simulation Results - Analysis', fontsize=16, fontweight='bold')
    
    # 2.1: Distance to goal over time
    ax = axes2[0, 0]
    distances = np.sqrt(
        (data['robot_pos_x'] - data['goal_pos_x'])**2 + 
        (data['robot_pos_y'] - data['goal_pos_y'])**2
    )
    ax.plot(data['times'], distances, 'b-', linewidth=2, label='Distance to Goal')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Success Threshold (1m)')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Distance (m)', fontsize=12)
    ax.set_title('Distance to Goal Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # 2.2: Velocity magnitude
    ax = axes2[0, 1]
    vel_magnitude = np.sqrt(data['linear_vel']**2 + (data['angular_vel'] * 0.5)**2)  # Approximate
    ax.plot(data['times'], vel_magnitude, 'g-', linewidth=2, label='Velocity Magnitude')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity Magnitude (m/s)', fontsize=12)
    ax.set_title('Velocity Magnitude Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # 2.3: Waypoint direction (dx, dy) scatter
    ax = axes2[1, 0]
    ax.scatter(data['waypoint_dx'], data['waypoint_dy'], c=data['times'], cmap='viridis', 
               s=20, alpha=0.6, label='Waypoint Directions')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Waypoint dx (m)', fontsize=12)
    ax.set_ylabel('Waypoint dy (m)', fontsize=12)
    ax.set_title('Waypoint Directions (dx, dy)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Time (s)', fontsize=10)
    ax.legend(loc='best', fontsize=10)
    
    # 2.4: Heading error
    ax = axes2[1, 1]
    heading_error = data['goal_heading'] - data['robot_heading']
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # Wrap to [-pi, pi]
    ax.plot(data['times'], np.degrees(heading_error), 'r-', linewidth=2, label='Heading Error')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Heading Error (degrees)', fontsize=12)
    ax.set_title('Heading Error Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        # 최소 1080px 해상도로 저장
        fig2.savefig(save_path / 'analysis.png', dpi=100, bbox_inches='tight')
        print(f"Figure 2 saved: {save_path / 'analysis.png'}")
    
    # Figure 2를 별도 창으로 표시 (저장 후)
    # Note: Agg backend에서는 show()가 작동하지 않으므로, 저장만 수행
    # 별도로 실행할 때는 interactive backend를 사용
    
    # Print statistics
    print("\n" + "="*60)
    print("Simulation Statistics")
    print("="*60)
    print(f"Total duration: {data['times'][-1]:.2f} s")
    print(f"Total distance traveled: {np.sum(np.diff(np.sqrt(np.diff(data['robot_pos_x'])**2 + np.diff(data['robot_pos_y'])**2))):.2f} m")
    print(f"Final distance to goal: {distances[-1]:.2f} m")
    print(f"Min distance to goal: {distances.min():.2f} m")
    print(f"Average linear velocity: {data['linear_vel'].mean():.4f} m/s")
    print(f"Average angular velocity: {data['angular_vel'].mean():.4f} rad/s")
    print(f"Max linear velocity: {data['linear_vel'].max():.4f} m/s")
    print(f"Max angular velocity: {data['angular_vel'].max():.4f} rad/s")
    print("="*60)
    
    return fig1, fig2


if __name__ == "__main__":
    import argparse
    # Interactive backend for standalone execution
    matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on system
    
    parser = argparse.ArgumentParser(description="Visualize OmniVLA simulation results")
    parser.add_argument("csv_path", type=str, help="Path to simulation CSV file")
    parser.add_argument("--wall-size", type=float, default=10.0, help="Wall size for graph bounds")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    save_path = Path(args.save_dir) if args.save_dir else None
    
    visualize_simulation_results(csv_path, wall_size=args.wall_size, save_path=save_path)
    plt.show(block=True)  # Block until windows are closed

