# PD 컨트롤러 구조 분석

## 개요
이 문서는 `benchmark_omnivla-ARIL.py`의 462-545 라인에 있는 PD 컨트롤러의 구조와 동작 원리를 상세히 분석합니다.

## 1. 전체 흐름

### 1.1 타겟 계산 (Goal Pose Calculation)
```421:426:inference/benchmark_omnivla-ARIL.py
goal_pose_loc_norm = np.array([
    relative_y / metric_waypoint_spacing,
    -relative_x / metric_waypoint_spacing,
    np.cos(self.goal_compass - cur_compass),
    np.sin(self.goal_compass - cur_compass)
])
```

**타겟 값 (goal_pose_loc_norm)**:
- **위치**: `[relative_y / metric_waypoint_spacing, -relative_x / metric_waypoint_spacing]`
  - `relative_x, relative_y`: 로봇의 현재 위치를 기준으로 한 목표 위치 (로컬 좌표계, 미터 단위)
  - `metric_waypoint_spacing = 0.1`: waypoint 간격 (미터)
  - 정규화된 위치 값 (waypoint 단위)
  
- **방향**: `[cos(goal_compass - cur_compass), sin(goal_compass - cur_compass)]`
  - 목표 heading과 현재 heading의 차이를 cos/sin으로 표현
  - 정규화된 방향 벡터

### 1.2 모델 예측 (Model Prediction)
```492:496:inference/benchmark_omnivla-ARIL.py
waypoints = actions.float().cpu().numpy()
waypoint_select = 4
chosen_waypoint = waypoints[0][waypoint_select].copy()
chosen_waypoint[:2] *= metric_waypoint_spacing
dx, dy, hx, hy = chosen_waypoint
```

**모델 출력 (chosen_waypoint)**:
- `waypoints`: 모델이 예측한 8개의 waypoint 시퀀스 `(batch, 8, 4)`
- `waypoint_select = 4`: 5번째 waypoint 선택 (인덱스 4)
- `chosen_waypoint[:2] *= metric_waypoint_spacing`: 정규화 해제 → 실제 미터 단위로 변환
- `dx, dy`: waypoint의 상대 위치 (미터 단위)
- `hx, hy`: waypoint의 heading 방향 (cos, sin)

## 2. PD 컨트롤러 구조 분석

### 2.1 컨트롤러 파라미터
```498:500:inference/benchmark_omnivla-ARIL.py
EPS = 1e-8
DT = 1 / 3
```

- **EPS**: 0에 가까운 값 판단 임계값
- **DT**: 시간 간격 (초) = 1/3 ≈ 0.333초 (3Hz 제어 주기)

### 2.2 속도 계산 로직

#### 케이스 1: dx ≈ 0, dy ≈ 0 (목표 위치에 도달)
```501:503:inference/benchmark_omnivla-ARIL.py
if np.abs(dx) < EPS and np.abs(dy) < EPS:
    linear_vel_value = 0
    angular_vel_value = 1.0 * np.arctan2(hy, hx) / DT
```

- **선속도**: 0 (이동 불필요)
- **각속도**: heading 방향으로 회전만 수행
  - `np.arctan2(hy, hx)`: heading 각도 계산
  - `1.0`: 각속도 게인
  - `/ DT`: 각속도로 변환 (rad/s)

#### 케이스 2: dx ≈ 0, dy ≠ 0 (옆으로만 이동 필요)
```504:506:inference/benchmark_omnivla-ARIL.py
elif np.abs(dx) < EPS:
    linear_vel_value = 0
    angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
```

- **선속도**: 0
- **각속도**: dy 방향으로 90도 회전
  - `np.sign(dy)`: dy의 부호 (방향)
  - `np.pi / 2`: 90도 (라디안)
  - `/ DT`: 각속도로 변환

#### 케이스 3: 일반 케이스 (dx ≠ 0)
```507:509:inference/benchmark_omnivla-ARIL.py
else:
    linear_vel_value = dx / DT
    angular_vel_value = np.arctan(dy / dx) / DT
```

- **선속도**: `dx / DT` (전진 속도, m/s)
  - dx를 시간 DT로 나누어 속도 계산
  
- **각속도**: `np.arctan(dy / dx) / DT` (회전 속도, rad/s)
  - `np.arctan(dy / dx)`: 목표 방향 각도 (라디안)
  - `/ DT`: 각속도로 변환

### 2.3 속도 제한 (Clipping)
```511:512:inference/benchmark_omnivla-ARIL.py
linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)
```

- **선속도**: 0 ~ 0.5 m/s로 제한
- **각속도**: -1.0 ~ 1.0 rad/s로 제한

### 2.4 속도 제약 조건 (Velocity Limitation)
```514:535:inference/benchmark_omnivla-ARIL.py
maxv, maxw = 0.3, 0.3
if np.abs(linear_vel_value) <= maxv:
    if np.abs(angular_vel_value) <= maxw:
        linear_vel_value_limit = linear_vel_value
        angular_vel_value_limit = angular_vel_value
    else:
        rd = linear_vel_value / angular_vel_value
        linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
        angular_vel_value_limit = maxw * np.sign(angular_vel_value)
else:
    if np.abs(angular_vel_value) <= 0.001:
        linear_vel_value_limit = maxv * np.sign(linear_vel_value)
        angular_vel_value_limit = 0.0
    else:
        rd = linear_vel_value / angular_vel_value
        if np.abs(rd) >= maxv / maxw:
            linear_vel_value_limit = maxv * np.sign(linear_vel_value)
            angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
        else:
            linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
            angular_vel_value_limit = maxw * np.sign(angular_vel_value)
```

**제약 조건**:
- `maxv = 0.3 m/s`: 최대 선속도
- `maxw = 0.3 rad/s`: 최대 각속도

**로직**:
1. **둘 다 제한 내**: 그대로 사용
2. **각속도만 초과**: 선속도와 각속도의 비율(`rd`) 유지하며 각속도를 `maxw`로 제한
3. **선속도만 초과**:
   - 각속도가 거의 0: 선속도만 `maxv`로 제한
   - 각속도 존재: 비율에 따라 제한
4. **둘 다 초과**: 비율에 따라 제한

## 3. 타겟과 비교되는 값 분석

### 3.1 타겟 (Target)
**goal_pose_loc_norm** (421-426 라인):
- **위치**: `[relative_y / metric_waypoint_spacing, -relative_x / metric_waypoint_spacing]`
  - 목표 위치를 waypoint 단위로 정규화
  - 로봇의 현재 위치 기준 상대 좌표
  
- **방향**: `[cos(goal_compass - cur_compass), sin(goal_compass - cur_compass)]`
  - 목표 heading과 현재 heading의 차이

### 3.2 모델 예측 (Predicted)
**chosen_waypoint** (492-496 라인):
- **위치**: `[dx, dy]` (미터 단위)
  - 모델이 예측한 waypoint의 상대 위치
  - `chosen_waypoint[:2] *= metric_waypoint_spacing`로 정규화 해제
  
- **방향**: `[hx, hy]` (cos, sin)
  - 모델이 예측한 waypoint의 heading 방향

### 3.3 비교 관계

**직접 비교는 하지 않음**:
- PD 컨트롤러는 타겟과 예측값을 직접 비교하지 않습니다
- 대신, **모델이 예측한 waypoint만을 사용**하여 속도를 계산합니다
- 타겟(`goal_pose_loc_norm`)은 모델의 입력으로만 사용됩니다

**간접적 비교**:
- 모델이 타겟을 참고하여 waypoint를 예측
- 예측된 waypoint가 타겟에 가까울수록 좋은 성능
- PD 컨트롤러는 예측된 waypoint를 기반으로 속도 명령 생성

## 4. 좌표계 변환

### 4.1 전역 좌표계 → 로컬 좌표계
```412:415:inference/benchmark_omnivla-ARIL.py
delta_x, delta_y = self.calculate_relative_position(
    cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1]
)
relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
```

- UTM 좌표계에서 상대 위치 계산
- 로봇의 현재 heading을 기준으로 로컬 좌표계로 회전

### 4.2 로컬 좌표계 → Waypoint 좌표계
```421:423:inference/benchmark_omnivla-ARIL.py
goal_pose_loc_norm = np.array([
    relative_y / metric_waypoint_spacing,
    -relative_x / metric_waypoint_spacing,
```

- `relative_y` → 첫 번째 요소 (앞/뒤)
- `-relative_x` → 두 번째 요소 (좌/우, 부호 반전)
- `metric_waypoint_spacing`으로 정규화

## 5. 제어 주기와 타이밍

- **제어 주기**: `DT = 1/3` 초 ≈ 0.333초
- **제어 주파수**: 3 Hz
- **tick_rate**: 3 (331 라인)

각 제어 주기마다:
1. 현재 위치와 목표 위치 계산
2. 모델로 waypoint 예측
3. PD 컨트롤러로 속도 계산
4. 속도 제한 적용
5. 로봇에 명령 전송

## 6. 결론

이 PD 컨트롤러는:
- **전통적인 PD 컨트롤러가 아님**: 오차를 직접 계산하지 않음
- **Waypoint 기반 제어**: 모델이 예측한 waypoint를 기하학적으로 속도로 변환
- **타겟은 모델 입력**: `goal_pose_loc_norm`은 모델 학습/추론에만 사용
- **예측값만 사용**: PD 컨트롤러는 예측된 waypoint만을 사용하여 속도 계산

**핵심 아이디어**:
- 모델이 타겟을 고려하여 waypoint를 예측
- PD 컨트롤러는 예측된 waypoint를 속도 명령으로 변환
- 타겟과 예측값의 직접적 비교는 모델 내부에서 이루어짐

