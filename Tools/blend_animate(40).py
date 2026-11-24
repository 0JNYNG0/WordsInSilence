"""
Tools/blend_animate.py
------------------------------------------------------
MediaPipe 랜드마크 JSON → Blender Mixamo 아마추어 애니메이션

팔  : IK chain_count=2 (팔꿈치 굽힘)
손  : FK 직접 회전 — set_hand_rotation (방향 + 손바닥 법선)
손가락: FK 직접 회전 — 연속 랜드마크 간 방향 벡터 사용
       (IK 절대위치 타게팅 방식 제거 — 뼈 길이 불일치로 관절 꺾임 발생)
       주먹 확정 구간은 OPEN↔FIST 시계열 상태기계로 네 손가락 자세 안정화
       주먹을 유지한 채 엄지만 펴짐↔접힘으로 바뀌는 구간은 상태 대비 강조
       엄지 끝 추적이 뒤집히는 뿌리 주도 전환은 내부 관절을 고정한 강체 회전
       엄지척 상태는 양손 모두 세계 위쪽을 향하게 유지한 뒤 검지 옆까지 회전
"""

import bpy, json, math, mathutils, statistics
print("[SCRIPT-VERSION] generic_thumb_vertical_hold_to_index_stop_v20")

# ================================================================
# 설정
# ================================================================
JSON_PATH        = r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\Recordings\인사(1).json"
BODY_SCALE       = 1.0
DEPTH_SCALE      = 0.0
PALM_Z_SCALE     = 1.0
FORWARD_OFFSET   = 0.15
FPS              = 30
SMOOTH_WINDOW    = 5
JUMP_THRESHOLD   = 0.5
MIRROR_X         = True
SWAP_HAND_LABELS = False  # 전문가 비미러 영상: mp_right→캐릭터LEFT, mp_left→캐릭터RIGHT
L_PALM_FLIP      = True
R_PALM_FLIP      = False
FLIP_FINGER_DIR_L = False   # 왼손 손가락 방향이 반대면 True
FLIP_FINGER_DIR_R = False   # 오른손 손가락 방향이 반대면 True
HAND_DEPTH_GAIN  = 1.0      # Hand landmark 깊이 스케일 (0.25 / 0.5 / 1.0 비교)
USE_HAND_WRIST_XZ_FOR_IK = True  # IK 타겟 XZ를 Pose wrist 대신 Hand pt0으로 교체 (접촉 정확도)
HAND_WRIST_IK_MAX_OFFSET_SW = 0.30  # Hand pt0 ↔ Pose wrist XZ 거리 gate (char_sw 단위)
USE_POSE_WRIST_DEPTH_FOR_IK = True  # IK Pose wrist Y에 상대 깊이 복원
POSE_WRIST_DEPTH_GAIN       = 1.0   # 깊이 스케일 (0=평면, 1=원본)
POSE_WRIST_DEPTH_MAX_SW     = 0.45  # 깊이 clamp 범위 (char_sw 단위)
USE_ARM_REACH_LIMIT  = True   # smooth IK 타겟 도달 거리 상한 적용
ARM_REACH_MAX_RATIO  = 0.97   # 팔 전체 길이 대비 최대 도달 비율 (hard limit)
USE_SOURCE_REACH_GUIDANCE = True  # source ratio 기반 reach retargeting 활성화
SOURCE_REACH_BLEND     = 1.0   # 0=hard limit, 1=source ratio 100% 적용
SOURCE_REACH_MIN_RATIO = 0.05  # source ratio 하한 clamp
SOURCE_REACH_2D_WEIGHT = 0.0   # 3D ratio만 사용 (2D 혼합 없음)
USE_ELBOW_POLE           = True   # 팔꿈치 방향 pole target 사용
ELBOW_POLE_DISTANCE_RATIO = 0.80  # char_arm_total 대비 pole 거리
L_ELBOW_POLE_ANGLE_DEG   = -90.0 # LEFT forearm IK pole_angle (도)
R_ELBOW_POLE_ANGLE_DEG   = 90.0  # RIGHT forearm IK pole_angle (도)
IK_RESET_DOF     = True   # IK constraint 생성 전 arm/forearm DOF 명시 초기화
ELBOW_BEND_DIAGNOSTICS         = True   # 팔꿈치 굽힘 방향 진단 (애니메이션 변경 없음)
ELBOW_BEND_MIN_MAGNITUDE       = 0.03   # 신뢰 가능한 최소 굽힘 크기
ELBOW_BEND_STRAIGHT_ANGLE_DEG  = 165.0  # 이 각도 이상이면 편 팔로 판정 (진단 제외)
ELBOW_BEND_NEGATIVE_DOT_THRESHOLD = -0.10  # 이 값 미만이면 반대 방향 판정
IK_PREBEND_ENABLED = True   # straight-chain singularity 방지 pre-bend
IK_PREBEND_DEG   = 5.0    # forearm local-X pre-bend 각도 (LEFT=+, RIGHT=-)
USE_TWO_HAND_CONTACT_PRESERVATION = True   # 양손 접촉점 보존 보정
TWO_HAND_CONTACT_ENTER_SW  = 0.07  # 접촉 판정 진입 거리 (char_sw 단위)
TWO_HAND_CONTACT_EXIT_SW   = 0.10  # 접촉 유지 이탈 거리 (char_sw 단위)
TWO_HAND_CONTACT_MIN_FRAMES = 3    # 최소 연속 프레임 (단일 프레임 오검출 방지)
TWO_HAND_CONTACT_BLEND     = 1.0   # gap → correction 비율
TWO_HAND_CONTACT_MAX_CORRECTION_SW = 0.45  # 접촉점 gap 전체를 닫을 수 있는 상한
TWO_HAND_CONTACT_FADE_EDGE_WEIGHT  = 1.00  # 활성 접촉 프레임은 모두 완전 보정
TWO_HAND_CONTACT_TRANSITION_FRAMES = 4     # 접촉 전후 보정 오프셋 진입/이탈 완화 프레임
TWO_HAND_CONTACT_REACH_SPLIT_STEPS = 1000  # 좌/우 팔 도달 여유 탐색 해상도
TWO_HAND_CONTACT_DEFAULT_MODE = "auto"    # auto=원본 접촉 연속구간만 자동 보정
TWO_HAND_CONTACT_MOTION_PROFILES = {
    # 필요할 때만 동작별로
    # "auto" / "touch" / "source" / "support_slide" / "off"를 지정한다.
    # "touch"도 원본 접촉 검출을 통과한 프레임만 보정한다.
    # "source"는 손이 맞닿더라도 미끄러지거나 회전하는 동작의 원본 궤적을
    # 보존하며, 정적인 랜드마크 한 쌍으로 양손을 묶지 않는다.
    # "support_slide"는 한 손의 중심을 반대쪽 주먹 윗면에 유지하되,
    # 펼친 손이 수평면 안에서 회전하는 것은 허용한다.
    "사랑": "support_slide",
}
SUPPORT_SLIDE_MOTION_PROFILES = {
    "사랑": {
        "support_side": "L",
        "moving_side": "R",
        # 펼친 오른손 손바닥이 아래쪽 주먹을 향하도록 한다.
        "moving_palm_normal_world": (0.0, 0.0, -1.0),
        # 원본의 중앙값을 쓰되 잘못된 깊이/높이 추정은 이 범위로 제한한다.
        "height_min_sw": 0.22,
        "height_max_sw": 0.31,
        "min_full_weight_frames": 6,
        # 오른손 중심이 왼 주먹 위에서 제자리 회전하지 않고 작은 타원을
        # 그리도록 한다. 깊이는 추정 잡음에 민감하므로 좌우의 절반만 사용.
        "trajectory_mode": "ellipse",
        "orbit_radius_x_sw": 0.11,
        "orbit_radius_y_sw": 0.055,
        "orbit_cycles": 3.0,
        "orbit_phase_deg": 0.0,
    },
}
INTRO_FRAMES     = 20
USE_SOURCE_UPPER_ARM_GUIDANCE      = True    # shoulder→elbow 방향 기반 Pole guidance 활성화
SOURCE_UPPER_ARM_GUIDANCE_BLEND    = 1.0     # 0=legacy(bend), 1=guided(upper-arm)
SOURCE_UPPER_ARM_SMOOTH_WINDOW     = 5       # upper-arm 방향 스무딩 윈도우
SOURCE_UPPER_ARM_MIN_MAGNITUDE     = 0.03    # source upper-arm 유효 최소 길이 (정규화 공간)
USE_WRIST_AWARE_ELBOW_GUIDANCE     = True       # retarget 오차로 늘어난 손목 swing만 줄이는 pole 보정
WRIST_AWARE_TARGET_SIDES           = {'L', 'R'} # 좌우를 독립적으로 자동 판정
WRIST_AWARE_MIN_VALID_FRAMES       = 5          # 손 검출이 너무 적은 쪽은 보정하지 않음
WRIST_AWARE_MIN_TARGET_ANGLE_DEG   = 42.0       # 리그 안정성을 위한 절대 하한
WRIST_AWARE_SOURCE_TOLERANCE_DEG   = 8.0        # 원본 손목각보다 이 정도 큰 값까지 허용
WRIST_AWARE_SOURCE_SMOOTH_WINDOW   = 5          # 원본 손목각 트랙 스무딩
WRIST_AWARE_MAX_POLE_SHIFT_DEG     = 40.0       # source 팔꿈치 평면에서 벗어날 수 있는 최대 각도
WRIST_AWARE_SEARCH_STEPS           = 14         # 목표 굽힘각을 찾는 이분 탐색 반복 횟수
USE_PALM_NORMAL_CONTINUITY = True   # palm normal hemisphere continuity 안정화
PALM_NORMAL_SMOOTH_WINDOW  = 5      # palm normal moving average 윈도우
PALM_NORMAL_MIN_MAGNITUDE  = 1e-5   # 유효 법선 최소 길이
USE_FOREARM_TWIST_REDISTRIBUTION = True       # 손 세계 방향을 유지하며 과도한 손목 roll을 전완으로 이전
FOREARM_TWIST_TARGET_SIDES       = {'L', 'R'} # 좌우를 독립 calibration 후 적용
FOREARM_TWIST_MIN_VALID_FRAMES   = 5          # 검출 부족/잡음 손은 자동 비활성화
HAND_TWIST_SOFT_LIMIT_DEG        = 100.0  # 이 각도 초과분만 ForeArm으로 이전
FOREARM_TWIST_MAX_TRANSFER_DEG   = 90.0   # ForeArm 이전 최대 각도 clamp
FOREARM_TWIST_SMOOTH_WINDOW      = 7      # correction track 스무딩 윈도우
FOREARM_TWIST_MAX_STEP_DEG       = 8.0    # controller의 프레임당 최대 변화량
DIAGNOSTIC_TIMESTAMPS_MS = (999, 1299, 1333, 1366)   # Hand-ForeArm 진단 대상 ts

# ── 손 모양 시계열 보정 ──────────────────────────────────────────
# 손 전체 위치/회전은 원본을 유지하고, 주먹으로 확정된 구간의
# 검지·중지·약지·소지 관절 방향만 안정적인 기준 자세로 보간한다.
# 엄지는 수어별 위치 차이가 크므로 자동 고정하지 않는다.
USE_FIST_HANDSHAPE_STABILIZATION = True
FIST_HANDSHAPE_DEFAULT_MODE      = "auto"  # auto / off
FIST_HANDSHAPE_TARGET_SIDES      = {'L', 'R'}
FIST_ENTER_MEAN_FOLD_RATIO       = 0.72
FIST_ENTER_SINGLE_FOLD_RATIO     = 0.78
FIST_ENTER_REQUIRED_FINGERS      = 3
FIST_EXIT_MEAN_FOLD_RATIO        = 0.88
FIST_EXIT_SINGLE_FOLD_RATIO      = 0.90
FIST_EXIT_REQUIRED_FINGERS       = 2
FIST_ENTER_FRAMES                = 4
FIST_EXIT_FRAMES                 = 4
FIST_TRANSITION_FRAMES           = 5
# 확정 주먹 바로 앞에서 네 손가락이 함께 접히는 구간도 찾는다.
# 동작명이나 프레임 번호가 아니라, "이후 실제 주먹으로 확정됨"과
# 네 손가락의 동시 접힘을 모두 만족할 때만 주먹 전환을 앞당긴다.
FIST_PRETRANSITION_MAX_FRAMES       = 14
FIST_PRETRANSITION_OPEN_MEAN_RATIO  = 0.98
FIST_PRETRANSITION_MIN_VALID_FRAMES = 5
FIST_PRETRANSITION_MIN_PROGRESS     = 0.10
FIST_PRETRANSITION_REQUIRED_FINGERS = 3
# 이 Mixamo 리그에서는 effective palm normal 쪽이 손바닥 안쪽이다.
# v11(-1.0)은 주먹을 손등 방향으로 말았으므로 v12에서 축 부호를 교정했다.
FIST_CURL_DIRECTION_SIGN         = +1.0
# 각 값은 손바닥 평면의 손가락 진행 방향에서 안쪽으로 회전한 누적 각도.
# chain index: 1=검지, 2=중지, 3=약지, 4=소지
FIST_CURL_ANGLES_DEG = {
    1: (62.0, 150.0, 205.0),
    2: (68.0, 156.0, 211.0),
    3: (72.0, 160.0, 215.0),
    4: (76.0, 164.0, 219.0),
}
# 주먹 동안 엄지는 실제 움직임을 기본적으로 그대로 둔다. 다만 안정 자세에서
# 갑자기 크게 이탈했다가 같은 자세로 돌아오는 짧은 내부 구간만 추적 오류로
# 판정해 보간한다. 따라서 엄지 위→아래처럼 시작/종료 자세가 달라지는 변화는
# 비율상 다수 자세가 있더라도 고정하지 않는다.
USE_FIST_THUMB_TEMPORAL_GUARD    = True
FIST_THUMB_MIN_VALID_FRAMES      = 8
FIST_THUMB_STABLE_ANGLE_DEG      = 35.0
FIST_THUMB_OUTLIER_PEAK_DEG      = 55.0
FIST_THUMB_OUTLIER_BOUNDARY_DEG  = 20.0
FIST_THUMB_OUTLIER_MAX_FRAMES    = 16
FIST_THUMB_OUTLIER_MAX_FRACTION  = 0.35
# 네 손가락이 확정 주먹으로 이어지는 동안 엄지 체인의 펴짐 비율이
# 두 안정 상태 사이에서 크게 바뀌면, 원본 전환 시점은 보존하면서
# "펴진 엄지"와 "접힌 엄지"의 자세 대비만 강화한다. 동작명/프레임 번호를
# 사용하지 않으므로 같은 손 모양 전환을 가진 다른 수어에도 적용된다.
USE_FIST_THUMB_GESTURE_EMPHASIS       = True
FIST_THUMB_GESTURE_WINDOW_FRAMES      = 6
FIST_THUMB_GESTURE_MIN_STATE_FRAMES   = 4
FIST_THUMB_GESTURE_EXTENDED_RATIO     = 0.92
FIST_THUMB_GESTURE_FOLDED_RATIO       = 0.82
FIST_THUMB_GESTURE_MIN_DELTA          = 0.20
FIST_THUMB_GESTURE_TRANSITION_BEFORE  = 5
FIST_THUMB_GESTURE_TRANSITION_AFTER   = 4
FIST_THUMB_GESTURE_BLEND              = 0.92
FIST_THUMB_GESTURE_EDGE_FRAMES        = 4
FIST_THUMB_GESTURE_PREHOLD_FRAMES     = 4
# 펴진 상태는 wrist→middle-MCP 방향으로 곧게 세우고, 접힌 상태는
# 원본의 좌우/손바닥 안쪽 방향은 보존하면서 전방 성분만 줄여 주먹 위로 눕힌다.
FIST_THUMB_EXTENDED_FORWARD_BLEND = (0.55, 0.72, 0.82)
FIST_THUMB_FOLDED_FORWARD_SCALE   = (0.38, 0.65, 0.30)
FIST_THUMB_FOLDED_SIDE_SCALE      = (1.35, 1.20, 1.25)
FIST_THUMB_FOLDED_NORMAL_SCALE    = (1.05, 1.10, 1.15)
# 엄지 뿌리가 실제로 움직이는데 가려진 끝마디 방향만 크게 뒤집히면,
# IP/MCP 굽힘이 아니라 CMC(손바닥 연결 관절) 주도 동작으로 해석한다.
# 이 리그에서는 side 성분이 큰 바깥축이 엄지척, forward 성분이 큰 축이
# 검지 옆 접촉 자세다. 두 축 사이에서만 회전하고 반대 side로 넘어가지
# 않게 하여, 접촉 자세를 지나 새끼손가락 아래까지 내려가는 것을 막는다.
FIST_THUMB_ROOT_PIVOT_MIN_DEG          = 15.0
FIST_THUMB_TIP_REVERSAL_MIN_DEG        = 65.0
FIST_THUMB_EXTENDED_OUTWARD_COMPONENT  = 0.40
FIST_THUMB_EXTENDED_NORMAL_MAX         = 0.12
FIST_THUMB_FOLDED_ROOT_FORWARD_SCALE   = 0.45
FIST_THUMB_FOLDED_ROOT_SIDE_SCALE      = 1.20
FIST_THUMB_FOLDED_ROOT_SIDE_MIN        = 0.72
FIST_THUMB_FOLDED_ROOT_NORMAL_SCALE    = 0.80
# 뿌리 주도 강체 엄지 전환에서 '엄지척' 상태인 동안에는 손바닥 로컬축의
# 단안 깊이 오차를 따르지 않고 캐릭터 세계 좌표의 위쪽(+Z)을 사용한다.
# 전환 중에는 상태 보간값에 맞춰 자동으로 해제되어 검지 옆 자세로 이어진다.
FIST_THUMB_RAISED_WORLD_UP_BLEND       = 1.0

# 동작별 예외가 꼭 필요할 때만 사용한다. 지정하지 않은 동작은 모두 자동 판정한다.
# 예:
# MOTION_OVERRIDES = {
#     "동작명": {
#         "contact_mode": "source",
#         "wrist_aware_sides": {'R'},
#         "forearm_twist_sides": {'R'},
#         "fist_mode": "off",
#         "fist_sides": {'L'},
#     },
# }
MOTION_OVERRIDES = {}

# ── 테스트 모드 ──────────────────────────────────────────────────
# True: 정적 단일 프레임 진단 (전체 애니메이션 스킵)
# False: 전체 프레임 애니메이션 생성
TEST_MODE       = False
TEST_FRAME_IDX  = 0     # 유효 LEFT hand 검출 프레임 중 n번째 (0=첫 번째)
TEST_INDEX_ONLY = True  # True: 검지만 FK, False: 전체 손가락

# ================================================================
# MediaPipe 인덱스
# ================================================================
P_L_SHOULDER = 11; P_R_SHOULDER = 12
P_L_ELBOW    = 13; P_R_ELBOW    = 14
P_L_WRIST    = 15; P_R_WRIST    = 16

H_WRIST     = 0
H_THUMB_CMC = 1;  H_THUMB_MCP = 2;  H_THUMB_IP  = 3;  H_THUMB_TIP  = 4
H_INDEX_MCP = 5;  H_INDEX_PIP = 6;  H_INDEX_DIP = 7;  H_INDEX_TIP  = 8
H_MID_MCP   = 9;  H_MID_PIP   = 10; H_MID_DIP   = 11; H_MID_TIP    = 12
H_RING_MCP  = 13; H_RING_PIP  = 14; H_RING_DIP  = 15; H_RING_TIP   = 16
H_PINK_MCP  = 17; H_PINK_PIP  = 18; H_PINK_DIP  = 19; H_PINK_TIP   = 20

# ================================================================
# Mixamo 뼈 이름
# ================================================================
BONE_L_ARM     = 'mixamorig:LeftArm'
BONE_L_FOREARM = 'mixamorig:LeftForeArm'
BONE_L_HAND    = 'mixamorig:LeftHand'
BONE_R_ARM     = 'mixamorig:RightArm'
BONE_R_FOREARM = 'mixamorig:RightForeArm'
BONE_R_HAND    = 'mixamorig:RightHand'

# 손가락 체인: [(뼈이름, from_landmark, to_landmark), ...]
# 각 뼈를 from→to 방향으로 FK 설정
# 첫 뼈의 부모 = Hand 뼈,  이후 뼈의 부모 = 이전 뼈
FINGER_CHAINS_L = [
    [('mixamorig:LeftHandThumb1',  H_THUMB_CMC, H_THUMB_MCP),
     ('mixamorig:LeftHandThumb2',  H_THUMB_MCP, H_THUMB_IP),
     ('mixamorig:LeftHandThumb3',  H_THUMB_IP,  H_THUMB_TIP)],
    [('mixamorig:LeftHandIndex1',  H_INDEX_MCP, H_INDEX_PIP),
     ('mixamorig:LeftHandIndex2',  H_INDEX_PIP, H_INDEX_DIP),
     ('mixamorig:LeftHandIndex3',  H_INDEX_DIP, H_INDEX_TIP)],
    [('mixamorig:LeftHandMiddle1', H_MID_MCP,   H_MID_PIP),
     ('mixamorig:LeftHandMiddle2', H_MID_PIP,   H_MID_DIP),
     ('mixamorig:LeftHandMiddle3', H_MID_DIP,   H_MID_TIP)],
    [('mixamorig:LeftHandRing1',   H_RING_MCP,  H_RING_PIP),
     ('mixamorig:LeftHandRing2',   H_RING_PIP,  H_RING_DIP),
     ('mixamorig:LeftHandRing3',   H_RING_DIP,  H_RING_TIP)],
    [('mixamorig:LeftHandPinky1',  H_PINK_MCP,  H_PINK_PIP),
     ('mixamorig:LeftHandPinky2',  H_PINK_PIP,  H_PINK_DIP),
     ('mixamorig:LeftHandPinky3',  H_PINK_DIP,  H_PINK_TIP)],
]
FINGER_CHAINS_R = [
    [('mixamorig:RightHandThumb1',  H_THUMB_CMC, H_THUMB_MCP),
     ('mixamorig:RightHandThumb2',  H_THUMB_MCP, H_THUMB_IP),
     ('mixamorig:RightHandThumb3',  H_THUMB_IP,  H_THUMB_TIP)],
    [('mixamorig:RightHandIndex1',  H_INDEX_MCP, H_INDEX_PIP),
     ('mixamorig:RightHandIndex2',  H_INDEX_PIP, H_INDEX_DIP),
     ('mixamorig:RightHandIndex3',  H_INDEX_DIP, H_INDEX_TIP)],
    [('mixamorig:RightHandMiddle1', H_MID_MCP,   H_MID_PIP),
     ('mixamorig:RightHandMiddle2', H_MID_PIP,   H_MID_DIP),
     ('mixamorig:RightHandMiddle3', H_MID_DIP,   H_MID_TIP)],
    [('mixamorig:RightHandRing1',   H_RING_MCP,  H_RING_PIP),
     ('mixamorig:RightHandRing2',   H_RING_PIP,  H_RING_DIP),
     ('mixamorig:RightHandRing3',   H_RING_DIP,  H_RING_TIP)],
    [('mixamorig:RightHandPinky1',  H_PINK_MCP,  H_PINK_PIP),
     ('mixamorig:RightHandPinky2',  H_PINK_PIP,  H_PINK_DIP),
     ('mixamorig:RightHandPinky3',  H_PINK_DIP,  H_PINK_TIP)],
]


# ================================================================
# 유틸
# ================================================================
def get_pt(pts, idx):
    if not pts or idx >= len(pts):
        return None
    p = pts[idx]
    return (p.get('x',0), p.get('y',0), p.get('z',0)) if isinstance(p, dict) else None


_FIST_NON_THUMB_CHAINS = (
    (H_INDEX_MCP, H_INDEX_PIP, H_INDEX_DIP, H_INDEX_TIP),
    (H_MID_MCP,   H_MID_PIP,   H_MID_DIP,   H_MID_TIP),
    (H_RING_MCP,  H_RING_PIP,  H_RING_DIP,  H_RING_TIP),
    (H_PINK_MCP,  H_PINK_PIP,  H_PINK_DIP,  H_PINK_TIP),
)


def _smoothstep01(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _fist_fold_metrics(raw, prefix, frame_index):
    """네 손가락의 MCP→TIP 직선거리 / 실제 체인 길이를 계산한다.

    1에 가까우면 펴진 손가락, 0에 가까울수록 강하게 접힌 손가락이다.
    길이 비율이므로 캐릭터 크기나 카메라 거리에 거의 영향을 받지 않는다.
    """
    points = []
    for point_index in range(21):
        track = raw.get(f'{prefix}_p{point_index}', [])
        point = track[frame_index] if frame_index < len(track) else None
        if point is None:
            return None
        points.append(point)

    ratios = []
    for mcp, pip, dip, tip in _FIST_NON_THUMB_CHAINS:
        chain_length = (
            (points[pip] - points[mcp]).length +
            (points[dip] - points[pip]).length +
            (points[tip] - points[dip]).length
        )
        if chain_length <= 1e-6:
            return None
        ratios.append((points[tip] - points[mcp]).length / chain_length)

    mean_ratio = sum(ratios) / len(ratios)
    enter_count = sum(
        ratio <= FIST_ENTER_SINGLE_FOLD_RATIO for ratio in ratios)
    stay_count = sum(
        ratio <= FIST_EXIT_SINGLE_FOLD_RATIO for ratio in ratios)
    return {
        'ratios': ratios,
        'mean': mean_ratio,
        'enter': (
            mean_ratio <= FIST_ENTER_MEAN_FOLD_RATIO and
            enter_count >= FIST_ENTER_REQUIRED_FINGERS
        ),
        'stay': (
            mean_ratio <= FIST_EXIT_MEAN_FOLD_RATIO and
            stay_count >= FIST_EXIT_REQUIRED_FINGERS
        ),
    }


def build_fist_weight_track(raw, prefix, valid_mask, timestamps, enabled):
    """OPEN↔FIST 히스테리시스 상태기계와 경계 보간 가중치를 만든다."""
    count = len(valid_mask)
    if not enabled or count == 0:
        return [0.0] * count, [], []

    metrics = []
    for frame_index in range(count):
        metrics.append(
            _fist_fold_metrics(raw, prefix, frame_index)
            if valid_mask[frame_index] else None)

    active = [False] * count
    in_fist = False
    enter_count = 0
    exit_count = 0

    for frame_index, metric in enumerate(metrics):
        enter_evidence = bool(metric and metric['enter'])
        stay_evidence = bool(metric and metric['stay'])

        if not in_fist:
            enter_count = enter_count + 1 if enter_evidence else 0
            if enter_count >= FIST_ENTER_FRAMES:
                in_fist = True
                start = frame_index - FIST_ENTER_FRAMES + 1
                for back_index in range(start, frame_index + 1):
                    active[back_index] = True
                exit_count = 0
        else:
            active[frame_index] = True
            if stay_evidence:
                exit_count = 0
            else:
                exit_count += 1
                if exit_count >= FIST_EXIT_FRAMES:
                    start = frame_index - FIST_EXIT_FRAMES + 1
                    for back_index in range(start, frame_index + 1):
                        active[back_index] = False
                    in_fist = False
                    enter_count = 0
                    exit_count = 0

    runs = []
    run_start = None
    for frame_index in range(count + 1):
        is_active = active[frame_index] if frame_index < count else False
        if is_active and run_start is None:
            run_start = frame_index
        elif not is_active and run_start is not None:
            runs.append((run_start, frame_index - 1))
            run_start = None

    weights = [0.0] * count
    transition = max(1, FIST_TRANSITION_FRAMES)
    for run_start, run_end in runs:
        for frame_index in range(run_start, run_end + 1):
            enter_t = min(1.0, (frame_index - run_start + 1) / transition)
            exit_t = min(1.0, (run_end - frame_index + 1) / transition)
            weights[frame_index] = _smoothstep01(min(enter_t, exit_t))

    # 확정 주먹으로 이어지는 "네 손가락 동시 접힘" 구간만 역방향으로
    # 탐색한다. 단순히 일부 손가락만 접는 수어나 확정 주먹이 없는 동작에는
    # 적용되지 않는다. 검출된 구간은 시간축 smoothstep으로 기존 주먹
    # 가중치에 연결해, 원본 랜드마크와 기준 주먹 사이의 급격한 전환을 막는다.
    pretransitions = {}
    for run_start, run_end in runs:
        pre_start = run_start
        search_start = max(
            0, run_start - max(0, FIST_PRETRANSITION_MAX_FRAMES))
        for frame_index in range(run_start - 1, search_start - 1, -1):
            metric = metrics[frame_index]
            if (metric is None or
                    metric['mean'] > FIST_PRETRANSITION_OPEN_MEAN_RATIO):
                break
            pre_start = frame_index

        full_frame = min(
            run_end, run_start + max(1, FIST_TRANSITION_FRAMES) - 1)
        start_metric = metrics[pre_start] if pre_start < run_start else None
        target_metric = metrics[full_frame]
        valid_count = sum(
            metrics[index] is not None
            for index in range(pre_start, run_start + 1))
        coherent_fingers = 0
        mean_progress = 0.0
        if start_metric is not None and target_metric is not None:
            coherent_fingers = sum(
                start_ratio - target_ratio >=
                FIST_PRETRANSITION_MIN_PROGRESS
                for start_ratio, target_ratio in zip(
                    start_metric['ratios'], target_metric['ratios']))
            mean_progress = (
                start_metric['mean'] - target_metric['mean'])

        bridge_enabled = bool(
            pre_start < run_start and
            valid_count >= FIST_PRETRANSITION_MIN_VALID_FRAMES and
            coherent_fingers >= FIST_PRETRANSITION_REQUIRED_FINGERS and
            mean_progress >= FIST_PRETRANSITION_MIN_PROGRESS and
            full_frame > pre_start)
        if bridge_enabled:
            span = full_frame - pre_start
            for frame_index in range(pre_start, full_frame + 1):
                bridge_t = (
                    (frame_index - pre_start) / span
                    if span > 0 else 1.0)
                bridge_weight = _smoothstep01(bridge_t)
                weights[frame_index] = max(
                    weights[frame_index], bridge_weight)

        pretransitions[(run_start, run_end)] = {
            'enabled': bridge_enabled,
            'start': pre_start if bridge_enabled else run_start,
            'full': full_frame,
            'valid': valid_count,
            'coherent_fingers': coherent_fingers,
            'mean_progress': mean_progress,
        }

    diagnostics = []
    for run_start, run_end in runs:
        run_means = [
            metrics[index]['mean']
            for index in range(run_start, run_end + 1)
            if metrics[index] is not None
        ]
        start_ts = timestamps[run_start] if run_start < len(timestamps) else None
        end_ts = timestamps[run_end] if run_end < len(timestamps) else None
        diagnostics.append({
            'start': run_start,
            'end': run_end,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'min_mean': min(run_means) if run_means else None,
            'pretransition': pretransitions.get((run_start, run_end)),
        })
    return weights, diagnostics, metrics


def to_blender(x, y, z, cx, cy, sw, origin, char_sw, depth_scale=None):
    """Pose landmark → Blender world (깊이 기본 0 — 손목 IK 타겟용)."""
    if depth_scale is None:
        depth_scale = DEPTH_SCALE
    sc = char_sw * BODY_SCALE
    xs = 1 if MIRROR_X else -1
    return origin + mathutils.Vector((
        xs * (x - cx) / sw * sc,
        z * sc * depth_scale - FORWARD_OFFSET,
        -(y - cy) / sw * sc,
    ))


def convert_hand_pt(x, y, z, cx, cy, sw, origin, char_sw):
    """
    MediaPipe Hand normalized landmark → Blender world 좌표.
    X, Y(depth), Z 모두 동일하게 sw 로 정규화하여 일관된 스케일 유지.
    손가락 관절 위치와 손바닥 법선 계산에 동일하게 사용.
    깊이 스케일: HAND_DEPTH_GAIN (설정값, 기본 1.0)
    """
    sc = char_sw * BODY_SCALE
    xs = 1 if MIRROR_X else -1
    return origin + mathutils.Vector((
        xs * (x - cx) / sw * sc,
        z / sw * sc * HAND_DEPTH_GAIN - FORWARD_OFFSET,
        -(y - cy) / sw * sc,
    ))


def smooth_scalar_track(values, window=5):
    """None 선형보간 + scalar moving average (source reach ratio 전용)."""
    n = len(values)
    if n == 0: return []
    # 유효 인덱스 수집
    valid_idx = [i for i, v in enumerate(values) if v is not None]
    if not valid_idx: return [None] * n
    filled = list(values)
    # 두 유효값 사이 선형 보간
    for a, b in zip(valid_idx, valid_idx[1:]):
        if b - a > 1:
            va, vb = filled[a], filled[b]
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                filled[k] = va + t * (vb - va)
    # 앞/뒤 None → 첫/마지막 유효값으로 채움
    first_v, last_v = filled[valid_idx[0]], filled[valid_idx[-1]]
    for i in range(valid_idx[0]): filled[i] = first_v
    for i in range(valid_idx[-1] + 1, n): filled[i] = last_v
    # 5-frame moving average
    half = window // 2
    result = []
    for i in range(n):
        chunk = filled[max(0, i-half):min(n, i+half+1)]
        result.append(sum(chunk) / len(chunk))
    return result


def smooth_direction_track(values, window=5):
    """None 방향 lerp+normalize + moving average + normalize (upper-arm 방향 전용)."""
    n = len(values)
    if n == 0: return []
    valid_idx = [i for i, v in enumerate(values) if v is not None]
    if not valid_idx: return [None] * n
    filled = list(values)
    # 두 유효값 사이 방향 lerp + normalize
    for a, b in zip(valid_idx, valid_idx[1:]):
        if b - a > 1:
            va, vb = filled[a], filled[b]
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                lerped = va.lerp(vb, t)
                filled[k] = lerped.normalized() if lerped.length > 1e-6 else va.copy()
    # 앞/뒤 None → 첫/마지막 유효값으로 채움
    for i in range(valid_idx[0]):           filled[i] = filled[valid_idx[0]].copy()
    for i in range(valid_idx[-1] + 1, n):  filled[i] = filled[valid_idx[-1]].copy()
    # moving average + normalize
    half = window // 2
    result = []
    for i in range(n):
        chunk = filled[max(0, i-half):min(n, i+half+1)]
        avg = mathutils.Vector((
            sum(v.x for v in chunk) / len(chunk),
            sum(v.y for v in chunk) / len(chunk),
            sum(v.z for v in chunk) / len(chunk),
        ))
        result.append(avg.normalized() if avg.length > 1e-6 else chunk[len(chunk)//2].copy())
    return result


def stabilize_normal_track(values, window=5):
    """
    Palm normal hemisphere continuity 안정화 + moving average.
    Returns (stabilized_track, flip_count, valid_count, missing_count)
    - 이전 유효 법선과 dot<0이면 현재 법선 negate (180도 뒤집힘 방지)
    - 중간 None은 부호 정렬된 앞뒤 법선 lerp 후 normalize
    - moving average 후 normalize (평균 길이 너무 작으면 중앙값 사용)
    """
    n = len(values)
    if n == 0: return [], 0, 0, 0
    # normalize 후 None 유지
    normed = []
    for v in values:
        if v is not None and v.length > PALM_NORMAL_MIN_MAGNITUDE:
            normed.append(v.normalized())
        else:
            normed.append(None)
    valid_count   = sum(1 for v in normed if v is not None)
    missing_count = n - valid_count
    if valid_count == 0:
        return [None] * n, 0, 0, missing_count
    # Pass 1: hemisphere continuity (시간 순)
    flip_count = 0
    prev = None
    for i in range(n):
        if normed[i] is None: continue
        if prev is not None and prev.dot(normed[i]) < 0:
            normed[i] = -normed[i]; flip_count += 1
        prev = normed[i]
    # Gap fill: 앞/뒤 상수, 중간 lerp+normalize
    valid_idx = [i for i, v in enumerate(normed) if v is not None]
    filled = list(normed)
    for i in range(valid_idx[0]):          filled[i] = filled[valid_idx[0]].copy()
    for i in range(valid_idx[-1]+1, n):    filled[i] = filled[valid_idx[-1]].copy()
    for a, b in zip(valid_idx, valid_idx[1:]):
        if b - a > 1:
            va, vb = filled[a], filled[b]
            for k in range(a+1, b):
                t = (k-a) / (b-a)
                lerped = va.lerp(vb, t)
                filled[k] = lerped.normalized() if lerped.length > 1e-6 else va.copy()
    # Moving average + normalize (평균 너무 작으면 중앙값)
    half = window // 2
    result = []
    for i in range(n):
        chunk = filled[max(0, i-half):min(n, i+half+1)]
        avg = mathutils.Vector((
            sum(v.x for v in chunk)/len(chunk),
            sum(v.y for v in chunk)/len(chunk),
            sum(v.z for v in chunk)/len(chunk),
        ))
        if avg.length > 1e-6:
            result.append(avg.normalized())
        else:
            result.append(chunk[len(chunk)//2].copy())
    return result, flip_count, valid_count, missing_count


def decompose_swing_twist(q, axis):
    """
    Quaternion을 axis 기준 twist와 그에 수직인 swing으로 분해.
    axis: Mixamo Hand Bone local Y = Vector((0,1,0))
    반환: (swing_q, twist_q, twist_angle_deg, swing_angle_deg, total_angle_deg)
    twist_angle_deg: [-180, +180] 범위, axis 방향 기준 부호 포함
    """
    axis = axis.normalized()
    q = q.normalized()
    total_angle_deg = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, abs(q.w)))))
    # twist: q의 회전벡터를 axis에 투영
    r = mathutils.Vector((q.x, q.y, q.z))
    proj = r.dot(axis) * axis
    twist_q = mathutils.Quaternion((q.w, proj.x, proj.y, proj.z))
    if twist_q.magnitude < 1e-10:
        twist_q = mathutils.Quaternion()
    else:
        twist_q.normalize()
    # signed twist angle
    twist_half = math.acos(max(-1.0, min(1.0, twist_q.w)))
    twist_vec   = mathutils.Vector((twist_q.x, twist_q.y, twist_q.z))
    sign        = 1.0 if twist_vec.dot(axis) >= 0 else -1.0
    twist_angle_deg = sign * math.degrees(2.0 * twist_half)
    while twist_angle_deg >  180.0: twist_angle_deg -= 360.0
    while twist_angle_deg < -180.0: twist_angle_deg += 360.0
    # swing = twist⁻¹ · q
    swing_q = twist_q.inverted() @ q
    swing_q.normalize()
    swing_angle_deg = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, abs(swing_q.w)))))
    return swing_q, twist_q, twist_angle_deg, swing_angle_deg, total_angle_deg


def smooth_track(vecs, window=SMOOTH_WINDOW, jump=JUMP_THRESHOLD):
    n = len(vecs)
    if n == 0:
        return []
    f = list(vecs)
    first = next((i for i, v in enumerate(f) if v is not None), None)
    if first is None:
        return [mathutils.Vector((0,0,0))] * n
    for i in range(first): f[i] = f[first].copy()
    last = next((i for i in range(n-1,-1,-1) if f[i] is not None), 0)
    for i in range(last+1, n): f[i] = f[last].copy()
    i = 0
    while i < n:
        if f[i] is None:
            a, b = i-1, i
            while b < n and f[b] is None: b += 1
            for k in range(i, b): f[k] = f[a].lerp(f[b], (k-a)/(b-a))
            i = b
        else:
            i += 1
    for _ in range(2):
        for i in range(1, n-1):
            avg = f[i-1].lerp(f[i+1], 0.5)
            if (f[i]-avg).length > jump: f[i] = avg
    half = window // 2
    result = []
    for i in range(n):
        chunk = f[max(0,i-half):min(n,i+half+1)]
        result.append(mathutils.Vector((
            sum(v.x for v in chunk)/len(chunk),
            sum(v.y for v in chunk)/len(chunk),
            sum(v.z for v in chunk)/len(chunk),
        )))
    return result


def make_empty(name, col):
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
        obj.hide_viewport = True
        obj.animation_data_clear()
        return obj
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_size = 0.04
    obj.hide_viewport = True
    col.objects.link(obj)
    return obj


def add_ik(arm_obj, bone_name, target, chain_count, name=None):
    b = arm_obj.pose.bones.get(bone_name)
    if not b:
        print(f"[경고] 뼈 없음: {bone_name}"); return None
    # 기존 IK constraint 전부 제거 (중복 방지)
    for _c in list(b.constraints):
        if _c.type == 'IK': b.constraints.remove(_c)
    c = b.constraints.new('IK')
    if name: c.name = name
    c.target = target; c.chain_count = chain_count; c.use_tail = True
    c.use_rotation = False; c.use_stretch = False; c.influence = 1.0
    c.iterations = 500
    return c


def animate_empty(emp, positions, frames):
    emp.animation_data_create()
    if emp.animation_data.action is None:
        emp.animation_data.action = bpy.data.actions.new(emp.name+"_act")
    for frame, pos in zip(frames, positions):
        emp.location = pos
        emp.keyframe_insert('location', frame=frame)


def bone_arma_mat(arm_obj, bone_name, parent_mat=None):
    """뼈의 현재 rotation_quaternion으로 armature-space 3x3 matrix 계산."""
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        return parent_mat or mathutils.Matrix.Identity(3)
    rest      = bd.matrix_local.to_3x3()
    local_rot = b.rotation_quaternion.to_matrix()
    if b.parent:
        p_bd   = arm_obj.data.bones.get(b.parent.name)
        p_rest = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        p_mat  = parent_mat if parent_mat is not None else b.parent.matrix.to_3x3()
        return p_mat @ p_rest.inverted() @ rest @ local_rot
    return rest @ local_rot


def set_bone_dir(arm_obj, bone_name, from_w, to_w, parent_mat=None, roll_pn=None):
    """
    뼈를 from_w→to_w 방향으로 FK 설정.
    roll_pn: palm_normal (world) — 지정 시 Z축을 손바닥 방향으로 roll 보정.
    parent_mat: 부모 뼈의 현재 armature-space 3x3 (propagation용).
    반환: (성공, 이 뼈의 새 armature-space 3x3)
    """
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        return False, parent_mat

    direction = (to_w - from_w)
    if direction.length < 1e-6:
        return False, parent_mat
    direction = direction.normalized()

    a_inv  = arm_obj.matrix_world.to_3x3().inverted()
    dir_a  = (a_inv @ direction).normalized()

    rest = bd.matrix_local.to_3x3()
    ry   = rest.col[1].normalized()
    rx   = rest.col[0].normalized()
    rz   = rest.col[2].normalized()

    q  = ry.rotation_difference(dir_a)
    ny = dir_a
    nx = (q @ rx).normalized()
    nz = (q @ rz).normalized()

    # palm_normal roll 보정 — Z축을 손바닥 방향으로 정렬
    if roll_pn is not None and roll_pn.length > 1e-5:
        pna = (a_inv @ roll_pn).normalized()
        pp  = pna - pna.dot(ny) * ny
        zp  = nz  - nz.dot(ny)  * ny
        if pp.length > 1e-4 and zp.length > 1e-4:
            pp = pp.normalized()
            zp = zp.normalized()
            cr = max(-1.0, min(1.0, zp.dot(pp)))
            ra = math.atan2(zp.cross(pp).dot(ny), cr)
            rq = mathutils.Quaternion(ny, ra)
            nx = rq @ nx
            nz = rq @ nz

    desired_a = mathutils.Matrix([
        [nx.x, ny.x, nz.x],
        [nx.y, ny.y, nz.y],
        [nx.z, ny.z, nz.z],
    ])

    if b.parent:
        p_bd   = arm_obj.data.bones.get(b.parent.name)
        p_rest = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        p_mat  = parent_mat if parent_mat is not None else b.parent.matrix.to_3x3()
        pose_mat = rest.inverted() @ p_rest @ p_mat.inverted() @ desired_a
    else:
        pose_mat = rest.inverted() @ desired_a

    b.rotation_mode = 'QUATERNION'
    b.rotation_quaternion = pose_mat.to_quaternion()

    # 자식 뼈를 위해 새 matrix propagation
    local_rot_mat = b.rotation_quaternion.to_matrix()
    if b.parent:
        p_mat_used = parent_mat if parent_mat is not None else b.parent.matrix.to_3x3()
        p_bd       = arm_obj.data.bones.get(b.parent.name)
        p_rest     = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        new_mat    = p_mat_used @ p_rest.inverted() @ rest @ local_rot_mat
    else:
        new_mat = rest @ local_rot_mat

    return True, new_mat


def get_parent_mat(arm_obj, bone_name, anchor_name, anchor_mat):
    """
    bone_name의 실제 부모 뼈의 armature-space 3x3을 anchor_mat 기준으로 계산.
    anchor_name → (메타카팔 등 중간 뼈) → bone_name.parent 까지 propagate.
    메타카팔이 없을 경우 anchor_mat을 그대로 반환.
    """
    b = arm_obj.pose.bones.get(bone_name)
    if not b or not b.parent or b.parent.name == anchor_name:
        return anchor_mat
    # anchor_name 에서 bone.parent 까지의 중간 뼈 수집
    chain = []
    cur = b.parent
    while cur and cur.name != anchor_name:
        chain.append(cur.name)
        cur = cur.parent
    chain.reverse()   # anchor 쪽이 앞으로
    mat = anchor_mat
    for inter in chain:
        mat = bone_arma_mat(arm_obj, inter, mat)
    return mat


def set_hand_rotation(arm_obj, bone_name, wrist_w, mid9_w, palm_n_w, flip_palm):
    """손뼈: 방향(wrist→mid9) + 손바닥 법선으로 roll 설정."""
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd: return False

    hand_dir = (mid9_w - wrist_w)
    if hand_dir.length < 1e-6: return False
    hand_dir = hand_dir.normalized()

    a_inv  = arm_obj.matrix_world.to_3x3().inverted()
    dir_a  = (a_inv @ hand_dir).normalized()

    rest = bd.matrix_local.to_3x3()
    ry   = rest.col[1].normalized()
    rx   = rest.col[0].normalized()
    rz   = rest.col[2].normalized()

    q  = ry.rotation_difference(dir_a)
    ny = dir_a
    nx = (q @ rx).normalized()
    nz = (q @ rz).normalized()

    pn_a = (a_inv @ palm_n_w)
    if pn_a.length > 1e-4:
        pn_a = pn_a.normalized()
        if flip_palm: pn_a = -pn_a
        pn_perp = pn_a - pn_a.dot(ny)*ny
        nz_perp = nz - nz.dot(ny)*ny
        if pn_perp.length > 1e-4 and nz_perp.length > 1e-4:
            pn_perp    = pn_perp.normalized()
            nz_perp    = nz_perp.normalized()
            cos_r      = max(-1.0, min(1.0, nz_perp.dot(pn_perp)))
            roll_angle = math.atan2(nz_perp.cross(pn_perp).dot(ny), cos_r)
            rq         = mathutils.Quaternion(ny, roll_angle)
            nx         = rq @ nx
            nz         = rq @ nz

    desired_a = mathutils.Matrix([
        [nx.x, ny.x, nz.x],
        [nx.y, ny.y, nz.y],
        [nx.z, ny.z, nz.z],
    ])

    if b.parent:
        p_bd     = arm_obj.data.bones.get(b.parent.name)
        p_rest   = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        pose_mat = rest.inverted() @ p_rest @ b.parent.matrix.to_3x3().inverted() @ desired_a
    else:
        pose_mat = rest.inverted() @ desired_a

    b.rotation_mode = 'QUATERNION'
    b.rotation_quaternion = pose_mat.to_quaternion()
    return True


# ================================================================
# 시각화 / 진단 헬퍼
# ================================================================
_HAND_EDGES = [
    # 손바닥
    (0,1),(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17),
    # 엄지
    (1,2),(2,3),(3,4),
    # 검지
    (5,6),(6,7),(7,8),
    # 중지
    (9,10),(10,11),(11,12),
    # 약지
    (13,14),(14,15),(15,16),
    # 소지
    (17,18),(18,19),(19,20),
]


def make_hand_skeleton(pts21, name='HandSkeleton_L'):
    """
    21개 Hand 랜드마크를 Blender 메쉬로 시각화.
    정면/측면/3/4뷰에서 손 모양과 깊이 분포를 확인한다.
    """
    verts = [tuple(p) for p in pts21]
    old_mesh = bpy.data.meshes.get(name)
    if old_mesh:
        bpy.data.meshes.remove(old_mesh)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, _HAND_EDGES, [])
    mesh.update()
    old_obj = bpy.data.objects.get(name)
    if old_obj:
        bpy.data.objects.remove(old_obj)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    print(f"  [시각화] '{name}' 생성 — 정면/측면/3/4뷰에서 확인")
    return obj


def diag_index_finger(arm_obj, pts, hand_bone_name, index_chain, roll_pn=None):
    """
    검지 3개 뼈 FK 적용 + 수식 오차 / Blender 평가 오차 출력.
    arm_obj 는 이미 POSE 모드이어야 한다.
    Mixamo rest axis (tail-head, col[0..2]) 도 함께 출력.
    roll_pn: 손바닥 법선 (실제 애니메이션과 동일하게 적용하려면 l_pn 전달)
    """
    pn_label = "O (palm-normal 적용)" if roll_pn is not None else "X (없음)"
    print(f"\n  ── 검지 FK 진단  (hand_bone={hand_bone_name}  roll_pn={pn_label}) ──")

    # ① Mixamo rest axis 진단 (Hand 포함)
    print("  [A] rest axis 진단:")
    for check_bn in [hand_bone_name] + [bn for bn, _, _ in index_chain]:
        b  = arm_obj.pose.bones.get(check_bn)
        bd = arm_obj.data.bones.get(check_bn)
        if not b or not bd:
            print(f"    {check_bn}: 뼈 없음"); continue
        parent_name = b.parent.name if b.parent else 'None'
        ml  = bd.matrix_local.to_3x3()
        th  = (bd.tail_local - bd.head_local)
        th_n = th.normalized() if th.length > 1e-6 else mathutils.Vector((0,0,0))
        print(f"    {check_bn}")
        print(f"      parent={parent_name}  connected={bd.use_connect}")
        print(f"      tail-head: {th_n.x:+.3f} {th_n.y:+.3f} {th_n.z:+.3f}")
        print(f"      col[0]:    {ml.col[0].x:+.3f} {ml.col[0].y:+.3f} {ml.col[0].z:+.3f}")
        print(f"      col[1]:    {ml.col[1].x:+.3f} {ml.col[1].y:+.3f} {ml.col[1].z:+.3f}")
        print(f"      col[2]:    {ml.col[2].x:+.3f} {ml.col[2].y:+.3f} {ml.col[2].z:+.3f}")
        print(f"      constraints: {[c.type for c in b.constraints]}")

    # ② 검지 FK 설정 + 수식 내부 오차
    print("\n  [B] FK 적용 및 수식 오차:")
    hand_mat = bone_arma_mat(arm_obj, hand_bone_name)
    # apply_hand 와 동일한 방식: anchor_mat은 고정, p_mat만 체인 내에서 propagate
    p_mat = get_parent_mat(arm_obj, index_chain[0][0], hand_bone_name, hand_mat)

    for (bn, from_idx, to_idx) in index_chain:
        from_w = pts[from_idx]
        to_w   = pts[to_idx]

        direction = (to_w - from_w)
        if direction.length < 1e-6:
            print(f"    {bn}: 방향 계산 불가"); continue
        direction = direction.normalized()

        a_inv = arm_obj.matrix_world.to_3x3().inverted()
        dir_a = (a_inv @ direction).normalized()

        bd = arm_obj.data.bones.get(bn)
        rest_y = bd.matrix_local.to_3x3().col[1].normalized() if bd else mathutils.Vector((0,1,0))
        b = arm_obj.pose.bones.get(bn)
        parent_name = b.parent.name if b and b.parent else 'None'

        ok, new_mat = set_bone_dir(arm_obj, bn, from_w, to_w, parent_mat=p_mat, roll_pn=roll_pn)

        if ok:
            p_mat = new_mat  # 다음 뼈를 위해 propagate (apply_hand와 동일)
            q = b.rotation_quaternion
            # new_mat.col[1] should ≈ dir_a (수식 검증)
            formula_y = new_mat.col[1].normalized()
            try:
                formula_err = math.degrees(formula_y.angle(dir_a))
            except Exception:
                formula_err = 999.0

            print(f"\n    [{bn}]  MP {from_idx}→{to_idx}  parent={parent_name}")
            print(f"      from_w: ({from_w.x:.4f}, {from_w.y:.4f}, {from_w.z:.4f})")
            print(f"      to_w:   ({to_w.x:.4f},   {to_w.y:.4f},   {to_w.z:.4f})")
            print(f"      target (arma): {dir_a.x:+.3f} {dir_a.y:+.3f} {dir_a.z:+.3f}")
            print(f"      rest_Y (arma): {rest_y.x:+.3f} {rest_y.y:+.3f} {rest_y.z:+.3f}")
            print(f"      quat: w={q.w:.3f} x={q.x:.3f} y={q.y:.3f} z={q.z:.3f}")
            print(f"      수식 Y:  {formula_y.x:+.3f} {formula_y.y:+.3f} {formula_y.z:+.3f}")
            print(f"      수식 오차: {formula_err:.2f}도")
            b.keyframe_insert('rotation_quaternion', frame=1)
        else:
            print(f"    {bn}: set_bone_dir 실패 (뼈 없음 또는 거리 0)")

    # ③ Blender 평가 후 실제 방향 오차 (Y=forward, Z=roll)
    bpy.context.view_layer.update()
    print("\n  [C] Blender 평가 후 각도 오차  (Y=forward  Z=roll):")
    a_inv_c = arm_obj.matrix_world.to_3x3().inverted()
    for (bn, from_idx, to_idx) in index_chain:
        b  = arm_obj.pose.bones.get(bn)
        if not b: continue
        direction = (pts[to_idx] - pts[from_idx])
        if direction.length < 1e-6: continue
        dir_a  = (a_inv_c @ direction.normalized()).normalized()
        eval_y = b.matrix.to_3x3().col[1].normalized()
        eval_z = b.matrix.to_3x3().col[2].normalized()
        try:    fwd_err = math.degrees(eval_y.angle(dir_a))
        except: fwd_err = 999.0
        # roll 오차: palm_normal 사용 시만 의미 있음
        if roll_pn is not None and roll_pn.length > 1e-5:
            pn_a = (a_inv_c @ roll_pn.normalized()).normalized()
            pn_perp = pn_a - pn_a.dot(dir_a) * dir_a
            z_perp  = eval_z - eval_z.dot(dir_a) * dir_a
            if pn_perp.length > 1e-4 and z_perp.length > 1e-4:
                try:    roll_err = math.degrees(z_perp.normalized().angle(pn_perp.normalized()))
                except: roll_err = 999.0
            else: roll_err = 999.0
            roll_str = f"  roll_err={roll_err:.1f}도"
        else:
            roll_str = "  roll_err=N/A (roll_pn 없음)"
        print(f"    {bn}: fwd_err={fwd_err:.1f}도{roll_str}")
        print(f"      target_Y: {dir_a.x:+.3f} {dir_a.y:+.3f} {dir_a.z:+.3f}")
        print(f"      eval_Y:   {eval_y.x:+.3f} {eval_y.y:+.3f} {eval_y.z:+.3f}")


# ================================================================
# 메인
# ================================================================
def main():
    arm_obj = next((o for o in bpy.data.objects if o.type=='ARMATURE'), None)
    if not arm_obj: print("[오류] 아마추어 없음"); return
    print(f"[정보] 아마추어: {arm_obj.name}")

    with open(JSON_PATH, encoding='utf-8') as f:
        recording = json.load(f)
    frames_data = recording.get('Frames', [])
    _motion_id = recording.get('MotionId', '')
    _motion_override = MOTION_OVERRIDES.get(_motion_id, {})
    _contact_mode = _motion_override.get(
        'contact_mode',
        TWO_HAND_CONTACT_MOTION_PROFILES.get(
            _motion_id, TWO_HAND_CONTACT_DEFAULT_MODE))
    _support_slide_profile = (
        SUPPORT_SLIDE_MOTION_PROFILES.get(_motion_id)
        if _contact_mode == "support_slide" else None)
    _wrist_aware_target_sides = set(_motion_override.get(
        'wrist_aware_sides', WRIST_AWARE_TARGET_SIDES))
    _forearm_twist_target_sides = set(_motion_override.get(
        'forearm_twist_sides', FOREARM_TWIST_TARGET_SIDES))
    _fist_mode = _motion_override.get(
        'fist_mode', FIST_HANDSHAPE_DEFAULT_MODE)
    _fist_target_sides = set(_motion_override.get(
        'fist_sides', FIST_HANDSHAPE_TARGET_SIDES))
    _motion_contact_enabled = (
        USE_TWO_HAND_CONTACT_PRESERVATION and
        _contact_mode in {"auto", "touch"})
    _motion_support_slide_enabled = (
        USE_TWO_HAND_CONTACT_PRESERVATION and
        _contact_mode == "support_slide" and
        _support_slide_profile is not None)
    _motion_fist_enabled = (
        USE_FIST_HANDSHAPE_STABILIZATION and
        _fist_mode == "auto")
    print(f"[MOTION-AUTOGUARD]"
          f"  motion_id={_motion_id!r}"
          f"  contact_mode={_contact_mode!r}"
          f"  support_slide={_motion_support_slide_enabled}"
          f"  wrist_aware_sides={sorted(_wrist_aware_target_sides)}"
          f"  forearm_twist_sides={sorted(_forearm_twist_target_sides)}"
          f"  fist_mode={_fist_mode!r}"
          f"  fist_sides={sorted(_fist_target_sides)}")
    bpy.context.scene.render.fps = FPS

    # 캐릭터 어깨 기준
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    lb = arm_obj.pose.bones.get(BONE_L_ARM)
    rb = arm_obj.pose.bones.get(BONE_R_ARM)
    _lb_fa = arm_obj.pose.bones.get(BONE_L_FOREARM)
    _rb_fa = arm_obj.pose.bones.get(BONE_R_FOREARM)
    _mw = arm_obj.matrix_world
    def _blen(pb): return (_mw @ pb.tail - _mw @ pb.head).length if pb else 0.0
    char_l_upper_len   = _blen(lb)
    char_l_forearm_len = _blen(_lb_fa)
    char_r_upper_len   = _blen(rb)
    char_r_forearm_len = _blen(_rb_fa)
    char_l_arm_total   = char_l_upper_len + char_l_forearm_len
    char_r_arm_total   = char_r_upper_len + char_r_forearm_len
    if lb and rb:
        l_sh_w = arm_obj.matrix_world @ lb.head
        r_sh_w = arm_obj.matrix_world @ rb.head
        char_origin = (l_sh_w + r_sh_w) / 2
        char_sw     = (l_sh_w - r_sh_w).length
    else:
        char_origin = mathutils.Vector((0,0,0)); char_sw = 1.5
        l_sh_w = char_origin + mathutils.Vector((-char_sw/2,0,0))
        r_sh_w = char_origin + mathutils.Vector(( char_sw/2,0,0))
    # rest-pose 팔꿈치 world (data bone tail_local = armature-space rest 좌표)
    _l_arm_data = arm_obj.data.bones.get(BONE_L_ARM)
    _r_arm_data = arm_obj.data.bones.get(BONE_R_ARM)
    _rest_elbow_l_w = (_mw @ _l_arm_data.tail_local) if _l_arm_data else (l_sh_w + mathutils.Vector((0.4, 0, -0.1)))
    _rest_elbow_r_w = (_mw @ _r_arm_data.tail_local) if _r_arm_data else (r_sh_w + mathutils.Vector((-0.4, 0, -0.1)))
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[정보] 캐릭터 팔 길이 L:{char_l_arm_total:.4f}  R:{char_r_arm_total:.4f}")
    print(f"[정보] 어깨 중심: {char_origin}  너비: {char_sw:.3f}")

    # ── 데이터 추출 ──────────────────────────────────────────────
    # l_p0..l_p20: 캐릭터 LEFT 손의 21개 랜드마크 Blender 월드 좌표
    # r_p0..r_p20: 캐릭터 RIGHT 손의 21개 랜드마크 Blender 월드 좌표
    raw = {
        'lw':[], 'rw':[],
        'lw_ik':[], 'rw_ik':[],
        'lh9':[], 'rh9':[],
        'l_pn':[], 'r_pn':[],
        **{f'l_p{i}':[] for i in range(21)},
        **{f'r_p{i}':[] for i in range(21)},
        'l_valid':[], 'r_valid':[],
    }

    blender_frame = 0
    # [hybrid_cnt, missing_fallback_cnt, [(ts, diff_sw), ...rejected], [diff_sw, ...raw_all_valid]]
    _gate_l = [0, 0, [], []]; _gate_r = [0, 0, [], []]
    _depth_l = []; _depth_r = []  # depth_sw 누적 (IK-DEPTH 진단)
    _reach_l = []; _reach_r = []  # (ts, src_ratio, char_ratio) 누적
    _src_ts     = []  # 유효 pose 프레임별 TimestampMs
    _src_ratio_l   = []; _src_ratio_r   = []  # 3D source reach ratio
    _src_ratio2d_l = []; _src_ratio2d_r = []  # 2D source reach ratio (depth 미사용)
    _pole_src_l    = []; _pole_src_r    = []  # pole bend 방향 Vector
    _upper_src_l   = []  # source upper-arm 방향 (normalized, per source frame)
    _upper_src_r   = []
    _forearm_src_l = []  # source forearm 방향 (elbow→wrist, normalized, per source frame)
    _forearm_src_r = []
    # Pose elbow → 실제 IK에 사용한 hybrid wrist(Hand pt0 XZ + Pose wrist Y).
    # 손 방향은 Hand pt0→pt9를 쓰므로, Pose wrist 기준 진단과 분리해 비교한다.
    _forearm_hybrid_src_l = []
    _forearm_hybrid_src_r = []
    _contact_raw2d_l  = []  # raw MediaPipe Hand (x,y) × 21, per source frame
    _contact_raw2d_r  = []
    _contact_pose_sw  = []  # 해당 frame의 pose shoulder width (image 좌표)
    _DIAG_TS = 1333
    _diag_best_diff = float('inf'); _diag_info = None
    _ik_depth_diag_best = float('inf'); _ik_depth_diag = None
    for fd in frames_data:
        if not fd.get('HasPose', False): continue
        pose_pts = fd.get('PosePoints') or []
        mp_l_pts = fd.get('LeftHandPoints') or []
        mp_r_pts = fd.get('RightHandPoints') or []
        has_mp_l = fd.get('HasLeftHand', False)
        has_mp_r = fd.get('HasRightHand', False)

        if SWAP_HAND_LABELS:
            ch_l_pts = mp_r_pts; has_ch_l = has_mp_r
            ch_r_pts = mp_l_pts; has_ch_r = has_mp_l
        else:
            ch_l_pts = mp_l_pts; has_ch_l = has_mp_l
            ch_r_pts = mp_r_pts; has_ch_r = has_mp_r

        ls = get_pt(pose_pts, P_L_SHOULDER)
        rs = get_pt(pose_pts, P_R_SHOULDER)
        if not ls or not rs: continue
        cx = (ls[0]+rs[0])/2; cy = (ls[1]+rs[1])/2
        sw = math.sqrt((ls[0]-rs[0])**2+(ls[1]-rs[1])**2)
        if sw < 1e-4: continue
        blender_frame += 1

        def bp(idx):
            """Pose landmark → Blender world (손목 IK 타겟, 깊이 0)."""
            pt = get_pt(pose_pts, idx)
            return to_blender(*pt, cx, cy, sw, char_origin, char_sw) if pt else None

        def chp(idx, pts):
            """Hand landmark → Blender world (convert_hand_pt 래퍼)."""
            pt = get_pt(pts, idx)
            return convert_hand_pt(pt[0],pt[1],pt[2], cx,cy,sw,char_origin,char_sw) if pt else None

        raw['lw'].append(bp(P_L_WRIST))
        raw['rw'].append(bp(P_R_WRIST))

        # IK 타겟: Hand pt0 XZ + Pose wrist Y (hybrid) — gate로 이상치 차단
        _lp0 = chp(H_WRIST, ch_l_pts) if has_ch_l else None
        _rp0 = chp(H_WRIST, ch_r_pts) if has_ch_r else None
        _ts = fd.get('TimestampMs', blender_frame)

        # Pose wrist Y에 상대 깊이 복원
        _shoulder_z = (ls[2] + rs[2]) / 2
        def _pose_ik_vec(base_vec, raw_pt, depth_log):
            if not USE_POSE_WRIST_DEPTH_FOR_IK or base_vec is None or raw_pt is None:
                return base_vec
            ds = (raw_pt[2] - _shoulder_z) * POSE_WRIST_DEPTH_GAIN
            ds = max(-POSE_WRIST_DEPTH_MAX_SW, min(POSE_WRIST_DEPTH_MAX_SW, ds))
            depth_log.append(ds)
            depth_y = char_origin.y - FORWARD_OFFSET + ds * char_sw
            return mathutils.Vector((base_vec.x, depth_y, base_vec.z))
        _lw_pose = _pose_ik_vec(raw['lw'][-1], get_pt(pose_pts, P_L_WRIST), _depth_l)
        _rw_pose = _pose_ik_vec(raw['rw'][-1], get_pt(pose_pts, P_R_WRIST), _depth_r)

        def _hybrid_ik(hp0, pw, gate_log):
            if USE_HAND_WRIST_XZ_FOR_IK and hp0 is not None and pw is not None:
                diff_sw = math.sqrt((hp0.x-pw.x)**2 + (hp0.z-pw.z)**2) / char_sw
                gate_log[3].append(diff_sw)  # RAW: gate 전 원본 차이
                if diff_sw <= HAND_WRIST_IK_MAX_OFFSET_SW:
                    gate_log[0] += 1  # hybrid 사용
                    return mathutils.Vector((hp0.x, pw.y, hp0.z))
                else:
                    gate_log[2].append((_ts, diff_sw))  # gate 거부 → Pose fallback
                    return pw
            gate_log[1] += 1  # missing_fallback (hand 미검출 또는 pose 없음)
            return pw

        raw['lw_ik'].append(_hybrid_ik(_lp0, _lw_pose, _gate_l))
        raw['rw_ik'].append(_hybrid_ik(_rp0, _rw_pose, _gate_r))

        _diff_ts = abs(_ts - _DIAG_TS)
        if _diff_ts < _ik_depth_diag_best:
            _ik_depth_diag_best = _diff_ts
            _lv = raw['lw_ik'][-1]; _rv = raw['rw_ik'][-1]
            _ik_depth_diag = {'ts': _ts,
                              'l_y': _lv.y if _lv else None,
                              'r_y': _rv.y if _rv else None}

        _src_ts.append(_ts)  # 유효 pose 프레임 타임스탬프 (smooth index 정렬)
        # raw MediaPipe Hand x/y (접촉 판정 전용 — SWAP 적용 후, z 미포함)
        _contact_raw2d_l.append(
            [(p.get('x', 0), p.get('y', 0)) for p in ch_l_pts[:21]] if has_ch_l and len(ch_l_pts) >= 21 else None)
        _contact_raw2d_r.append(
            [(p.get('x', 0), p.get('y', 0)) for p in ch_r_pts[:21]] if has_ch_r and len(ch_r_pts) >= 21 else None)
        _contact_pose_sw.append(sw)
        # Reach source ratio: 3D 정규화 좌표 (depth 포함, bp() 미사용)
        _r_sz = (ls[2] + rs[2]) / 2
        def _np(idx):
            pt = get_pt(pose_pts, idx)
            return mathutils.Vector(((pt[0]-cx)/sw, (pt[2]-_r_sz)/sw, -(pt[1]-cy)/sw)) if pt else None
        _sr_frame = {}
        for _p_sh, _p_el, _p_wr, _rlog, _srk in [
                (P_L_SHOULDER, P_L_ELBOW, P_L_WRIST, _reach_l, 'L'),
                (P_R_SHOULDER, P_R_ELBOW, P_R_WRIST, _reach_r, 'R')]:
            _np_sh = _np(_p_sh); _np_el = _np(_p_el); _np_wr = _np(_p_wr)
            if _np_sh and _np_el and _np_wr:
                _dse = (_np_sh-_np_el).length; _dew = (_np_el-_np_wr).length
                _src_r = (_np_sh-_np_wr).length / (_dse+_dew) if (_dse+_dew) > 1e-6 else 0.0
                _rlog.append((_ts, _src_r)); _sr_frame[_srk] = _src_r
        _src_ratio_l.append(_sr_frame.get('L')); _src_ratio_r.append(_sr_frame.get('R'))

        # 2D source reach ratio (pt.z 미사용)
        def _np2d(idx):
            pt = get_pt(pose_pts, idx)
            return mathutils.Vector(((pt[0]-cx)/sw, 0.0, -(pt[1]-cy)/sw)) if pt else None
        _sr2d = {}
        for _p_sh2, _p_el2, _p_wr2, _rk2 in [
                (P_L_SHOULDER, P_L_ELBOW, P_L_WRIST, 'L'),
                (P_R_SHOULDER, P_R_ELBOW, P_R_WRIST, 'R')]:
            _s2 = _np2d(_p_sh2); _e2 = _np2d(_p_el2); _w2 = _np2d(_p_wr2)
            if _s2 and _e2 and _w2:
                _dse2 = (_s2-_e2).length; _dew2 = (_e2-_w2).length
                _sr2d[_rk2] = (_s2-_w2).length / (_dse2+_dew2) if (_dse2+_dew2) > 1e-6 else 0.0
        _src_ratio2d_l.append(_sr2d.get('L')); _src_ratio2d_r.append(_sr2d.get('R'))

        # Pole source: elbow bend 방향 (정규화 공간 → Blender world 방향)
        _xs_p = 1 if MIRROR_X else -1
        def _bend_src(p_sh, p_el, p_wr):
            sh = _np(p_sh); el = _np(p_el); wr = _np(p_wr)
            if not (sh and el and wr): return None
            arm_v = wr - sh
            if arm_v.length < 1e-6: return None
            arm_n = arm_v / arm_v.length
            bv = (el - sh) - arm_n * (el - sh).dot(arm_n)
            return mathutils.Vector((_xs_p * bv.x, bv.y, bv.z)) if bv.length > 1e-4 else None
        _pole_src_l.append(_bend_src(P_L_SHOULDER, P_L_ELBOW, P_L_WRIST))
        _pole_src_r.append(_bend_src(P_R_SHOULDER, P_R_ELBOW, P_R_WRIST))

        # Source upper-arm 방향: shoulder → elbow (정규화 공간 → Blender world 방향)
        def _upper_arm_dir(p_sh, p_el):
            sh = _np(p_sh); el = _np(p_el)
            if not (sh and el): return None
            uv = el - sh
            world_uv = mathutils.Vector((_xs_p * uv.x, uv.y, uv.z))
            return world_uv.normalized() if world_uv.length >= SOURCE_UPPER_ARM_MIN_MAGNITUDE else None
        _upper_src_l.append(_upper_arm_dir(P_L_SHOULDER, P_L_ELBOW))
        _upper_src_r.append(_upper_arm_dir(P_R_SHOULDER, P_R_ELBOW))

        # Source forearm 방향: elbow → wrist (정규화 공간 → Blender world 방향)
        def _forearm_dir(p_el, p_wr):
            el = _np(p_el); wr = _np(p_wr)
            if not (el and wr): return None
            fv = wr - el
            world_fv = mathutils.Vector((_xs_p * fv.x, fv.y, fv.z))
            return world_fv.normalized() if world_fv.length > 1e-6 else None
        _forearm_src_l.append(_forearm_dir(P_L_ELBOW, P_L_WRIST))
        _forearm_src_r.append(_forearm_dir(P_R_ELBOW, P_R_WRIST))

        # IK 입력과 동일한 좌표계로 Pose elbow를 만들고, 이미 gate가 적용된
        # raw['*_ik']의 hybrid wrist까지의 방향을 기록한다.
        def _pose_depth_world(p_idx):
            _base = bp(p_idx)
            _pt = get_pt(pose_pts, p_idx)
            if _base is None or _pt is None:
                return None
            if not USE_POSE_WRIST_DEPTH_FOR_IK:
                return _base
            _ds = (_pt[2] - _shoulder_z) * POSE_WRIST_DEPTH_GAIN
            _ds = max(-POSE_WRIST_DEPTH_MAX_SW, min(POSE_WRIST_DEPTH_MAX_SW, _ds))
            return mathutils.Vector((
                _base.x,
                char_origin.y - FORWARD_OFFSET + _ds * char_sw,
                _base.z,
            ))

        def _hybrid_forearm_dir(p_el, hybrid_wrist):
            _el_w = _pose_depth_world(p_el)
            if _el_w is None or hybrid_wrist is None:
                return None
            _fv = hybrid_wrist - _el_w
            return _fv.normalized() if _fv.length > 1e-6 else None

        _forearm_hybrid_src_l.append(
            _hybrid_forearm_dir(P_L_ELBOW, raw['lw_ik'][-1]))
        _forearm_hybrid_src_r.append(
            _hybrid_forearm_dir(P_R_ELBOW, raw['rw_ik'][-1]))

        raw['lh9'].append(chp(H_MID_MCP, ch_l_pts) if has_ch_l else None)
        raw['rh9'].append(chp(H_MID_MCP, ch_r_pts) if has_ch_r else None)

        # 21개 랜드마크 전부 저장 (convert_hand_pt — 손가락·법선 동일 스케일)
        for i in range(21):
            raw[f'l_p{i}'].append(chp(i, ch_l_pts) if has_ch_l else None)
            raw[f'r_p{i}'].append(chp(i, ch_r_pts) if has_ch_r else None)

        # 손바닥 법선: (점5-점0)×(점17-점0)  손가락 좌표와 동일 함수 사용
        lw3 = chp(H_WRIST,ch_l_pts); li3=chp(H_INDEX_MCP,ch_l_pts); lp3=chp(H_PINK_MCP,ch_l_pts)
        if has_ch_l and lw3 and li3 and lp3:
            pn = (li3-lw3).cross(lp3-lw3)
            raw['l_pn'].append(pn if pn.length>1e-6 else None)
        else:
            raw['l_pn'].append(None)

        rw3 = chp(H_WRIST,ch_r_pts); ri3=chp(H_INDEX_MCP,ch_r_pts); rp3=chp(H_PINK_MCP,ch_r_pts)
        if has_ch_r and rw3 and ri3 and rp3:
            pn = (ri3-rw3).cross(rp3-rw3)
            raw['r_pn'].append(pn if pn.length>1e-6 else None)
        else:
            raw['r_pn'].append(None)

        raw['l_valid'].append(has_ch_l)
        raw['r_valid'].append(has_ch_r)

        ts = fd.get('TimestampMs', None)
        if ts is not None:
            diff = abs(ts - _DIAG_TS)
            if diff < _diag_best_diff:
                _diag_best_diff = diff
                lw3_d = chp(H_WRIST,   ch_l_pts) if has_ch_l else None
                lh9_d = chp(H_MID_MCP, ch_l_pts) if has_ch_l else None
                rw3_d = chp(H_WRIST,   ch_r_pts) if has_ch_r else None
                rh9_d = chp(H_MID_MCP, ch_r_pts) if has_ch_r else None
                _diag_info = {
                    'ts':      ts,
                    'l_label': 'mp_right' if SWAP_HAND_LABELS else 'mp_left',
                    'r_label': 'mp_left'  if SWAP_HAND_LABELS else 'mp_right',
                    'l_fwd': (lh9_d - lw3_d).normalized() if (lw3_d and lh9_d) else None,
                    'r_fwd': (rh9_d - rw3_d).normalized() if (rw3_d and rh9_d) else None,
                }

    total   = blender_frame
    l_det   = sum(raw['l_valid']); r_det = sum(raw['r_valid'])
    print(f"[정보] 유효 프레임: {total}  손 검출 L:{l_det} R:{r_det}")

    # XZ 진단: RAW(gate 전) / APPLIED(gate 후) + gate 통계
    if USE_HAND_WRIST_XZ_FOR_IK and char_sw > 1e-4:
        for side, ik_key, pose_key, valid_key, g in [
                ('L','lw_ik','lw','l_valid',_gate_l),
                ('R','rw_ik','rw','r_valid',_gate_r)]:
            # RAW: Hand pt0 ↔ Pose wrist gate 전 원본 차이 (유효 프레임 전체)
            if g[3]:
                avg_r = sum(g[3])/len(g[3]); max_r = max(g[3])
                print(f"[IK-XZ-RAW]     {side}: avg={avg_r:.3f}  max={max_r:.3f}  (프레임 {len(g[3])}개)")
            # APPLIED: gate 후 lw_ik ↔ Pose wrist (거부된 프레임은 diff=0)
            app = []
            for i, v in enumerate(raw[valid_key]):
                if not v: continue
                ik = raw[ik_key][i]; pw = raw[pose_key][i]
                if ik is None or pw is None: continue
                app.append(math.sqrt((ik.x-pw.x)**2 + (ik.z-pw.z)**2) / char_sw)
            if app:
                print(f"[IK-XZ-APPLIED] {side}: avg={sum(app)/len(app):.3f}  max={max(app):.3f}  (프레임 {len(app)}개)")
            # Gate 통계 (hybrid + missing_fallback + rejected == 전체 프레임)
            rejected = g[2]
            print(f"[IK-Gate] {side}: hybrid={g[0]}  missing_fallback={g[1]}  rejected={len(rejected)}")
            for ts, ds in rejected[:10]:
                print(f"  거부: ts={ts}  diff_sw={ds:.3f}")

    if USE_POSE_WRIST_DEPTH_FOR_IK:
        for side, dl in [('L', _depth_l), ('R', _depth_r)]:
            if dl:
                print(f"[IK-DEPTH] {side}: avg={sum(dl)/len(dl):+.3f}  min={min(dl):+.3f}  max={max(dl):+.3f}  (char_sw 단위, 프레임 {len(dl)}개)")
        if _ik_depth_diag:
            d2 = _ik_depth_diag
            ly = d2['l_y']; ry = d2['r_y']
            print(f"[IK-DEPTH-TS] ts={d2['ts']}  LEFT_IK_Y={ly:.4f}  RIGHT_IK_Y={ry:.4f}" +
                  (f"  R-L={ry-ly:+.4f}" if (ly is not None and ry is not None) else ""))

    if _diag_info:
        d = _diag_info
        lf = d['l_fwd']; rf = d['r_fwd']
        print(f"\n[진단] TimestampMs≈{_DIAG_TS} → 실제 {d['ts']}ms")
        print(f"  SWAP_HAND_LABELS : {SWAP_HAND_LABELS}")
        print(f"  캐릭터 LEFT  ← {d['l_label']}")
        print(f"  캐릭터 RIGHT ← {d['r_label']}")
        if lf: print(f"  LEFT  wrist→mid9 (world): ({lf.x:+.3f}, {lf.y:+.3f}, {lf.z:+.3f})")
        else:  print(f"  LEFT  wrist→mid9 : 검출 없음")
        if rf: print(f"  RIGHT wrist→mid9 (world): ({rf.x:+.3f}, {rf.y:+.3f}, {rf.z:+.3f})")
        else:  print(f"  RIGHT wrist→mid9 : 검출 없음")
        print(f"  [기대] LEFT X<0(몸 오른쪽방향), RIGHT X>0(몸 왼쪽방향)")

    # ── 스무딩 ───────────────────────────────────────────────────
    # l_pn, r_pn은 hemisphere-stabilized 별도 처리
    smooth = {k: smooth_track(v) for k,v in raw.items() if k not in ('l_valid','r_valid','l_pn','r_pn')}
    # Palm normal: hemisphere continuity 안정화
    if USE_PALM_NORMAL_CONTINUITY:
        _pn_l_track, _pn_l_flips, _pn_l_valid, _pn_l_miss = stabilize_normal_track(raw['l_pn'], window=PALM_NORMAL_SMOOTH_WINDOW)
        _pn_r_track, _pn_r_flips, _pn_r_valid, _pn_r_miss = stabilize_normal_track(raw['r_pn'], window=PALM_NORMAL_SMOOTH_WINDOW)
    else:
        _pn_l_track = smooth_track(raw['l_pn'])
        _pn_r_track = smooth_track(raw['r_pn'])
        _pn_l_flips = _pn_r_flips = _pn_l_valid = _pn_r_valid = _pn_l_miss = _pn_r_miss = 0
    smooth['l_pn'] = _pn_l_track
    smooth['r_pn'] = _pn_r_track

    # ── 손 모양 시계열 상태기계 ─────────────────────────────────
    _fist_weights_l, _fist_runs_l, _fist_metrics_l = build_fist_weight_track(
        raw, 'l', raw['l_valid'], _src_ts,
        _motion_fist_enabled and 'L' in _fist_target_sides)
    _fist_weights_r, _fist_runs_r, _fist_metrics_r = build_fist_weight_track(
        raw, 'r', raw['r_valid'], _src_ts,
        _motion_fist_enabled and 'R' in _fist_target_sides)
    print(f"[FIST-HANDSHAPE-MODE]"
          f"  motion_id={_motion_id!r}"
          f"  global_enabled={USE_FIST_HANDSHAPE_STABILIZATION}"
          f"  mode={_fist_mode!r}"
          f"  target_sides={sorted(_fist_target_sides)}"
          f"  enabled={_motion_fist_enabled}")
    for _fist_side, _fist_runs in (
            ('L', _fist_runs_l), ('R', _fist_runs_r)):
        if not _fist_runs:
            print(f"[FIST-HANDSHAPE-RUN] {_fist_side}: none")
            continue
        for _fist_run in _fist_runs:
            _fist_min = _fist_run['min_mean']
            _fist_min_str = (
                f"{_fist_min:.3f}" if _fist_min is not None else "None")
            print(f"[FIST-HANDSHAPE-RUN] {_fist_side}:"
                  f"  source_index={_fist_run['start']}..{_fist_run['end']}"
                  f"  ts={_fist_run['start_ts']}..{_fist_run['end_ts']}"
                  f"  min_mean_fold_ratio={_fist_min_str}")
            _pre = _fist_run.get('pretransition') or {}
            _pre_start = _pre.get('start', _fist_run['start'])
            _pre_start_ts = (
                _src_ts[_pre_start]
                if _pre_start < len(_src_ts) else None)
            print(f"[FIST-HANDSHAPE-PRETRANSITION] {_fist_side}:"
                  f"  enabled={_pre.get('enabled', False)}"
                  f"  source_index={_pre_start}..{_pre.get('full')}"
                  f"  ts={_pre_start_ts}.."
                  f"{(_src_ts[_pre.get('full')] if isinstance(_pre.get('full'), int) and _pre.get('full') < len(_src_ts) else None)}"
                  f"  valid={_pre.get('valid', 0)}"
                  f"  coherent_fingers="
                  f"{_pre.get('coherent_fingers', 0)}"
                  f"  mean_progress="
                  f"{_pre.get('mean_progress', 0.0):.3f}")

    # ── 주먹 엄지 시계열 이상치 보정 ─────────────────────────────
    # 엄지의 실제 움직임은 원본을 유지한다. 안정 자세에서 갑자기 크게
    # 이탈했다가 같은 자세로 돌아오는 짧은 내부 구간만 추적 이상치로 보고,
    # 앞뒤의 정상 자세 사이를 보간한다. 시작 또는 끝 자세가 다른 엄지
    # 변화와 완만한 엄지 전환은 건드리지 않는다.
    def _thumb_local_feature(_prefix, _frame_index, _flip):
        if (_frame_index >= len(raw[f'{_prefix}_valid']) or
                not raw[f'{_prefix}_valid'][_frame_index]):
            return None
        _pts = [
            smooth[f'{_prefix}_p{_pi}'][_frame_index]
            for _pi in range(21)
        ]
        _pn = smooth[f'{_prefix}_pn'][_frame_index]
        if any(_p is None for _p in _pts) or _pn is None:
            return None
        _normal = (-_pn if _flip else _pn)
        if _normal.length <= 1e-5:
            return None
        _normal = _normal.normalized()
        _forward = _pts[H_MID_MCP] - _pts[H_WRIST]
        _forward -= _normal * _forward.dot(_normal)
        if _forward.length <= 1e-5:
            return None
        _forward.normalize()
        _side = _normal.cross(_forward)
        if _side.length <= 1e-5:
            return None
        _side.normalize()

        _components = []
        for _a, _b in (
                (H_THUMB_CMC, H_THUMB_MCP),
                (H_THUMB_MCP, H_THUMB_IP),
                (H_THUMB_IP, H_THUMB_TIP)):
            _direction = _pts[_b] - _pts[_a]
            if _direction.length <= 1e-5:
                return None
            _direction.normalize()
            _components.append(mathutils.Vector((
                _direction.dot(_forward),
                _direction.dot(_side),
                _direction.dot(_normal),
            )))
        return _components

    def _thumb_feature_distance(_feature_a, _feature_b):
        if _feature_a is None or _feature_b is None:
            return None
        _deviations = []
        for _segment_index in range(3):
            _a = _feature_a[_segment_index]
            _b = _feature_b[_segment_index]
            if _a.length <= 1e-5 or _b.length <= 1e-5:
                return None
            _dot = max(-1.0, min(
                1.0, _a.normalized().dot(_b.normalized())))
            _deviations.append(math.degrees(math.acos(_dot)))
        return sum(_deviations) / len(_deviations)

    def _thumb_chain_straightness(_prefix, _frame_index):
        """엄지 CMC→TIP 직선거리 / 세 관절 체인 길이.

        1에 가까우면 엄지가 펴졌고, 작아질수록 관절이 접혔다.
        길이 비율이라 캐릭터 크기와 카메라 거리에 거의 무관하다.
        """
        if (_frame_index >= len(raw[f'{_prefix}_valid']) or
                not raw[f'{_prefix}_valid'][_frame_index]):
            return None
        _points = [
            smooth[f'{_prefix}_p{_point_index}'][_frame_index]
            for _point_index in (
                H_THUMB_CMC, H_THUMB_MCP,
                H_THUMB_IP, H_THUMB_TIP)
        ]
        if any(_point is None for _point in _points):
            return None
        _chain_length = sum(
            (_points[_segment + 1] - _points[_segment]).length
            for _segment in range(3))
        if _chain_length <= 1e-6:
            return None
        return (_points[3] - _points[0]).length / _chain_length

    def _median_thumb_reference(_indices, _features):
        _valid_features = [
            _features[_frame_index]
            for _frame_index in _indices
            if _frame_index in _features
        ]
        if len(_valid_features) < FIST_THUMB_GESTURE_MIN_STATE_FRAMES:
            return None
        _reference = []
        for _segment_index in range(3):
            _median_direction = mathutils.Vector(tuple(
                statistics.median(
                    _feature[_segment_index][_axis]
                    for _feature in _valid_features)
                for _axis in range(3)
            ))
            if _median_direction.length <= 1e-5:
                return None
            _reference.append(_median_direction.normalized())
        return tuple(_reference)

    def _shape_thumb_state_reference(_reference, _extended):
        """검출한 엄지 상태를 손바닥 로컬 좌표에서 조금 더 명확하게 만든다."""
        if not _reference:
            return None
        _shaped = []
        for _segment_index, _direction in enumerate(_reference):
            if _extended:
                _forward = mathutils.Vector((1.0, 0.0, 0.0))
                _blend = FIST_THUMB_EXTENDED_FORWARD_BLEND[
                    _segment_index]
                _target = _direction.lerp(_forward, _blend)
            else:
                _target = mathutils.Vector((
                    _direction.x *
                    FIST_THUMB_FOLDED_FORWARD_SCALE[_segment_index],
                    _direction.y *
                    FIST_THUMB_FOLDED_SIDE_SCALE[_segment_index],
                    _direction.z *
                    FIST_THUMB_FOLDED_NORMAL_SCALE[_segment_index],
                ))
            if _target.length <= 1e-5:
                _target = _direction.copy()
            else:
                _target.normalize()
            _shaped.append(_target)
        return tuple(_shaped)

    def _thumb_direction_angle_deg(_direction_a, _direction_b):
        if (
                _direction_a is None or _direction_b is None or
                _direction_a.length <= 1e-5 or
                _direction_b.length <= 1e-5):
            return None
        _dot = max(-1.0, min(
            1.0,
            _direction_a.normalized().dot(
                _direction_b.normalized())))
        return math.degrees(math.acos(_dot))

    def _shape_thumb_gesture_pair(
            _early_reference, _late_reference, _early_extended):
        """두 엄지 상태를 일반 대비 또는 CMC 주도 강체 회전으로 만든다.

        MediaPipe에서 엄지가 검지에 가려지면 TIP 방향만 80~100도 뒤집혀
        실제로는 곧은 엄지가 IP에서 꺾인 것처럼 보일 수 있다. 시작 엄지가
        펴져 있고, 뿌리 방향은 의미 있게 이동했으며, 접힌 상태의 끝 관절만
        크게 뒤집힌 경우에는 뿌리(CMC)의 방향 변화만 채택한다. 세 세그먼트
        방향을 같은 축으로 두므로 MCP/IP 상대 굽힘은 전환 내내 고정된다.
        """
        if not _early_reference or not _late_reference:
            return None, None, {
                'mode': None,
                'root_motion_deg': None,
                'late_tip_joint_deg': None,
            }

        _extended_reference = (
            _early_reference if _early_extended
            else _late_reference)
        _folded_reference = (
            _late_reference if _early_extended
            else _early_reference)
        _root_motion_deg = _thumb_direction_angle_deg(
            _extended_reference[0], _folded_reference[0])
        _late_tip_joint_deg = _thumb_direction_angle_deg(
            _folded_reference[1], _folded_reference[2])
        _use_root_pivot = bool(
            _root_motion_deg is not None and
            _late_tip_joint_deg is not None and
            _root_motion_deg >= FIST_THUMB_ROOT_PIVOT_MIN_DEG and
            _late_tip_joint_deg >=
            FIST_THUMB_TIP_REVERSAL_MIN_DEG)

        if not _use_root_pivot:
            return (
                _shape_thumb_state_reference(
                    _early_reference, _early_extended),
                _shape_thumb_state_reference(
                    _late_reference, not _early_extended),
                {
                    'mode': 'source_articulation',
                    'root_motion_deg': _root_motion_deg,
                    'late_tip_joint_deg': _late_tip_joint_deg,
                },
            )

        # 검지 옆 접촉 자세: wrist→middle-MCP 축을 주축으로 사용한다.
        # 이 축은 실제 리그에서 검지 옆에 곧게 놓이는 자세이며, 이전
        # 버전의 57프레임에서 사용자가 올바른 마무리 모양으로 확인했다.
        _extended_side_sample = statistics.median(
            _direction.y for _direction in _extended_reference)
        if abs(_extended_side_sample) <= 1e-4:
            _extended_side_sample = _extended_reference[0].y
        _extended_side_sign = (
            -1.0 if _extended_side_sample < 0.0 else 1.0)
        _extended_normal = statistics.median(
            _direction.z for _direction in _extended_reference)
        _extended_normal = max(
            -FIST_THUMB_EXTENDED_NORMAL_MAX,
            min(FIST_THUMB_EXTENDED_NORMAL_MAX,
                _extended_normal * 0.35))
        _index_contact_axis = mathutils.Vector((
            1.0,
            _extended_side_sign *
            FIST_THUMB_EXTENDED_OUTWARD_COMPONENT,
            _extended_normal,
        ))
        _index_contact_axis.normalize()

        # 엄지척 자세: 가려져 뒤집힌 TIP은 버리고 뿌리의 이동량만
        # 사용한다. side 부호는 초기 엄지의 해부학적 바깥쪽을 유지한다.
        # 이전 버전에서 이 축이 마지막에 엄지가 쭉 펴진 자세로 나타났으므로
        # 이번에는 그 자세를 전환 전의 올바른 엄지척 상태로 배정한다.
        _folded_root = _folded_reference[0].normalized()
        _raised_side = max(
            abs(_folded_root.y) *
            FIST_THUMB_FOLDED_ROOT_SIDE_SCALE,
            FIST_THUMB_FOLDED_ROOT_SIDE_MIN)
        _raised_axis = mathutils.Vector((
            _folded_root.x *
            FIST_THUMB_FOLDED_ROOT_FORWARD_SCALE,
            _extended_side_sign * _raised_side,
            _folded_root.z *
            FIST_THUMB_FOLDED_ROOT_NORMAL_SCALE,
        ))
        if _raised_axis.length <= 1e-5:
            _raised_axis = _folded_root.copy()
        else:
            _raised_axis.normalize()

        # 모든 세그먼트가 같은 축을 사용한다. 따라서 엄지척→검지 접촉
        # 전환은 CMC에서만 일어나며 Thumb2/Thumb3의 상대 굽힘은 없다.
        # 접촉 축이 최종 한계라 반대 side(손바닥/새끼손가락 아래)로
        # 과회전하는 목표 자체가 생성되지 않는다.
        _raised_shaped = tuple(
            _raised_axis.copy() for _ in range(3))
        _contact_shaped = tuple(
            _index_contact_axis.copy() for _ in range(3))
        if _early_extended:
            _early_shaped = _raised_shaped
            _late_shaped = _contact_shaped
        else:
            _early_shaped = _contact_shaped
            _late_shaped = _raised_shaped
        return _early_shaped, _late_shaped, {
            'mode': 'root_pivot_rigid_contact_stop',
            'root_motion_deg': _root_motion_deg,
            'late_tip_joint_deg': _late_tip_joint_deg,
        }

    def _build_fist_thumb_gesture_track(
            _prefix, _runs, _flip, _enabled):
        """확정 주먹 안의 명확한 엄지 펴짐↔접힘 전환을 찾아 대비를 강화한다."""
        _count = len(raw[f'{_prefix}_valid'])
        _references = [None] * _count
        _weights = [0.0] * _count
        _world_up_weights = [0.0] * _count
        _logs = []
        if not _enabled:
            return _references, _weights, _world_up_weights, _logs

        for _run in _runs:
            _run_start = _run['start']
            _run_end = _run['end']
            _pretransition = _run.get('pretransition') or {}
            _active_start = (
                _pretransition.get('start', _run_start)
                if _pretransition.get('enabled', False)
                else _run_start)
            _feature_start = max(
                0,
                _active_start -
                FIST_THUMB_GESTURE_PREHOLD_FRAMES)

            _features = {}
            _straightness = {}
            for _frame_index in range(
                    _feature_start, _run_end + 1):
                _feature = _thumb_local_feature(
                    _prefix, _frame_index, _flip)
                _ratio = _thumb_chain_straightness(
                    _prefix, _frame_index)
                if _feature is not None:
                    _features[_frame_index] = _feature
                if _ratio is not None:
                    _straightness[_frame_index] = _ratio

            _window = max(2, FIST_THUMB_GESTURE_WINDOW_FRAMES)
            _best = None
            for _split in range(
                    _run_start + FIST_THUMB_GESTURE_MIN_STATE_FRAMES,
                    _run_end - FIST_THUMB_GESTURE_MIN_STATE_FRAMES + 2):
                _early_indices = [
                    _frame_index for _frame_index in range(
                        max(_run_start, _split - _window), _split)
                    if _frame_index in _straightness]
                _late_indices = [
                    _frame_index for _frame_index in range(
                        _split, min(_run_end + 1, _split + _window))
                    if _frame_index in _straightness]
                if (
                        len(_early_indices) <
                        FIST_THUMB_GESTURE_MIN_STATE_FRAMES or
                        len(_late_indices) <
                        FIST_THUMB_GESTURE_MIN_STATE_FRAMES):
                    continue
                _early_ratio = statistics.median(
                    _straightness[_frame_index]
                    for _frame_index in _early_indices)
                _late_ratio = statistics.median(
                    _straightness[_frame_index]
                    for _frame_index in _late_indices)
                _delta = abs(_early_ratio - _late_ratio)
                _high_ratio = max(_early_ratio, _late_ratio)
                _low_ratio = min(_early_ratio, _late_ratio)
                if (
                        _high_ratio >=
                        FIST_THUMB_GESTURE_EXTENDED_RATIO and
                        _low_ratio <=
                        FIST_THUMB_GESTURE_FOLDED_RATIO and
                        _delta >= FIST_THUMB_GESTURE_MIN_DELTA and
                        (_best is None or _delta > _best['delta'])):
                    _best = {
                        'split': _split,
                        'early_indices': _early_indices,
                        'late_indices': _late_indices,
                        'early_ratio': _early_ratio,
                        'late_ratio': _late_ratio,
                        'delta': _delta,
                    }

            _enabled_run = False
            if _best:
                _early_reference = _median_thumb_reference(
                    _best['early_indices'], _features)
                _late_reference = _median_thumb_reference(
                    _best['late_indices'], _features)
                _early_extended = (
                    _best['early_ratio'] > _best['late_ratio'])
                (
                    _early_reference,
                    _late_reference,
                    _shape_diagnostic,
                ) = _shape_thumb_gesture_pair(
                    _early_reference,
                    _late_reference,
                    _early_extended)

                if _early_reference and _late_reference:
                    # 엄지척 유지가 네 손가락의 주먹 pretransition보다
                    # 조금 먼저 시작되면, 펴진 엄지가 안정적으로 검출되는
                    # 범위 안에서만 최대 몇 프레임 앞까지 유지한다.
                    _gesture_active_start = _active_start
                    if _early_extended:
                        for _frame_index in range(
                                _active_start - 1,
                                _feature_start - 1,
                                -1):
                            _ratio = _straightness.get(
                                _frame_index)
                            if (
                                    _ratio is None or
                                    _ratio <
                                    FIST_THUMB_GESTURE_EXTENDED_RATIO):
                                break
                            _gesture_active_start = _frame_index
                    _transition_start = max(
                        _gesture_active_start,
                        _best['split'] -
                        FIST_THUMB_GESTURE_TRANSITION_BEFORE)
                    _transition_end = min(
                        _run_end,
                        _best['split'] +
                        FIST_THUMB_GESTURE_TRANSITION_AFTER)
                    _transition_span = max(
                        1, _transition_end - _transition_start)
                    for _frame_index in range(
                            _gesture_active_start, _run_end + 1):
                        if _frame_index <= _transition_start:
                            _state_t = 0.0
                        elif _frame_index >= _transition_end:
                            _state_t = 1.0
                        else:
                            _state_t = _smoothstep01(
                                (_frame_index - _transition_start) /
                                _transition_span)
                        _frame_reference = []
                        for _segment_index in range(3):
                            _direction = _early_reference[
                                _segment_index].lerp(
                                    _late_reference[_segment_index],
                                    _state_t)
                            if _direction.length <= 1e-5:
                                _direction = _early_reference[
                                    _segment_index].copy()
                            else:
                                _direction.normalize()
                            _frame_reference.append(_direction)
                        _references[_frame_index] = tuple(
                            _frame_reference)
                        # 이 특수 모드의 엄지척 상태는 손바닥 로컬 side
                        # 추정에 남은 좌우 기울기를 사용하지 않는다. 양손
                        # 모두 세계 +Z를 향하게 한 뒤, 접촉 상태로 넘어가는
                        # 만큼만 고정을 자연스럽게 해제한다.
                        if (
                                _shape_diagnostic.get('mode') ==
                                'root_pivot_rigid_contact_stop'):
                            _raised_state_weight = (
                                1.0 - _state_t
                                if _early_extended else _state_t)
                            _world_up_weights[_frame_index] = max(
                                0.0, min(
                                    1.0,
                                    _raised_state_weight *
                                    FIST_THUMB_RAISED_WORLD_UP_BLEND))
                        # 의미 있는 엄지 전환은 비엄지 주먹 lock보다 먼저
                        # 시작될 수 있다. 별도 edge fade를 써서 자연스럽게
                        # 진입하며, 강체 뿌리 회전은 내부 관절을 완전히
                        # 고정하도록 100% 목표 자세를 사용한다.
                        _edge_frames = max(
                            1, FIST_THUMB_GESTURE_EDGE_FRAMES)
                        _edge_distance = min(
                            _frame_index -
                            _gesture_active_start + 1,
                            _run_end - _frame_index + 1)
                        _edge_weight = _smoothstep01(
                            _edge_distance / _edge_frames)
                        _state_blend = (
                            1.0
                            if str(_shape_diagnostic.get('mode')).startswith(
                                'root_pivot_rigid')
                            else FIST_THUMB_GESTURE_BLEND)
                        _weights[_frame_index] = (
                            _state_blend * _edge_weight)
                    _best['transition_start'] = _transition_start
                    _best['transition_end'] = _transition_end
                    _best['gesture_active_start'] = (
                        _gesture_active_start)
                    _best['direction'] = (
                        'extended_to_folded'
                        if _early_extended
                        else 'folded_to_extended')
                    _best['shape_mode'] = (
                        _shape_diagnostic.get('mode'))
                    _best['root_motion_deg'] = (
                        _shape_diagnostic.get('root_motion_deg'))
                    _best['late_tip_joint_deg'] = (
                        _shape_diagnostic.get(
                            'late_tip_joint_deg'))
                    _enabled_run = True

            _logs.append({
                'enabled': _enabled_run,
                'run_start': _run_start,
                'run_end': _run_end,
                'active_start': _active_start,
                'gesture_active_start': (
                    _best.get('gesture_active_start', _active_start)
                    if _best else _active_start),
                'split': _best.get('split') if _best else None,
                'transition_start': (
                    _best.get('transition_start') if _best else None),
                'transition_end': (
                    _best.get('transition_end') if _best else None),
                'direction': (
                    _best.get('direction') if _best else None),
                'early_ratio': (
                    _best.get('early_ratio') if _best else None),
                'late_ratio': (
                    _best.get('late_ratio') if _best else None),
                'delta': _best.get('delta') if _best else None,
                'shape_mode': (
                    _best.get('shape_mode') if _best else None),
                'root_motion_deg': (
                    _best.get('root_motion_deg')
                    if _best else None),
                'late_tip_joint_deg': (
                    _best.get('late_tip_joint_deg')
                    if _best else None),
            })
        return _references, _weights, _world_up_weights, _logs

    def _build_fist_thumb_reference_track(
            _prefix, _runs, _flip, _enabled):
        _count = len(raw[f'{_prefix}_valid'])
        _references = [None] * _count
        _logs = []
        if not _enabled:
            return _references, _logs

        for _run in _runs:
            _start = _run['start']
            _end = _run['end']
            _samples = []
            _features = {}
            for _fi in range(_start, _end + 1):
                _feature = _thumb_local_feature(
                    _prefix, _fi, _flip)
                if _feature is not None:
                    _samples.append((_fi, _feature))
                    _features[_fi] = _feature

            _reference = None
            _stable_fraction = 0.0
            _stable_count = 0
            _max_mean_deviation = 0.0
            if len(_samples) >= FIST_THUMB_MIN_VALID_FRAMES:
                _reference = []
                for _segment_index in range(3):
                    _median_components = mathutils.Vector(tuple(
                        statistics.median(
                            _sample[1][_segment_index][_axis]
                            for _sample in _samples)
                        for _axis in range(3)
                    ))
                    if _median_components.length <= 1e-5:
                        _reference = None
                        break
                    _reference.append(
                        _median_components.normalized())

            _deviation_by_frame = {}
            if _reference:
                for _fi, _feature in _samples:
                    _mean_deviation = _thumb_feature_distance(
                        _feature, _reference)
                    if _mean_deviation is None:
                        continue
                    _deviation_by_frame[_fi] = _mean_deviation
                    _max_mean_deviation = max(
                        _max_mean_deviation, _mean_deviation)
                    if (_mean_deviation <=
                            FIST_THUMB_STABLE_ANGLE_DEG):
                        _stable_count += 1
                _stable_fraction = (
                    _stable_count / len(_samples)
                    if _samples else 0.0)

            # 중앙값에서 크게 벗어난 연속 구간을 만든다.
            _candidate_runs = []
            _candidate_start = None
            for _fi in range(_start, _end + 2):
                _is_candidate = (
                    _fi <= _end and
                    _deviation_by_frame.get(_fi, 0.0) >
                    FIST_THUMB_STABLE_ANGLE_DEG)
                if _is_candidate and _candidate_start is None:
                    _candidate_start = _fi
                elif not _is_candidate and _candidate_start is not None:
                    _candidate_runs.append(
                        (_candidate_start, _fi - 1))
                    _candidate_start = None

            _corrected_frames = 0
            _corrected_segments = 0
            for _outlier_start, _outlier_end in _candidate_runs:
                # 앞뒤 모두 같은 안정 군집으로 돌아오는 내부 이탈만 허용한다.
                _before = next((
                    _fi for _fi in range(
                        _outlier_start - 1, _start - 1, -1)
                    if (_fi in _features and
                        _deviation_by_frame.get(
                            _fi, float('inf')) <=
                        FIST_THUMB_STABLE_ANGLE_DEG)
                ), None)
                _after = next((
                    _fi for _fi in range(
                        _outlier_end + 1, _end + 1)
                    if (_fi in _features and
                        _deviation_by_frame.get(
                            _fi, float('inf')) <=
                        FIST_THUMB_STABLE_ANGLE_DEG)
                ), None)
                _valid_outlier_frames = [
                    _fi for _fi in range(
                        _outlier_start, _outlier_end + 1)
                    if _fi in _features]
                _segment_length = (
                    _outlier_end - _outlier_start + 1)
                _segment_fraction = (
                    len(_valid_outlier_frames) / len(_samples)
                    if _samples else 1.0)
                _peak_deviation = max((
                    _deviation_by_frame.get(_fi, 0.0)
                    for _fi in _valid_outlier_frames
                ), default=0.0)
                _entry_jump = (
                    _thumb_feature_distance(
                        _features.get(_before),
                        _features.get(_outlier_start))
                    if _before is not None else None)
                _exit_jump = (
                    _thumb_feature_distance(
                        _features.get(_outlier_end),
                        _features.get(_after))
                    if _after is not None else None)
                _is_temporal_outlier = bool(
                    _before is not None and
                    _after is not None and
                    _segment_length <=
                    FIST_THUMB_OUTLIER_MAX_FRAMES and
                    _segment_fraction <=
                    FIST_THUMB_OUTLIER_MAX_FRACTION and
                    _peak_deviation >=
                    FIST_THUMB_OUTLIER_PEAK_DEG and
                    _entry_jump is not None and
                    _exit_jump is not None and
                    _entry_jump >=
                    FIST_THUMB_OUTLIER_BOUNDARY_DEG and
                    _exit_jump >=
                    FIST_THUMB_OUTLIER_BOUNDARY_DEG)
                if not _is_temporal_outlier:
                    continue

                _before_feature = _features[_before]
                _after_feature = _features[_after]
                _span = _after - _before
                for _fi in range(
                        _outlier_start, _outlier_end + 1):
                    _t = (
                        (_fi - _before) / _span
                        if _span > 0 else 0.5)
                    _interpolated = []
                    for _segment_index in range(3):
                        _direction = (
                            _before_feature[_segment_index].lerp(
                                _after_feature[_segment_index], _t))
                        if _direction.length <= 1e-5:
                            _direction = _reference[
                                _segment_index].copy()
                        else:
                            _direction.normalize()
                        _interpolated.append(_direction)
                    _references[_fi] = tuple(_interpolated)
                    _corrected_frames += 1
                _corrected_segments += 1

            _guard_enabled = _corrected_frames > 0

            _logs.append({
                'start': _start,
                'end': _end,
                'start_ts': _run['start_ts'],
                'end_ts': _run['end_ts'],
                'valid': len(_samples),
                'stable': _stable_count,
                'fraction': _stable_fraction,
                'max_mean_deviation': _max_mean_deviation,
                'enabled': _guard_enabled,
                'corrected_frames': _corrected_frames,
                'corrected_segments': _corrected_segments,
            })
        return _references, _logs

    _fist_thumb_refs_l, _fist_thumb_logs_l = (
        _build_fist_thumb_reference_track(
            'l', _fist_runs_l, L_PALM_FLIP,
            USE_FIST_THUMB_TEMPORAL_GUARD and
            _motion_fist_enabled and 'L' in _fist_target_sides))
    _fist_thumb_refs_r, _fist_thumb_logs_r = (
        _build_fist_thumb_reference_track(
            'r', _fist_runs_r, R_PALM_FLIP,
            USE_FIST_THUMB_TEMPORAL_GUARD and
            _motion_fist_enabled and 'R' in _fist_target_sides))
    for _thumb_side, _thumb_logs in (
            ('L', _fist_thumb_logs_l),
            ('R', _fist_thumb_logs_r)):
        if not _thumb_logs:
            print(f"[FIST-THUMB-GUARD] {_thumb_side}: none")
            continue
        for _thumb_log in _thumb_logs:
            print(f"[FIST-THUMB-GUARD] {_thumb_side}:"
                  f"  ts={_thumb_log['start_ts']}..{_thumb_log['end_ts']}"
                  f"  valid={_thumb_log['valid']}"
                  f"  stable={_thumb_log['stable']}"
                  f"  stable_fraction={_thumb_log['fraction']:.3f}"
                  f"  max_mean_deviation_deg="
                  f"{_thumb_log['max_mean_deviation']:.1f}"
                  f"  corrected_segments="
                  f"{_thumb_log['corrected_segments']}"
                  f"  corrected_frames="
                  f"{_thumb_log['corrected_frames']}"
                  f"  enabled={_thumb_log['enabled']}")

    # ── 의미 있는 엄지 펴짐↔접힘 상태 대비 강화 ─────────────────
    (_fist_thumb_gesture_refs_l,
     _fist_thumb_gesture_weights_l,
     _fist_thumb_gesture_world_up_l,
     _fist_thumb_gesture_logs_l) = _build_fist_thumb_gesture_track(
            'l', _fist_runs_l, L_PALM_FLIP,
            USE_FIST_THUMB_GESTURE_EMPHASIS and
            _motion_fist_enabled and 'L' in _fist_target_sides)
    (_fist_thumb_gesture_refs_r,
     _fist_thumb_gesture_weights_r,
     _fist_thumb_gesture_world_up_r,
     _fist_thumb_gesture_logs_r) = _build_fist_thumb_gesture_track(
            'r', _fist_runs_r, R_PALM_FLIP,
            USE_FIST_THUMB_GESTURE_EMPHASIS and
            _motion_fist_enabled and 'R' in _fist_target_sides)
    for _thumb_side, _gesture_logs in (
            ('L', _fist_thumb_gesture_logs_l),
            ('R', _fist_thumb_gesture_logs_r)):
        if not _gesture_logs:
            print(f"[FIST-THUMB-GESTURE] {_thumb_side}: none")
            continue
        for _gesture_log in _gesture_logs:
            _early_ratio = _gesture_log['early_ratio']
            _late_ratio = _gesture_log['late_ratio']
            _delta = _gesture_log['delta']
            _early_ratio_text = (
                f"{_early_ratio:.3f}"
                if _early_ratio is not None else "None")
            _late_ratio_text = (
                f"{_late_ratio:.3f}"
                if _late_ratio is not None else "None")
            _delta_text = (
                f"{_delta:.3f}" if _delta is not None else "None")
            _root_motion_deg = _gesture_log['root_motion_deg']
            _late_tip_joint_deg = (
                _gesture_log['late_tip_joint_deg'])
            _root_motion_text = (
                f"{_root_motion_deg:.1f}"
                if _root_motion_deg is not None else "None")
            _late_tip_joint_text = (
                f"{_late_tip_joint_deg:.1f}"
                if _late_tip_joint_deg is not None else "None")
            print(f"[FIST-THUMB-GESTURE] {_thumb_side}:"
                  f"  enabled={_gesture_log['enabled']}"
                  f"  source_index="
                  f"{_gesture_log['gesture_active_start']}.."
                  f"{_gesture_log['run_end']}"
                  f"  split={_gesture_log['split']}"
                  f"  transition="
                  f"{_gesture_log['transition_start']}.."
                  f"{_gesture_log['transition_end']}"
                  f"  direction={_gesture_log['direction']}"
                  f"  straightness="
                  f"{_early_ratio_text}->{_late_ratio_text}"
                  f"  delta={_delta_text}"
                  f"  shape_mode="
                  f"{_gesture_log['shape_mode']}"
                  f"  root_motion_deg={_root_motion_text}"
                  f"  late_tip_joint_deg="
                  f"{_late_tip_joint_text}")

    # [PALM-NORMAL-CONTINUITY] / [PALM-NORMAL-TS] 진단
    if _src_ts:
        _pn_diag_bi = min(range(len(_src_ts)), key=lambda k: abs(_src_ts[k] - _DIAG_TS))
        _pn_diag_ts = _src_ts[_pn_diag_bi]
        for _pns, _pn_track, _pn_flips, _pn_valid, _pn_miss, _pn_raw, _pn_flip_flag in [
                ('L', _pn_l_track, _pn_l_flips, _pn_l_valid, _pn_l_miss, raw['l_pn'], L_PALM_FLIP),
                ('R', _pn_r_track, _pn_r_flips, _pn_r_valid, _pn_r_miss, raw['r_pn'], R_PALM_FLIP)]:
            print(f"[PALM-NORMAL-CONTINUITY] {_pns}:"
                  f"  source_frames={len(_pn_track)}"
                  f"  valid_frames={_pn_valid}"
                  f"  missing_frames={_pn_miss}"
                  f"  hemisphere_flips={_pn_flips}"
                  f"  stabilized_track_length={len(_pn_track)}")
            _pn_raw_v  = _pn_raw[_pn_diag_bi]  if _pn_diag_bi < len(_pn_raw)  else None
            _pn_stab_v = _pn_track[_pn_diag_bi] if _pn_diag_bi < len(_pn_track) else None
            _pn_prev_v = (_pn_track[_pn_diag_bi-1] if _pn_diag_bi > 0 and (_pn_diag_bi-1) < len(_pn_track) else None)
            _pn_eff_v  = ((-_pn_stab_v if _pn_flip_flag else _pn_stab_v) if _pn_stab_v is not None else None)
            _pn_dot    = (_pn_prev_v.dot(_pn_stab_v) if (_pn_prev_v is not None and _pn_stab_v is not None) else None)
            _raw_str  = (f"({_pn_raw_v.normalized().x:+.3f},{_pn_raw_v.normalized().y:+.3f},{_pn_raw_v.normalized().z:+.3f})"
                         if (_pn_raw_v is not None and _pn_raw_v.length > 1e-6) else "None")
            _stab_str = (f"({_pn_stab_v.x:+.3f},{_pn_stab_v.y:+.3f},{_pn_stab_v.z:+.3f})"
                         if _pn_stab_v is not None else "None")
            _eff_str  = (f"({_pn_eff_v.x:+.3f},{_pn_eff_v.y:+.3f},{_pn_eff_v.z:+.3f})"
                         if _pn_eff_v is not None else "None")
            _dot_str  = f"{_pn_dot:+.4f}" if _pn_dot is not None else "None"
            print(f"[PALM-NORMAL-TS] {_pns}: ts={_pn_diag_ts}"
                  f"  raw_normal={_raw_str}"
                  f"  stabilized_normal={_stab_str}"
                  f"  effective_normal(flip={_pn_flip_flag})={_eff_str}"
                  f"  prev_dot={_dot_str}")

    # ── 미검출 앞 구간 → 자연 자세 ───────────────────────────────
    l_first = next((i for i,v in enumerate(raw['l_valid']) if v), None)
    r_first = next((i for i,v in enumerate(raw['r_valid']) if v), None)
    hang    = mathutils.Vector((0,0,-0.15))

    def fix_hang(wrist_key, h9_key, pt_keys, first_idx):
        n     = len(smooth[wrist_key])
        n_fix = first_idx if first_idx is not None else n
        for i in range(n_fix):
            w = smooth[wrist_key][i]
            smooth[h9_key][i] = w + hang
            for k in pt_keys: smooth[k][i] = w + hang
        if first_idx is None:
            for i in range(n):
                w = smooth[wrist_key][i]
                smooth[h9_key][i] = w + hang
                for k in pt_keys: smooth[k][i] = w + hang

    fix_hang('lw','lh9',[f'l_p{i}' for i in range(21)], l_first)
    fix_hang('rw','rh9',[f'r_p{i}' for i in range(21)], r_first)

    # ── Intro ────────────────────────────────────────────────────
    if INTRO_FRAMES > 0:
        idle_l = l_sh_w + mathutils.Vector((0,-FORWARD_OFFSET,-0.40))
        idle_r = r_sh_w + mathutils.Vector((0,-FORWARD_OFFSET,-0.40))
        idle = {'lw':idle_l,'rw':idle_r,'lw_ik':idle_l,'rw_ik':idle_r,
                'lh9':idle_l+hang,'rh9':idle_r+hang,
                'l_pn':mathutils.Vector((0,0,0)),'r_pn':mathutils.Vector((0,0,0)),
                **{f'l_p{i}':idle_l+hang for i in range(21)},
                **{f'r_p{i}':idle_r+hang for i in range(21)}}
        for k,v in idle.items():
            smooth[k] = [v.copy() for _ in range(INTRO_FRAMES)] + smooth[k]
        _fist_weights_l = [0.0] * INTRO_FRAMES + _fist_weights_l
        _fist_weights_r = [0.0] * INTRO_FRAMES + _fist_weights_r
        _fist_thumb_refs_l = [None] * INTRO_FRAMES + _fist_thumb_refs_l
        _fist_thumb_refs_r = [None] * INTRO_FRAMES + _fist_thumb_refs_r
        _fist_thumb_gesture_refs_l = (
            [None] * INTRO_FRAMES + _fist_thumb_gesture_refs_l)
        _fist_thumb_gesture_refs_r = (
            [None] * INTRO_FRAMES + _fist_thumb_gesture_refs_r)
        _fist_thumb_gesture_weights_l = (
            [0.0] * INTRO_FRAMES + _fist_thumb_gesture_weights_l)
        _fist_thumb_gesture_weights_r = (
            [0.0] * INTRO_FRAMES + _fist_thumb_gesture_weights_r)
        _fist_thumb_gesture_world_up_l = (
            [0.0] * INTRO_FRAMES + _fist_thumb_gesture_world_up_l)
        _fist_thumb_gesture_world_up_r = (
            [0.0] * INTRO_FRAMES + _fist_thumb_gesture_world_up_r)
        total += INTRO_FRAMES
        if l_first is not None: l_first += INTRO_FRAMES
        if r_first is not None: r_first += INTRO_FRAMES

    # character_reach_ratio: smooth['lw/rw_ik'] 기준, INTRO_FRAMES 제외
    def _combine_reach(src_log, ik_key, sh_w, arm_tot):
        result = []
        for i, (ts, src_r) in enumerate(src_log):
            si = INTRO_FRAMES + i
            if si >= len(smooth[ik_key]): break
            sv = smooth[ik_key][si]
            if sv is None: continue
            chr_r = (sh_w - sv).length / arm_tot if arm_tot > 1e-6 else 0.0
            result.append((ts, src_r, chr_r))
        return result
    _reach_final_l = _combine_reach(_reach_l, 'lw_ik', l_sh_w, char_l_arm_total)
    _reach_final_r = _combine_reach(_reach_r, 'rw_ik', r_sh_w, char_r_arm_total)
    for side, rlog in [('L', _reach_final_l), ('R', _reach_final_r)]:
        if not rlog: continue
        srcs = [r[1] for r in rlog]; chrs = [r[2] for r in rlog]
        over = sorted([(ts, sr, cr) for ts, sr, cr in rlog if cr >= 0.98], key=lambda x: -x[2])
        print(f"[IK-REACH] {side}: source3d avg={sum(srcs)/len(srcs):.3f} min={min(srcs):.3f} max={max(srcs):.3f}"
              f"  char avg={sum(chrs)/len(chrs):.3f} min={min(chrs):.3f} max={max(chrs):.3f}")
        print(f"[IK-REACH] {side}: char>=0.98 {len(over)}프레임")
        for ts, sr, cr in over[:10]:
            print(f"  over: ts={ts}  src={sr:.3f}  char={cr:.3f}")
        if rlog:
            closest = min(rlog, key=lambda r: abs(r[0] - _DIAG_TS))
            print(f"[IK-REACH-TS] {side}: ts={closest[0]}  src={closest[1]:.3f}  char={closest[2]:.3f}")

    # ── Source Reach Retarget (스무딩된 3D source ratio를 캐릭터 팔 길이에 적용) ──
    if USE_ARM_REACH_LIMIT:
        # source ratio 5-frame scalar smoothing (source frame 단위, INTRO 제외)
        _src_ratio_l_sm = smooth_scalar_track(_src_ratio_l, window=5)
        _src_ratio_r_sm = smooth_scalar_track(_src_ratio_r, window=5)
        if len(_src_ratio_l_sm) != len(_src_ratio_l) or len(_src_ratio_r_sm) != len(_src_ratio_r):
            print(f"[IK-REACH-RETARGET] 오류: source ratio smooth 길이 불일치"
                  f" L={len(_src_ratio_l_sm)}/{len(_src_ratio_l)}"
                  f" R={len(_src_ratio_r_sm)}/{len(_src_ratio_r)}")
            _src_ratio_l_sm = list(_src_ratio_l)
            _src_ratio_r_sm = list(_src_ratio_r)

        for ik_key, sh_w, arm_tot, _side, _src_rl_sm in [
                ('lw_ik', l_sh_w, char_l_arm_total, 'L', _src_ratio_l_sm),
                ('rw_ik', r_sh_w, char_r_arm_total, 'R', _src_ratio_r_sm)]:
            if arm_tot <= 1e-6: continue
            _rt_log = []  # (ts, before_r, src_sm, final_r)
            _rt_fallback = 0; _rt_expanded = 0; _rt_contracted = 0
            _rt_diag_bi = (min(range(len(_src_ts)), key=lambda k: abs(_src_ts[k]-_DIAG_TS))
                           if _src_ts else None)
            _rt_diag = None

            for i in range(INTRO_FRAMES, len(smooth[ik_key])):
                v = smooth[ik_key][i]
                if v is None: continue
                offset = v - sh_w
                off_len = offset.length
                if off_len < 1e-6: continue
                before_r = off_len / arm_tot
                safe_r = min(before_r, ARM_REACH_MAX_RATIO)
                src_i = i - INTRO_FRAMES
                ts = _src_ts[src_i] if src_i < len(_src_ts) else 0
                src_sm = _src_rl_sm[src_i] if src_i < len(_src_rl_sm) else None

                if USE_SOURCE_REACH_GUIDANCE and src_sm is not None:
                    src_target = max(SOURCE_REACH_MIN_RATIO, min(ARM_REACH_MAX_RATIO, src_sm))
                    final_r = safe_r + SOURCE_REACH_BLEND * (src_target - safe_r)
                    final_r = min(final_r, ARM_REACH_MAX_RATIO)  # 절대 상한
                    if final_r > before_r + 1e-6: _rt_expanded += 1
                    elif final_r < before_r - 1e-6: _rt_contracted += 1
                else:
                    src_target = None; final_r = safe_r; _rt_fallback += 1

                if abs(final_r - before_r) > 1e-6:
                    smooth[ik_key][i] = sh_w + (offset / off_len) * arm_tot * final_r
                _rt_log.append((ts, before_r, src_sm if src_sm is not None else 0.0, final_r))
                if _rt_diag_bi is not None and src_i == _rt_diag_bi:
                    _rt_diag = (ts, before_r, src_sm, final_r)

            # [IK-REACH-RETARGET] 통계
            _n = len(_rt_log)
            _bs = [r[1] for r in _rt_log]; _ss = [r[2] for r in _rt_log]; _fs = [r[3] for r in _rt_log]
            _deltas = [abs(r[3]-r[1]) for r in _rt_log]
            print(f"[IK-REACH-RETARGET] {_side}:"
                  f"  source_frames={_n}"
                  f"  applied_frames={_n - _rt_fallback}"
                  f"  fallback_frames={_rt_fallback}"
                  f"  expanded={_rt_expanded}"
                  f"  contracted={_rt_contracted}")
            if _n:
                print(f"  before avg={sum(_bs)/_n:.4f} min={min(_bs):.4f} max={max(_bs):.4f}")
                print(f"  source avg={sum(_ss)/_n:.4f} min={min(_ss):.4f} max={max(_ss):.4f}")
                print(f"  final  avg={sum(_fs)/_n:.4f} min={min(_fs):.4f} max={max(_fs):.4f}")
                print(f"  max_delta_ratio={max(_deltas):.4f}")
            # [IK-REACH-RETARGET-TS]
            if _rt_diag:
                _d_ts, _d_bef, _d_src, _d_fin = _rt_diag
                _d_delta = _d_fin - _d_bef
                _src_str = f"{_d_src:.4f}" if _d_src is not None else "None"
                print(f"[IK-REACH-RETARGET-TS] {_side}: ts={_d_ts}"
                      f"  before_ratio={_d_bef:.4f}"
                      f"  smoothed_src={_src_str}"
                      f"  final_ratio={_d_fin:.4f}"
                      f"  delta_ratio={_d_delta:+.4f}")

    # ── Support-Slide Contact ─────────────────────────────────────
    # 받침 손의 MCP 중심 위에 움직이는 손의 MCP 중심을 유지한다.
    # 움직이는 손 랜드마크 전체를 강체 회전시켜 손바닥 면만 수평으로
    # 맞추므로, 손목→손가락의 수평 회전(yaw)은 원본대로 남는다.
    _support_slide_weights = [0.0] * total
    print(f"[SUPPORT-SLIDE-MODE]"
          f"  motion_id={_motion_id!r}"
          f"  enabled={_motion_support_slide_enabled}"
          f"  profile={_support_slide_profile}")
    if _motion_support_slide_enabled and char_sw > 1e-6:
        _side_data = {
            'L': {
                'prefix': 'l', 'ik': 'lw_ik', 'pn': 'l_pn',
                'h9': 'lh9', 'flip': L_PALM_FLIP,
                'valid': raw['l_valid'], 'shoulder': l_sh_w,
                'arm_total': char_l_arm_total,
                'fist_weights': _fist_weights_l,
            },
            'R': {
                'prefix': 'r', 'ik': 'rw_ik', 'pn': 'r_pn',
                'h9': 'rh9', 'flip': R_PALM_FLIP,
                'valid': raw['r_valid'], 'shoulder': r_sh_w,
                'arm_total': char_r_arm_total,
                'fist_weights': _fist_weights_r,
            },
        }
        _support_side = _support_slide_profile.get(
            'support_side', 'L')
        _moving_side = _support_slide_profile.get(
            'moving_side', 'R')
        _support_data = _side_data.get(_support_side)
        _moving_data = _side_data.get(_moving_side)
        if (_support_data is None or _moving_data is None or
                _support_side == _moving_side):
            print("[SUPPORT-SLIDE] skipped — invalid side profile")
        else:
            _sp = _support_data['prefix']
            _mp = _moving_data['prefix']
            _support_fist_weights = _support_data['fist_weights']
            _target_normal = mathutils.Vector(
                _support_slide_profile.get(
                    'moving_palm_normal_world',
                    (0.0, 0.0, -1.0)))
            if _target_normal.length <= 1e-6:
                _target_normal = mathutils.Vector((0.0, 0.0, -1.0))
            _target_normal.normalize()
            _center_landmarks = (
                H_INDEX_MCP, H_MID_MCP, H_RING_MCP, H_PINK_MCP)

            def _palm_center_relative(_prefix, _si):
                _p0 = smooth[f'{_prefix}_p0'][_si]
                _mcp_points = [
                    smooth[f'{_prefix}_p{_lm}'][_si]
                    for _lm in _center_landmarks
                ]
                if (_p0 is None or
                        any(_point is None for _point in _mcp_points)):
                    return None
                _center = sum(
                    _mcp_points, mathutils.Vector((0.0, 0.0, 0.0))
                ) / len(_mcp_points)
                return _center - _p0

            # 원본의 중앙값 높이를 보존한다. 비정상 깊이 추정만 profile
            # 범위로 clamp하여 손 사이 간격이 뜨거나 관통하는 것을 막는다.
            _source_heights_sw = []
            for _source_index in range(len(_src_ts)):
                _si = INTRO_FRAMES + _source_index
                if _si >= total:
                    continue
                _weight = (
                    _support_fist_weights[_si]
                    if _si < len(_support_fist_weights) else 0.0)
                if _weight < 0.5:
                    continue
                if (_source_index >= len(_support_data['valid']) or
                        _source_index >= len(_moving_data['valid']) or
                        not _support_data['valid'][_source_index] or
                        not _moving_data['valid'][_source_index]):
                    continue
                _support_rel = _palm_center_relative(_sp, _si)
                _moving_rel = _palm_center_relative(_mp, _si)
                _support_wrist = smooth[_support_data['ik']][_si]
                _moving_wrist = smooth[_moving_data['ik']][_si]
                if any(_value is None for _value in (
                        _support_rel, _moving_rel,
                        _support_wrist, _moving_wrist)):
                    continue
                _source_heights_sw.append(
                    ((_moving_wrist + _moving_rel).z -
                     (_support_wrist + _support_rel).z) / char_sw)

            _height_min = float(
                _support_slide_profile.get('height_min_sw', 0.22))
            _height_max = float(
                _support_slide_profile.get('height_max_sw', 0.31))
            if _height_min > _height_max:
                _height_min, _height_max = _height_max, _height_min
            _min_height_samples = int(
                _support_slide_profile.get(
                    'min_full_weight_frames', 6))
            if len(_source_heights_sw) >= _min_height_samples:
                _source_height_median = statistics.median(
                    _source_heights_sw)
            else:
                _source_height_median = (
                    _height_min + _height_max) * 0.5
            _support_height_sw = max(
                _height_min,
                min(_height_max, _source_height_median))

            # support-slide가 양손 중심을 완전히 포개면 손바닥 각도만
            # 변하고 손목 위치는 고정되어 '비비기'처럼 보인다. profile이
            # ellipse를 요청한 경우 주먹 유지 구간 전체를 시계로 삼아
            # 움직이는 손 중심에 작은 수평 타원 궤적을 부여한다.
            _trajectory_mode = str(
                _support_slide_profile.get(
                    'trajectory_mode', 'centered')).lower()
            _orbit_radius_x_sw = max(
                0.0, float(_support_slide_profile.get(
                    'orbit_radius_x_sw', 0.0)))
            _orbit_radius_y_sw = max(
                0.0, float(_support_slide_profile.get(
                    'orbit_radius_y_sw', 0.0)))
            _orbit_cycles = float(
                _support_slide_profile.get('orbit_cycles', 0.0))
            _orbit_phase_rad = math.radians(float(
                _support_slide_profile.get(
                    'orbit_phase_deg', 0.0)))
            _orbit_source_indices = []
            for _orbit_source_index in range(len(_src_ts)):
                _orbit_si = INTRO_FRAMES + _orbit_source_index
                if _orbit_si >= total:
                    continue
                _orbit_weight = (
                    _support_fist_weights[_orbit_si]
                    if _orbit_si < len(_support_fist_weights)
                    else 0.0)
                if _orbit_weight <= 1e-5:
                    continue
                if (_orbit_source_index >=
                        len(_support_data['valid']) or
                        _orbit_source_index >=
                        len(_moving_data['valid']) or
                        not _support_data['valid'][
                            _orbit_source_index] or
                        not _moving_data['valid'][
                            _orbit_source_index]):
                    continue
                _orbit_source_indices.append(
                    _orbit_source_index)
            _orbit_start = (
                min(_orbit_source_indices)
                if _orbit_source_indices else None)
            _orbit_end = (
                max(_orbit_source_indices)
                if _orbit_source_indices else None)
            _orbit_enabled = (
                _trajectory_mode == 'ellipse' and
                _orbit_start is not None and
                _orbit_end is not None and
                _orbit_end > _orbit_start and
                _orbit_cycles != 0.0 and
                (_orbit_radius_x_sw > 1e-6 or
                 _orbit_radius_y_sw > 1e-6))

            _weighted_count = 0
            _full_count = 0
            _post_clamp_count = 0
            _max_wrist_correction_sw = 0.0
            _max_center_gap_sw = 0.0
            _full_orbit_x_sw = []
            _full_orbit_y_sw = []
            _source_normal_angles = []
            _after_normal_angles = []
            _diag_support_slide = None
            _identity_q = mathutils.Quaternion(
                (1.0, 0.0, 0.0, 0.0))

            for _source_index in range(len(_src_ts)):
                _si = INTRO_FRAMES + _source_index
                if _si >= total:
                    continue
                _weight = (
                    _support_fist_weights[_si]
                    if _si < len(_support_fist_weights) else 0.0)
                if _weight <= 1e-5:
                    continue

                _moving_pn_raw = smooth[_moving_data['pn']][_si]
                if (_moving_pn_raw is None or
                        _moving_pn_raw.length <= 1e-5):
                    continue
                _moving_pn_eff = (
                    -_moving_pn_raw
                    if _moving_data['flip'] else _moving_pn_raw)
                _moving_pn_eff = _moving_pn_eff.normalized()
                _source_dot = max(
                    -1.0, min(
                        1.0,
                        _moving_pn_eff.dot(_target_normal)))
                _source_angle = math.degrees(
                    math.acos(_source_dot))
                _full_rotation = (
                    _moving_pn_eff.rotation_difference(
                        _target_normal))
                _blend_rotation = _identity_q.slerp(
                    _full_rotation, max(0.0, min(1.0, _weight)))
                _blend_rotation.normalize()

                _moving_p0 = smooth[f'{_mp}_p0'][_si]
                if _moving_p0 is None:
                    continue
                _rotated_points = {}
                _valid_points = True
                for _landmark_index in range(21):
                    _point_key = f'{_mp}_p{_landmark_index}'
                    _point = smooth[_point_key][_si]
                    if _point is None:
                        _valid_points = False
                        break
                    _rotated_points[_landmark_index] = (
                        _moving_p0 +
                        _blend_rotation @ (_point - _moving_p0))
                if not _valid_points:
                    continue

                _rotated_pn_raw = (
                    _blend_rotation @ _moving_pn_raw)
                smooth[_moving_data['pn']][_si] = _rotated_pn_raw
                for _landmark_index, _point in _rotated_points.items():
                    smooth[f'{_mp}_p{_landmark_index}'][_si] = _point
                smooth[_moving_data['h9']][_si] = (
                    _rotated_points[H_MID_MCP])

                _support_rel = _palm_center_relative(_sp, _si)
                _moving_rel = _palm_center_relative(_mp, _si)
                _support_wrist = smooth[_support_data['ik']][_si]
                _moving_wrist_before = smooth[_moving_data['ik']][_si]
                if any(_value is None for _value in (
                        _support_rel, _moving_rel,
                        _support_wrist, _moving_wrist_before)):
                    continue
                _support_center = (
                    _support_wrist + _support_rel)
                _orbit_offset = mathutils.Vector((
                    0.0, 0.0, 0.0))
                if _orbit_enabled:
                    _orbit_progress = (
                        (_source_index - _orbit_start) /
                        (_orbit_end - _orbit_start))
                    _orbit_progress = max(
                        0.0, min(1.0, _orbit_progress))
                    _orbit_phase = (
                        _orbit_phase_rad +
                        2.0 * math.pi *
                        _orbit_cycles * _orbit_progress)
                    _orbit_offset.x = (
                        math.cos(_orbit_phase) *
                        _orbit_radius_x_sw * char_sw)
                    _orbit_offset.y = (
                        math.sin(_orbit_phase) *
                        _orbit_radius_y_sw * char_sw)
                _target_center = (
                    _support_center +
                    _orbit_offset +
                    mathutils.Vector((
                        0.0, 0.0,
                        _support_height_sw * char_sw)))
                _moving_wrist_target = (
                    _target_center - _moving_rel)
                _moving_wrist_after = (
                    _moving_wrist_before +
                    (_moving_wrist_target -
                     _moving_wrist_before) * _weight)

                _post_clamp = False
                _moving_arm_total = _moving_data['arm_total']
                if _moving_arm_total > 1e-6:
                    _reach_offset = (
                        _moving_wrist_after -
                        _moving_data['shoulder'])
                    if (_reach_offset.length / _moving_arm_total >
                            ARM_REACH_MAX_RATIO):
                        _moving_wrist_after = (
                            _moving_data['shoulder'] +
                            _reach_offset.normalized() *
                            _moving_arm_total *
                            ARM_REACH_MAX_RATIO)
                        _post_clamp = True
                        _post_clamp_count += 1
                smooth[_moving_data['ik']][_si] = (
                    _moving_wrist_after)
                _support_slide_weights[_si] = _weight

                _moving_center_after = (
                    _moving_wrist_after + _moving_rel)
                _center_gap_sw = (
                    _moving_center_after -
                    _target_center).length / char_sw
                _wrist_correction_sw = (
                    _moving_wrist_after -
                    _moving_wrist_before).length / char_sw
                _after_pn_eff = (
                    -_rotated_pn_raw
                    if _moving_data['flip']
                    else _rotated_pn_raw).normalized()
                _after_dot = max(
                    -1.0, min(
                        1.0,
                        _after_pn_eff.dot(_target_normal)))
                _after_angle = math.degrees(
                    math.acos(_after_dot))

                _weighted_count += 1
                if _weight >= 0.999:
                    _full_count += 1
                    _full_relative_center = (
                        _moving_center_after -
                        _support_center) / char_sw
                    _full_orbit_x_sw.append(
                        _full_relative_center.x)
                    _full_orbit_y_sw.append(
                        _full_relative_center.y)
                _max_center_gap_sw = max(
                    _max_center_gap_sw, _center_gap_sw)
                _max_wrist_correction_sw = max(
                    _max_wrist_correction_sw,
                    _wrist_correction_sw)
                _source_normal_angles.append(_source_angle)
                _after_normal_angles.append(_after_angle)
                if (abs(_src_ts[_source_index] - _DIAG_TS) <
                        (abs(_diag_support_slide['ts'] - _DIAG_TS)
                         if _diag_support_slide else float('inf'))):
                    _diag_support_slide = {
                        'ts': _src_ts[_source_index],
                        'frame': _si + 1,
                        'weight': _weight,
                        'source_angle': _source_angle,
                        'after_angle': _after_angle,
                        'wrist_correction_sw':
                            _wrist_correction_sw,
                        'center_gap_sw': _center_gap_sw,
                        'post_clamp': _post_clamp,
                    }

            _source_angle_avg = (
                sum(_source_normal_angles) /
                len(_source_normal_angles)
                if _source_normal_angles else 0.0)
            _after_angle_avg = (
                sum(_after_normal_angles) /
                len(_after_normal_angles)
                if _after_normal_angles else 0.0)
            _full_orbit_x_range_sw = (
                max(_full_orbit_x_sw) -
                min(_full_orbit_x_sw)
                if _full_orbit_x_sw else 0.0)
            _full_orbit_y_range_sw = (
                max(_full_orbit_y_sw) -
                min(_full_orbit_y_sw)
                if _full_orbit_y_sw else 0.0)
            print(f"[SUPPORT-SLIDE]"
                  f"  support={_support_side}"
                  f"  moving={_moving_side}"
                  f"  height_source_median_sw="
                  f"{_source_height_median:.4f}"
                  f"  height_applied_sw={_support_height_sw:.4f}"
                  f"  height_samples={len(_source_heights_sw)}"
                  f"  weighted_frames={_weighted_count}"
                  f"  full_frames={_full_count}"
                  f"  source_normal_angle_avg_deg="
                  f"{_source_angle_avg:.2f}"
                  f"  after_normal_angle_avg_deg="
                  f"{_after_angle_avg:.2f}"
                  f"  max_wrist_correction_sw="
                  f"{_max_wrist_correction_sw:.4f}"
                  f"  max_center_gap_sw={_max_center_gap_sw:.4f}"
                  f"  post_clamp={_post_clamp_count}")
            print(f"[SUPPORT-SLIDE-TRAJECTORY]"
                  f"  mode={_trajectory_mode}"
                  f"  enabled={_orbit_enabled}"
                  f"  cycles={_orbit_cycles:.2f}"
                  f"  radius_x_sw={_orbit_radius_x_sw:.4f}"
                  f"  radius_y_sw={_orbit_radius_y_sw:.4f}"
                  f"  full_x_range_sw="
                  f"{_full_orbit_x_range_sw:.4f}"
                  f"  full_y_range_sw="
                  f"{_full_orbit_y_range_sw:.4f}"
                  f"  source_index={_orbit_start}..{_orbit_end}"
                  f"  ts="
                  f"{(_src_ts[_orbit_start] if _orbit_start is not None else None)}"
                  f".."
                  f"{(_src_ts[_orbit_end] if _orbit_end is not None else None)}")
            if _diag_support_slide:
                _sd = _diag_support_slide
                print(f"[SUPPORT-SLIDE-TS]"
                      f"  ts={_sd['ts']}"
                      f"  frame={_sd['frame']}"
                      f"  weight={_sd['weight']:.3f}"
                      f"  source_normal_angle_deg="
                      f"{_sd['source_angle']:.2f}"
                      f"  after_normal_angle_deg="
                      f"{_sd['after_angle']:.2f}"
                      f"  wrist_correction_sw="
                      f"{_sd['wrist_correction_sw']:.4f}"
                      f"  center_gap_sw="
                      f"{_sd['center_gap_sw']:.4f}"
                      f"  post_clamp={_sd['post_clamp']}")

    # ── Two-Hand Contact Preservation ───────────────────────────────
    print(f"[TWO-HAND-CONTACT-MODE]"
          f"  motion_id={_motion_id!r}"
          f"  global_enabled={USE_TWO_HAND_CONTACT_PRESERVATION}"
          f"  mode={_contact_mode!r}"
          f"  enabled={_motion_contact_enabled}")
    if not _motion_contact_enabled:
        if _contact_mode == "source":
            print("[TWO-HAND-CONTACT] skipped — dynamic/sliding contact; "
                  "source wrist trajectory preserved")
        elif _contact_mode == "support_slide":
            print("[TWO-HAND-CONTACT] skipped — support-slide profile "
                  "handles palm-over-fist contact")
        else:
            print("[TWO-HAND-CONTACT] skipped — motion contact mode is off")
    if _motion_contact_enabled and char_sw > 1e-6:
        _contact_source_count = len(_src_ts)  # source frame 수 (INTRO 제외)

        # 길이 진단
        _clen_ok = (
            _contact_source_count == len(raw.get('l_valid', [])) and
            len(smooth.get('lw_ik', [])) == total and
            total == _contact_source_count + INTRO_FRAMES
        )
        print(f"[TWO-HAND-CONTACT-LENGTH]"
              f"  source_count={_contact_source_count}"
              f"  raw_valid_len={len(raw.get('l_valid', []))}"
              f"  smooth_lw_ik_len={len(smooth.get('lw_ik', []))}"
              f"  total={total}")
        print(f"[TWO-HAND-CONTACT-RAW-LENGTH]"
              f"  source_count={_contact_source_count}"
              f"  raw2d_l_len={len(_contact_raw2d_l)}"
              f"  raw2d_r_len={len(_contact_raw2d_r)}"
              f"  pose_sw_len={len(_contact_pose_sw)}"
              f"  {'OK' if len(_contact_raw2d_l)==_contact_source_count==len(_contact_raw2d_r)==len(_contact_pose_sw) else '오류: 길이 불일치'}")
        if not _clen_ok:
            print(f"[TWO-HAND-CONTACT-LENGTH] 오류: 길이 불일치 — contact 블록 스킵")
        else:

            # Pass 1: per-source-frame closest raw MediaPipe (x,y) pair (z 미사용)
            _raw_contact = []  # None or (li, rj, dist_sw)
            for _ci in range(_contact_source_count):
                _lpts = _contact_raw2d_l[_ci] if _ci < len(_contact_raw2d_l) else None
                _rpts = _contact_raw2d_r[_ci] if _ci < len(_contact_raw2d_r) else None
                _psw  = _contact_pose_sw[_ci]  if _ci < len(_contact_pose_sw) else None
                if not (_lpts and _rpts and _psw and _psw > 1e-6):
                    _raw_contact.append(None); continue
                _best_d = float('inf'); _best_li = 0; _best_rj = 0
                for _li in range(21):
                    _lxy = _lpts[_li]
                    for _rj in range(21):
                        _rxy = _rpts[_rj]
                        _d = math.sqrt((_lxy[0]-_rxy[0])**2 + (_lxy[1]-_rxy[1])**2) / _psw
                        if _d < _best_d:
                            _best_d = _d; _best_li = _li; _best_rj = _rj
                _raw_contact.append((_best_li, _best_rj, _best_d) if _best_d < float('inf') else None)

            # Pass 2: hysteresis → _contact_active[i]
            _contact_active = [False] * _contact_source_count
            _in_contact_h = False; _consec_h = 0
            for _ci in range(_contact_source_count):
                _rc = _raw_contact[_ci]
                _dist_h = _rc[2] if _rc else float('inf')
                if not _in_contact_h:
                    if _dist_h <= TWO_HAND_CONTACT_ENTER_SW:
                        _consec_h += 1
                        if _consec_h >= TWO_HAND_CONTACT_MIN_FRAMES:
                            _in_contact_h = True
                            for _k in range(_consec_h):
                                _contact_active[_ci - _k] = True
                    else:
                        _consec_h = 0
                else:
                    if _dist_h <= TWO_HAND_CONTACT_EXIT_SW:
                        _contact_active[_ci] = True
                    else:
                        _in_contact_h = False; _consec_h = 0

            # 길이 검증 출력
            print(f"[TWO-HAND-CONTACT-LENGTH]"
                  f"  raw_contact_len={len(_raw_contact)}"
                  f"  contact_active_len={len(_contact_active)}"
                  f"  (모두 source_count={_contact_source_count}이면 정상)")

            # [TWO-HAND-CONTACT-NEAREST] 가장 가까운 source frame 12개 (거리 오름차순)
            _nearest_all = [(i, rc[0], rc[1], rc[2]) for i, rc in enumerate(_raw_contact) if rc]
            _nearest_all.sort(key=lambda x: x[3])
            print(f"[TWO-HAND-CONTACT-NEAREST] (ENTER_SW={TWO_HAND_CONTACT_ENTER_SW}):")
            for _ni, _nli, _nrj, _nd in _nearest_all[:12]:
                _nts = _src_ts[_ni] if _ni < len(_src_ts) else '?'
                _flag = " ← ENTER" if _nd <= TWO_HAND_CONTACT_ENTER_SW else ""
                print(f"  ts={_nts}  L_lm={_nli}  R_lm={_nrj}  dist_sw={_nd:.4f}{_flag}")

            # [TWO-HAND-CONTACT-RUNS] ENTER_SW 이하 연속 구간
            print(f"[TWO-HAND-CONTACT-RUNS] (ENTER_SW={TWO_HAND_CONTACT_ENTER_SW}):")
            _run_s = None; _run_min = float('inf')
            for _ci in range(_contact_source_count + 1):
                _rc_r = _raw_contact[_ci] if _ci < _contact_source_count else None
                _in_run = _rc_r is not None and _rc_r[2] <= TWO_HAND_CONTACT_ENTER_SW
                if _in_run and _run_s is None:
                    _run_s = _ci; _run_min = _rc_r[2]
                elif _in_run:
                    _run_min = min(_run_min, _rc_r[2])
                elif not _in_run and _run_s is not None:
                    _run_e = _ci - 1
                    _ts_s = _src_ts[_run_s] if _run_s < len(_src_ts) else '?'
                    _ts_e = _src_ts[_run_e] if _run_e < len(_src_ts) else '?'
                    print(f"  ts={_ts_s}~{_ts_e}  frames={_run_e-_run_s+1}  min_dist_sw={_run_min:.4f}")
                    _run_s = None; _run_min = float('inf')

            # Pass 3: 접촉 구간마다 하나의 대표 랜드마크 쌍과 하나의
            # reach split을 사용한다. 프레임마다 closest pair와 split이
            # 바뀌면 접촉은 맞아도 wrist target이 좌우로 덜컥거릴 수 있다.
            _active_segments = []
            _seg_s = None
            for _ci in range(_contact_source_count + 1):
                _act = _contact_active[_ci] if _ci < _contact_source_count else False
                if _act and _seg_s is None:
                    _seg_s = _ci
                elif not _act and _seg_s is not None:
                    _active_segments.append((_seg_s, _ci - 1))
                    _seg_s = None

            _base_lw_c = [v.copy() if v is not None else None
                          for v in smooth['lw_ik']]
            _base_rw_c = [v.copy() if v is not None else None
                          for v in smooth['rw_ik']]
            _corr_w = [0.0] * _contact_source_count
            _corr_l_off = [None] * _contact_source_count
            _corr_r_off = [None] * _contact_source_count
            _corr_pair = [None] * _contact_source_count
            _corr_split = [None] * _contact_source_count
            _weight_log = []
            _segment_log = []

            def _contact_gap_vector(_ci, _li, _rj):
                _si = INTRO_FRAMES + _ci
                if not (0 <= _si < len(_base_lw_c)):
                    return None
                _lp0 = smooth['l_p0'][_si]
                _rp0 = smooth['r_p0'][_si]
                _lpi = smooth[f'l_p{_li}'][_si]
                _rpj = smooth[f'r_p{_rj}'][_si]
                _lw0 = _base_lw_c[_si]
                _rw0 = _base_rw_c[_si]
                if not (_lp0 and _rp0 and _lpi and _rpj and _lw0 and _rw0):
                    return None
                return ((_rw0 + (_rpj - _rp0)) -
                        (_lw0 + (_lpi - _lp0)))

            def _contact_clamp_vector(_cv):
                if _cv is None:
                    return None
                _out = _cv * TWO_HAND_CONTACT_BLEND
                _max_len = char_sw * TWO_HAND_CONTACT_MAX_CORRECTION_SW
                if _out.length > _max_len:
                    _out = _out.normalized() * _max_len
                return _out

            def _segment_split(_active_cv):
                # 구간 전체에서 동시에 reach 한도를 지키는 alpha 중
                # 0.5에 가장 가까운 값을 선택한다. 공통 feasible 값이
                # 없을 때만 총 초과량이 가장 작은 alpha를 사용한다.
                _best_feasible = None
                _best_fallback = None
                for _sp in range(TWO_HAND_CONTACT_REACH_SPLIT_STEPS + 1):
                    _alpha = _sp / TWO_HAND_CONTACT_REACH_SPLIT_STEPS
                    _viol_sum = 0.0
                    _viol_max = 0.0
                    for _fci, _fcv in _active_cv.items():
                        _fsi = INTRO_FRAMES + _fci
                        _fl = _base_lw_c[_fsi] + _fcv * _alpha
                        _fr = _base_rw_c[_fsi] - _fcv * (1.0 - _alpha)
                        _lr = ((_fl - l_sh_w).length / char_l_arm_total
                               if char_l_arm_total > 1e-6 else 0.0)
                        _rr = ((_fr - r_sh_w).length / char_r_arm_total
                               if char_r_arm_total > 1e-6 else 0.0)
                        _viol = (max(0.0, _lr - ARM_REACH_MAX_RATIO) ** 2 +
                                 max(0.0, _rr - ARM_REACH_MAX_RATIO) ** 2)
                        _viol_sum += _viol
                        _viol_max = max(_viol_max, _viol)
                    _balance = abs(_alpha - 0.5)
                    _score = (_viol_max, _viol_sum, _balance, _alpha)
                    if _viol_max <= 1e-12:
                        if (_best_feasible is None or
                                _balance < _best_feasible[2]):
                            _best_feasible = _score
                    if _best_fallback is None or _score[:3] < _best_fallback[:3]:
                        _best_fallback = _score
                _chosen = _best_feasible if _best_feasible is not None else _best_fallback
                return ((_chosen[3] if _chosen else 0.5),
                        _best_feasible is not None)

            def _contact_smoothstep01(_x):
                _x = max(0.0, min(1.0, _x))
                return _x * _x * (3.0 - 2.0 * _x)

            for _seg_index, (_as, _ae) in enumerate(_active_segments):
                # active 구간에서 가장 자주 closest로 선택된 쌍을 대표로 사용.
                # 동률이면 평균 source 거리가 작은 쌍을 선택한다.
                _pair_stats = {}
                for _ci in range(_as, _ae + 1):
                    _rc = _raw_contact[_ci]
                    if not _rc:
                        continue
                    _cp = (_rc[0], _rc[1])
                    _cnt, _dsum = _pair_stats.get(_cp, (0, 0.0))
                    _pair_stats[_cp] = (_cnt + 1, _dsum + _rc[2])
                if not _pair_stats:
                    continue
                _anchor_pair = min(
                    _pair_stats,
                    key=lambda p: (-_pair_stats[p][0],
                                   _pair_stats[p][1] / _pair_stats[p][0],
                                   p[0], p[1]))
                _ali, _arj = _anchor_pair

                _active_cv = {}
                for _ci in range(_as, _ae + 1):
                    _cv = _contact_clamp_vector(
                        _contact_gap_vector(_ci, _ali, _arj))
                    if _cv is not None:
                        _active_cv[_ci] = _cv
                if not _active_cv:
                    continue

                _split_l, _split_feasible = _segment_split(_active_cv)
                for _ci, _cv in _active_cv.items():
                    _corr_w[_ci] = TWO_HAND_CONTACT_FADE_EDGE_WEIGHT
                    _corr_l_off[_ci] = _cv * _split_l
                    _corr_r_off[_ci] = -_cv * (1.0 - _split_l)
                    _corr_pair[_ci] = _anchor_pair
                    _corr_split[_ci] = _split_l

                # transition 프레임에서는 그 프레임의 새 closest gap을
                # 다시 계산하지 않는다. 경계의 완성된 좌/우 offset 자체를
                # cosine-like smoothstep으로 0↔1 블렌드한다.
                _first_ci = min(_active_cv)
                _last_ci = max(_active_cv)
                _first_l = _corr_l_off[_first_ci]
                _first_r = _corr_r_off[_first_ci]
                _last_l = _corr_l_off[_last_ci]
                _last_r = _corr_r_off[_last_ci]
                for _dist_frames in range(1, TWO_HAND_CONTACT_TRANSITION_FRAMES + 1):
                    _t = ((TWO_HAND_CONTACT_TRANSITION_FRAMES -
                           _dist_frames + 1) /
                          (TWO_HAND_CONTACT_TRANSITION_FRAMES + 1))
                    _tw = _contact_smoothstep01(_t)
                    for _ti, _loff, _roff in [
                            (_as - _dist_frames, _first_l, _first_r),
                            (_ae + _dist_frames, _last_l, _last_r)]:
                        if not (0 <= _ti < _contact_source_count):
                            continue
                        if _corr_w[_ti] >= _tw:
                            continue
                        _corr_w[_ti] = _tw
                        _corr_l_off[_ti] = _loff * _tw
                        _corr_r_off[_ti] = _roff * _tw
                        _corr_pair[_ti] = _anchor_pair
                        _corr_split[_ti] = _split_l

                _log_s = max(0, _as - TWO_HAND_CONTACT_TRANSITION_FRAMES)
                _log_e = min(_contact_source_count - 1,
                             _ae + TWO_HAND_CONTACT_TRANSITION_FRAMES)
                _weight_log.append((
                    _src_ts[_log_s], _src_ts[_log_e], _log_e - _log_s + 1,
                    [_corr_w[_wi] for _wi in range(_log_s, _log_e + 1)]))
                _pair_summary = sorted(
                    [(p, v[0], v[1] / v[0]) for p, v in _pair_stats.items()],
                    key=lambda x: (-x[1], x[2], x[0][0], x[0][1]))
                _segment_log.append((
                    _seg_index, _as, _ae, _anchor_pair, _pair_summary,
                    _split_l, _split_feasible))

            print(f"[TWO-HAND-CONTACT-WEIGHTS]"
                  f"  transition_frames={TWO_HAND_CONTACT_TRANSITION_FRAMES}"
                  f"  mode=boundary_offset_smoothstep")
            for _wts_s, _wts_e, _wfcnt, _wfs in _weight_log:
                print(f"  ts={_wts_s}~{_wts_e}  frames={_wfcnt}"
                      f"  weights={[round(_wf, 3) for _wf in _wfs]}")
            print("[TWO-HAND-CONTACT-SEGMENT]")
            for (_sgi, _sas, _sae, _sap, _sps, _ssl, _sfeas) in _segment_log:
                print(f"  segment={_sgi}"
                      f"  ts={_src_ts[_sas]}~{_src_ts[_sae]}"
                      f"  anchor_pair=L{_sap[0]}-R{_sap[1]}"
                      f"  split L={_ssl:.3f} R={1.0-_ssl:.3f}"
                      f"  common_feasible={_sfeas}"
                      f"  source_pairs={_sps}")

            # Pass 4: 완성된 per-side offset을 base wrist target에 적용한다.
            _max_corr_sw_c = 0.0; _max_src_dsw_c = 0.0; _postclamp_cnt_c = 0
            _max_anchor_gap_after_sw_c = 0.0
            _max_nearest_gap_after_sw_c = 0.0
            _applied_corr_sw_c = [0.0] * _contact_source_count
            _split_l_values_c = []
            _diag_bi_c = (min(range(len(_src_ts)), key=lambda k: abs(_src_ts[k] - _DIAG_TS))
                          if _src_ts else None)
            _contact_diag_c = None
            for _ci in range(_contact_source_count):
                _loff = _corr_l_off[_ci]
                _roff = _corr_r_off[_ci]
                _w = _corr_w[_ci]
                _cpair = _corr_pair[_ci]
                if _loff is None or _roff is None or _cpair is None or _w < 1e-6:
                    continue
                _si = INTRO_FRAMES + _ci
                _lw_bef = _base_lw_c[_si]
                _rw_bef = _base_rw_c[_si]
                if _lw_bef is None or _rw_bef is None:
                    continue
                _lw_candidate = _lw_bef + _loff
                _rw_candidate = _rw_bef + _roff
                smooth['lw_ik'][_si] = _lw_candidate
                smooth['rw_ik'][_si] = _rw_candidate
                _corr_sw = (_loff - _roff).length / char_sw
                _applied_corr_sw_c[_ci] = _corr_sw
                _max_corr_sw_c = max(_max_corr_sw_c, _corr_sw)
                if _corr_split[_ci] is not None:
                    _split_l_values_c.append(_corr_split[_ci])
                _rc2 = _raw_contact[_ci]
                if _contact_active[_ci] and _rc2:
                    _max_src_dsw_c = max(_max_src_dsw_c, _rc2[2])

                # 공통 split이 active 전체에서 reach 한도를 만족하도록
                # 골라졌으므로 clamp는 보통 transition에서만 발생한다.
                _postclamp = False
                for _ik_k3, _sh_w3, _at3 in [('lw_ik', l_sh_w, char_l_arm_total),
                                               ('rw_ik', r_sh_w, char_r_arm_total)]:
                    _v3 = smooth[_ik_k3][_si]
                    if _v3 is None or _at3 <= 1e-6:
                        continue
                    _off3 = _v3 - _sh_w3
                    if _off3.length / _at3 > ARM_REACH_MAX_RATIO:
                        smooth[_ik_k3][_si] = (
                            _sh_w3 + _off3.normalized() *
                            _at3 * ARM_REACH_MAX_RATIO)
                        _postclamp = True
                if _postclamp:
                    _postclamp_cnt_c += 1

                _li2, _rj2 = _cpair
                _lp0_2 = smooth['l_p0'][_si]
                _rp0_2 = smooth['r_p0'][_si]
                _lpli2 = smooth[f'l_p{_li2}'][_si]
                _rprj2 = smooth[f'r_p{_rj2}'][_si]
                _lw_eval = smooth['lw_ik'][_si]
                _rw_eval = smooth['rw_ik'][_si]
                _gap_bef_sw = ((_rw_bef + (_rprj2 - _rp0_2)) -
                               (_lw_bef + (_lpli2 - _lp0_2))).length / char_sw
                _anchor_gap_after_sw = (
                    ((_rw_eval + (_rprj2 - _rp0_2)) -
                     (_lw_eval + (_lpli2 - _lp0_2))).length / char_sw)
                if _contact_active[_ci]:
                    _max_anchor_gap_after_sw_c = max(
                        _max_anchor_gap_after_sw_c, _anchor_gap_after_sw)
                    if _rc2:
                        _nli, _nrj = _rc2[0], _rc2[1]
                        _nlpi = smooth[f'l_p{_nli}'][_si]
                        _nrpj = smooth[f'r_p{_nrj}'][_si]
                        _nearest_gap_after_sw = (
                            ((_rw_eval + (_nrpj - _rp0_2)) -
                             (_lw_eval + (_nlpi - _lp0_2))).length / char_sw)
                        _max_nearest_gap_after_sw_c = max(
                            _max_nearest_gap_after_sw_c,
                            _nearest_gap_after_sw)

                if _diag_bi_c is not None and _ci == _diag_bi_c:
                    _anchor_src_dist = float('nan')
                    _lraw = _contact_raw2d_l[_ci]
                    _rraw = _contact_raw2d_r[_ci]
                    _psw = _contact_pose_sw[_ci]
                    if _lraw and _rraw and _psw and _psw > 1e-6:
                        _anchor_src_dist = math.sqrt(
                            (_lraw[_li2][0] - _rraw[_rj2][0]) ** 2 +
                            (_lraw[_li2][1] - _rraw[_rj2][1]) ** 2) / _psw
                    _unclamped_sw = _gap_bef_sw * TWO_HAND_CONTACT_BLEND
                    _clamped_sw = min(
                        _unclamped_sw, TWO_HAND_CONTACT_MAX_CORRECTION_SW)
                    _reach_l_c = ((_lw_eval - l_sh_w).length / char_l_arm_total
                                  if char_l_arm_total > 1e-6 else 0.0)
                    _reach_r_c = ((_rw_eval - r_sh_w).length / char_r_arm_total
                                  if char_r_arm_total > 1e-6 else 0.0)
                    _contact_diag_c = (
                        _src_ts[_ci], _li2, _rj2, _anchor_src_dist,
                        _gap_bef_sw, _corr_sw, _anchor_gap_after_sw,
                        _lw_bef, _lw_eval, _rw_bef, _rw_eval,
                        _w, _unclamped_sw, _clamped_sw,
                        _corr_split[_ci], _reach_l_c, _reach_r_c)

            # 보정 구간과 바로 바깥 1프레임까지 포함해 실제 target step을
            # 측정한다. v4에서 보였던 큰 진입/이탈 스텝의 재발 여부를 본다.
            _diag_window_indices = set()
            for _as, _ae in _active_segments:
                _ws = max(0, _as - TWO_HAND_CONTACT_TRANSITION_FRAMES - 1)
                _we = min(_contact_source_count - 1,
                          _ae + TWO_HAND_CONTACT_TRANSITION_FRAMES + 1)
                _diag_window_indices.update(range(_ws, _we + 1))
            _max_l_step_c = 0.0; _max_r_step_c = 0.0
            for _ci in sorted(_diag_window_indices):
                if _ci <= 0:
                    continue
                _si = INTRO_FRAMES + _ci
                _lcur = smooth['lw_ik'][_si]
                _lprev = smooth['lw_ik'][_si - 1]
                _rcur = smooth['rw_ik'][_si]
                _rprev = smooth['rw_ik'][_si - 1]
                if _lcur is not None and _lprev is not None:
                    _max_l_step_c = max(
                        _max_l_step_c, (_lcur - _lprev).length / char_sw)
                if _rcur is not None and _rprev is not None:
                    _max_r_step_c = max(
                        _max_r_step_c, (_rcur - _rprev).length / char_sw)

            _cand_c = sum(
                1 for _rc in _raw_contact
                if _rc and _rc[2] <= TWO_HAND_CONTACT_ENTER_SW)
            _conf_c = sum(_contact_active)
            _transition_c = sum(
                1 for _ci, _w in enumerate(_corr_w)
                if _w >= 1e-6 and not _contact_active[_ci])
            _split_avg_c = (sum(_split_l_values_c) / len(_split_l_values_c)
                            if _split_l_values_c else 0.5)
            _split_min_c = min(_split_l_values_c) if _split_l_values_c else 0.5
            _split_max_c = max(_split_l_values_c) if _split_l_values_c else 0.5
            print(f"[TWO-HAND-CONTACT] candidate={_cand_c}  confirmed={_conf_c}"
                  f"  transition={_transition_c}"
                  f"  max_src_dist_sw={_max_src_dsw_c:.4f}"
                  f"  max_corr_sw={_max_corr_sw_c:.4f}"
                  f"  max_active_anchor_gap_after_sw={_max_anchor_gap_after_sw_c:.6f}"
                  f"  max_active_nearest_gap_after_sw={_max_nearest_gap_after_sw_c:.6f}"
                  f"  post_clamp={_postclamp_cnt_c}")
            print(f"[TWO-HAND-CONTACT-REACH-SPLIT]"
                  f"  left_share_avg={_split_avg_c:.3f}"
                  f"  range=[{_split_min_c:.3f},{_split_max_c:.3f}]"
                  f"  right_share_avg={1.0-_split_avg_c:.3f}")
            print(f"[TWO-HAND-CONTACT-STEP-MAX]"
                  f"  L={_max_l_step_c:.4f}sw  R={_max_r_step_c:.4f}sw")
            print("[TWO-HAND-CONTACT-TRANSITION]")
            for _ci in sorted(_diag_window_indices):
                _si = INTRO_FRAMES + _ci
                _lstep = 0.0
                _rstep = 0.0
                if _ci > 0 and _si > 0:
                    _lcur = smooth['lw_ik'][_si]
                    _lprev = smooth['lw_ik'][_si - 1]
                    _rcur = smooth['rw_ik'][_si]
                    _rprev = smooth['rw_ik'][_si - 1]
                    if _lcur is not None and _lprev is not None:
                        _lstep = (_lcur - _lprev).length / char_sw
                    if _rcur is not None and _rprev is not None:
                        _rstep = (_rcur - _rprev).length / char_sw
                _state = ("active" if _contact_active[_ci] else
                          ("transition" if _corr_w[_ci] >= 1e-6 else "outside"))
                print(f"  ts={_src_ts[_ci]}  state={_state}"
                      f"  weight={_corr_w[_ci]:.3f}"
                      f"  corr/sw={_applied_corr_sw_c[_ci]:.4f}"
                      f"  wrist_step/sw L={_lstep:.4f} R={_rstep:.4f}")
            # [TWO-HAND-CONTACT-TS] _DIAG_TS 근접 프레임 — 접촉 활성 여부와 무관하게 항상 출력
            if _diag_bi_c is not None and _diag_bi_c < _contact_source_count:
                _dts_rc = _raw_contact[_diag_bi_c]
                _dts_ts = _src_ts[_diag_bi_c]
                _dts_act = _contact_active[_diag_bi_c]
                if _dts_rc:
                    _dts_li, _dts_rj, _dts_dsw = _dts_rc
                    print(f"[TWO-HAND-CONTACT-TS] ts={_dts_ts}"
                          f"  L_lm={_dts_li}  R_lm={_dts_rj}"
                          f"  raw_dist_sw={_dts_dsw:.4f}"
                          f"  active={_dts_act}"
                          f"  corrected={_contact_diag_c is not None}")
                else:
                    print(f"[TWO-HAND-CONTACT-TS] ts={_dts_ts}  raw_contact=None"
                          f"  active={_dts_act}  corrected=False")
            if _contact_diag_c:
                _cd = _contact_diag_c
                print(f"[TWO-HAND-CONTACT-TS-DETAIL] ts={_cd[0]}  L_lm={_cd[1]}  R_lm={_cd[2]}"
                      f"  src_dist_sw={_cd[3]:.4f}  gap_before/sw={_cd[4]:.4f}"
                      f"  corr/sw={_cd[5]:.4f}  gap_after/sw={_cd[6]:.4f}")
                print(f"  fade_weight={_cd[11]:.3f}"
                      f"  unclamped_corr/sw={_cd[12]:.4f}"
                      f"  clamped_corr/sw={_cd[13]:.4f}")
                print(f"  reach_aware_split L={_cd[14]:.3f} R={1.0-_cd[14]:.3f}"
                      f"  candidate_reach L={_cd[15]:.4f} R={_cd[16]:.4f}")
                _lb2=_cd[7]; _la2=_cd[8]; _rb2=_cd[9]; _ra2=_cd[10]
                print(f"  L_wrist before=({_lb2.x:.4f},{_lb2.y:.4f},{_lb2.z:.4f})"
                      f"  after=({_la2.x:.4f},{_la2.y:.4f},{_la2.z:.4f})")
                print(f"  R_wrist before=({_rb2.x:.4f},{_rb2.y:.4f},{_rb2.z:.4f})"
                      f"  after=({_ra2.x:.4f},{_ra2.y:.4f},{_ra2.z:.4f})")

    # ── Source Upper-Arm Guidance: 방향 스무딩 (Pole Track 전) ───────
    _upper_src_l_sm = (smooth_direction_track(_upper_src_l, window=SOURCE_UPPER_ARM_SMOOTH_WINDOW)
                       if USE_SOURCE_UPPER_ARM_GUIDANCE else [None] * len(_upper_src_l))
    _upper_src_r_sm = (smooth_direction_track(_upper_src_r, window=SOURCE_UPPER_ARM_SMOOTH_WINDOW)
                       if USE_SOURCE_UPPER_ARM_GUIDANCE else [None] * len(_upper_src_r))
    _guide_diag_l = None; _guide_diag_r = None  # [UPPER-ARM-EVAL-TS] 용

    # ── Elbow pole track ─────────────────────────────────────────
    _pole_track_l = []; _pole_track_r = []
    if USE_ELBOW_POLE:
        def _pole_dir_angle_deg(_a, _b):
            if _a is None or _b is None or _a.length < 1e-6 or _b.length < 1e-6:
                return None
            _dot = max(-1.0, min(1.0, _a.normalized().dot(_b.normalized())))
            return math.degrees(math.acos(_dot))

        def _pole_slerp_dir(_a, _b, _t):
            """두 단위 방향 사이를 구면 보간한다. 반평행 근처는 안전한 lerp fallback."""
            _aa = _a.normalized()
            _bb = _b.normalized()
            _t = max(0.0, min(1.0, float(_t)))
            _dot = max(-1.0, min(1.0, _aa.dot(_bb)))
            _ang = math.acos(_dot)
            if _ang < 1e-7:
                return _aa.copy()
            _axis = _aa.cross(_bb)
            if _axis.length < 1e-7:
                # 거의 반평행이면 선형 보간의 중간이 영벡터가 되므로,
                # 현재 방향과 가장 덜 평행한 기준축을 골라 안정적으로 회전한다.
                _refs = [
                    mathutils.Vector((1.0, 0.0, 0.0)),
                    mathutils.Vector((0.0, 1.0, 0.0)),
                    mathutils.Vector((0.0, 0.0, 1.0)),
                ]
                _ref = min(_refs, key=lambda _rv: abs(_aa.dot(_rv)))
                _axis = _aa.cross(_ref)
                if _axis.length < 1e-7:
                    return _aa.copy()
            _q = mathutils.Quaternion(_axis.normalized(), _ang * _t)
            return (_q @ _aa).normalized()

        def _predicted_wrist_angle_deg(_wrist, _circle_center, _circle_height,
                                       _radial, _hand_forward):
            _elbow = _circle_center + _radial.normalized() * _circle_height
            _forearm = _wrist - _elbow
            if _forearm.length < 1e-6 or _hand_forward.length < 1e-6:
                return None
            return _pole_dir_angle_deg(_forearm, _hand_forward)

        def _build_source_wrist_angle_track(_side, _forearm_dirs):
            """
            원본의 elbow→hybrid-wrist 방향과 hand pt0→pt9 방향 사이 각도.
            실제 손 검출 프레임에서만 측정하고, 짧은 누락 구간은 보간한다.
            """
            _prefix = 'l' if _side == 'L' else 'r'
            _valid = raw[f'{_prefix}_valid']
            _valid_count = sum(1 for _v in _valid if _v)
            if _valid_count < WRIST_AWARE_MIN_VALID_FRAMES:
                return [None] * len(_src_ts), _valid_count
            _values = []
            for _wi in range(len(_src_ts)):
                if _wi >= len(_valid) or not _valid[_wi]:
                    _values.append(None)
                    continue
                _si = INTRO_FRAMES + _wi
                _p0 = smooth[f'{_prefix}_p0'][_si]
                _p9 = smooth['lh9' if _side == 'L' else 'rh9'][_si]
                _fa = _forearm_dirs[_wi] if _wi < len(_forearm_dirs) else None
                _hd = (_p9 - _p0) if (_p0 is not None and _p9 is not None) else None
                _values.append(_pole_dir_angle_deg(_fa, _hd))
            return (smooth_scalar_track(
                        _values, window=WRIST_AWARE_SOURCE_SMOOTH_WINDOW),
                    _valid_count)

        _source_wrist_angle_l, _source_wrist_valid_l = (
            _build_source_wrist_angle_track('L', _forearm_hybrid_src_l))
        _source_wrist_angle_r, _source_wrist_valid_r = (
            _build_source_wrist_angle_track('R', _forearm_hybrid_src_r))
        print(f"[WRIST-AWARE-SOURCE]"
              f"  L_valid={_source_wrist_valid_l}"
              f"  R_valid={_source_wrist_valid_r}"
              f"  min_required={WRIST_AWARE_MIN_VALID_FRAMES}")

        for (_ik_key, _sh_w, _arm_tot, _srb, _rest_el, _pside, _p_tr,
             _upsm, _up_len, _fa_len, _source_wrist_angles) in [
                ('lw_ik', l_sh_w, char_l_arm_total, _pole_src_l, _rest_elbow_l_w, 'L', _pole_track_l,
                 _upper_src_l_sm, char_l_upper_len, char_l_forearm_len,
                 _source_wrist_angle_l),
                ('rw_ik', r_sh_w, char_r_arm_total, _pole_src_r, _rest_elbow_r_w, 'R', _pole_track_r,
                 _upper_src_r_sm, char_r_upper_len, char_r_forearm_len,
                 _source_wrist_angle_r)]:
            _prev_dir = None; _flip_cnt = 0; _fb_cnt = 0; _val_cnt = 0
            _guide_cnt = 0; _legacy_cnt = 0; _rest_cnt = 0
            _pdiag = None; _pdiag_best = float('inf')
            _guide_diag = None
            _raw_poles = []; _rest_dr = _rest_el - _sh_w
            _wa_available = 0; _wa_used = 0; _wa_fallback = 0
            _wa_base_sum = 0.0; _wa_after_sum = 0.0; _wa_best_sum = 0.0
            _wa_source_sum = 0.0; _wa_target_sum = 0.0; _wa_source_valid = 0
            _wa_shift_max = 0.0; _wa_diag = None; _wa_diag_best = float('inf')
            _source_count = len(_src_ts)
            for _pi in range(_source_count):
                _si2 = INTRO_FRAMES + _pi
                _wv = smooth[_ik_key][_si2] if _si2 < len(smooth[_ik_key]) else None
                _bv = _srb[_pi] if _pi < len(_srb) else None
                _ts2 = _src_ts[_pi] if _pi < len(_src_ts) else 0
                _uv = _upsm[_pi] if _pi < len(_upsm) else None  # 스무딩된 upper-arm 방향

                _wdiff = _wv - _sh_w if _wv else None
                _cax = ((_wdiff.normalized() if _wdiff.length > 1e-6 else mathutils.Vector((0,0,1)))
                        if _wdiff else mathutils.Vector((0,0,1)))

                # Legacy radial (bend source 기반, shoulder→wrist 축에 수직)
                _legacy_pd = (_bv - _cax * _bv.dot(_cax)) if (_bv and _bv.length > 1e-4) else None
                if _legacy_pd is not None and _legacy_pd.length < 1e-4:
                    _legacy_pd = None

                # Guided radial (source upper-arm 방향 기반, shoulder→wrist 축에 수직)
                _guided_pd = None; _feas_el = None
                if USE_SOURCE_UPPER_ARM_GUIDANCE and _uv is not None and _wdiff is not None:
                    _gr = _uv - _cax * _uv.dot(_cax)
                    if _gr.length > 1e-4:
                        _gr = _gr.normalized()
                        _guided_pd = _gr
                        # IK 원 위의 feasible elbow 계산
                        _d_wr = _wdiff.length
                        if _d_wr > 1e-6:
                            _along = (_up_len*_up_len - _fa_len*_fa_len + _d_wr*_d_wr) / (2.0 * _d_wr)
                            _height = math.sqrt(max(0.0, _up_len*_up_len - _along*_along))
                            _cc = _sh_w + _cax * _along
                            _feas_el = _cc + _guided_pd * _height

                # Guided / legacy / rest fallback 선택 + blend
                if _guided_pd is not None:
                    if _legacy_pd is not None and SOURCE_UPPER_ARM_GUIDANCE_BLEND < 1.0 - 1e-6:
                        _blended = _legacy_pd.normalized().lerp(_guided_pd, SOURCE_UPPER_ARM_GUIDANCE_BLEND)
                        _pd = _blended.normalized() if _blended.length > 1e-6 else _guided_pd
                    else:
                        _pd = _guided_pd
                    _guide_cnt += 1; _val_cnt += 1
                elif _legacy_pd is not None:
                    _pd = _legacy_pd.normalized()
                    _legacy_cnt += 1; _val_cnt += 1
                else:
                    _fb_cnt += 1; _rest_cnt += 1
                    _rd2 = _rest_dr - _cax * _rest_dr.dot(_cax)
                    _pd = _rd2.normalized() if _rd2.length > 1e-4 else mathutils.Vector((0,-1,0))

                # Wrist-aware pole:
                # IK 원 위에서 손 방향과 가장 잘 정렬되는 전완을 구하고,
                # source 팔꿈치 평면에서 최대 허용 각도 안에서만 그 방향으로 이동한다.
                # 손의 world 방향은 이후 FK가 그대로 보존한다.
                if (USE_WRIST_AWARE_ELBOW_GUIDANCE and
                        _pside in _wrist_aware_target_sides and
                        _wv is not None and _wdiff is not None and _wdiff.length > 1e-6):
                    _hp0_key = 'l_p0' if _pside == 'L' else 'r_p0'
                    _h9_key  = 'lh9' if _pside == 'L' else 'rh9'
                    _hp0 = smooth[_hp0_key][_si2] if _si2 < len(smooth[_hp0_key]) else None
                    _h9v = smooth[_h9_key][_si2] if _si2 < len(smooth[_h9_key]) else None
                    _hdir = (_h9v - _hp0) if (_hp0 is not None and _h9v is not None) else None
                    _d_wr = _wdiff.length
                    _along = (_up_len*_up_len - _fa_len*_fa_len + _d_wr*_d_wr) / (2.0 * _d_wr)
                    _height = math.sqrt(max(0.0, _up_len*_up_len - _along*_along))
                    _circle_center = _sh_w + _cax * _along
                    _hperp = (_hdir - _cax * _hdir.dot(_cax)
                              if (_hdir is not None and _hdir.length > 1e-6) else None)

                    if (_hdir is not None and _hdir.length > 1e-6 and
                            _hperp is not None and _hperp.length > 1e-5 and
                            _height > 1e-6):
                        _hdir = _hdir.normalized()
                        _wrist_opt_pd = (-_hperp).normalized()
                        _base_pd = _pd.normalized()
                        _base_ang = _predicted_wrist_angle_deg(
                            _wv, _circle_center, _height, _base_pd, _hdir)
                        _best_ang = _predicted_wrist_angle_deg(
                            _wv, _circle_center, _height, _wrist_opt_pd, _hdir)
                        _sep_ang = _pole_dir_angle_deg(_base_pd, _wrist_opt_pd)
                        _after_ang = _base_ang
                        _used_shift = 0.0
                        _source_ang = (_source_wrist_angles[_pi]
                                       if _pi < len(_source_wrist_angles) else None)
                        _dynamic_target = (
                            max(WRIST_AWARE_MIN_TARGET_ANGLE_DEG,
                                _source_ang + WRIST_AWARE_SOURCE_TOLERANCE_DEG)
                            if _source_ang is not None else None)
                        _wa_available += 1
                        if _source_ang is not None:
                            _wa_source_sum += _source_ang
                            _wa_target_sum += _dynamic_target
                            _wa_source_valid += 1

                        if (_source_ang is not None and
                                _base_ang is not None and _best_ang is not None and
                                _sep_ang is not None and _sep_ang > 1e-5 and
                                _best_ang + 1e-4 < _base_ang and
                                _base_ang > _dynamic_target):
                            # 목표각에 도달하는 최소 구면 보간량을 찾는다.
                            _target_ang = max(_dynamic_target, _best_ang)
                            _lo = 0.0; _hi = 1.0
                            for _ in range(max(1, int(WRIST_AWARE_SEARCH_STEPS))):
                                _mid = 0.5 * (_lo + _hi)
                                _test_pd = _pole_slerp_dir(_base_pd, _wrist_opt_pd, _mid)
                                _test_ang = _predicted_wrist_angle_deg(
                                    _wv, _circle_center, _height, _test_pd, _hdir)
                                if _test_ang is not None and _test_ang <= _target_ang:
                                    _hi = _mid
                                else:
                                    _lo = _mid

                            _max_t = min(1.0, WRIST_AWARE_MAX_POLE_SHIFT_DEG / _sep_ang)
                            _use_t = min(_hi, _max_t)
                            if _use_t > 1e-6:
                                _pd = _pole_slerp_dir(_base_pd, _wrist_opt_pd, _use_t)
                                _after_ang = _predicted_wrist_angle_deg(
                                    _wv, _circle_center, _height, _pd, _hdir)
                                _used_shift = _sep_ang * _use_t
                                _wa_used += 1

                        _wa_base_sum += _base_ang if _base_ang is not None else 0.0
                        _wa_after_sum += _after_ang if _after_ang is not None else 0.0
                        _wa_best_sum += _best_ang if _best_ang is not None else 0.0
                        _wa_shift_max = max(_wa_shift_max, _used_shift)
                        if (_source_ang is not None and
                                abs(_ts2 - _DIAG_TS) < _wa_diag_best):
                            _wa_diag_best = abs(_ts2 - _DIAG_TS)
                            _wa_diag = {
                                'ts': _ts2, 'base': _base_ang, 'after': _after_ang,
                                'best': _best_ang, 'sep': _sep_ang,
                                'source': _source_ang, 'target': _dynamic_target,
                                'shift': _used_shift, 'radial': _pd.copy(),
                            }
                    else:
                        _wa_fallback += 1

                # 시간 연속성
                if _prev_dir is not None and _pd.dot(_prev_dir) < 0:
                    _pd = -_pd; _flip_cnt += 1
                _prev_dir = _pd
                _pp = _sh_w + _pd * _arm_tot * ELBOW_POLE_DISTANCE_RATIO
                _raw_poles.append(_pp)
                if abs(_ts2 - _DIAG_TS) < _pdiag_best:
                    _pdiag_best = abs(_ts2 - _DIAG_TS)
                    _pdiag = {'ts':_ts2,'bv':_bv,'dir':_pd,'pos':_pp}
                    _guide_diag = {
                        'ts': _ts2, 'src_upper': _uv, 'wrist_axis': _cax,
                        'guided_radial': _guided_pd, 'feas_el': _feas_el, 'pole_pos': _pp,
                        'up_len': _up_len, 'fa_len': _fa_len,
                        'target_reach': (_wdiff.length / _arm_tot if (_wdiff and _arm_tot > 1e-6) else 0.0),
                    }
            _smpoles = smooth_track(_raw_poles)
            _pole_step_max_deg = 0.0
            for _spi in range(1, len(_smpoles)):
                _sd0 = _smpoles[_spi-1] - _sh_w
                _sd1 = _smpoles[_spi] - _sh_w
                _sang = _pole_dir_angle_deg(_sd0, _sd1)
                if _sang is not None:
                    _pole_step_max_deg = max(_pole_step_max_deg, _sang)
            if INTRO_FRAMES > 0:
                _rdn = _rest_dr.normalized() if _rest_dr.length > 1e-4 else mathutils.Vector((0,-1,0))
                _rp0 = _sh_w + _rdn * _arm_tot * ELBOW_POLE_DISTANCE_RATIO
                _fp0 = _smpoles[0] if _smpoles else _rp0
                _smpoles = [_rp0.lerp(_fp0,(k+1)/INTRO_FRAMES) for k in range(INTRO_FRAMES)] + _smpoles
            _p_tr.extend(_smpoles)
            if len(_p_tr) != total:
                print(f"[오류] pole track length {len(_p_tr)} != total {total} ({_pside})")
            # [UPPER-ARM-GUIDE] 통계
            if _pside == 'L': _guide_diag_l = _guide_diag
            else:             _guide_diag_r = _guide_diag
            print(f"[UPPER-ARM-GUIDE] {_pside}:"
                  f"  source_frames={_source_count}"
                  f"  guided_frames={_guide_cnt}"
                  f"  legacy_fallback_frames={_legacy_cnt}"
                  f"  rest_fallback_frames={_rest_cnt}"
                  f"  temporal_flips={_flip_cnt}"
                  f"  source_track_length={len(_upsm)}"
                  f"  smoothed_track_length={len(_upsm)}")
            if _guide_diag:
                _gd = _guide_diag
                _su_str = (f"({_gd['src_upper'].x:+.3f},{_gd['src_upper'].y:+.3f},{_gd['src_upper'].z:+.3f})"
                           if _gd['src_upper'] else "None")
                _gr_str = (f"({_gd['guided_radial'].x:+.3f},{_gd['guided_radial'].y:+.3f},{_gd['guided_radial'].z:+.3f})"
                           if _gd['guided_radial'] else "None")
                _fe_str = (f"({_gd['feas_el'].x:.4f},{_gd['feas_el'].y:.4f},{_gd['feas_el'].z:.4f})"
                           if _gd['feas_el'] else "None")
                _ax_str = f"({_gd['wrist_axis'].x:+.3f},{_gd['wrist_axis'].y:+.3f},{_gd['wrist_axis'].z:+.3f})"
                _pp_str = (f"({_gd['pole_pos'].x:.4f},{_gd['pole_pos'].y:.4f},{_gd['pole_pos'].z:.4f})"
                           if _gd['pole_pos'] else "None")
                print(f"[UPPER-ARM-GUIDE-TS] {_pside}: ts={_gd['ts']}"
                      f"  source_upper_dir={_su_str}"
                      f"  final_wrist_axis={_ax_str}"
                      f"  guided_radial={_gr_str}"
                      f"  feasible_desired_elbow={_fe_str}"
                      f"  final_pole_position={_pp_str}"
                      f"  upper_len={_gd['up_len']:.4f}"
                      f"  forearm_len={_gd['fa_len']:.4f}"
                      f"  target_reach={_gd['target_reach']:.4f}")
            if _raw_poles:
                _xs2=[p.x for p in _raw_poles]; _ys2=[p.y for p in _raw_poles]; _zs2=[p.z for p in _raw_poles]
                print(f"[IK-POLE] {_pside}: valid={_val_cnt}  fallback={_fb_cnt}  flips={_flip_cnt}")
                print(f"  X:[{min(_xs2):.3f},{max(_xs2):.3f}]  Y:[{min(_ys2):.3f},{max(_ys2):.3f}]  Z:[{min(_zs2):.3f},{max(_zs2):.3f}]")
            if _pdiag:
                _d3=_pdiag; _bvs=f"({_d3['bv'].x:+.3f},{_d3['bv'].y:+.3f},{_d3['bv'].z:+.3f})" if _d3['bv'] else "None"
                print(f"[IK-POLE-TS] {_pside}: ts={_d3['ts']}  src_bend={_bvs}  dir=({_d3['dir'].x:+.3f},{_d3['dir'].y:+.3f},{_d3['dir'].z:+.3f})  pos=({_d3['pos'].x:.3f},{_d3['pos'].y:.3f},{_d3['pos'].z:.3f})")
            if USE_WRIST_AWARE_ELBOW_GUIDANCE and _pside in _wrist_aware_target_sides:
                _wa_den = max(1, _wa_available)
                _wa_src_den = max(1, _wa_source_valid)
                print(f"[WRIST-AWARE-POLE] {_pside}:"
                      f"  available_frames={_wa_available}"
                      f"  corrected_frames={_wa_used}"
                      f"  fallback_frames={_wa_fallback}"
                      f"  source_angle_frames={_wa_source_valid}"
                      f"  avg_source_wrist_deg={_wa_source_sum/_wa_src_den:.2f}"
                      f"  avg_dynamic_target_deg={_wa_target_sum/_wa_src_den:.2f}"
                      f"  avg_base_wrist_deg={_wa_base_sum/_wa_den:.2f}"
                      f"  avg_after_wrist_deg={_wa_after_sum/_wa_den:.2f}"
                      f"  avg_best_possible_deg={_wa_best_sum/_wa_den:.2f}"
                      f"  max_pole_shift_deg={_wa_shift_max:.2f}"
                      f"  max_smoothed_pole_step_deg={_pole_step_max_deg:.2f}")
                if _wa_diag:
                    print(f"[WRIST-AWARE-POLE-TS] {_pside}:"
                          f"  ts={_wa_diag['ts']}"
                          f"  base_wrist_deg={_wa_diag['base']:.2f}"
                          f"  after_wrist_deg={_wa_diag['after']:.2f}"
                          f"  best_possible_deg={_wa_diag['best']:.2f}"
                          f"  source_wrist_deg={_wa_diag['source']:.2f}"
                          f"  dynamic_target_deg={_wa_diag['target']:.2f}"
                          f"  source_to_optimal_deg={_wa_diag['sep']:.2f}"
                          f"  applied_shift_deg={_wa_diag['shift']:.2f}"
                          f"  final_radial=({_wa_diag['radial'].x:+.3f},"
                          f"{_wa_diag['radial'].y:+.3f},{_wa_diag['radial'].z:+.3f})")

    valid_frames = list(range(1, total+1))

    # ── IK Empty (팔 손목만) ─────────────────────────────────────
    col = bpy.data.collections.get("IK_Targets")
    if not col:
        col = bpy.data.collections.new("IK_Targets")
        bpy.context.scene.collection.children.link(col)
    emp_lw = make_empty("IK_L_Wrist", col)
    emp_rw = make_empty("IK_R_Wrist", col)
    if USE_ELBOW_POLE:
        emp_pole_l = make_empty("SL_LeftElbowPole", col)
        emp_pole_r = make_empty("SL_RightElbowPole", col)
    # ── Controller Bone 정리/생성 ─────────────────────────────────
    # 이전 실행의 생성 뼈가 남아 있으면 먼저 제거한다.
    # 새 controller는 IK chain의 Arm을 parent로 두지 않는다.
    # Arm parent → ForeArm Copy Rotation은 dependency cycle을 만들기 때문이다.
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    for _old_ctbn in ['SL_LeftForearmTwistCtrlBone', 'SL_RightForearmTwistCtrlBone']:
        _old_cteb = arm_obj.data.edit_bones.get(_old_ctbn)
        if _old_cteb is not None:
            arm_obj.data.edit_bones.remove(_old_cteb)
    if USE_FOREARM_TWIST_REDISTRIBUTION:
        for _ct_side, _ctbn, _src_fbn in [
                ('L', 'SL_LeftForearmTwistCtrlBone',  BONE_L_FOREARM),
                ('R', 'SL_RightForearmTwistCtrlBone', BONE_R_FOREARM)]:
            if _ct_side not in _forearm_twist_target_sides:
                continue
            _src_eb = arm_obj.data.edit_bones.get(_src_fbn)
            if not _src_eb:
                print(f"[FOREARM-TWIST-V2] 경고: source bone 없음 {_src_fbn}"); continue
            _ct_eb = arm_obj.data.edit_bones.get(_ctbn)
            if _ct_eb is None:
                _ct_eb = arm_obj.data.edit_bones.new(_ctbn)
            _ct_eb.head        = _src_eb.head.copy()
            _ct_eb.tail        = _src_eb.tail.copy()
            _ct_eb.roll        = _src_eb.roll
            _ct_eb.parent      = None
            _ct_eb.use_connect = False
            _ct_eb.use_deform  = False
    bpy.ops.object.mode_set(mode='OBJECT')

    # ── 컨스트레인트: 팔 IK만, 손/손가락은 모두 제거 ─────────────
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    # 이전 실행에서 남았을 수 있는 twist constraint는 기능 활성 여부와
    # 관계없이 항상 제거한다. 실패한 calibration이 IK에 잔류하지 않게 한다.
    for _cleanup_fbn in [BONE_L_FOREARM, BONE_R_FOREARM]:
        _cleanup_pb = arm_obj.pose.bones.get(_cleanup_fbn)
        if not _cleanup_pb:
            continue
        for _cleanup_con in list(_cleanup_pb.constraints):
            if _cleanup_con.name in (
                    'SL_ForearmTwist_L', 'SL_ForearmTwist_R',
                    'SL_ForearmTwistV2_L', 'SL_ForearmTwistV2_R'):
                _cleanup_pb.constraints.remove(_cleanup_con)

    # IK DOF 명시 초기화 (lock/limit/stiffness/stretch 전부 클리어)
    if IK_RESET_DOF:
        for _dbn in [BONE_L_ARM, BONE_L_FOREARM, BONE_R_ARM, BONE_R_FOREARM]:
            _dpb = arm_obj.pose.bones.get(_dbn)
            if not _dpb: continue
            _dpb.lock_ik_x = False; _dpb.lock_ik_y = False; _dpb.lock_ik_z = False
            _dpb.use_ik_limit_x = False; _dpb.use_ik_limit_y = False; _dpb.use_ik_limit_z = False
            _dpb.ik_stiffness_x = 0.0; _dpb.ik_stiffness_y = 0.0; _dpb.ik_stiffness_z = 0.0
            _dpb.ik_stretch = 0.0
            _dpb.use_ik_rotation_control = False
            _dpb.use_ik_linear_control   = False

    _ik_con_l = add_ik(arm_obj, BONE_L_FOREARM, emp_lw, chain_count=2, name='SL_ArmIK_L')
    _ik_con_r = add_ik(arm_obj, BONE_R_FOREARM, emp_rw, chain_count=2, name='SL_ArmIK_R')
    if USE_ELBOW_POLE:
        for _ik_con2, _pe2, _ang2 in [(_ik_con_l, emp_pole_l, L_ELBOW_POLE_ANGLE_DEG),
                                       (_ik_con_r, emp_pole_r, R_ELBOW_POLE_ANGLE_DEG)]:
            if _ik_con2:
                _ik_con2.pole_target = _pe2; _ik_con2.pole_subtarget = ""
                _ik_con2.pole_angle = math.radians(_ang2)

    # IK solver: LEGACY 명시 (속성 없는 버전은 무시)
    try: arm_obj.pose.ik_solver = 'LEGACY'
    except AttributeError: pass

    # IK stretch 비활성화 (DOF 초기화 후 재확인)
    for _sbn in [BONE_L_ARM, BONE_L_FOREARM, BONE_R_ARM, BONE_R_FOREARM]:
        _pb = arm_obj.pose.bones.get(_sbn)
        if _pb: _pb.ik_stretch = 0.0

    # Straight-chain singularity pre-bend (키프레임 없음)
    if IK_PREBEND_ENABLED:
        for _pbn, _psign in [(BONE_L_FOREARM, 1.0), (BONE_R_FOREARM, -1.0)]:
            _ppb = arm_obj.pose.bones.get(_pbn)
            if _ppb:
                _ppb.rotation_mode = 'XYZ'
                _ppb.rotation_euler = mathutils.Euler(
                    (math.radians(IK_PREBEND_DEG * _psign), 0.0, 0.0), 'XYZ')

    # [IK-DOF] 진단
    _ik_solver_str = getattr(arm_obj.pose, 'ik_solver', 'N/A')
    for _ds, _dabn, _dfbn, _dikc in [
            ('L', BONE_L_ARM, BONE_L_FOREARM, _ik_con_l),
            ('R', BONE_R_ARM, BONE_R_FOREARM, _ik_con_r)]:
        _dua = arm_obj.pose.bones.get(_dabn); _duf = arm_obj.pose.bones.get(_dfbn)
        _dpb_fa = arm_obj.pose.bones.get(_dfbn)
        _bend_deg = (math.degrees(_dpb_fa.rotation_euler[0])
                     if _dpb_fa and _dpb_fa.rotation_mode in ('XYZ','XZY','YXZ','YZX','ZXY','ZYX')
                     else 'N/A')
        print(f"[IK-DOF] {_ds}:")
        for _bn2, _pb2 in [('upper', _dua), ('forearm', _duf)]:
            if _pb2:
                print(f"  {_bn2}: lock_ik={_pb2.lock_ik_x}/{_pb2.lock_ik_y}/{_pb2.lock_ik_z}"
                      f"  limit={_pb2.use_ik_limit_x}/{_pb2.use_ik_limit_y}/{_pb2.use_ik_limit_z}"
                      f"  stiff={_pb2.ik_stiffness_x:.2f}/{_pb2.ik_stiffness_y:.2f}/{_pb2.ik_stiffness_z:.2f}"
                      f"  ik_stretch={_pb2.ik_stretch:.3f}")
        print(f"  forearm pre-bend X={_bend_deg}deg"
              f"  use_tail={_dikc.use_tail if _dikc else 'N/A'}"
              f"  iterations={_dikc.iterations if _dikc else 'N/A'}"
              f"  ik_solver={_ik_solver_str}")

    # [IK-CONSTRAINTS] ForeArm constraint 전체 목록
    for _cs, _cfbn in [('L', BONE_L_FOREARM), ('R', BONE_R_FOREARM)]:
        _cfb = arm_obj.pose.bones.get(_cfbn)
        _clist = _cfb.constraints if _cfb else []
        print(f"[IK-CONSTRAINTS] {_cs} ({_cfbn}) — {len(_clist)}개:")
        for _cc in _clist:
            _tname = _cc.target.name if getattr(_cc, 'target', None) else 'None'
            _line = (f"  name={_cc.name!r} type={_cc.type} target={_tname}"
                     f" influence={_cc.influence:.2f} mute={_cc.mute}")
            if _cc.type == 'IK':
                _pname = _cc.pole_target.name if getattr(_cc, 'pole_target', None) else 'None'
                _line += (f" chain_count={_cc.chain_count} use_stretch={_cc.use_stretch}"
                          f" pole={_pname} pole_angle={math.degrees(_cc.pole_angle):.1f}deg")
            print(_line)

    # ForeArm Twist Redistribution v2: Controller Bone 기반 Copy Rotation
    if USE_FOREARM_TWIST_REDISTRIBUTION:
        # Controller Bone 초기 회전 = 0
        for _ctbn2 in ['SL_LeftForearmTwistCtrlBone', 'SL_RightForearmTwistCtrlBone']:
            _ct_pb2 = arm_obj.pose.bones.get(_ctbn2)
            if _ct_pb2:
                _ct_pb2.rotation_mode = 'XYZ'
                _ct_pb2.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0), 'XYZ')

        # 선택된 쪽에만 v2 constraint 추가 (IK 다음 순서)
        for _ft_side, _ft_bn, _ft_ctbn, _ft_cname in [
                ('L', BONE_L_FOREARM, 'SL_LeftForearmTwistCtrlBone',  'SL_ForearmTwistV2_L'),
                ('R', BONE_R_FOREARM, 'SL_RightForearmTwistCtrlBone', 'SL_ForearmTwistV2_R')]:
            if _ft_side not in _forearm_twist_target_sides:
                continue
            _ft_pb = arm_obj.pose.bones.get(_ft_bn)
            if not _ft_pb:
                print(f"[FOREARM-TWIST-V2] 경고: 뼈 없음 {_ft_bn}"); continue
            _ft_con = _ft_pb.constraints.new('COPY_ROTATION')
            _ft_con.name       = _ft_cname
            _ft_con.target     = arm_obj
            _ft_con.subtarget  = _ft_ctbn
            _ft_con.use_x = False; _ft_con.use_y = True; _ft_con.use_z = False
            _ft_con.target_space = 'LOCAL'
            _ft_con.owner_space  = 'LOCAL'
            _ft_con.mix_mode     = 'AFTER'
            _ft_con.influence    = 1.0

    # 손뼈 + 손가락 뼈: 기존 컨스트레인트 전부 제거, FK 전환
    hand_and_finger_bones = (
        [BONE_L_HAND, BONE_R_HAND] +
        [bn for chain in FINGER_CHAINS_L for (bn,_,_) in chain] +
        [bn for chain in FINGER_CHAINS_R for (bn,_,_) in chain]
    )
    for bn in hand_and_finger_bones:
        b = arm_obj.pose.bones.get(bn)
        if b:
            for c in list(b.constraints): b.constraints.remove(c)
            b.rotation_mode = 'QUATERNION'

    bpy.ops.object.mode_set(mode='OBJECT')

    # 아마추어 액션
    arm_obj.animation_data_create()
    action_name = "SL_" + recording.get('MotionId','anim')
    arm_obj.animation_data.action = bpy.data.actions.new(name=action_name)

    # ====================================================================
    # TEST_MODE: 정적 단일 프레임 — 깊이 복구 확인 + 검지 FK 진단
    # ====================================================================
    if TEST_MODE:
        # 유효 LEFT hand 검출 프레임 선택
        l_cands = [i for i, v in enumerate(raw['l_valid']) if v]
        if not l_cands:
            print("[테스트] LEFT hand 검출 프레임 없음 — 종료")
            bpy.ops.object.mode_set(mode='OBJECT')
            return

        raw_idx = l_cands[min(TEST_FRAME_IDX, len(l_cands)-1)]
        s_idx   = raw_idx + INTRO_FRAMES   # smooth 배열 인덱스
        tframe  = s_idx + 1                # Blender 프레임 번호 (1-indexed)

        l_pts = [smooth[f'l_p{j}'][s_idx] for j in range(21)]
        l_pn  = smooth['l_pn'][s_idx]
        lw    = smooth['lw'][s_idx]
        lh9   = smooth['lh9'][s_idx]

        print(f"\n{'='*62}")
        print(f"[1단계] Hand landmark 깊이 복구 확인")
        print(f"  raw_idx={raw_idx}  smooth_idx={s_idx}  frame={tframe}")
        print(f"  HAND_DEPTH_GAIN={HAND_DEPTH_GAIN}")
        print(f"\n  landmark 좌표 예시 (pt0=손목, pt5=Index_MCP, pt8=Index_Tip):")
        for idx in [0, 5, 6, 7, 8, 17]:
            p = l_pts[idx]
            print(f"    pt{idx:2d}: x={p.x:+.4f}  y={p.y:+.4f}  z={p.z:+.4f}")
        y_vals = [l_pts[i].y for i in [0, 5, 6, 7, 8]]
        y_range = max(y_vals) - min(y_vals)
        print(f"\n  검지(pt0,5,6,7,8) Y 범위: {y_range:.5f}")
        if y_range < 1e-4:
            print("  ⚠ Y 범위 거의 0 → 깊이 여전히 flat (HAND_DEPTH_GAIN 확인)")
        else:
            print("  ✓ Y 범위 존재 → 깊이 복구 성공")

        # IK empty 배치 + arm IK 평가
        emp_lw.location = smooth['lw_ik'][s_idx]
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='POSE')
        bpy.context.scene.frame_set(tframe)
        bpy.context.view_layer.update()

        # 랜드마크 메쉬 시각화
        make_hand_skeleton(l_pts, 'HandSkeleton_L')

        # 손바닥 법선 + effective_pn (flip 적용)
        eff_pn_l = None
        if l_pn and l_pn.length > 1e-6:
            pn_n = l_pn.normalized()
            eff_pn_l = -pn_n if L_PALM_FLIP else pn_n
            print(f"\n  손바닥 법선 raw:        ({pn_n.x:+.3f}, {pn_n.y:+.3f}, {pn_n.z:+.3f})")
            print(f"  effective_pn (flip={L_PALM_FLIP}): ({eff_pn_l.x:+.3f}, {eff_pn_l.y:+.3f}, {eff_pn_l.z:+.3f})")

        # 손뼈 회전: Hand pt0(손목) → lh9(mid9), effective_pn 사용, flip_palm=False(이미 적용)
        if eff_pn_l is not None:
            set_hand_rotation(arm_obj, BONE_L_HAND, l_pts[H_WRIST], lh9, eff_pn_l, False)
        arm_obj.pose.bones[BONE_L_HAND].keyframe_insert('rotation_quaternion', frame=tframe)

        print(f"\n{'='*62}")
        print(f"[2단계] 검지 FK 진단  (TEST_INDEX_ONLY={TEST_INDEX_ONLY})")
        # FINGER_CHAINS_L[1] = Index 체인 [Index1, Index2, Index3]
        # roll_pn=eff_pn_l → hand bone과 동일한 effective_pn 사용
        index_chain_l = FINGER_CHAINS_L[1]
        diag_index_finger(arm_obj, l_pts, BONE_L_HAND, index_chain_l, roll_pn=eff_pn_l)

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end   = tframe
        bpy.context.scene.frame_set(tframe)
        print(f"\n{'='*62}")
        print(f"[테스트 완료]  frame={tframe}")
        print("  확인 사항:")
        print("  1) 'HandSkeleton_L' mesh — 정면·측면·3/4뷰로 손 형태 확인")
        print("     측면에서 깊이 분포가 납작하지 않아야 함")
        print("  2) [C] 각도 오차가 작으면 (< 5도) 검지 방향 정확")
        print("     오차가 크면 rest axis 방향 또는 parent_mat 불일치 의심")
        print("  3) 각도 오차가 정상이면 roll 보정 단계로 진행 가능")
        return

    # ====================================================================
    # PASS 1: 팔 IK Empty 키프레임 (전체 애니메이션)
    # ====================================================================
    print("[정보] Pass 1: 팔 IK 키프레임...")
    animate_empty(emp_lw, smooth['lw_ik'], valid_frames)
    animate_empty(emp_rw, smooth['rw_ik'], valid_frames)
    if USE_ELBOW_POLE:
        animate_empty(emp_pole_l, _pole_track_l, valid_frames)
        animate_empty(emp_pole_r, _pole_track_r, valid_frames)

    # [IK-EVAL-TS] / [IK-BONE-ROT-TS] _DIAG_TS 프레임 실제 평가
    if _src_ts:
        _diag_bi = min(range(len(_src_ts)), key=lambda k: abs(_src_ts[k] - _DIAG_TS))
        _diag_bf = INTRO_FRAMES + _diag_bi + 1
        _diag_ts_actual = _src_ts[_diag_bi]
        bpy.context.scene.frame_set(_diag_bf)
        bpy.context.view_layer.update()
        _mw = arm_obj.matrix_world
        for _es, _abn2, _fbn2, _emp2, _sh_w2, _arm_tot2 in [
                ('L', BONE_L_ARM, BONE_L_FOREARM, emp_lw, l_sh_w, char_l_arm_total),
                ('R', BONE_R_ARM, BONE_R_FOREARM, emp_rw, r_sh_w, char_r_arm_total)]:
            _pba2 = arm_obj.pose.bones.get(_abn2)
            _pbf2 = arm_obj.pose.bones.get(_fbn2)
            if not _pba2 or not _pbf2:
                print(f"[IK-EVAL-TS] {_es}: 뼈 없음"); continue
            _sh2  = _mw @ _pba2.head   # shoulder world
            _el2  = _mw @ _pba2.tail   # elbow world (= forearm head)
            _wr2  = _mw @ _pbf2.tail   # wrist world (= forearm tail)
            _tgt2 = _emp2.matrix_world.translation.copy()
            # reach ratios
            _tgt_r = (_tgt2 - _sh2).length / _arm_tot2 if _arm_tot2 > 1e-6 else 0.0
            _act_r = (_wr2  - _sh2).length / _arm_tot2 if _arm_tot2 > 1e-6 else 0.0
            _err   = (_wr2 - _tgt2).length / char_sw
            # elbow angle
            _v1 = (_sh2 - _el2); _v2 = (_wr2 - _el2)
            _l1 = _v1.length; _l2 = _v2.length
            _el_deg = math.degrees(math.acos(max(-1.0, min(1.0,
                (_v1/_l1).dot(_v2/_l2))))) if (_l1 > 1e-6 and _l2 > 1e-6) else 0.0
            print(f"[IK-EVAL-TS] {_es} ts={_diag_ts_actual} frame={_diag_bf}:")
            print(f"  target_reach={_tgt_r:.4f}  actual_reach={_act_r:.4f}"
                  f"  wrist_err/sw={_err:.4f}  elbow_angle={_el_deg:.1f}deg")
            print(f"  shoulder_w=({_sh2.x:.4f},{_sh2.y:.4f},{_sh2.z:.4f})"
                  f"  elbow_w=({_el2.x:.4f},{_el2.y:.4f},{_el2.z:.4f})"
                  f"  wrist_w=({_wr2.x:.4f},{_wr2.y:.4f},{_wr2.z:.4f})"
                  f"  target_w=({_tgt2.x:.4f},{_tgt2.y:.4f},{_tgt2.z:.4f})")
            # bone rotation quaternion
            _qa = tuple(_pba2.rotation_quaternion)
            _qf = tuple(_pbf2.rotation_quaternion)
            print(f"[IK-BONE-ROT-TS] {_es}: upper_arm q=({_qa[0]:.4f},{_qa[1]:.4f},{_qa[2]:.4f},{_qa[3]:.4f})"
                  f"  forearm q=({_qf[0]:.4f},{_qf[1]:.4f},{_qf[2]:.4f},{_qf[3]:.4f})")
            # [UPPER-ARM-EVAL-TS]
            _gusd = _guide_diag_l if _es == 'L' else _guide_diag_r
            if _gusd and _gusd.get('feas_el') is not None:
                _feas_el_ev = _gusd['feas_el']
                _feas_sh_v  = _feas_el_ev - _sh2
                _act_sh_v   = _el2 - _sh2
                _feas_upper_dir = _feas_sh_v.normalized() if _feas_sh_v.length > 1e-6 else mathutils.Vector((0,0,1))
                _act_upper_dir  = _act_sh_v.normalized()  if _act_sh_v.length  > 1e-6 else mathutils.Vector((0,0,1))
                _eval_dot = max(-1.0, min(1.0, _feas_upper_dir.dot(_act_upper_dir)))
                try: _eval_ang = math.degrees(math.acos(_eval_dot))
                except: _eval_ang = 0.0
                _feas_err = (_el2 - _feas_el_ev).length / char_sw
                _inward_x = (_el2.x - _sh2.x) / char_sw
                print(f"[UPPER-ARM-EVAL-TS] {_es}: ts={_diag_ts_actual}"
                      f"  feasible_desired_upper_dir=({_feas_upper_dir.x:+.3f},{_feas_upper_dir.y:+.3f},{_feas_upper_dir.z:+.3f})"
                      f"  actual_upper_dir=({_act_upper_dir.x:+.3f},{_act_upper_dir.y:+.3f},{_act_upper_dir.z:+.3f})"
                      f"  direction_dot={_eval_dot:+.4f}"
                      f"  direction_angle_error_deg={_eval_ang:.2f}"
                      f"  feasible_desired_elbow=({_feas_el_ev.x:.4f},{_feas_el_ev.y:.4f},{_feas_el_ev.z:.4f})"
                      f"  actual_elbow=({_el2.x:.4f},{_el2.y:.4f},{_el2.z:.4f})"
                      f"  desired_elbow_error/sw={_feas_err:.4f}"
                      f"  inward_x_from_shoulder/sw={_inward_x:+.4f}")
            else:
                print(f"[UPPER-ARM-EVAL-TS] {_es}: ts={_diag_ts_actual}  feas_el=None (no guide at this frame)")

    # ── Elbow Bend Diagnostics (애니메이션 변경 없음) ───────────────────
    if ELBOW_BEND_DIAGNOSTICS and _src_ts:
        _mw_bd = arm_obj.matrix_world

        # [HUMANOID-RIG-DIAG] 리그 정보
        for _bd_s, _bd_abn, _bd_fbn, _bd_hbn, _bd_sh_w, _bd_at, _bd_ikc in [
                ('L', BONE_L_ARM, BONE_L_FOREARM, BONE_L_HAND,
                 l_sh_w, char_l_arm_total, _ik_con_l),
                ('R', BONE_R_ARM, BONE_R_FOREARM, BONE_R_HAND,
                 r_sh_w, char_r_arm_total, _ik_con_r)]:
            _bd_dba = arm_obj.data.bones.get(_bd_abn)
            _bd_dbf = arm_obj.data.bones.get(_bd_fbn)
            _bd_la = (_mw_bd @ _bd_dba.tail_local - _mw_bd @ _bd_dba.head_local).length if _bd_dba else 0.0
            _bd_lf = (_mw_bd @ _bd_dbf.tail_local - _mw_bd @ _bd_dbf.head_local).length if _bd_dbf else 0.0
            _bd_lt = _bd_la + _bd_lf
            _bd_ur = _bd_la / _bd_lf if _bd_lf > 1e-6 else 0.0
            _bd_ad = (_mw_bd @ _bd_dba.tail_local - _mw_bd @ _bd_dba.head_local).normalized() if _bd_dba else None
            _bd_fd = (_mw_bd @ _bd_dbf.tail_local - _mw_bd @ _bd_dbf.head_local).normalized() if _bd_dbf else None
            _bd_pa = math.degrees(_bd_ikc.pole_angle) if _bd_ikc else 0.0
            print(f"[HUMANOID-RIG-DIAG] {_bd_s}:"
                  f"  upper={_bd_abn}  forearm={_bd_fbn}  hand={_bd_hbn}")
            print(f"  upper_len={_bd_la:.4f}  forearm_len={_bd_lf:.4f}"
                  f"  total={_bd_lt:.4f}  u/f_ratio={_bd_ur:.3f}"
                  f"  pole_angle={_bd_pa:.1f}deg")
            if _bd_ad: print(f"  upper_rest_dir=({_bd_ad.x:+.4f},{_bd_ad.y:+.4f},{_bd_ad.z:+.4f})")
            if _bd_fd: print(f"  forearm_rest_dir=({_bd_fd.x:+.4f},{_bd_fd.y:+.4f},{_bd_fd.z:+.4f})")

        # 프레임별 굽힘 방향 평가
        _bd_all = {'L': [], 'R': []}  # (ts, bf, dot, el_deg, src_mag, act_mag, src_bv, act_bv, tgt_r, reliable)
        for _bi, _bts in enumerate(_src_ts):
            _bf2 = INTRO_FRAMES + _bi + 1
            bpy.context.scene.frame_set(_bf2)
            bpy.context.view_layer.update()
            _mw3 = arm_obj.matrix_world
            for _bd_s, _bd_abn, _bd_fbn, _bd_src_list, _bd_sh_w, _bd_at, _bd_ik_key in [
                    ('L', BONE_L_ARM, BONE_L_FOREARM, _pole_src_l,
                     l_sh_w, char_l_arm_total, 'lw_ik'),
                    ('R', BONE_R_ARM, BONE_R_FOREARM, _pole_src_r,
                     r_sh_w, char_r_arm_total, 'rw_ik')]:
                _bd_pba = arm_obj.pose.bones.get(_bd_abn)
                _bd_pbf = arm_obj.pose.bones.get(_bd_fbn)
                if not (_bd_pba and _bd_pbf): continue
                _bd_sh = _mw3 @ _bd_pba.head
                _bd_el = _mw3 @ _bd_pba.tail
                _bd_wr = _mw3 @ _bd_pbf.tail
                # elbow angle
                _bd_v1 = _bd_sh - _bd_el; _bd_v2 = _bd_wr - _bd_el
                _bd_l1 = _bd_v1.length; _bd_l2 = _bd_v2.length
                _bd_el_deg = (math.degrees(math.acos(max(-1.0, min(1.0,
                    (_bd_v1/_bd_l1).dot(_bd_v2/_bd_l2)))))
                    if (_bd_l1>1e-6 and _bd_l2>1e-6) else 180.0)
                # target reach ratio
                _bd_ik_v = smooth[_bd_ik_key][INTRO_FRAMES+_bi] if (INTRO_FRAMES+_bi)<len(smooth[_bd_ik_key]) else None
                _bd_tgt_r = (_bd_ik_v-_bd_sh_w).length/_bd_at if (_bd_ik_v and _bd_at>1e-6) else 0.0
                # actual bend (perpendicular to shoulder→wrist axis)
                _bd_ax_v = _bd_wr - _bd_sh; _bd_ax_l = _bd_ax_v.length
                if _bd_ax_l < 1e-6: continue
                _bd_ax_n = _bd_ax_v / _bd_ax_l
                _bd_el_rel = _bd_el - _bd_sh
                _bd_act_bv = _bd_el_rel - _bd_ax_n * _bd_el_rel.dot(_bd_ax_n)
                _bd_act_mag = _bd_act_bv.length
                # source bend
                _bd_src_bv = _bd_src_list[_bi] if _bi < len(_bd_src_list) else None
                _bd_src_mag = _bd_src_bv.length if _bd_src_bv else 0.0
                # reliability
                _bd_rel = (_bd_src_mag >= ELBOW_BEND_MIN_MAGNITUDE and
                           _bd_act_mag >= ELBOW_BEND_MIN_MAGNITUDE and
                           _bd_el_deg < ELBOW_BEND_STRAIGHT_ANGLE_DEG)
                if _bd_rel:
                    _bd_dot = (_bd_src_bv/_bd_src_mag).dot(_bd_act_bv/_bd_act_mag)
                else:
                    _bd_dot = None
                _bd_all[_bd_s].append((_bts, _bf2, _bd_dot, _bd_el_deg,
                                        _bd_src_mag, _bd_act_mag,
                                        _bd_src_bv, _bd_act_bv, _bd_tgt_r, _bd_rel))

        # 통계 출력
        for _bd_s in ['L', 'R']:
            _bd_res  = _bd_all[_bd_s]
            _bd_reli = [r for r in _bd_res if r[9]]
            _bd_unrl = len(_src_ts) - len(_bd_reli)
            _bd_pos  = [r for r in _bd_reli if r[2] >  0.10]
            _bd_neg  = [r for r in _bd_reli if r[2] <  ELBOW_BEND_NEGATIVE_DOT_THRESHOLD]
            _bd_nz   = [r for r in _bd_reli if ELBOW_BEND_NEGATIVE_DOT_THRESHOLD <= r[2] <= 0.10]
            _bd_dots = [r[2] for r in _bd_reli]
            print(f"[ELBOW-BEND-DIAG] {_bd_s}:"
                  f"  total_source_frames={len(_src_ts)}"
                  f"  reliable_frames={len(_bd_reli)}"
                  f"  positive_count={len(_bd_pos)}"
                  f"  negative_count={len(_bd_neg)}"
                  f"  near_zero_count={len(_bd_nz)}"
                  f"  straight_or_unreliable_count={_bd_unrl}")
            if _bd_dots:
                print(f"  avg_dot={sum(_bd_dots)/len(_bd_dots):+.4f}"
                      f"  min_dot={min(_bd_dots):+.4f}"
                      f"  max_dot={max(_bd_dots):+.4f}")
            # 음수 프레임 (최대 15개, dot 오름차순)
            if _bd_neg:
                print(f"[ELBOW-BEND-NEGATIVE] {_bd_s}: {len(_bd_neg)}프레임")
                for _r in sorted(_bd_neg, key=lambda r: r[2])[:15]:
                    _rts,_rbf,_rdot,_rdeg,_rsm,_ram,_rsbv,_rabv,_rtr,_ = _r
                    _sv = f"({_rsbv.x:+.3f},{_rsbv.y:+.3f},{_rsbv.z:+.3f})" if _rsbv else "None"
                    _av = f"({_rabv.x:+.3f},{_rabv.y:+.3f},{_rabv.z:+.3f})"
                    print(f"  ts={_rts}  frame={_rbf}  dot={_rdot:+.4f}"
                          f"  elbow_angle={_rdeg:.1f}deg"
                          f"  src_mag={_rsm:.4f}  act_mag={_ram:.4f}"
                          f"  src_bend={_sv}  act_bend={_av}"
                          f"  tgt_reach={_rtr:.4f}")
            # 대표 프레임 (src_mag 최대, reliable)
            _bd_best = max(_bd_reli, key=lambda r: r[4]) if _bd_reli else None
            if _bd_best:
                _rts,_rbf,_rdot,_rdeg,_rsm,_ram,_rsbv,_rabv,_rtr,_ = _bd_best
                _sv = f"({_rsbv.x:+.3f},{_rsbv.y:+.3f},{_rsbv.z:+.3f})" if _rsbv else "None"
                _av = f"({_rabv.x:+.3f},{_rabv.y:+.3f},{_rabv.z:+.3f})"
                print(f"[ELBOW-BEND-BEST] {_bd_s}: ts={_rts}  frame={_rbf}"
                      f"  dot={_rdot:+.4f}  elbow_angle={_rdeg:.1f}deg"
                      f"  src_mag={_rsm:.4f}  act_mag={_ram:.4f}"
                      f"  src_bend={_sv}  act_bend={_av}"
                      f"  tgt_reach={_rtr:.4f}")

    # ====================================================================
    # FOREARM TWIST V2: Calibration → Pre-pass → correction track
    # ====================================================================
    _ft_correction_l    = [0.0] * total
    _ft_correction_r    = [0.0] * total
    _ft_v2_enabled      = False
    _ft_v2_enabled_l    = False
    _ft_v2_enabled_r    = False
    _ft_response_gain_l = 1.0
    _ft_response_gain_r = 1.0
    _ft_pre_l           = [0.0] * total   # hand twist per frame (controller=0), for diagnostics
    _ft_pre_r           = [0.0] * total

    if USE_FOREARM_TWIST_REDISTRIBUTION and _src_ts:
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='POSE')

        # ── Helper: read hand twist (local-Y) after set_hand_rotation ──
        def _ft_read_twist(bone_name, valid_first, frame_i, pts, h9, pn, flip):
            if valid_first is not None and frame_i < valid_first:
                return 0.0
            eff = ((-pn if flip else pn) if (pn and pn.length > 1e-4) else None)
            if eff is not None:
                set_hand_rotation(arm_obj, bone_name, pts[H_WRIST], h9, eff, False)
            hb = arm_obj.pose.bones.get(bone_name)
            if not hb: return 0.0
            hq = hb.rotation_quaternion.copy()
            if hq.w < 0: hq.negate()
            _, _, tdeg, _, _ = decompose_swing_twist(hq, mathutils.Vector((0, 1, 0)))
            return tdeg

        # ── Helper: wrist position error (IK empty vs actual ForeArm tail) ──
        def _ft_wrist_err(forearm_bn, emp_ik):
            fb = arm_obj.pose.bones.get(forearm_bn)
            if not fb: return 0.0
            act_wr = (arm_obj.matrix_world @ fb.tail)
            tgt_wr = emp_ik.matrix_world.translation
            return (act_wr - tgt_wr).length / (char_sw if char_sw > 1e-6 else 1.0)

        # ── Helper: hand forward + palm alignment errors ──
        def _ft_hand_errors(bone_name, pts, h9, pn, flip):
            hb = arm_obj.pose.bones.get(bone_name)
            if not hb: return 0.0, 0.0
            mw3 = arm_obj.matrix_world
            act_fwd = (mw3 @ hb.tail - mw3 @ hb.head)
            if act_fwd.length < 1e-6: return 0.0, 0.0
            act_fwd = act_fwd.normalized()
            src_fwd = (h9 - pts[H_WRIST])
            if src_fwd.length < 1e-6: return 0.0, 0.0
            src_fwd = src_fwd.normalized()
            try:   fwd_err = math.degrees(math.acos(max(-1.0, min(1.0, act_fwd.dot(src_fwd)))))
            except: fwd_err = 0.0
            palm_err = 0.0
            eff_pn = ((-pn if flip else pn) if (pn and pn.length > 1e-4) else None)
            if eff_pn is not None:
                az = hb.matrix.col[2].to_3d().normalized()
                act_hz = (mw3.to_3x3() @ az).normalized()
                pnp = eff_pn - eff_pn.dot(act_fwd) * act_fwd
                hzp = act_hz - act_hz.dot(act_fwd) * act_fwd
                if pnp.length > 1e-4 and hzp.length > 1e-4:
                    try: palm_err = math.degrees(math.acos(max(-1.0, min(1.0, pnp.normalized().dot(hzp.normalized())))))
                    except: palm_err = 0.0
            return fwd_err, palm_err

        # ── Per-side calibration ────────────────────────────────────
        # 고정 timestamp 대신 각 손이 실제 검출된 중간 프레임을 사용한다.

        def _unwrap_twist(baseline, test):
            delta = test - baseline
            while delta >  180.0: delta -= 360.0
            while delta < -180.0: delta += 360.0
            return baseline + delta

        for _cal_side, _cal_fbn, _cal_hbn, _cal_ctbn, _cal_flip, _cal_vf, _cal_emp_ik in [
                ('L', BONE_L_FOREARM, BONE_L_HAND, 'SL_LeftForearmTwistCtrlBone',
                 L_PALM_FLIP, l_first, emp_lw),
                ('R', BONE_R_FOREARM, BONE_R_HAND, 'SL_RightForearmTwistCtrlBone',
                 R_PALM_FLIP, r_first, emp_rw)]:
            if _cal_side not in _forearm_twist_target_sides:
                print(f"[FOREARM-TWIST-CALIBRATION] {_cal_side}: skipped — not selected")
                continue
            _cal_prefix = 'l' if _cal_side == 'L' else 'r'
            _cal_candidates = [
                _ci for _ci, _cv in enumerate(raw[f'{_cal_prefix}_valid'])
                if _cv and _ci < len(raw[f'{_cal_prefix}_pn'])
                and raw[f'{_cal_prefix}_pn'][_ci] is not None
            ]
            if len(_cal_candidates) < FOREARM_TWIST_MIN_VALID_FRAMES:
                print(f"[FOREARM-TWIST-CALIBRATION] {_cal_side}:"
                      f"  selected_enabled=False"
                      f"  failure_reason=insufficient_valid_frames"
                      f"  valid_frames={len(_cal_candidates)}"
                      f"  min_required={FOREARM_TWIST_MIN_VALID_FRAMES}")
                if _cal_side == 'L': _ft_v2_enabled_l = False
                else:                _ft_v2_enabled_r = False
                continue
            _cal_bi = _cal_candidates[len(_cal_candidates) // 2]
            _cal_bf = INTRO_FRAMES + _cal_bi + 1
            _cal_ts = _src_ts[_cal_bi]
            _cal_si = INTRO_FRAMES + _cal_bi
            _cal_fb  = arm_obj.pose.bones.get(_cal_fbn)
            _cal_ctb = arm_obj.pose.bones.get(_cal_ctbn)
            if not _cal_fb or not _cal_ctb:
                print(f"[FOREARM-TWIST-CALIBRATION] {_cal_side}: bone missing — disabled")
                if _cal_side == 'L': _ft_v2_enabled_l = False
                else:                _ft_v2_enabled_r = False
                continue
            _cal_pts = [smooth[f'{"l" if _cal_side=="L" else "r"}_p{j}'][_cal_si] for j in range(21)]
            _cal_h9  = smooth['lh9' if _cal_side == 'L' else 'rh9'][_cal_si]
            _cal_pnv = smooth['l_pn' if _cal_side == 'L' else 'r_pn'][_cal_si]

            # Step 1: baseline (controller = 0°) — frame_set FIRST
            bpy.context.scene.frame_set(_cal_bf)
            _cal_ctb.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0), 'XYZ')
            bpy.context.view_layer.update()
            _cal_fa_mat0  = _cal_fb.matrix.copy()
            _cal_twist0   = _ft_read_twist(_cal_hbn, _cal_vf, _cal_si, _cal_pts, _cal_h9, _cal_pnv, _cal_flip)
            # set_hand_rotation()은 pose channel을 즉시 바꾸지만 evaluated
            # head/tail/matrix는 depsgraph 갱신 전 값일 수 있다. 갱신 없이
            # 오차를 읽으면 정상적인 전완 roll 응답도 hand_forward_error로
            # 오판되어 constraint가 제거된다.
            bpy.context.view_layer.update()
            _cal_fwd0, _cal_palm0 = _ft_hand_errors(_cal_hbn, _cal_pts, _cal_h9, _cal_pnv, _cal_flip)
            _cal_wrist0   = _ft_wrist_err(_cal_fbn, _cal_emp_ik)

            # Step 2: controller = +30° — frame_set FIRST, then override
            bpy.context.scene.frame_set(_cal_bf)
            _cal_ctb.rotation_euler = mathutils.Euler((0.0, math.radians(30.0), 0.0), 'XYZ')
            bpy.context.view_layer.update()
            _cal_fa_mat_p30 = _cal_fb.matrix.copy()
            _cal_twist_p30  = _ft_read_twist(_cal_hbn, _cal_vf, _cal_si, _cal_pts, _cal_h9, _cal_pnv, _cal_flip)
            bpy.context.view_layer.update()
            _cal_fwd_p30, _cal_palm_p30 = _ft_hand_errors(_cal_hbn, _cal_pts, _cal_h9, _cal_pnv, _cal_flip)
            _cal_wrist_p30  = _ft_wrist_err(_cal_fbn, _cal_emp_ik)

            # Step 3: controller = -30° — frame_set FIRST, then override
            bpy.context.scene.frame_set(_cal_bf)
            _cal_ctb.rotation_euler = mathutils.Euler((0.0, math.radians(-30.0), 0.0), 'XYZ')
            bpy.context.view_layer.update()
            _cal_twist_m30  = _ft_read_twist(_cal_hbn, _cal_vf, _cal_si, _cal_pts, _cal_h9, _cal_pnv, _cal_flip)
            bpy.context.view_layer.update()
            _cal_fwd_m30, _cal_palm_m30 = _ft_hand_errors(_cal_hbn, _cal_pts, _cal_h9, _cal_pnv, _cal_flip)
            _cal_wrist_m30  = _ft_wrist_err(_cal_fbn, _cal_emp_ik)

            # Reset controller to 0
            bpy.context.scene.frame_set(_cal_bf)
            _cal_ctb.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0), 'XYZ')
            bpy.context.view_layer.update()

            # Unwrap ±30° hand twist around baseline (avoid ±180° boundary artifacts)
            _cal_uw_p30 = _unwrap_twist(_cal_twist0, _cal_twist_p30)
            _cal_uw_m30 = _unwrap_twist(_cal_twist0, _cal_twist_m30)

            # Forearm twist delta for +30° test (local Y — relative quat is in local space)
            _cal_dq_p30 = (_cal_fa_mat0.to_3x3().to_quaternion().inverted()
                           @ _cal_fa_mat_p30.to_3x3().to_quaternion())
            _cal_dq_p30.normalize()
            _, _, _cal_fa_delta_p30, _, _ = decompose_swing_twist(
                _cal_dq_p30, mathutils.Vector((0.0, 1.0, 0.0)))

            # response_gain: hand-twist degrees per controller degree (signed)
            _cal_rg = (_cal_uw_p30 - _cal_uw_m30) / 60.0

            # Pass/fail
            _cal_fa_ok    = abs(_cal_fa_delta_p30) >= 20.0
            _cal_rg_ok    = abs(_cal_rg) >= 0.1
            _cal_wrist_ok = _cal_wrist_p30 <= 0.001 and _cal_wrist_m30 <= 0.001
            _cal_fwd_ok   = _cal_fwd_p30 <= 1.0 and _cal_fwd_m30 <= 1.0
            _cal_palm_ok  = _cal_palm_p30 <= 1.0 and _cal_palm_m30 <= 1.0
            _cal_pass     = _cal_fa_ok and _cal_rg_ok and _cal_wrist_ok and _cal_fwd_ok and _cal_palm_ok
            _cal_reason   = ("ok" if _cal_pass else
                             "forearm_no_response" if not _cal_fa_ok else
                             "response_gain_too_small" if not _cal_rg_ok else
                             "wrist_error" if not _cal_wrist_ok else
                             "hand_forward_error" if not _cal_fwd_ok else
                             "palm_error")

            print(f"[FOREARM-TWIST-CALIBRATION] {_cal_side}:"
                  f"  ts={_cal_ts}"
                  f"  baseline_hand_twist_deg={_cal_twist0:.1f}"
                  f"  plus30_hand_twist_deg={_cal_twist_p30:.1f}"
                  f"  minus30_hand_twist_deg={_cal_twist_m30:.1f}"
                  f"  unwrapped_plus30_deg={_cal_uw_p30:.1f}"
                  f"  unwrapped_minus30_deg={_cal_uw_m30:.1f}"
                  f"  plus30_forearm_delta_deg={_cal_fa_delta_p30:.1f}"
                  f"  minus30_forearm_delta_deg=N/A"
                  f"  response_gain={_cal_rg:.4f}"
                  f"  plus30_forward_error_deg={_cal_fwd_p30:.3f}"
                  f"  minus30_forward_error_deg={_cal_fwd_m30:.3f}"
                  f"  plus30_palm_error_deg={_cal_palm_p30:.3f}"
                  f"  minus30_palm_error_deg={_cal_palm_m30:.3f}"
                  f"  plus30_wrist_error_sw={_cal_wrist_p30:.6f}"
                  f"  minus30_wrist_error_sw={_cal_wrist_m30:.6f}"
                  f"  selected_enabled={_cal_pass}"
                  f"  failure_reason={_cal_reason}")

            if _cal_side == 'L':
                _ft_v2_enabled_l    = _cal_pass
                _ft_response_gain_l = _cal_rg if _cal_pass else 1.0
            else:
                _ft_v2_enabled_r    = _cal_pass
                _ft_response_gain_r = _cal_rg if _cal_pass else 1.0

        # Calibration 실패 쪽은 constraint를 즉시 제거한다.
        # 보정값 0인 controller를 남기는 것보다 평가 경로 자체를 분리해야
        # IK 결과와 손 FK가 실패한 실험 설정의 영향을 받지 않는다.
        for _fail_side, _fail_fbn, _fail_cname, _fail_enabled in [
                ('L', BONE_L_FOREARM, 'SL_ForearmTwistV2_L', _ft_v2_enabled_l),
                ('R', BONE_R_FOREARM, 'SL_ForearmTwistV2_R', _ft_v2_enabled_r)]:
            if _fail_enabled:
                continue
            _fail_pb = arm_obj.pose.bones.get(_fail_fbn)
            if _fail_pb:
                for _fail_con in list(_fail_pb.constraints):
                    if _fail_con.name == _fail_cname:
                        _fail_pb.constraints.remove(_fail_con)
            print(f"[FOREARM-TWIST-CLEANUP] {_fail_side}: failed/unused constraint removed")
        bpy.context.view_layer.update()

        _ft_v2_enabled = _ft_v2_enabled_l or _ft_v2_enabled_r
        if not _ft_v2_enabled:
            print("[FOREARM-TWIST-V2] disabled — both sides failed calibration")
            bpy.ops.object.mode_set(mode='OBJECT')
        else:
            if not _ft_v2_enabled_l: print("[FOREARM-TWIST-V2] L: disabled — calibration failed")
            if not _ft_v2_enabled_r: print("[FOREARM-TWIST-V2] R: disabled — calibration failed")

            # ── Pre-pass: measure hand twist per frame (controller=0) ──
            _ft_pq_l = {}; _ft_pq_r = {}
            for _ft_i, _ft_frame in enumerate(valid_frames):
                bpy.context.scene.frame_set(_ft_frame)
                bpy.context.view_layer.update()
                for _ft_bn2, _ft_vf2, _ft_flip2, _ft_pq2, _ft_list2 in [
                        (BONE_L_HAND, l_first, L_PALM_FLIP, _ft_pq_l, _ft_pre_l),
                        (BONE_R_HAND, r_first, R_PALM_FLIP, _ft_pq_r, _ft_pre_r)]:
                    if _ft_vf2 is not None and _ft_i < _ft_vf2: continue
                    _ft_pts2 = [smooth[f'{"l" if _ft_bn2==BONE_L_HAND else "r"}_p{j}'][_ft_i] for j in range(21)]
                    _ft_h9_2 = smooth['lh9' if _ft_bn2 == BONE_L_HAND else 'rh9'][_ft_i]
                    _ft_pn2  = smooth['l_pn' if _ft_bn2 == BONE_L_HAND else 'r_pn'][_ft_i]
                    _ft_eff2 = ((-_ft_pn2 if _ft_flip2 else _ft_pn2)
                                if (_ft_pn2 and _ft_pn2.length > 1e-4) else None)
                    if _ft_eff2 is not None:
                        set_hand_rotation(arm_obj, _ft_bn2, _ft_pts2[H_WRIST], _ft_h9_2, _ft_eff2, False)
                    _ft_hb2 = arm_obj.pose.bones.get(_ft_bn2)
                    if not _ft_hb2: continue
                    _ft_pqv2 = _ft_pq2.get(_ft_bn2)
                    if _ft_pqv2 and _ft_pqv2.dot(_ft_hb2.rotation_quaternion) < 0:
                        _ft_hb2.rotation_quaternion.negate()
                    _ft_pq2[_ft_bn2] = _ft_hb2.rotation_quaternion.copy()
                    _ft_hq2 = _ft_hb2.rotation_quaternion.copy()
                    if _ft_hq2.w < 0: _ft_hq2.negate()
                    _, _, _ft_tdeg2, _, _ = decompose_swing_twist(_ft_hq2, mathutils.Vector((0, 1, 0)))
                    _ft_list2[_ft_i] = _ft_tdeg2

            bpy.ops.object.mode_set(mode='OBJECT')

            # ── Correction track (연속 twist + response_gain 역산) ────
            def _ft_unwrap_track(values):
                """±180° 표현 경계에서 correction 부호가 뒤집히지 않도록 연속화."""
                result = []
                prev = None
                for value in values:
                    current = float(value)
                    if prev is not None:
                        while current - prev > 180.0:
                            current -= 360.0
                        while current - prev < -180.0:
                            current += 360.0
                    result.append(current)
                    prev = current
                return result

            def _ft_build_correction(pre_list, response_gain):
                continuous = _ft_unwrap_track(pre_list)
                raw = []
                for td in continuous:
                    # 목표는 hand local twist를 [-soft_limit, +soft_limit] 안으로
                    # 이동하는 것이다. 따라서 필요한 hand 변화량은 항상
                    # target - current이며, 기존 실험의 부호 반전을 바로잡는다.
                    target = max(-HAND_TWIST_SOFT_LIMIT_DEG,
                                 min(HAND_TWIST_SOFT_LIMIT_DEG, td))
                    desired_hand_delta = target - td
                    if abs(response_gain) > 1e-4:
                        corr = desired_hand_delta / response_gain
                    else:
                        corr = 0.0
                    raw.append(max(-FOREARM_TWIST_MAX_TRANSFER_DEG,
                                   min(FOREARM_TWIST_MAX_TRANSFER_DEG, corr)))
                sm = smooth_scalar_track(raw, window=FOREARM_TWIST_SMOOTH_WINDOW)
                smoothed = [v if v is not None else 0.0 for v in sm] if sm else raw[:]

                # 인트로와 모션 시작부의 목표 correction부터 완만하게 만든 뒤
                # rate limit을 적용해야 실제 최종 track도 step 제한을 지킨다.
                for _fi in range(INTRO_FRAMES):
                    if _fi < len(smoothed):
                        smoothed[_fi] = 0.0
                for _fi in range(5):
                    _idx = INTRO_FRAMES + _fi
                    if _idx < len(smoothed):
                        smoothed[_idx] *= (_fi + 1) / 5

                # 급격한 landmark/branch 변화가 남아 있어도 controller가
                # 한 프레임에 크게 변하지 않도록 slew-rate를 제한한다.
                full = []
                previous = 0.0
                for value in smoothed:
                    delta = max(-FOREARM_TWIST_MAX_STEP_DEG,
                                min(FOREARM_TWIST_MAX_STEP_DEG, value - previous))
                    current = previous + delta
                    full.append(current)
                    previous = current
                return full

            _ft_full_l = (_ft_build_correction(_ft_pre_l, _ft_response_gain_l)
                          if _ft_v2_enabled_l else [0.0] * total)
            _ft_full_r = (_ft_build_correction(_ft_pre_r, _ft_response_gain_r)
                          if _ft_v2_enabled_r else [0.0] * total)
            for _idx2 in range(total):
                _ft_correction_l[_idx2] = _ft_full_l[_idx2] if _idx2 < len(_ft_full_l) else 0.0
                _ft_correction_r[_idx2] = _ft_full_r[_idx2] if _idx2 < len(_ft_full_r) else 0.0

    # ====================================================================
    # [HAND-FOREARM-REST-DIAG] 1회 출력: rest bone 축·constraint 목록
    # ====================================================================
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    print("[HAND-FOREARM-REST-DIAG] ===== REST BONE & CONSTRAINT DIAGNOSTIC =====")
    for _rd_side, _rd_fbn, _rd_hbn in [
            ('L', BONE_L_FOREARM, BONE_L_HAND),
            ('R', BONE_R_FOREARM, BONE_R_HAND)]:
        _rd_fb_data = arm_obj.data.bones.get(_rd_fbn)
        _rd_hb_data = arm_obj.data.bones.get(_rd_hbn)
        _rd_fp  = arm_obj.pose.bones.get(_rd_fbn)
        _rd_hp  = arm_obj.pose.bones.get(_rd_hbn)
        print(f"[HAND-FOREARM-REST-DIAG] {_rd_side} ForeArm parent="
              f"{_rd_fb_data.parent.name if (_rd_fb_data and _rd_fb_data.parent) else 'None'}")
        print(f"[HAND-FOREARM-REST-DIAG] {_rd_side} Hand parent="
              f"{_rd_hb_data.parent.name if (_rd_hb_data and _rd_hb_data.parent) else 'None'}")
        for _rd_lbl, _rd_bd in [('ForeArm', _rd_fb_data), ('Hand', _rd_hb_data)]:
            if _rd_bd:
                _rm = _rd_bd.matrix_local
                def _col3(m, c): v = m.col[c].to_3d(); return f"({v.x:+.6f},{v.y:+.6f},{v.z:+.6f})"
                _ht = (_rd_bd.tail_local - _rd_bd.head_local)
                _ht_n = _ht.normalized() if _ht.length > 1e-6 else _ht
                print(f"[HAND-FOREARM-REST-DIAG] {_rd_side} {_rd_lbl} rest col[0]={_col3(_rm,0)}"
                      f" col[1]={_col3(_rm,1)} col[2]={_col3(_rm,2)}"
                      f" head_to_tail={_col3(_rm,1)}")
                print(f"[HAND-FOREARM-REST-DIAG] {_rd_side} {_rd_lbl} rest head_tail_dir=({_ht_n.x:+.6f},{_ht_n.y:+.6f},{_ht_n.z:+.6f})")
        for _rd_lbl2, _rd_pp in [('ForeArm', _rd_fp), ('Hand', _rd_hp)]:
            _rd_clist = _rd_pp.constraints if _rd_pp else []
            print(f"[HAND-FOREARM-REST-DIAG] {_rd_side} {_rd_lbl2} constraints ({len(_rd_clist)}):")
            for _rc in _rd_clist:
                _rc_tgt = _rc.target.name if getattr(_rc, 'target', None) else 'None'
                _rc_sub = getattr(_rc, 'subtarget', 'N/A')
                print(f"  name={_rc.name!r} type={_rc.type} influence={_rc.influence:.2f}"
                      f" mute={_rc.mute} target={_rc_tgt} subtarget={_rc_sub!r}")
                _rd_twist_names = ('SL_ForearmTwist_L', 'SL_ForearmTwist_R',
                                   'SL_ForearmTwistV2_L', 'SL_ForearmTwistV2_R')
                if _rc.name in _rd_twist_names:
                    _rd_expected = (
                        (_rd_side == 'L' and _ft_v2_enabled_l and
                         _rc.name == 'SL_ForearmTwistV2_L') or
                        (_rd_side == 'R' and _ft_v2_enabled_r and
                         _rc.name == 'SL_ForearmTwistV2_R')
                    )
                    _rd_state = "ACTIVE: calibrated Twist constraint" if _rd_expected else "FAIL: unexpected Twist constraint"
                    print(f"  *** {_rd_state}: {_rc.name!r} ***")
    # Twist controller bone / object scan
    _rd_twist_bones = [b.name for b in arm_obj.data.bones
                       if 'ForearmTwist' in b.name or 'TwistCtrl' in b.name]
    _rd_twist_objs  = [o.name for o in bpy.data.objects
                       if 'ForearmTwist' in o.name or 'TwistCtrl' in o.name]
    print(f"[HAND-FOREARM-REST-DIAG] ForearmTwist/TwistCtrl bones: "
          f"{_rd_twist_bones if _rd_twist_bones else '(none)'}")
    print(f"[HAND-FOREARM-REST-DIAG] ForearmTwist/TwistCtrl objects: "
          f"{_rd_twist_objs if _rd_twist_objs else '(none — expected)'}")
    print("[HAND-FOREARM-REST-DIAG] ===== END =====")

    # ====================================================================
    # 진단 프레임 맵: DIAGNOSTIC_TIMESTAMPS_MS → smooth_index
    # ====================================================================
    _diag_frame_map = {}   # smooth_index (=i in valid_frames) → (ts_actual, src_idx, smooth_idx, bf)
    if _src_ts:
        for _dts in DIAGNOSTIC_TIMESTAMPS_MS:
            _d_si = min(range(len(_src_ts)), key=lambda k: abs(_src_ts[k] - _dts))
            _d_smi = INTRO_FRAMES + _d_si   # smooth_index = i in valid_frames loop
            _d_bf  = _d_smi + 1             # blender_frame
            _diag_frame_map[_d_smi] = (_src_ts[_d_si], _d_si, _d_smi, _d_bf)
    # 참조 quaternion (999ms 기준): {side: Quaternion or None}
    _diag_ref_q = {'L': None, 'R': None}

    def _vec_fmt(v):
        """Vector를 (+0.000000, +0.000000, +0.000000) 형식으로."""
        if v is None: return "N/A"
        return f"({v.x:+.6f}, {v.y:+.6f}, {v.z:+.6f})"

    def _hand_forearm_diag(i, l_pts, r_pts, l_pn, r_pn, lh9, rh9):
        """Pass 2 프레임 i에서 Hand–ForeArm 회전 진단값 출력.
        호출 전 view_layer.update() 완료 필요.
        set_hand_rotation/set_bone_dir 재호출 금지 — 기존 평가 결과만 읽는다.
        """
        if i not in _diag_frame_map:
            return
        _ts_act, _src_idx, _smi, _bf = _diag_frame_map[i]

        for (_ds, _fbn, _hbn, _pts, _pn, _h9, _flip,
             _fa_src_list, _fa_hybrid_list, _emp_ik) in [
                ('L', BONE_L_FOREARM, BONE_L_HAND, l_pts, l_pn, lh9,
                 L_PALM_FLIP, _forearm_src_l, _forearm_hybrid_src_l, emp_lw),
                ('R', BONE_R_FOREARM, BONE_R_HAND, r_pts, r_pn, rh9,
                 R_PALM_FLIP, _forearm_src_r, _forearm_hybrid_src_r, emp_rw)]:

            print(f"[HAND-FOREARM-DIAG] side={_ds} source_ts={_ts_act}"
                  f" source_index={_src_idx} smooth_index={_smi} blender_frame={_bf}")

            # ── A. Source 데이터 ─────────────────────────────────────
            # A1. hand forward (pt0 → pt9)
            _hfwd = _h9 - _pts[H_WRIST] if (_h9 and _pts[H_WRIST]) else None
            if _hfwd and _hfwd.length > 1e-6:
                _hfwd_n = _hfwd.normalized()
            else:
                _hfwd_n = None
            print(f"  source_hand_forward_world={_vec_fmt(_hfwd_n)}")

            # A2. raw palm normal (pre-stabilization, pre-flip)
            _pn_raw_key = 'l_pn' if _ds == 'L' else 'r_pn'
            _pn_raw = raw[_pn_raw_key][_src_idx] if _src_idx < len(raw[_pn_raw_key]) else None
            _pn_raw_n = _pn_raw.normalized() if (_pn_raw and _pn_raw.length > 1e-6) else None
            print(f"  source_palm_normal_world_raw={_vec_fmt(_pn_raw_n)}")

            # A3. effective palm normal (post-stabilization + flip)
            _pn_eff = (-_pn if _flip else _pn) if (_pn and _pn.length > 1e-4) else None
            _pn_eff_n = _pn_eff.normalized() if (_pn_eff and _pn_eff.length > 1e-6) else None
            print(f"  source_palm_normal_world_effective={_vec_fmt(_pn_eff_n)}")

            # A4. source forearm direction (Pose elbow→wrist, Blender world)
            _fa_src = _fa_src_list[_src_idx] if _src_idx < len(_fa_src_list) else None
            _fa_hybrid = (_fa_hybrid_list[_src_idx]
                          if _src_idx < len(_fa_hybrid_list) else None)
            print(f"  source_forearm_direction_world_pose_wrist={_vec_fmt(_fa_src)}")
            print(f"  source_forearm_direction_world_hybrid_wrist={_vec_fmt(_fa_hybrid)}")

            def _angle_deg(_va, _vb):
                if _va is None or _vb is None or _va.length <= 1e-6 or _vb.length <= 1e-6:
                    return None
                _dot = max(-1.0, min(1.0, _va.normalized().dot(_vb.normalized())))
                return math.degrees(math.acos(_dot))

            _src_basis_diff = _angle_deg(_fa_src, _fa_hybrid)
            print(f"  source_pose_vs_hybrid_forearm_angle_deg="
                  f"{_src_basis_diff:+.4f}" if _src_basis_diff is not None else
                  "  source_pose_vs_hybrid_forearm_angle_deg=N/A")

            # ── B. 평가된 Bone world 방향 ────────────────────────────
            _mw = arm_obj.matrix_world
            _fa_pb = arm_obj.pose.bones.get(_fbn)
            _hd_pb = arm_obj.pose.bones.get(_hbn)

            def _bone_world_axes(pb):
                if pb is None:
                    return None, None, None, None
                _world_mat = _mw @ pb.matrix
                _, _rot_q, _scale = _world_mat.decompose()
                if _rot_q.magnitude < 1e-10:
                    return None, None, None, None
                _rot_q.normalize()
                _rot_mat = _rot_q.to_matrix()
                _col0 = _rot_mat.col[0].normalized()
                _col1 = _rot_mat.col[1].normalized()
                _col2 = _rot_mat.col[2].normalized()
                return _col0, _col1, _col2, _rot_mat

            _fa_X, _fa_Y, _fa_Z, _fa_3x3 = _bone_world_axes(_fa_pb)
            _hd_X, _hd_Y, _hd_Z, _hd_3x3 = _bone_world_axes(_hd_pb)

            print(f"  forearm_world_Y={_vec_fmt(_fa_Y)}  forearm_world_Z={_vec_fmt(_fa_Z)}")
            print(f"  hand_world_Y={_vec_fmt(_hd_Y)}  hand_world_Z={_vec_fmt(_hd_Z)}")

            # ── C. 오차 ──────────────────────────────────────────────
            # C1. forearm forward error
            if _fa_Y and _fa_src:
                _cos = max(-1.0, min(1.0, _fa_Y.dot(_fa_src)))
                print(f"  forearm_forward_error_vs_pose_wrist_deg="
                      f"{math.degrees(math.acos(_cos)):+.4f}")
            else:
                print(f"  forearm_forward_error_vs_pose_wrist_deg="
                      f"N/A(fa_Y={_fa_Y is not None} fa_src={_fa_src is not None})")
            _fa_hybrid_err = _angle_deg(_fa_Y, _fa_hybrid)
            print(f"  forearm_forward_error_vs_hybrid_wrist_deg="
                  f"{_fa_hybrid_err:+.4f}" if _fa_hybrid_err is not None else
                  "  forearm_forward_error_vs_hybrid_wrist_deg=N/A")

            # C2. hand forward error
            if _hd_Y and _hfwd_n:
                _cos = max(-1.0, min(1.0, _hd_Y.dot(_hfwd_n)))
                print(f"  hand_forward_error_deg={math.degrees(math.acos(_cos)):+.4f}")
            else:
                print(f"  hand_forward_error_deg=N/A(hd_Y={_hd_Y is not None} hfwd={_hfwd_n is not None})")

            # C3. palm alignment error (project onto plane perp to hand forward)
            if _pn_eff_n and _hd_Y and _hd_Z:
                _hfwd_ref = _hd_Y
                _pn_proj = _pn_eff_n - _pn_eff_n.dot(_hfwd_ref) * _hfwd_ref
                _hz_proj  = _hd_Z    - _hd_Z.dot(_hfwd_ref)     * _hfwd_ref
                if _pn_proj.length > 1e-4 and _hz_proj.length > 1e-4:
                    _cos = max(-1.0, min(1.0, _pn_proj.normalized().dot(_hz_proj.normalized())))
                    print(f"  palm_alignment_error_deg={math.degrees(math.acos(_cos)):+.4f}")
                else:
                    _cause = ("pn_proj_zero" if _pn_proj.length <= 1e-4 else "hz_proj_zero")
                    print(f"  palm_alignment_error_deg=N/A({_cause})")
            else:
                print(f"  palm_alignment_error_deg=N/A(pn_eff={_pn_eff_n is not None}"
                      f" hd_Y={_hd_Y is not None} hd_Z={_hd_Z is not None})")

            # C4. wrist position error
            if _fa_pb and _emp_ik and char_sw > 1e-6:
                _act_wr = _mw @ _fa_pb.tail
                _tgt_wr = _emp_ik.matrix_world.translation
                print(f"  wrist_position_error_sw={(_act_wr - _tgt_wr).length / char_sw:+.6f}")
            else:
                print(f"  wrist_position_error_sw=N/A")

            # C5. 동일한 Hand pt0→pt9 방향에 대해 Pose wrist 기준과
            # 실제 IK 입력(hybrid wrist) 기준의 손목 각도를 분리한다.
            _src_wrist_pose = _angle_deg(_fa_src, _hfwd_n)
            _src_wrist_hybrid = _angle_deg(_fa_hybrid, _hfwd_n)
            _actual_wrist = _angle_deg(_fa_Y, _hd_Y)
            print(f"  source_wrist_angle_pose_wrist_deg="
                  f"{_src_wrist_pose:+.4f}" if _src_wrist_pose is not None else
                  "  source_wrist_angle_pose_wrist_deg=N/A")
            print(f"  source_wrist_angle_hybrid_wrist_deg="
                  f"{_src_wrist_hybrid:+.4f}" if _src_wrist_hybrid is not None else
                  "  source_wrist_angle_hybrid_wrist_deg=N/A")
            print(f"  actual_wrist_angle_deg="
                  f"{_actual_wrist:+.4f}" if _actual_wrist is not None else
                  "  actual_wrist_angle_deg=N/A")
            _werr_pose = (abs(_actual_wrist - _src_wrist_pose)
                          if _actual_wrist is not None and _src_wrist_pose is not None else None)
            _werr_hybrid = (abs(_actual_wrist - _src_wrist_hybrid)
                            if _actual_wrist is not None and _src_wrist_hybrid is not None else None)
            print(f"  wrist_angle_error_vs_pose_wrist_deg="
                  f"{_werr_pose:+.4f}" if _werr_pose is not None else
                  "  wrist_angle_error_vs_pose_wrist_deg=N/A")
            print(f"  wrist_angle_error_vs_hybrid_wrist_deg="
                  f"{_werr_hybrid:+.4f}" if _werr_hybrid is not None else
                  "  wrist_angle_error_vs_hybrid_wrist_deg=N/A")

            # ── D. ForeArm–Hand 상대 회전 ────────────────────────────
            if _fa_3x3 is not None and _hd_3x3 is not None:
                _fa_wq = _fa_3x3.to_quaternion().normalized()
                _hd_wq = _hd_3x3.to_quaternion().normalized()
                _rel_q = _fa_wq.inverted() @ _hd_wq
                _rel_q.normalize()
                if _rel_q.w < 0: _rel_q.negate()
                _, _, _tw_y, _sw_deg, _tot_deg = decompose_swing_twist(
                    _rel_q, mathutils.Vector((0.0, 1.0, 0.0)))
                print(f"  relative_quaternion=({_rel_q.w:+.6f},{_rel_q.x:+.6f},"
                      f"{_rel_q.y:+.6f},{_rel_q.z:+.6f})")
                print(f"  relative_total_angle_deg={_tot_deg:+.4f}"
                      f"  relative_twist_y_deg={_tw_y:+.4f} [rig ForeArm-Hand relative, not anatomical]"
                      f"  relative_swing_deg={_sw_deg:+.4f}")

                # ── E. 기준(999ms) 대비 변화 ─────────────────────────
                if _diag_ref_q[_ds] is None:
                    _diag_ref_q[_ds] = _rel_q.copy()
                _dref = _diag_ref_q[_ds]
                _dq = _rel_q @ _dref.inverted()
                _dq.normalize()
                if _dq.w < 0: _dq.negate()
                _, _, _d_tw, _d_sw, _d_tot = decompose_swing_twist(
                    _dq, mathutils.Vector((0.0, 1.0, 0.0)))
                print(f"  relative_delta_from_999_total_deg={_d_tot:+.4f}"
                      f"  relative_delta_from_999_twist_y_deg={_d_tw:+.4f}"
                      f"  relative_delta_from_999_swing_deg={_d_sw:+.4f}")
            else:
                print(f"  relative_quaternion=N/A(fa_mat={_fa_3x3 is not None}"
                      f" hd_mat={_hd_3x3 is not None})")

            # ── F. Pose channel 값 ───────────────────────────────────
            if _hd_pb:
                _hpq = _hd_pb.rotation_quaternion.copy()
                _hpq.normalize()
                _, _, _pose_ty, _, _ = decompose_swing_twist(
                    _hpq, mathutils.Vector((0.0, 1.0, 0.0)))
                print(f"  hand_pose_channel_quaternion=({_hpq.w:+.6f},{_hpq.x:+.6f},"
                      f"{_hpq.y:+.6f},{_hpq.z:+.6f})")
                print(f"  old_pose_channel_local_y_twist_deg={_pose_ty:+.4f}"
                      f" [pose channel local-Y decomposition diagnostic only]")
            else:
                print(f"  hand_pose_channel_quaternion=N/A")

    # ====================================================================
    # PASS 2: 손 + 손가락 FK 직접 설정 (전체 애니메이션)
    # ====================================================================
    print(f"[정보] Pass 2: 손/손가락 FK ({total}프레임)...")
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    prev_quats_l = {}   # quaternion sign 연속성 추적 (LEFT)
    prev_quats_r = {}   # quaternion sign 연속성 추적 (RIGHT)

    def sign_norm(bone_name, prev_quats):
        """이전 키프레임과 부호가 다르면 쿼터니언 부호 반전 (보간 경로 최단화)."""
        b  = arm_obj.pose.bones.get(bone_name)
        pq = prev_quats.get(bone_name)
        if b and pq and pq.dot(b.rotation_quaternion) < 0:
            b.rotation_quaternion.negate()
        if b:
            prev_quats[bone_name] = b.rotation_quaternion.copy()

    # Forearm twist v2 tracking
    _ft_post_l = [0.0] * total
    _ft_post_r = [0.0] * total
    _ft_corrected_l = 0; _ft_corrected_r = 0
    _ft_unchanged_l = 0; _ft_unchanged_r = 0
    _ft_fallback_l  = 0; _ft_fallback_r  = 0

    for i, frame in enumerate(valid_frames):
        # V2: frame_set FIRST so F-Curve eval runs, then override controller bone Y
        bpy.context.scene.frame_set(frame)
        _ft_cy_l = 0.0; _ft_cy_r = 0.0
        if _ft_v2_enabled:
            if _ft_v2_enabled_l:
                _ft_cy_l = _ft_correction_l[i] if i < len(_ft_correction_l) else 0.0
                _ft_ctb_l = arm_obj.pose.bones.get('SL_LeftForearmTwistCtrlBone')
                if _ft_ctb_l: _ft_ctb_l.rotation_euler.y = math.radians(_ft_cy_l)
            if _ft_v2_enabled_r:
                _ft_cy_r = _ft_correction_r[i] if i < len(_ft_correction_r) else 0.0
                _ft_ctb_r = arm_obj.pose.bones.get('SL_RightForearmTwistCtrlBone')
                if _ft_ctb_r: _ft_ctb_r.rotation_euler.y = math.radians(_ft_cy_r)
        bpy.context.view_layer.update()   # 팔 IK + ForeArm Copy Rotation 평가

        l_pts = [smooth[f'l_p{j}'][i] for j in range(21)]
        r_pts = [smooth[f'r_p{j}'][i] for j in range(21)]
        l_pn  = smooth['l_pn'][i]
        r_pn  = smooth['r_pn'][i]
        lw    = smooth['lw'][i]
        rw    = smooth['rw'][i]
        lh9   = smooth['lh9'][i]
        rh9   = smooth['rh9'][i]

        def _canonical_fist_targets(pts, eff_pn):
            """현재 손바닥 좌표계를 유지한 네 손가락 기준 방향을 만든다."""
            if eff_pn is None or eff_pn.length <= 1e-5:
                return {}
            wrist = pts[H_WRIST]
            middle_mcp = pts[H_MID_MCP]
            if wrist is None or middle_mcp is None:
                return {}

            palm_normal = eff_pn.normalized()
            hand_forward = middle_mcp - wrist
            hand_forward -= palm_normal * hand_forward.dot(palm_normal)
            if hand_forward.length <= 1e-5:
                return {}
            hand_forward.normalize()

            curl_direction = palm_normal * FIST_CURL_DIRECTION_SIGN
            targets = {}
            for chain_index, mcp_index in (
                    (1, H_INDEX_MCP),
                    (2, H_MID_MCP),
                    (3, H_RING_MCP),
                    (4, H_PINK_MCP)):
                mcp = pts[mcp_index]
                if mcp is None:
                    continue
                radial = mcp - wrist
                radial -= palm_normal * radial.dot(palm_normal)
                if radial.length > 1e-5:
                    radial.normalize()
                    base_direction = hand_forward * 0.72 + radial * 0.28
                else:
                    base_direction = hand_forward.copy()
                base_direction -= (
                    palm_normal * base_direction.dot(palm_normal))
                if base_direction.length <= 1e-5:
                    continue
                base_direction.normalize()

                inward = curl_direction - (
                    base_direction * curl_direction.dot(base_direction))
                if inward.length <= 1e-5:
                    continue
                inward.normalize()
                bend_axis = base_direction.cross(inward)
                if bend_axis.length <= 1e-5:
                    continue
                bend_axis.normalize()

                directions = []
                roll_normals = []
                for angle_deg in FIST_CURL_ANGLES_DEG[chain_index]:
                    bend_rotation = mathutils.Quaternion(
                        bend_axis, math.radians(angle_deg))
                    direction = bend_rotation @ base_direction
                    roll_normal = bend_rotation @ palm_normal
                    directions.append(direction.normalized())
                    roll_normals.append(roll_normal.normalized())
                targets[chain_index] = (directions, roll_normals)
            return targets

        def _canonical_thumb_targets(
                pts, eff_pn, thumb_reference,
                thumb_world_up_weight):
            """주먹 구간의 지배적인 엄지 자세를 현재 손바닥 좌표계로 복원."""
            if (not thumb_reference or eff_pn is None or
                    eff_pn.length <= 1e-5):
                return None
            wrist = pts[H_WRIST]
            middle_mcp = pts[H_MID_MCP]
            if wrist is None or middle_mcp is None:
                return None
            palm_normal = eff_pn.normalized()
            hand_forward = middle_mcp - wrist
            hand_forward -= (
                palm_normal * hand_forward.dot(palm_normal))
            if hand_forward.length <= 1e-5:
                return None
            hand_forward.normalize()
            hand_side = palm_normal.cross(hand_forward)
            if hand_side.length <= 1e-5:
                return None
            hand_side.normalize()

            directions = []
            roll_normals = []
            world_up_blend = max(
                0.0, min(1.0, thumb_world_up_weight))
            world_up = mathutils.Vector((0.0, 0.0, 1.0))
            for local_direction in thumb_reference:
                world_direction = (
                    hand_forward * local_direction.x +
                    hand_side * local_direction.y +
                    palm_normal * local_direction.z)
                if world_direction.length <= 1e-5:
                    return None
                if world_up_blend > 1e-5:
                    world_direction = world_direction.lerp(
                        world_up, world_up_blend)
                    if world_direction.length <= 1e-5:
                        world_direction = world_up.copy()
                directions.append(world_direction.normalized())
                roll_normals.append(palm_normal.copy())
            return directions, roll_normals

        def apply_hand(bone_name, wrist_ik, mid9, pn, flip, flip_dir,
                       valid_first, pts, chains, prev_quats, fist_weight,
                       thumb_reference, thumb_blend_weight,
                       thumb_world_up_weight):
            """손뼈 + 손가락 체인 FK 설정.

            effective_pn = -pn if flip else pn — hand bone과 finger roll_pn에 동일 적용.
            hand direction 기준: pts[H_WRIST](Hand pt0) → mid9(Hand pt9) — Pose wrist 미사용.
            wrist_ik: 팔 IK Empty 위치용(Pose 기반), hand rotation에는 사용하지 않음.
            fist_weight: 0=원본 랜드마크, 1=안정적인 주먹 기준 자세.
            thumb_reference: 지배적인 palm-local 엄지 자세. None이면 원본 유지.
            thumb_blend_weight: 엄지만 원본에서 기준 자세로 보간하는 강도.
            """
            if valid_first is not None and i < valid_first:
                return
            # effective_pn: hand bone과 finger roll_pn에 동일하게 적용
            eff_pn = (-pn if flip else pn) if (pn and pn.length > 1e-4) else None
            # hand direction: Hand pt0(wrist) → mid9(Hand pt9) 일관 사용
            if eff_pn is not None:
                set_hand_rotation(arm_obj, bone_name, pts[H_WRIST], mid9, eff_pn, False)
            hand_mat = bone_arma_mat(arm_obj, bone_name)
            sign_norm(bone_name, prev_quats)
            arm_obj.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=frame)
            fist_targets = (
                _canonical_fist_targets(pts, eff_pn)
                if fist_weight > 1e-5 else {})
            thumb_target = (
                _canonical_thumb_targets(
                    pts, eff_pn, thumb_reference,
                    thumb_world_up_weight)
                if thumb_blend_weight > 1e-5 else None)
            for chain_index, chain in enumerate(chains):
                first_parent_mat = get_parent_mat(arm_obj, chain[0][0], bone_name, hand_mat)
                p_mat = first_parent_mat
                canonical = (
                    thumb_target if chain_index == 0
                    else fist_targets.get(chain_index))
                chain_blend_weight = (
                    thumb_blend_weight
                    if chain_index == 0 else fist_weight)
                for segment_index, (bn, from_idx, to_idx) in enumerate(chain):
                    from_w = pts[from_idx]
                    to_w   = pts[to_idx]
                    if flip_dir:
                        from_w, to_w = to_w, from_w

                    ok, new_mat = set_bone_dir(
                        arm_obj, bn, from_w, to_w,
                        parent_mat=p_mat, roll_pn=eff_pn)
                    if ok and canonical and chain_blend_weight > 1e-5:
                        pose_bone = arm_obj.pose.bones.get(bn)
                        tracked_quaternion = (
                            pose_bone.rotation_quaternion.copy()
                            if pose_bone else None)
                        canonical_direction = canonical[0][segment_index]
                        canonical_roll = canonical[1][segment_index]
                        if flip_dir:
                            canonical_direction = -canonical_direction
                        canonical_ok, _ = set_bone_dir(
                            arm_obj, bn,
                            from_w, from_w + canonical_direction,
                            parent_mat=p_mat, roll_pn=canonical_roll)
                        if canonical_ok and pose_bone and tracked_quaternion:
                            canonical_quaternion = (
                                pose_bone.rotation_quaternion.copy())
                            if (tracked_quaternion.dot(
                                    canonical_quaternion) < 0.0):
                                canonical_quaternion.negate()
                            blend_weight = max(
                                0.0, min(1.0, chain_blend_weight))
                            pose_bone.rotation_quaternion = (
                                tracked_quaternion.slerp(
                                    canonical_quaternion, blend_weight))
                            pose_bone.rotation_quaternion.normalize()
                            new_mat = bone_arma_mat(
                                arm_obj, bn, p_mat)
                    if ok:
                        sign_norm(bn, prev_quats)
                        arm_obj.pose.bones[bn].keyframe_insert('rotation_quaternion', frame=frame)
                        p_mat = new_mat
                    else:
                        p_mat = bone_arma_mat(arm_obj, bn, p_mat)

        _fist_weight_l = (
            _fist_weights_l[i] if i < len(_fist_weights_l) else 0.0)
        _fist_weight_r = (
            _fist_weights_r[i] if i < len(_fist_weights_r) else 0.0)
        _fist_thumb_ref_l = (
            _fist_thumb_refs_l[i]
            if i < len(_fist_thumb_refs_l) else None)
        _fist_thumb_ref_r = (
            _fist_thumb_refs_r[i]
            if i < len(_fist_thumb_refs_r) else None)
        _fist_thumb_gesture_ref_l = (
            _fist_thumb_gesture_refs_l[i]
            if i < len(_fist_thumb_gesture_refs_l) else None)
        _fist_thumb_gesture_ref_r = (
            _fist_thumb_gesture_refs_r[i]
            if i < len(_fist_thumb_gesture_refs_r) else None)
        _fist_thumb_gesture_weight_l = (
            _fist_thumb_gesture_weights_l[i]
            if i < len(_fist_thumb_gesture_weights_l) else 0.0)
        _fist_thumb_gesture_weight_r = (
            _fist_thumb_gesture_weights_r[i]
            if i < len(_fist_thumb_gesture_weights_r) else 0.0)
        _fist_thumb_gesture_world_up_weight_l = (
            _fist_thumb_gesture_world_up_l[i]
            if i < len(_fist_thumb_gesture_world_up_l) else 0.0)
        _fist_thumb_gesture_world_up_weight_r = (
            _fist_thumb_gesture_world_up_r[i]
            if i < len(_fist_thumb_gesture_world_up_r) else 0.0)

        # 추적 이상치 보정이 필요한 프레임은 기존 guard가 우선한다.
        # 나머지에서는 검출된 의미 있는 엄지 상태 대비를 사용한다.
        if _fist_thumb_ref_l is not None:
            _thumb_reference_l = _fist_thumb_ref_l
            _thumb_blend_weight_l = _fist_weight_l
            _thumb_world_up_weight_l = 0.0
        else:
            _thumb_reference_l = _fist_thumb_gesture_ref_l
            _thumb_blend_weight_l = (
                _fist_thumb_gesture_weight_l)
            _thumb_world_up_weight_l = (
                _fist_thumb_gesture_world_up_weight_l)
        if _fist_thumb_ref_r is not None:
            _thumb_reference_r = _fist_thumb_ref_r
            _thumb_blend_weight_r = _fist_weight_r
            _thumb_world_up_weight_r = 0.0
        else:
            _thumb_reference_r = _fist_thumb_gesture_ref_r
            _thumb_blend_weight_r = (
                _fist_thumb_gesture_weight_r)
            _thumb_world_up_weight_r = (
                _fist_thumb_gesture_world_up_weight_r)
        apply_hand(
            BONE_L_HAND, lw, lh9, l_pn, L_PALM_FLIP,
            FLIP_FINGER_DIR_L, l_first, l_pts, FINGER_CHAINS_L,
            prev_quats_l, _fist_weight_l,
            _thumb_reference_l, _thumb_blend_weight_l,
            _thumb_world_up_weight_l)
        apply_hand(
            BONE_R_HAND, rw, rh9, r_pn, R_PALM_FLIP,
            FLIP_FINGER_DIR_R, r_first, r_pts, FINGER_CHAINS_R,
            prev_quats_r, _fist_weight_r,
            _thumb_reference_r, _thumb_blend_weight_r,
            _thumb_world_up_weight_r)

        # 진단 대상 프레임이면 dependency graph 갱신 후 진단값 출력
        if i in _diag_frame_map:
            bpy.context.view_layer.update()
            _hand_forearm_diag(i, l_pts, r_pts, l_pn, r_pn, lh9, rh9)

        # V2: keyframe controller bones + record post-correction twist
        if _ft_v2_enabled:
            for (_ft_ctbn4, _ft_cy4, _ft_hbn4, _ft_vf4, _ft_post4,
                 _ft_corr4, _ft_side_en4, _ft_cnt_c, _ft_cnt_u) in [
                    ('SL_LeftForearmTwistCtrlBone',  _ft_cy_l, BONE_L_HAND, l_first,
                     _ft_post_l, _ft_correction_l, _ft_v2_enabled_l,
                     '_ft_corrected_l', '_ft_unchanged_l'),
                    ('SL_RightForearmTwistCtrlBone', _ft_cy_r, BONE_R_HAND, r_first,
                     _ft_post_r, _ft_correction_r, _ft_v2_enabled_r,
                     '_ft_corrected_r', '_ft_unchanged_r')]:
                if not _ft_side_en4: continue
                _ft_ctb4 = arm_obj.pose.bones.get(_ft_ctbn4)
                if _ft_ctb4:
                    _ft_ctb4.keyframe_insert('rotation_euler', frame=frame)
                if _ft_vf4 is not None and i < _ft_vf4: continue
                _ft_hb4 = arm_obj.pose.bones.get(_ft_hbn4)
                if not _ft_hb4: continue
                _ft_hq4 = _ft_hb4.rotation_quaternion.copy()
                if _ft_hq4.w < 0: _ft_hq4.negate()
                _, _, _ft_ptdeg4, _, _ = decompose_swing_twist(_ft_hq4, mathutils.Vector((0, 1, 0)))
                _ft_post4[i] = _ft_ptdeg4
                if abs(_ft_cy4) > 0.5:
                    if _ft_cnt_c == '_ft_corrected_l': _ft_corrected_l += 1
                    else:                              _ft_corrected_r += 1
                else:
                    if _ft_cnt_u == '_ft_unchanged_l': _ft_unchanged_l += 1
                    else:                              _ft_unchanged_r += 1

    for (_fist_side, _fist_weights, _thumb_refs,
         _gesture_refs, _gesture_weights) in (
            ('L', _fist_weights_l, _fist_thumb_refs_l,
             _fist_thumb_gesture_refs_l,
             _fist_thumb_gesture_weights_l),
            ('R', _fist_weights_r, _fist_thumb_refs_r,
             _fist_thumb_gesture_refs_r,
             _fist_thumb_gesture_weights_r)):
        _fist_weighted = sum(weight > 1e-5 for weight in _fist_weights)
        _fist_full = sum(weight >= 0.999 for weight in _fist_weights)
        _fist_max = max(_fist_weights) if _fist_weights else 0.0
        _thumb_guarded = sum(
            1 for _weight, _reference in zip(
                _fist_weights, _thumb_refs)
            if _weight > 1e-5 and _reference is not None)
        _thumb_emphasized = sum(
            1 for _fist_weight, _gesture_weight, _reference in zip(
                _fist_weights, _gesture_weights, _gesture_refs)
            if (_fist_weight * _gesture_weight > 1e-5 and
                _reference is not None))
        print(f"[FIST-HANDSHAPE-APPLY] {_fist_side}:"
              f"  weighted_frames={_fist_weighted}"
              f"  full_lock_frames={_fist_full}"
              f"  max_weight={_fist_max:.3f}"
              f"  thumb_temporal_guard_frames={_thumb_guarded}"
              f"  thumb_gesture_emphasis_frames={_thumb_emphasized}")

    # ── Forearm Twist V2 Diagnostics ────────────────────────────────────
    if USE_FOREARM_TWIST_REDISTRIBUTION:
        for _frd_s, _frd_pre, _frd_post, _frd_corr, _frd_cnt_c, _frd_cnt_u, _frd_fb in [
                ('L', _ft_pre_l, _ft_post_l, _ft_correction_l,
                 _ft_corrected_l, _ft_unchanged_l, _ft_fallback_l),
                ('R', _ft_pre_r, _ft_post_r, _ft_correction_r,
                 _ft_corrected_r, _ft_unchanged_r, _ft_fallback_r)]:
            _src_slice = slice(INTRO_FRAMES, total)
            _pre_src  = [abs(_frd_pre[k])  for k in range(INTRO_FRAMES, total)]
            _post_src = [abs(_frd_post[k]) for k in range(INTRO_FRAMES, total)]
            _corr_src = [abs(_frd_corr[k]) for k in range(INTRO_FRAMES, total)]
            _n = len(_pre_src)
            if _n == 0:
                print(f"[FOREARM-TWIST-V2] {_frd_s}: no data"); continue
            _frd_side_en = _ft_v2_enabled_l if _frd_s == 'L' else _ft_v2_enabled_r
            print(f"[FOREARM-TWIST-V2] {_frd_s}:"
                  f"  enabled={_frd_side_en}"
                  f"  total_frames={_n}"
                  f"  corrected_frames={_frd_cnt_c}"
                  f"  unchanged_frames={_frd_cnt_u}"
                  f"  fallback_frames={_frd_fb}"
                  f"  max_pre_twist_deg={max(_pre_src):.1f}"
                  f"  max_transfer_deg={max(_corr_src):.1f}"
                  f"  max_post_twist_deg={max(_post_src):.1f}")

        # [FOREARM-TWIST-V2-TS] _DIAG_TS 근접 프레임
        if _src_ts and _ft_v2_enabled:
            _frd_bi2 = min(range(len(_src_ts)), key=lambda k: abs(_src_ts[k] - _DIAG_TS))
            _frd_bf2 = INTRO_FRAMES + _frd_bi2 + 1
            _frd_ts2 = _src_ts[_frd_bi2]
            bpy.context.scene.frame_set(_frd_bf2)
            bpy.context.view_layer.update()
            for _frd_s2, _frd_hbn2, _frd_fbn2, _frd_ctbn2, _frd_flip2, _frd_vf2, _frd_emp2 in [
                    ('L', BONE_L_HAND, BONE_L_FOREARM, 'SL_LeftForearmTwistCtrlBone',
                     L_PALM_FLIP, l_first, emp_lw),
                    ('R', BONE_R_HAND, BONE_R_FOREARM, 'SL_RightForearmTwistCtrlBone',
                     R_PALM_FLIP, r_first, emp_rw)]:
                _frd_side_enabled2 = (_ft_v2_enabled_l if _frd_s2 == 'L'
                                      else _ft_v2_enabled_r)
                if not _frd_side_enabled2:
                    continue
                _frd_si2 = INTRO_FRAMES + _frd_bi2
                _frd_pts2 = [smooth[f'{"l" if _frd_s2=="L" else "r"}_p{j}'][_frd_si2] for j in range(21)]
                _frd_h92  = smooth['lh9' if _frd_s2=='L' else 'rh9'][_frd_si2]
                _frd_pnv2 = smooth['l_pn' if _frd_s2=='L' else 'r_pn'][_frd_si2]
                # evaluated forearm twist at _DIAG_TS (controller=0 baseline)
                _frd_ctb2 = arm_obj.pose.bones.get(_frd_ctbn2)
                _frd_fb2  = arm_obj.pose.bones.get(_frd_fbn2)
                _frd_hb2  = arm_obj.pose.bones.get(_frd_hbn2)
                if not (_frd_ctb2 and _frd_fb2 and _frd_hb2):
                    print(f"[FOREARM-TWIST-V2-TS] {_frd_s2}: bone missing — skipped")
                    continue
                _frd_ctb2.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0), 'XYZ')
                bpy.context.view_layer.update()
                _frd_fa_mat_base = _frd_fb2.matrix.copy()
                # actual controller Y at this frame
                _frd_ctb2.rotation_euler.y = math.radians(
                    _ft_correction_l[_frd_bf2-1] if _frd_s2=='L' else _ft_correction_r[_frd_bf2-1])
                bpy.context.view_layer.update()
                _frd_fa_mat_cur = _frd_fb2.matrix.copy()
                # forearm twist delta
                _frd_dq2 = (_frd_fa_mat_base.to_3x3().to_quaternion().inverted()
                            @ _frd_fa_mat_cur.to_3x3().to_quaternion())
                _frd_dq2.normalize()
                _, _, _frd_fa_delta2, _, _ = decompose_swing_twist(
                    _frd_dq2, mathutils.Vector((0.0, 1.0, 0.0)))
                _frd_actual_ctrl_y = math.degrees(_frd_ctb2.rotation_euler.y)
                # hand errors
                _frd_fwd2, _frd_palm2 = _ft_hand_errors(_frd_hbn2, _frd_pts2, _frd_h92, _frd_pnv2, _frd_flip2)
                _frd_wrist2 = _ft_wrist_err(_frd_fbn2, _frd_emp2)
                # post hand twist
                _frd_pre_td2  = _ft_pre_l[_frd_si2]  if _frd_s2=='L' else _ft_pre_r[_frd_si2]
                _frd_post_td2 = _ft_post_l[_frd_bf2-1] if _frd_s2=='L' else _ft_post_r[_frd_bf2-1]
                _frd_cy2 = (_ft_correction_l[_frd_bf2-1] if _frd_s2=='L'
                            else _ft_correction_r[_frd_bf2-1])
                if abs(_frd_actual_ctrl_y - _frd_cy2) > 0.5:
                    print(f"[FOREARM-TWIST-V2-TS] {_frd_s2}: WARNING actual_ctrl_y={_frd_actual_ctrl_y:.2f}"
                          f" differs from requested={_frd_cy2:.2f} by {abs(_frd_actual_ctrl_y-_frd_cy2):.2f}°")
                print(f"[FOREARM-TWIST-V2-TS] {_frd_s2}: ts={_frd_ts2}  frame={_frd_bf2}"
                      f"  pre_hand_twist_deg={_frd_pre_td2:.1f}"
                      f"  controller_bone_y_deg={_frd_cy2:.1f}"
                      f"  actual_controller_euler_y_deg={_frd_actual_ctrl_y:.1f}"
                      f"  evaluated_forearm_twist_delta_deg={_frd_fa_delta2:.1f}"
                      f"  post_hand_twist_deg={_frd_post_td2:.1f}"
                      f"  hand_forward_error_deg={_frd_fwd2:.2f}"
                      f"  palm_alignment_error_deg={_frd_palm2:.2f}"
                      f"  wrist_position_error_sw={_frd_wrist2:.5f}")

    # ── Wrist Relative Rotation Diagnostics (Pass 2 완료 후, POSE 모드 유지) ──
    if _src_ts:
        _wrd_bi = min(range(len(_src_ts)), key=lambda k: abs(_src_ts[k] - _DIAG_TS))
        _wrd_bf = INTRO_FRAMES + _wrd_bi + 1
        _wrd_ts = _src_ts[_wrd_bi]
        # [TWIST-BONE-DIAG]
        _twist_bones = [b.name for b in arm_obj.data.bones
                        if any(kw in b.name.lower() for kw in ('twist', 'forearmtwist', 'handtwist'))]
        print(f"[TWIST-BONE-DIAG]: {_twist_bones if _twist_bones else 'none'}")
        bpy.context.scene.frame_set(_wrd_bf)
        bpy.context.view_layer.update()
        _wrd_mw = arm_obj.matrix_world
        _wrd_window_start = max(0, _wrd_bi - 3)
        _wrd_window_end   = min(len(_src_ts) - 1, _wrd_bi + 3)
        for (_wrd_side, _wrd_fbn, _wrd_hbn, _wrd_fa_src,
             _wrd_fa_hybrid_src, _wrd_pn_tr, _wrd_pn_flip) in [
                ('L', BONE_L_FOREARM, BONE_L_HAND, _forearm_src_l,
                 _forearm_hybrid_src_l, _pn_l_track, L_PALM_FLIP),
                ('R', BONE_R_FOREARM, BONE_R_HAND, _forearm_src_r,
                 _forearm_hybrid_src_r, _pn_r_track, R_PALM_FLIP)]:
            _wrd_pfb = arm_obj.pose.bones.get(_wrd_fbn)
            _wrd_phb = arm_obj.pose.bones.get(_wrd_hbn)
            if not (_wrd_pfb and _wrd_phb):
                print(f"[WRIST-RELATIVE-DIAG] {_wrd_side}: bone not found"); continue
            # actual forearm / hand world directions
            _wrd_act_fa = (_wrd_mw @ _wrd_pfb.tail - _wrd_mw @ _wrd_pfb.head)
            _wrd_act_hd = (_wrd_mw @ _wrd_phb.tail - _wrd_mw @ _wrd_phb.head)
            _wrd_act_fa = _wrd_act_fa.normalized() if _wrd_act_fa.length > 1e-6 else mathutils.Vector((0,0,1))
            _wrd_act_hd = _wrd_act_hd.normalized() if _wrd_act_hd.length > 1e-6 else mathutils.Vector((0,0,1))
            # actual wrist angle
            try: _wrd_act_wang = math.degrees(math.acos(max(-1.0, min(1.0, _wrd_act_fa.dot(_wrd_act_hd)))))
            except: _wrd_act_wang = 0.0
            # source forearm dir
            _wrd_src_fa = _wrd_fa_src[_wrd_bi] if _wrd_bi < len(_wrd_fa_src) else None
            _wrd_src_fa_hybrid = (_wrd_fa_hybrid_src[_wrd_bi]
                                  if _wrd_bi < len(_wrd_fa_hybrid_src) else None)
            # source hand dir (Hand pt0 → pt9)
            _wrd_si = INTRO_FRAMES + _wrd_bi
            _wrd_p0k = 'l_p0' if _wrd_side == 'L' else 'r_p0'
            _wrd_h9k = 'lh9'  if _wrd_side == 'L' else 'rh9'
            _wrd_sp0 = smooth[_wrd_p0k][_wrd_si] if _wrd_si < len(smooth[_wrd_p0k]) else None
            _wrd_sh9 = smooth[_wrd_h9k][_wrd_si] if _wrd_si < len(smooth[_wrd_h9k]) else None
            _wrd_src_hd = None
            if _wrd_sp0 is not None and _wrd_sh9 is not None:
                _hv = _wrd_sh9 - _wrd_sp0
                _wrd_src_hd = _hv.normalized() if _hv.length > 1e-6 else None
            # source wrist angle
            _wrd_src_wang = None
            if _wrd_src_fa is not None and _wrd_src_hd is not None:
                try: _wrd_src_wang = math.degrees(math.acos(max(-1.0, min(1.0, _wrd_src_fa.dot(_wrd_src_hd)))))
                except: _wrd_src_wang = 0.0
            _wrd_src_wang_hybrid = None
            if _wrd_src_fa_hybrid is not None and _wrd_src_hd is not None:
                try:
                    _wrd_src_wang_hybrid = math.degrees(math.acos(
                        max(-1.0, min(1.0, _wrd_src_fa_hybrid.dot(_wrd_src_hd)))))
                except:
                    _wrd_src_wang_hybrid = 0.0
            # effective palm normal
            _wrd_pnv = _wrd_pn_tr[_wrd_bi] if _wrd_bi < len(_wrd_pn_tr) else None
            _wrd_eff_pn = ((-_wrd_pnv if _wrd_pn_flip else _wrd_pnv) if _wrd_pnv is not None else None)
            # actual hand world Z
            _wrd_az = _wrd_phb.matrix.col[2].to_3d().normalized()
            _wrd_act_hz = (_wrd_mw.to_3x3() @ _wrd_az).normalized()
            # palm alignment error
            _wrd_palm_err = None
            if _wrd_eff_pn is not None:
                _pnp = _wrd_eff_pn - _wrd_eff_pn.dot(_wrd_act_hd) * _wrd_act_hd
                _hzp = _wrd_act_hz - _wrd_act_hz.dot(_wrd_act_hd) * _wrd_act_hd
                if _pnp.length > 1e-4 and _hzp.length > 1e-4:
                    try: _wrd_palm_err = math.degrees(math.acos(max(-1.0, min(1.0, _pnp.normalized().dot(_hzp.normalized())))))
                    except: _wrd_palm_err = 0.0
            # hand bone local rotation swing-twist
            _wrd_hq = _wrd_phb.rotation_quaternion.copy()
            if _wrd_hq.w < 0: _wrd_hq.negate()
            _wrd_sw, _wrd_tq, _wrd_tdeg, _wrd_sdeg, _wrd_totdeg = decompose_swing_twist(
                _wrd_hq, mathutils.Vector((0, 1, 0)))
            def _vs(v): return f"({v.x:+.3f},{v.y:+.3f},{v.z:+.3f})" if v is not None else "None"
            def _fs(v): return f"{v:.2f}" if v is not None else "None"
            print(f"[WRIST-RELATIVE-DIAG] {_wrd_side}: ts={_wrd_ts}  frame={_wrd_bf}")
            print(f"  source_forearm_dir_pose_wrist={_vs(_wrd_src_fa)}")
            print(f"  source_forearm_dir_hybrid_wrist={_vs(_wrd_src_fa_hybrid)}")
            print(f"  source_hand_dir={_vs(_wrd_src_hd)}")
            print(f"  source_wrist_angle_pose_wrist_deg={_fs(_wrd_src_wang)}")
            print(f"  source_wrist_angle_hybrid_wrist_deg={_fs(_wrd_src_wang_hybrid)}")
            print(f"  actual_forearm_dir={_vs(_wrd_act_fa)}")
            print(f"  actual_hand_dir={_vs(_wrd_act_hd)}")
            print(f"  actual_wrist_angle_deg={_wrd_act_wang:.2f}")
            print(f"  wrist_angle_error_vs_pose_wrist_deg="
                  f"{_fs(abs(_wrd_act_wang - _wrd_src_wang) if _wrd_src_wang is not None else None)}")
            print(f"  wrist_angle_error_vs_hybrid_wrist_deg="
                  f"{_fs(abs(_wrd_act_wang - _wrd_src_wang_hybrid) if _wrd_src_wang_hybrid is not None else None)}")
            print(f"  hand_local_quaternion=({_wrd_hq.w:.4f},{_wrd_hq.x:.4f},{_wrd_hq.y:.4f},{_wrd_hq.z:.4f})")
            print(f"  hand_local_total_angle_deg={_wrd_totdeg:.2f}")
            print(f"  hand_local_twist_deg={_wrd_tdeg:.2f}")
            print(f"  hand_local_swing_deg={_wrd_sdeg:.2f}")
            print(f"  effective_palm_normal={_vs(_wrd_eff_pn)}")
            print(f"  actual_hand_world_z={_vs(_wrd_act_hz)}")
            print(f"  palm_alignment_error_deg={_fs(_wrd_palm_err)}")
        # [WRIST-RELATIVE-WINDOW] per side
        for (_wrw_side, _wrw_fbn, _wrw_hbn, _wrw_fa_src,
             _wrw_fa_hybrid_src, _wrw_pn_tr, _wrw_pn_flip) in [
                ('L', BONE_L_FOREARM, BONE_L_HAND, _forearm_src_l,
                 _forearm_hybrid_src_l, _pn_l_track, L_PALM_FLIP),
                ('R', BONE_R_FOREARM, BONE_R_HAND, _forearm_src_r,
                 _forearm_hybrid_src_r, _pn_r_track, R_PALM_FLIP)]:
            print(f"[WRIST-RELATIVE-WINDOW] {_wrw_side}:")
            _wrw_prev_twist = None
            _wrw_p0k = 'l_p0' if _wrw_side == 'L' else 'r_p0'
            _wrw_h9k = 'lh9'  if _wrw_side == 'L' else 'rh9'
            for _wi in range(_wrd_window_start, _wrd_window_end + 1):
                _wi_bf = INTRO_FRAMES + _wi + 1
                _wi_ts = _src_ts[_wi] if _wi < len(_src_ts) else '?'
                bpy.context.scene.frame_set(_wi_bf)
                bpy.context.view_layer.update()
                _wi_mw = arm_obj.matrix_world
                _wi_phb = arm_obj.pose.bones.get(_wrw_hbn)
                _wi_pfb = arm_obj.pose.bones.get(_wrw_fbn)
                if not (_wi_phb and _wi_pfb): continue
                # twist decomposition
                _wi_hq = _wi_phb.rotation_quaternion.copy()
                if _wi_hq.w < 0: _wi_hq.negate()
                _, _, _wi_twist_raw, _, _ = decompose_swing_twist(_wi_hq, mathutils.Vector((0, 1, 0)))
                # unwrap
                if _wrw_prev_twist is not None:
                    _wi_dlt = _wi_twist_raw - _wrw_prev_twist
                    if _wi_dlt > 180.0:    _wi_twist_raw -= 360.0
                    elif _wi_dlt < -180.0: _wi_twist_raw += 360.0
                _wi_twist_delta = (_wi_twist_raw - _wrw_prev_twist) if _wrw_prev_twist is not None else 0.0
                _wrw_prev_twist = _wi_twist_raw
                # source wrist angle
                _wi_fa_dir = _wrw_fa_src[_wi] if _wi < len(_wrw_fa_src) else None
                _wi_fa_hybrid_dir = (_wrw_fa_hybrid_src[_wi]
                                     if _wi < len(_wrw_fa_hybrid_src) else None)
                _wi_si = INTRO_FRAMES + _wi
                _wi_sp0 = smooth[_wrw_p0k][_wi_si] if _wi_si < len(smooth[_wrw_p0k]) else None
                _wi_sh9 = smooth[_wrw_h9k][_wi_si] if _wi_si < len(smooth[_wrw_h9k]) else None
                _wi_src_wang = None
                _wi_src_hybrid_wang = None
                if _wi_fa_dir is not None and _wi_sp0 is not None and _wi_sh9 is not None:
                    _hv2 = _wi_sh9 - _wi_sp0
                    _wi_hd2 = _hv2.normalized() if _hv2.length > 1e-6 else None
                    if _wi_hd2:
                        try: _wi_src_wang = math.degrees(math.acos(max(-1.0, min(1.0, _wi_fa_dir.dot(_wi_hd2)))))
                        except: _wi_src_wang = 0.0
                        if _wi_fa_hybrid_dir is not None:
                            try:
                                _wi_src_hybrid_wang = math.degrees(math.acos(
                                    max(-1.0, min(1.0, _wi_fa_hybrid_dir.dot(_wi_hd2)))))
                            except:
                                _wi_src_hybrid_wang = 0.0
                # actual wrist angle
                _wi_fa_d = (_wi_mw @ _wi_pfb.tail - _wi_mw @ _wi_pfb.head)
                _wi_hb_d = (_wi_mw @ _wi_phb.tail - _wi_mw @ _wi_phb.head)
                _wi_fa_d = _wi_fa_d.normalized() if _wi_fa_d.length > 1e-6 else None
                _wi_hb_d = _wi_hb_d.normalized() if _wi_hb_d.length > 1e-6 else None
                _wi_act_wang = None
                if _wi_fa_d and _wi_hb_d:
                    try: _wi_act_wang = math.degrees(math.acos(max(-1.0, min(1.0, _wi_fa_d.dot(_wi_hb_d)))))
                    except: _wi_act_wang = 0.0
                # palm alignment
                _wi_palm_err_w = None
                _wi_pnv = _wrw_pn_tr[_wi] if _wi < len(_wrw_pn_tr) else None
                _wi_eff2 = ((-_wi_pnv if _wrw_pn_flip else _wi_pnv) if _wi_pnv is not None else None)
                if _wi_eff2 is not None and _wi_hb_d:
                    _wi_az2 = _wi_phb.matrix.col[2].to_3d().normalized()
                    _wi_ahz = (_wi_mw.to_3x3() @ _wi_az2).normalized()
                    _pnp3 = _wi_eff2 - _wi_eff2.dot(_wi_hb_d) * _wi_hb_d
                    _hzp3 = _wi_ahz - _wi_ahz.dot(_wi_hb_d) * _wi_hb_d
                    if _pnp3.length > 1e-4 and _hzp3.length > 1e-4:
                        try: _wi_palm_err_w = math.degrees(math.acos(max(-1.0, min(1.0, _pnp3.normalized().dot(_hzp3.normalized())))))
                        except: _wi_palm_err_w = 0.0
                _wi_sw_s = f"{_wi_src_wang:.1f}" if _wi_src_wang is not None else "None"
                _wi_shw_s = (f"{_wi_src_hybrid_wang:.1f}"
                             if _wi_src_hybrid_wang is not None else "None")
                _wi_aw_s = f"{_wi_act_wang:.1f}" if _wi_act_wang is not None else "None"
                _wi_pe_s = f"{_wi_palm_err_w:.1f}" if _wi_palm_err_w is not None else "None"
                _ctr = " ← center" if _wi == _wrd_bi else ""
                print(f"  ts={_wi_ts}  frame={_wi_bf}"
                      f"  twist_deg={_wi_twist_raw:+.1f}"
                      f"  twist_delta={_wi_twist_delta:+.1f}"
                      f"  src_pose_wrist={_wi_sw_s}"
                      f"  src_hybrid_wrist={_wi_shw_s}"
                      f"  act_wrist={_wi_aw_s}"
                      f"  palm_err={_wi_pe_s}{_ctr}")

    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end   = total
    bpy.context.scene.frame_set(1)

    print(f"[완료] {total}프레임  액션: {action_name}")
    print("  손바닥 뒤집힘 → L_PALM_FLIP / R_PALM_FLIP = True")
    print("  손가락 방향 반대 → FLIP_FINGER_DIR_L / FLIP_FINGER_DIR_R = True")
    print("  손 반대쪽에 붙음 → SWAP_HAND_LABELS = not SWAP_HAND_LABELS")


main()
