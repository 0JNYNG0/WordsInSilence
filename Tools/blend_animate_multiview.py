"""
Tools/blend_animate.py
------------------------------------------------------
MediaPipe 랜드마크 JSON → Blender Mixamo 아마추어 애니메이션

팔  : IK chain_count=2 (팔꿈치 굽힘)
손  : FK 직접 회전 — set_hand_rotation (방향 + 손바닥 법선)
손가락: FK 직접 회전 — 연속 랜드마크 간 방향 벡터 사용
       (IK 절대위치 타게팅 방식 제거 — 뼈 길이 불일치로 관절 꺾임 발생)
"""

import bpy, json, math, mathutils
print("[SCRIPT-VERSION] blend_animate_multiview_v1")

# ================================================================
# 설정
# ================================================================
import os as _os
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
JSON_PATH = _os.path.join(_THIS_DIR, "s04_out", "multiview_fused.json")
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
TWO_HAND_CONTACT_MAX_CORRECTION_SW = 0.16  # correction 최대 크기 (char_sw 단위)
TWO_HAND_CONTACT_FADE_EDGE_WEIGHT  = 0.50  # 구간 첫/마지막 프레임 fade weight
TWO_HAND_CONTACT_DEFAULT_MODE = "off"     # 등록되지 않은 모션의 기본값
TWO_HAND_CONTACT_MOTION_PROFILES = {
    "감사합니다": "touch",
    # "touch" = 접촉 보정 실행, "off" = 스킵
}
INTRO_FRAMES     = 20
USE_SOURCE_UPPER_ARM_GUIDANCE      = True    # shoulder→elbow 방향 기반 Pole guidance 활성화
SOURCE_UPPER_ARM_GUIDANCE_BLEND    = 1.0     # 0=legacy(bend), 1=guided(upper-arm)
SOURCE_UPPER_ARM_SMOOTH_WINDOW     = 5       # upper-arm 방향 스무딩 윈도우
SOURCE_UPPER_ARM_MIN_MAGNITUDE     = 0.03    # source upper-arm 유효 최소 길이 (정규화 공간)
USE_PALM_NORMAL_CONTINUITY = True   # palm normal hemisphere continuity 안정화
PALM_NORMAL_SMOOTH_WINDOW  = 5      # palm normal moving average 윈도우
PALM_NORMAL_MIN_MAGNITUDE  = 1e-5   # 유효 법선 최소 길이
USE_FOREARM_TWIST_REDISTRIBUTION = False  # 진단 실행 중 비활성화 (기존 실험 코드)
HAND_TWIST_SOFT_LIMIT_DEG        = 100.0  # 이 각도 초과분만 ForeArm으로 이전
FOREARM_TWIST_MAX_TRANSFER_DEG   = 90.0   # ForeArm 이전 최대 각도 clamp
FOREARM_TWIST_SMOOTH_WINDOW      = 5      # correction track 스무딩 윈도우
DIAGNOSTIC_TIMESTAMPS_MS = (999, 1299, 1333, 1366)   # Hand-ForeArm 진단 대상 ts

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
    _contact_mode = TWO_HAND_CONTACT_MOTION_PROFILES.get(
        _motion_id, TWO_HAND_CONTACT_DEFAULT_MODE)
    _motion_contact_enabled = (
        USE_TWO_HAND_CONTACT_PRESERVATION and _contact_mode == "touch")
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

    # ── Two-Hand Contact Preservation ───────────────────────────────
    print(f"[TWO-HAND-CONTACT-MODE]"
          f"  motion_id={_motion_id!r}"
          f"  global_enabled={USE_TWO_HAND_CONTACT_PRESERVATION}"
          f"  mode={_contact_mode!r}"
          f"  enabled={_motion_contact_enabled}")
    if not _motion_contact_enabled:
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

            # Pass 3: 접촉 구간별 raw correction vector
            _corr_raw = [None] * _contact_source_count
            _prev_pair_c = None; _pair_changes_c = 0
            for _ci in range(_contact_source_count):
                if not _contact_active[_ci]: continue
                _si = INTRO_FRAMES + _ci
                _rc = _raw_contact[_ci]
                if not _rc: continue
                _li, _rj, _ = _rc
                _lp0v = smooth['l_p0'][_si]      if _si < len(smooth['l_p0']) else None
                _rp0v = smooth['r_p0'][_si]      if _si < len(smooth['r_p0']) else None
                _lpiv = smooth[f'l_p{_li}'][_si] if _si < len(smooth[f'l_p{_li}']) else None
                _rpjv = smooth[f'r_p{_rj}'][_si] if _si < len(smooth[f'r_p{_rj}']) else None
                _lwv  = smooth['lw_ik'][_si]     if _si < len(smooth['lw_ik']) else None
                _rwv  = smooth['rw_ik'][_si]     if _si < len(smooth['rw_ik']) else None
                if not (_lp0v and _rp0v and _lpiv and _rpjv and _lwv and _rwv): continue
                _gap_v = (_rwv + (_rpjv - _rp0v)) - (_lwv + (_lpiv - _lp0v))
                _cv = _gap_v * TWO_HAND_CONTACT_BLEND
                _max_len = char_sw * TWO_HAND_CONTACT_MAX_CORRECTION_SW
                if _cv.length > _max_len: _cv = _cv.normalized() * _max_len
                _corr_raw[_ci] = _cv
                _cp = (_li, _rj)
                if _prev_pair_c is not None and _cp != _prev_pair_c: _pair_changes_c += 1
                _prev_pair_c = _cp

            # Pass 4a: correction track 3-frame moving average (접촉 구간 내부만)
            _corr_sm = [None] * _contact_source_count
            for _ci in range(_contact_source_count):
                if not _contact_active[_ci] or _corr_raw[_ci] is None: continue
                _vs = mathutils.Vector((0.0, 0.0, 0.0)); _vc = 0
                for _dk in range(-1, 2):
                    _ck2 = _ci + _dk
                    if 0 <= _ck2 < _contact_source_count and _contact_active[_ck2] and _corr_raw[_ck2]:
                        _vs += _corr_raw[_ck2]; _vc += 1
                _corr_sm[_ci] = _vs / _vc if _vc else _corr_raw[_ci]

            # Pass 4b: 접촉 구간 경계 fade weight (구간 길이별)
            _corr_w = [0.0] * _contact_source_count
            _weight_log = []  # [(ts_s, ts_e, slen, weights)]
            _seg_s = None
            for _ci in range(_contact_source_count + 1):
                _act = _contact_active[_ci] if _ci < _contact_source_count else False
                if _act and _seg_s is None:
                    _seg_s = _ci
                elif not _act and _seg_s is not None:
                    _slen = _ci - _seg_s
                    if _slen == 1:
                        _seg_w = [1.0]
                    elif _slen == 2:
                        _seg_w = [0.75, 0.75]
                    else:
                        _seg_w = ([TWO_HAND_CONTACT_FADE_EDGE_WEIGHT] +
                                  [1.0] * (_slen - 2) +
                                  [TWO_HAND_CONTACT_FADE_EDGE_WEIGHT])
                    for _k, _wv in enumerate(_seg_w):
                        _corr_w[_seg_s + _k] = _wv
                    _ts_s = _src_ts[_seg_s]     if _seg_s < len(_src_ts) else '?'
                    _ts_e = _src_ts[_ci - 1]   if (_ci - 1) < len(_src_ts) else '?'
                    _weight_log.append((_ts_s, _ts_e, _slen, _seg_w))
                    _seg_s = None

            # [TWO-HAND-CONTACT-WEIGHTS]
            print(f"[TWO-HAND-CONTACT-WEIGHTS]")
            for _wts_s, _wts_e, _wfcnt, _wfs in _weight_log:
                print(f"  ts={_wts_s}~{_wts_e}  frames={_wfcnt}"
                      f"  weights={[round(_wf, 3) for _wf in _wfs]}")

            # Pass 5: 보정 적용 + post-clamp + 진단 수집
            _max_corr_sw_c = 0.0; _max_src_dsw_c = 0.0; _postclamp_cnt_c = 0
            _diag_bi_c = (min(range(len(_src_ts)), key=lambda k: abs(_src_ts[k] - _DIAG_TS))
                          if _src_ts else None)
            _contact_diag_c = None
            for _ci in range(_contact_source_count):
                _cv = _corr_sm[_ci]; _w = _corr_w[_ci]
                if _cv is None or _w < 1e-6: continue
                _cv_w = _cv * _w
                _corr_sw = _cv_w.length / char_sw
                _max_corr_sw_c = max(_max_corr_sw_c, _corr_sw)
                _rc2 = _raw_contact[_ci]   # guaranteed not None when _cv not None
                _max_src_dsw_c = max(_max_src_dsw_c, _rc2[2])
                _si = INTRO_FRAMES + _ci
                _li2, _rj2, _dsw2 = _rc2
                _lp0_2 = smooth['l_p0'][_si]; _rp0_2 = smooth['r_p0'][_si]
                _lpli2 = smooth[f'l_p{_li2}'][_si]; _rprj2 = smooth[f'r_p{_rj2}'][_si]
                _lw_bef = smooth['lw_ik'][_si]; _rw_bef = smooth['rw_ik'][_si]
                _gap_bef_sw = ((_rw_bef + (_rprj2 - _rp0_2)) -
                               (_lw_bef + (_lpli2 - _lp0_2))).length / char_sw
                smooth['lw_ik'][_si] = _lw_bef + _cv_w * 0.5
                smooth['rw_ik'][_si] = _rw_bef - _cv_w * 0.5
                # post-correction reach re-clamp (초과 시에만)
                _postclamp = False
                for _ik_k3, _sh_w3, _at3 in [('lw_ik', l_sh_w, char_l_arm_total),
                                               ('rw_ik', r_sh_w, char_r_arm_total)]:
                    _v3 = smooth[_ik_k3][_si]
                    if _v3 is None or _at3 <= 1e-6: continue
                    _off3 = _v3 - _sh_w3
                    if _off3.length / _at3 > ARM_REACH_MAX_RATIO:
                        smooth[_ik_k3][_si] = _sh_w3 + _off3.normalized() * _at3 * ARM_REACH_MAX_RATIO
                        _postclamp = True
                if _postclamp: _postclamp_cnt_c += 1
                # _DIAG_TS 진단
                if _diag_bi_c is not None and _ci == _diag_bi_c:
                    _lw_aft = smooth['lw_ik'][_si]; _rw_aft = smooth['rw_ik'][_si]
                    _gap_aft_sw = ((_rw_aft + (_rprj2 - _rp0_2)) -
                                   (_lw_aft + (_lpli2 - _lp0_2))).length / char_sw
                    _unclamped_sw = _gap_bef_sw * TWO_HAND_CONTACT_BLEND
                    _clamped_sw   = min(_unclamped_sw, TWO_HAND_CONTACT_MAX_CORRECTION_SW)
                    _contact_diag_c = (_src_ts[_ci], _li2, _rj2, _dsw2,
                                        _gap_bef_sw, _corr_sw, _gap_aft_sw,
                                        _lw_bef, _lw_aft, _rw_bef, _rw_aft,
                                        _w, _unclamped_sw, _clamped_sw)

            # 통계 출력
            _cand_c = sum(1 for _rc in _raw_contact if _rc and _rc[2] <= TWO_HAND_CONTACT_ENTER_SW)
            _conf_c = sum(_contact_active)
            print(f"[TWO-HAND-CONTACT] candidate={_cand_c}  confirmed={_conf_c}"
                  f"  max_src_dist_sw={_max_src_dsw_c:.4f}  max_corr_sw={_max_corr_sw_c:.4f}"
                  f"  pair_changes={_pair_changes_c}  post_clamp={_postclamp_cnt_c}")
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
        for _ik_key, _sh_w, _arm_tot, _srb, _rest_el, _pside, _p_tr, _upsm, _up_len, _fa_len in [
                ('lw_ik', l_sh_w, char_l_arm_total, _pole_src_l, _rest_elbow_l_w, 'L', _pole_track_l,
                 _upper_src_l_sm, char_l_upper_len, char_l_forearm_len),
                ('rw_ik', r_sh_w, char_r_arm_total, _pole_src_r, _rest_elbow_r_w, 'R', _pole_track_r,
                 _upper_src_r_sm, char_r_upper_len, char_r_forearm_len)]:
            _prev_dir = None; _flip_cnt = 0; _fb_cnt = 0; _val_cnt = 0
            _guide_cnt = 0; _legacy_cnt = 0; _rest_cnt = 0
            _pdiag = None; _pdiag_best = float('inf')
            _guide_diag = None
            _raw_poles = []; _rest_dr = _rest_el - _sh_w
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
    # ── Controller Bone 생성 (EDIT Mode, ForeArm과 동일한 head/tail/roll/parent) ──
    if USE_FOREARM_TWIST_REDISTRIBUTION:
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='EDIT')
        for _ctbn, _src_fbn in [
                ('SL_LeftForearmTwistCtrlBone',  BONE_L_FOREARM),
                ('SL_RightForearmTwistCtrlBone', BONE_R_FOREARM)]:
            _src_eb = arm_obj.data.edit_bones.get(_src_fbn)
            if not _src_eb:
                print(f"[FOREARM-TWIST-V2] 경고: source bone 없음 {_src_fbn}"); continue
            _ct_eb = arm_obj.data.edit_bones.get(_ctbn)
            if _ct_eb is None:
                _ct_eb = arm_obj.data.edit_bones.new(_ctbn)
            _ct_eb.head        = _src_eb.head.copy()
            _ct_eb.tail        = _src_eb.tail.copy()
            _ct_eb.roll        = _src_eb.roll
            _ct_eb.parent      = _src_eb.parent
            _ct_eb.use_connect = False
            _ct_eb.use_deform  = False
        bpy.ops.object.mode_set(mode='OBJECT')

    # ── 컨스트레인트: 팔 IK만, 손/손가락은 모두 제거 ─────────────
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

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
        # 기존 v1 constraint 제거, v2 constraint 추가 (IK 다음 순서)
        for _ft_bn, _ft_ctbn, _ft_cname in [
                (BONE_L_FOREARM, 'SL_LeftForearmTwistCtrlBone',  'SL_ForearmTwistV2_L'),
                (BONE_R_FOREARM, 'SL_RightForearmTwistCtrlBone', 'SL_ForearmTwistV2_R')]:
            _ft_pb = arm_obj.pose.bones.get(_ft_bn)
            if not _ft_pb:
                print(f"[FOREARM-TWIST-V2] 경고: 뼈 없음 {_ft_bn}"); continue
            for _fc in list(_ft_pb.constraints):
                if _fc.name in ('SL_ForearmTwist_L', 'SL_ForearmTwist_R', _ft_cname):
                    _ft_pb.constraints.remove(_fc)
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

        # ── Calibration at _DIAG_TS (per-side independent) ───────────
        _cal_bi = min(range(len(_src_ts)), key=lambda k: abs(_src_ts[k] - _DIAG_TS))
        _cal_bf = INTRO_FRAMES + _cal_bi + 1
        _cal_ts = _src_ts[_cal_bi]
        _cal_si = INTRO_FRAMES + _cal_bi   # smooth index (full array index)

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
            _cal_fwd0, _cal_palm0 = _ft_hand_errors(_cal_hbn, _cal_pts, _cal_h9, _cal_pnv, _cal_flip)
            _cal_wrist0   = _ft_wrist_err(_cal_fbn, _cal_emp_ik)

            # Step 2: controller = +30° — frame_set FIRST, then override
            bpy.context.scene.frame_set(_cal_bf)
            _cal_ctb.rotation_euler = mathutils.Euler((0.0, math.radians(30.0), 0.0), 'XYZ')
            bpy.context.view_layer.update()
            _cal_fa_mat_p30 = _cal_fb.matrix.copy()
            _cal_twist_p30  = _ft_read_twist(_cal_hbn, _cal_vf, _cal_si, _cal_pts, _cal_h9, _cal_pnv, _cal_flip)
            _cal_fwd_p30, _cal_palm_p30 = _ft_hand_errors(_cal_hbn, _cal_pts, _cal_h9, _cal_pnv, _cal_flip)
            _cal_wrist_p30  = _ft_wrist_err(_cal_fbn, _cal_emp_ik)

            # Step 3: controller = -30° — frame_set FIRST, then override
            bpy.context.scene.frame_set(_cal_bf)
            _cal_ctb.rotation_euler = mathutils.Euler((0.0, math.radians(-30.0), 0.0), 'XYZ')
            bpy.context.view_layer.update()
            _cal_twist_m30  = _ft_read_twist(_cal_hbn, _cal_vf, _cal_si, _cal_pts, _cal_h9, _cal_pnv, _cal_flip)
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
                  f"  selected_enabled={_cal_pass}"
                  f"  failure_reason={_cal_reason}")

            if _cal_side == 'L':
                _ft_v2_enabled_l    = _cal_pass
                _ft_response_gain_l = _cal_rg if _cal_pass else 1.0
            else:
                _ft_v2_enabled_r    = _cal_pass
                _ft_response_gain_r = _cal_rg if _cal_pass else 1.0

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

            # ── Correction track (response_gain-based inversion) ──────
            def _ft_build_correction(pre_list, response_gain):
                raw = []
                for td in pre_list:
                    sgn    = 1.0 if td >= 0 else -1.0
                    excess = max(0.0, abs(td) - HAND_TWIST_SOFT_LIMIT_DEG)
                    desired_delta = sgn * excess
                    if abs(response_gain) > 1e-4:
                        corr = desired_delta / response_gain
                    else:
                        corr = 0.0
                    raw.append(max(-FOREARM_TWIST_MAX_TRANSFER_DEG,
                                   min(FOREARM_TWIST_MAX_TRANSFER_DEG, corr)))
                sm = smooth_scalar_track(raw, window=FOREARM_TWIST_SMOOTH_WINDOW)
                full = [v if v is not None else 0.0 for v in sm] if sm else raw[:]
                for _fi in range(INTRO_FRAMES):
                    if _fi < len(full): full[_fi] = 0.0
                for _fi in range(5):            # 5-frame fade-in from motion start
                    _idx = INTRO_FRAMES + _fi
                    if _idx < len(full): full[_idx] *= (_fi + 1) / 5
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
                _rd_bad = ('SL_ForearmTwist_L', 'SL_ForearmTwist_R',
                           'SL_ForearmTwistV2_L', 'SL_ForearmTwistV2_R')
                if _rc.name in _rd_bad:
                    print(f"  *** FAIL: unexpected Twist constraint present: {_rc.name!r} ***")
    # Twist controller bone / object scan
    _rd_twist_bones = [b.name for b in arm_obj.data.bones
                       if 'ForearmTwist' in b.name or 'TwistCtrl' in b.name]
    _rd_twist_objs  = [o.name for o in bpy.data.objects
                       if 'ForearmTwist' in o.name or 'TwistCtrl' in o.name]
    print(f"[HAND-FOREARM-REST-DIAG] ForearmTwist/TwistCtrl bones: "
          f"{_rd_twist_bones if _rd_twist_bones else '(none — expected)'}")
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

        for _ds, _fbn, _hbn, _pts, _pn, _h9, _flip, _fa_src_list, _emp_ik in [
                ('L', BONE_L_FOREARM, BONE_L_HAND, l_pts, l_pn, lh9,
                 L_PALM_FLIP, _forearm_src_l, emp_lw),
                ('R', BONE_R_FOREARM, BONE_R_HAND, r_pts, r_pn, rh9,
                 R_PALM_FLIP, _forearm_src_r, emp_rw)]:

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
            print(f"  source_forearm_direction_world={_vec_fmt(_fa_src)}")

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
                print(f"  forearm_forward_error_deg={math.degrees(math.acos(_cos)):+.4f}")
            else:
                print(f"  forearm_forward_error_deg=N/A(fa_Y={_fa_Y is not None} fa_src={_fa_src is not None})")

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

        def apply_hand(bone_name, wrist_ik, mid9, pn, flip, flip_dir, valid_first, pts, chains, prev_quats):
            """손뼈 + 손가락 체인 FK 설정.

            effective_pn = -pn if flip else pn — hand bone과 finger roll_pn에 동일 적용.
            hand direction 기준: pts[H_WRIST](Hand pt0) → mid9(Hand pt9) — Pose wrist 미사용.
            wrist_ik: 팔 IK Empty 위치용(Pose 기반), hand rotation에는 사용하지 않음.
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
            for chain in chains:
                first_parent_mat = get_parent_mat(arm_obj, chain[0][0], bone_name, hand_mat)
                p_mat = first_parent_mat
                for (bn, from_idx, to_idx) in chain:
                    from_w = pts[from_idx]
                    to_w   = pts[to_idx]
                    if flip_dir:
                        from_w, to_w = to_w, from_w
                    ok, new_mat = set_bone_dir(arm_obj, bn, from_w, to_w, parent_mat=p_mat, roll_pn=eff_pn)
                    if ok:
                        sign_norm(bn, prev_quats)
                        arm_obj.pose.bones[bn].keyframe_insert('rotation_quaternion', frame=frame)
                        p_mat = new_mat
                    else:
                        p_mat = bone_arma_mat(arm_obj, bn, p_mat)

        apply_hand(BONE_L_HAND, lw, lh9, l_pn, L_PALM_FLIP, FLIP_FINGER_DIR_L, l_first, l_pts, FINGER_CHAINS_L, prev_quats_l)
        apply_hand(BONE_R_HAND, rw, rh9, r_pn, R_PALM_FLIP, FLIP_FINGER_DIR_R, r_first, r_pts, FINGER_CHAINS_R, prev_quats_r)

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
                _frd_si2 = INTRO_FRAMES + _frd_bi2
                _frd_pts2 = [smooth[f'{"l" if _frd_s2=="L" else "r"}_p{j}'][_frd_si2] for j in range(21)]
                _frd_h92  = smooth['lh9' if _frd_s2=='L' else 'rh9'][_frd_si2]
                _frd_pnv2 = smooth['l_pn' if _frd_s2=='L' else 'r_pn'][_frd_si2]
                # evaluated forearm twist at _DIAG_TS (controller=0 baseline)
                _frd_ctb2 = arm_obj.pose.bones.get(_frd_ctbn2)
                _frd_fb2  = arm_obj.pose.bones.get(_frd_fbn2)
                _frd_hb2  = arm_obj.pose.bones.get(_frd_hbn2)
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
        for _wrd_side, _wrd_fbn, _wrd_hbn, _wrd_fa_src, _wrd_pn_tr, _wrd_pn_flip in [
                ('L', BONE_L_FOREARM, BONE_L_HAND, _forearm_src_l, _pn_l_track, L_PALM_FLIP),
                ('R', BONE_R_FOREARM, BONE_R_HAND, _forearm_src_r, _pn_r_track, R_PALM_FLIP)]:
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
            print(f"  source_forearm_dir={_vs(_wrd_src_fa)}")
            print(f"  source_hand_dir={_vs(_wrd_src_hd)}")
            print(f"  source_wrist_angle_deg={_fs(_wrd_src_wang)}")
            print(f"  actual_forearm_dir={_vs(_wrd_act_fa)}")
            print(f"  actual_hand_dir={_vs(_wrd_act_hd)}")
            print(f"  actual_wrist_angle_deg={_wrd_act_wang:.2f}")
            print(f"  wrist_angle_error_deg={_fs(abs(_wrd_act_wang - _wrd_src_wang) if _wrd_src_wang is not None else None)}")
            print(f"  hand_local_quaternion=({_wrd_hq.w:.4f},{_wrd_hq.x:.4f},{_wrd_hq.y:.4f},{_wrd_hq.z:.4f})")
            print(f"  hand_local_total_angle_deg={_wrd_totdeg:.2f}")
            print(f"  hand_local_twist_deg={_wrd_tdeg:.2f}")
            print(f"  hand_local_swing_deg={_wrd_sdeg:.2f}")
            print(f"  effective_palm_normal={_vs(_wrd_eff_pn)}")
            print(f"  actual_hand_world_z={_vs(_wrd_act_hz)}")
            print(f"  palm_alignment_error_deg={_fs(_wrd_palm_err)}")
        # [WRIST-RELATIVE-WINDOW] per side
        for _wrw_side, _wrw_fbn, _wrw_hbn, _wrw_fa_src, _wrw_pn_tr, _wrw_pn_flip in [
                ('L', BONE_L_FOREARM, BONE_L_HAND, _forearm_src_l, _pn_l_track, L_PALM_FLIP),
                ('R', BONE_R_FOREARM, BONE_R_HAND, _forearm_src_r, _pn_r_track, R_PALM_FLIP)]:
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
                _wi_si = INTRO_FRAMES + _wi
                _wi_sp0 = smooth[_wrw_p0k][_wi_si] if _wi_si < len(smooth[_wrw_p0k]) else None
                _wi_sh9 = smooth[_wrw_h9k][_wi_si] if _wi_si < len(smooth[_wrw_h9k]) else None
                _wi_src_wang = None
                if _wi_fa_dir is not None and _wi_sp0 is not None and _wi_sh9 is not None:
                    _hv2 = _wi_sh9 - _wi_sp0
                    _wi_hd2 = _hv2.normalized() if _hv2.length > 1e-6 else None
                    if _wi_hd2:
                        try: _wi_src_wang = math.degrees(math.acos(max(-1.0, min(1.0, _wi_fa_dir.dot(_wi_hd2)))))
                        except: _wi_src_wang = 0.0
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
                _wi_aw_s = f"{_wi_act_wang:.1f}" if _wi_act_wang is not None else "None"
                _wi_pe_s = f"{_wi_palm_err_w:.1f}" if _wi_palm_err_w is not None else "None"
                _ctr = " ← center" if _wi == _wrd_bi else ""
                print(f"  ts={_wi_ts}  frame={_wi_bf}"
                      f"  twist_deg={_wi_twist_raw:+.1f}"
                      f"  twist_delta={_wi_twist_delta:+.1f}"
                      f"  src_wrist={_wi_sw_s}"
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

# ── 애니메이션 결과 .blend 저장 ──────────────────────────────────────────────
_out_blend = _os.path.join(_THIS_DIR, "s04_out", "인사_multiview_s01.blend")
bpy.ops.wm.save_as_mainfile(filepath=_out_blend)
print(f"[blend saved] {_out_blend}")
