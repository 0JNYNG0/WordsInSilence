"""
Tools/diag_hand_basis2.py
────────────────────────────────────────────────────────────────
Hand bone basis 진단 (2차) — 팔 자세 포함한 유효 단일 프레임 기준

1. HasPose + LEFT 유효 + RIGHT 유효인 원본 프레임 하나 선택
2. 동일 프레임에서 팔 IK (blend_animate.py 와 동일 방식) 적용
3. A/B/C/D 방식으로 Hand bone만 다르게 설정 (손가락 = Rest Pose 유지)
4. HandTest2_A/B/C/D_WithArmPose 액션 저장

실행: Blender Python 콘솔에서
    exec(open(r"C:\\MyUnityProjects\\WordsInSilence\\Tools\\diag_hand_basis2.py").read())

수정 금지 항목:
  - blend_animate.py 본체
  - L_PALM_FLIP / R_PALM_FLIP 확정
  - 전체 88프레임 액션 생성

────────────────────────────────────────────────────────────────
"""

import bpy
import json
import math
import mathutils

# ================================================================
# 설정 — blend_animate.py 와 동일
# ================================================================
JSON_PATH        = r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\Recordings\감사합니다.json"
BODY_SCALE       = 1.0
DEPTH_SCALE      = 0.0
PALM_Z_SCALE     = 1.0
FORWARD_OFFSET   = 0.15
HAND_DEPTH_GAIN  = 1.0
MIRROR_X         = True
SWAP_HAND_LABELS = True   # mp_right → 캐릭터 LEFT, mp_left → 캐릭터 RIGHT

# 선택 전략: 'first' = 양손 유효 첫 프레임, 'nth' = N번째 양손 유효 프레임
FRAME_SELECT_STRATEGY = 'first'
FRAME_SELECT_N        = 0   # 'nth' 일 때 사용

# ================================================================
# MediaPipe 인덱스
# ================================================================
P_L_SHOULDER = 11; P_R_SHOULDER = 12
P_L_WRIST    = 15; P_R_WRIST    = 16

H_WRIST     = 0
H_INDEX_MCP = 5
H_MID_MCP   = 9
H_PINK_MCP  = 17

# ================================================================
# Mixamo 뼈 이름
# ================================================================
BONE_L_ARM     = 'mixamorig:LeftArm'
BONE_L_FOREARM = 'mixamorig:LeftForeArm'
BONE_L_HAND    = 'mixamorig:LeftHand'
BONE_R_ARM     = 'mixamorig:RightArm'
BONE_R_FOREARM = 'mixamorig:RightForeArm'
BONE_R_HAND    = 'mixamorig:RightHand'

REST_BONES_L = {
    'hand':    'mixamorig:LeftHand',
    'index1':  'mixamorig:LeftHandIndex1',
    'middle1': 'mixamorig:LeftHandMiddle1',
    'pinky1':  'mixamorig:LeftHandPinky1',
}
REST_BONES_R = {
    'hand':    'mixamorig:RightHand',
    'index1':  'mixamorig:RightHandIndex1',
    'middle1': 'mixamorig:RightHandMiddle1',
    'pinky1':  'mixamorig:RightHandPinky1',
}


# ================================================================
# 유틸 — blend_animate.py 복사
# ================================================================
def get_pt(pts, idx):
    if not pts or idx >= len(pts):
        return None
    p = pts[idx]
    return (p.get('x', 0), p.get('y', 0), p.get('z', 0)) if isinstance(p, dict) else None


def to_blender(x, y, z, cx, cy, sw, origin, char_sw, depth_scale=None):
    """Pose landmark → Blender world (blend_animate.py 와 동일)."""
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
    """MediaPipe Hand landmark → Blender world (blend_animate.py 와 동일)."""
    sc = char_sw * BODY_SCALE
    xs = 1 if MIRROR_X else -1
    return origin + mathutils.Vector((
        xs * (x - cx) / sw * sc,
        z / sw * sc * HAND_DEPTH_GAIN - FORWARD_OFFSET,
        -(y - cy) / sw * sc,
    ))


def det3(c0, c1, c2):
    return c0.dot(c1.cross(c2))


def angle_deg(a, b):
    try:
        return math.degrees(a.angle(b))
    except Exception:
        return 999.0


# ================================================================
# IK 유틸 — blend_animate.py 복사
# ================================================================
def make_empty(name, col):
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
        # 기존 애니메이션 제거 (이전 액션 간섭 방지)
        obj.animation_data_clear()
        obj.hide_viewport = True
        return obj
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_size = 0.04
    obj.hide_viewport = True
    col.objects.link(obj)
    return obj


def add_ik(arm_obj, bone_name, target, chain_count):
    b = arm_obj.pose.bones.get(bone_name)
    if not b:
        print(f"[경고] 뼈 없음: {bone_name}"); return
    # 기존 IK 제거 후 재추가
    for c in list(b.constraints):
        if c.type == 'IK':
            b.constraints.remove(c)
    c = b.constraints.new('IK')
    c.target = target
    c.chain_count = chain_count
    c.use_rotation = False


# ================================================================
# Rest basis — bone head_local 위치에서 계산
# ================================================================
def rest_hand_basis(arm, bones_dict):
    def hl(name):
        bd = arm.data.bones.get(name)
        return mathutils.Vector(bd.head_local) if bd else None

    h0 = hl(bones_dict['hand'])
    hi = hl(bones_dict['index1'])
    hm = hl(bones_dict['middle1'])
    hp = hl(bones_dict['pinky1'])

    if not all([h0, hi, hm, hp]):
        return None, None, None

    fwd_raw = (hm - h0)
    if fwd_raw.length < 1e-6:
        return None, None, None
    fwd = fwd_raw.normalized()

    lat_raw = (hi - hp)
    if lat_raw.length < 1e-6:
        return None, None, None

    norm_raw = fwd.cross(lat_raw.normalized())
    if norm_raw.length < 1e-6:
        return None, None, None
    norm = norm_raw.normalized()
    lat  = norm.cross(fwd).normalized()

    return fwd, lat, norm


def get_rest_norm_and_bone_z(arm, bones_dict, label):
    """Rest norm (head_local) 와 bone matrix_local col[2] 비교 출력. rest_norm 반환."""
    fwd, lat, norm = rest_hand_basis(arm, bones_dict)
    if fwd is None:
        return None

    hand_bn = bones_dict['hand']
    bd = arm.data.bones.get(hand_bn)
    rest_z = bd.matrix_local.to_3x3().col[2].normalized() if bd else None

    if rest_z:
        try:
            ang = math.degrees(norm.angle(rest_z))
        except Exception:
            ang = 999.0
        print(f"  [{label}] computed_norm ∠ bone_Z : {ang:.1f}°  "
              f"({'동일 방향' if ang < 90 else '반대 방향'})")
    return norm   # armature-space


# ================================================================
# Target basis — MediaPipe 랜드마크에서 계산
# ================================================================
def target_hand_basis(pts21, pn_sign=1.0):
    wrist     = pts21[H_WRIST]
    index_mcp = pts21[H_INDEX_MCP]
    mid_mcp   = pts21[H_MID_MCP]
    pink_mcp  = pts21[H_PINK_MCP]

    fwd_raw = mid_mcp - wrist
    if fwd_raw.length < 1e-6:
        return None, None, None
    fwd = fwd_raw.normalized()

    lat_raw = index_mcp - pink_mcp
    if lat_raw.length < 1e-6:
        return None, None, None

    norm_raw = fwd.cross(lat_raw.normalized()) * pn_sign
    if norm_raw.length < 1e-6:
        return None, None, None
    norm = norm_raw.normalized()
    lat  = norm.cross(fwd).normalized()

    return fwd, lat, norm


# ================================================================
# 접근법 A/B — set_hand_rotation (blend_animate.py 복사)
# ================================================================
def set_hand_rotation_diag(arm_obj, bone_name, wrist_w, mid9_w, palm_n_w, flip_palm):
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        return False

    hand_dir = (mid9_w - wrist_w)
    if hand_dir.length < 1e-6:
        return False
    hand_dir = hand_dir.normalized()

    a_inv = arm_obj.matrix_world.to_3x3().inverted()
    dir_a = (a_inv @ hand_dir).normalized()

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
        if flip_palm:
            pn_a = -pn_a
        pn_perp = pn_a - pn_a.dot(ny) * ny
        nz_perp = nz  - nz.dot(ny)  * ny
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
# 접근법 C/D — set_hand_by_basis (직접 기저 매핑)
# ================================================================
def set_hand_by_basis(arm_obj, bone_name, fwd_w, lat_w, norm_w):
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        return False

    a_inv = arm_obj.matrix_world.to_3x3().inverted()

    ny = (a_inv @ fwd_w).normalized()
    nz = (a_inv @ norm_w)
    nz = (nz - nz.dot(ny) * ny)
    if nz.length < 1e-6:
        return False
    nz = nz.normalized()
    nx = ny.cross(nz).normalized()   # X = Y × Z (오른손 좌표계)

    desired_a = mathutils.Matrix([
        [nx.x, ny.x, nz.x],
        [nx.y, ny.y, nz.y],
        [nx.z, ny.z, nz.z],
    ])

    rest = bd.matrix_local.to_3x3()
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
# 진단 리포트
# ================================================================
def report_hand(arm_obj, bone_name, target_fwd_w, target_lat_w, target_norm_w, label):
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        print(f"  [{label}] 뼈 없음: {bone_name}")
        return

    bpy.context.view_layer.update()   # ← 설정 후 즉시 갱신 (stale 방지)

    bone_w_mat = (arm_obj.matrix_world @ b.matrix).to_3x3()
    actual_x = bone_w_mat.col[0].normalized()
    actual_y = bone_w_mat.col[1].normalized()
    actual_z = bone_w_mat.col[2].normalized()

    tf = target_fwd_w.normalized()
    tl = target_lat_w.normalized()
    tn = target_norm_w.normalized()

    fwd_err  = angle_deg(actual_y, tf)
    norm_err = angle_deg(actual_z, tn)
    lat_err  = angle_deg(actual_x, tl)
    d        = det3(actual_x, actual_y, actual_z)

    print(f"\n  [{label}] {bone_name}")
    print(f"    actual  X: {actual_x.x:+.3f} {actual_x.y:+.3f} {actual_x.z:+.3f}")
    print(f"    actual  Y: {actual_y.x:+.3f} {actual_y.y:+.3f} {actual_y.z:+.3f}  (forward)")
    print(f"    actual  Z: {actual_z.x:+.3f} {actual_z.y:+.3f} {actual_z.z:+.3f}  (normal)")
    print(f"    target fwd: {tf.x:+.3f} {tf.y:+.3f} {tf.z:+.3f}")
    print(f"    target lat: {tl.x:+.3f} {tl.y:+.3f} {tl.z:+.3f}")
    print(f"    target nm : {tn.x:+.3f} {tn.y:+.3f} {tn.z:+.3f}")
    print(f"    forward_err : {fwd_err:6.1f}°")
    print(f"    normal_err  : {norm_err:6.1f}°  ← 핵심")
    print(f"    lateral_err : {lat_err:6.1f}°")
    print(f"    actual det  : {d:+.4f}  ({'RIGHT-HANDED' if d > 0 else 'LEFT-HANDED'})")


# ================================================================
# 액션 저장
# ================================================================
def save_action_current_pose(arm_obj, action_name, frame=1):
    """현재 포즈 상태를 지정 액션에 단일 프레임으로 저장."""
    old = bpy.data.actions.get(action_name)
    if old:
        bpy.data.actions.remove(old)

    act = bpy.data.actions.new(action_name)
    arm_obj.animation_data_create()
    arm_obj.animation_data.action = act

    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.keyframe_insert('rotation_quaternion', frame=frame)

    print(f"  → 액션 저장: '{action_name}'")
    return act


def reset_pose(arm_obj):
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = mathutils.Quaternion()
        pb.location  = mathutils.Vector((0, 0, 0))
        pb.scale     = mathutils.Vector((1, 1, 1))


# ================================================================
# 메인
# ================================================================
def main():
    # ── 아마추어 ────────────────────────────────────────────────
    arm_obj = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if not arm_obj:
        print("[오류] 아마추어 없음"); return
    print(f"[아마추어] {arm_obj.name}")

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    # ── JSON 로드 ────────────────────────────────────────────────
    with open(JSON_PATH, encoding='utf-8') as f:
        recording = json.load(f)
    frames_data = recording.get('Frames', [])
    print(f"[JSON] 총 {len(frames_data)}개 프레임")

    # ── 캐릭터 스케일 (어깨 기준) ─────────────────────────────
    lb = arm_obj.pose.bones.get(BONE_L_ARM)
    rb = arm_obj.pose.bones.get(BONE_R_ARM)
    if lb and rb:
        l_sh_w      = arm_obj.matrix_world @ lb.head
        r_sh_w      = arm_obj.matrix_world @ rb.head
        char_origin = (l_sh_w + r_sh_w) / 2
        char_sw     = (l_sh_w - r_sh_w).length
    else:
        char_origin = mathutils.Vector((0, 0, 0)); char_sw = 1.5
    print(f"[캐릭터] 어깨 중심: {char_origin}  너비: {char_sw:.3f}")

    # ── 양손 유효 프레임 탐색 ────────────────────────────────────
    # 원본 frames_data 인덱스를 보존하여 LEFT·RIGHT가 항상 동일 프레임임을 보장
    both_valid_indices = []   # (fd_idx, cx, cy, sw, ch_l_pts, ch_r_pts, pose_pts)

    for fd_idx, fd in enumerate(frames_data):
        if not fd.get('HasPose', False):
            continue
        pose_pts  = fd.get('PosePoints') or []
        mp_l_pts  = fd.get('LeftHandPoints') or []
        mp_r_pts  = fd.get('RightHandPoints') or []
        has_mp_l  = fd.get('HasLeftHand', False)
        has_mp_r  = fd.get('HasRightHand', False)

        if SWAP_HAND_LABELS:
            ch_l_pts, has_ch_l = mp_r_pts, has_mp_r
            ch_r_pts, has_ch_r = mp_l_pts, has_mp_l
        else:
            ch_l_pts, has_ch_l = mp_l_pts, has_mp_l
            ch_r_pts, has_ch_r = mp_r_pts, has_mp_r

        if not (has_ch_l and has_ch_r):
            continue

        ls = get_pt(pose_pts, P_L_SHOULDER)
        rs = get_pt(pose_pts, P_R_SHOULDER)
        if not ls or not rs:
            continue
        cx = (ls[0] + rs[0]) / 2
        cy = (ls[1] + rs[1]) / 2
        sw = math.sqrt((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2)
        if sw < 1e-4:
            continue

        both_valid_indices.append((fd_idx, cx, cy, sw, ch_l_pts, ch_r_pts, pose_pts))

    print(f"[탐색] 양손 유효 프레임 수: {len(both_valid_indices)}")
    if not both_valid_indices:
        print("[오류] 양손이 동시에 유효한 프레임 없음"); return

    # 프레임 선택
    if FRAME_SELECT_STRATEGY == 'nth':
        sel = min(FRAME_SELECT_N, len(both_valid_indices) - 1)
    else:
        sel = 0
    fd_idx, cx, cy, sw, ch_l_pts, ch_r_pts, pose_pts = both_valid_indices[sel]

    print(f"\n[선택된 프레임]")
    print(f"  원본 JSON frame index : {fd_idx}")
    print(f"  LEFT Hand 원본 index  : {fd_idx}  (동일 프레임)")
    print(f"  RIGHT Hand 원본 index : {fd_idx}  (동일 프레임)")
    print(f"  양손 동일 프레임      : True (보장)")

    # ── 해당 프레임 랜드마크 변환 ────────────────────────────────
    def chp(idx, pts):
        pt = get_pt(pts, idx)
        return convert_hand_pt(pt[0], pt[1], pt[2], cx, cy, sw, char_origin, char_sw) if pt else None

    def bp(pose_idx):
        pt = get_pt(pose_pts, pose_idx)
        return to_blender(*pt, cx, cy, sw, char_origin, char_sw) if pt else None

    # Pose 손목 (팔 IK 타겟)
    pose_lw = bp(P_L_WRIST)
    pose_rw = bp(P_R_WRIST)

    # Hand 21개 랜드마크 (Blender world)
    l_pts21 = [chp(i, ch_l_pts) for i in range(21)]
    r_pts21 = [chp(i, ch_r_pts) for i in range(21)]

    if any(p is None for p in l_pts21) or any(p is None for p in r_pts21):
        print("[오류] 랜드마크 변환 실패"); return

    # Hand forward (Hand MCP 기반)
    lh9 = l_pts21[H_MID_MCP]   # Left Middle MCP (Blender world)
    rh9 = r_pts21[H_MID_MCP]   # Right Middle MCP

    # palm_n_w (원본 공식 — blend_animate.py 와 동일)
    lw3 = l_pts21[H_WRIST]; li3 = l_pts21[H_INDEX_MCP]; lp3 = l_pts21[H_PINK_MCP]
    rw3 = r_pts21[H_WRIST]; ri3 = r_pts21[H_INDEX_MCP]; rp3 = r_pts21[H_PINK_MCP]

    l_pn_raw = (li3 - lw3).cross(lp3 - lw3)
    r_pn_raw = (ri3 - rw3).cross(rp3 - rw3)

    if l_pn_raw.length < 1e-6 or r_pn_raw.length < 1e-6:
        print("[오류] palm_n_w 길이 0"); return
    l_pn = l_pn_raw.normalized()
    r_pn = r_pn_raw.normalized()

    # pose_lw → Hand 21pts[0] 차이 보고 (혼합 여부 확인)
    print(f"\n[Pose·Hand 손목 불일치]")
    if pose_lw:
        diff_l = (lw3 - pose_lw).length
        print(f"  LEFT  Pose_wrist vs Hand_wrist[0]: {diff_l:.4f} m")
    if pose_rw:
        diff_r = (rw3 - pose_rw).length
        print(f"  RIGHT Pose_wrist vs Hand_wrist[0]: {diff_r:.4f} m")

    # ── 팔 IK 준비 (blend_animate.py 와 동일 방식) ────────────────
    # IK Empty 생성/재사용
    col = bpy.data.collections.get("IK_Targets")
    if not col:
        col = bpy.data.collections.new("IK_Targets")
        bpy.context.scene.collection.children.link(col)
    emp_lw = make_empty("IK_L_Wrist", col)
    emp_rw = make_empty("IK_R_Wrist", col)

    # IK 컨스트레인트 설정 (ForeArm, chain=2)
    add_ik(arm_obj, BONE_L_FOREARM, emp_lw, chain_count=2)
    add_ik(arm_obj, BONE_R_FOREARM, emp_rw, chain_count=2)

    # Hand / Finger 뼈: FK 모드 (컨스트레인트 제거)
    finger_bones = (
        [BONE_L_HAND, BONE_R_HAND]
        + [bn for prefix in ['LeftHandThumb', 'LeftHandIndex', 'LeftHandMiddle',
                              'LeftHandRing', 'LeftHandPinky',
                              'RightHandThumb', 'RightHandIndex', 'RightHandMiddle',
                              'RightHandRing', 'RightHandPinky']
           for bn in [f'mixamorig:{prefix}{n}' for n in [1, 2, 3]]]
    )
    for bn in finger_bones:
        b = arm_obj.pose.bones.get(bn)
        if b:
            for c in list(b.constraints):
                b.constraints.remove(c)
            b.rotation_mode = 'QUATERNION'

    # ── 팔 IK 포즈 적용 ─────────────────────────────────────────
    # 모든 뼈 초기화 후 팔 IK 평가
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='POSE')
    reset_pose(arm_obj)

    # IK Empty 위치 설정 (pose_lw 없으면 Hand wrist 위치로 대체)
    emp_lw.location = pose_lw if pose_lw else lw3
    emp_rw.location = pose_rw if pose_rw else rw3
    emp_lw.hide_viewport = False   # IK 평가를 위해 임시 표시
    emp_rw.hide_viewport = False

    # Depsgraph 갱신 → 팔 IK 풀기
    bpy.context.view_layer.update()

    # 현재 팔 자세를 FK 회전값으로 고정 (Hand A/B/C/D 이후에도 유지)
    # → 팔 뼈 rotation_quaternion을 현재 IK 평가 결과로 스냅
    for arm_bn in [BONE_L_ARM, BONE_L_FOREARM, BONE_R_ARM, BONE_R_FOREARM]:
        pb = arm_obj.pose.bones.get(arm_bn)
        if pb:
            # b.matrix 에서 회전 추출 → rotation_quaternion 에 고정
            # (IK 컨스트레인트 제거 후에도 자세 유지)
            pass   # IK 컨스트레인트를 유지한 채 Hand만 FK로 설정

    # ← IK 컨스트레인트는 그대로 유지: 각 접근법 loop 에서 Hand만 변경
    # Empty 위치가 고정이므로 IK 결과가 바뀌지 않음

    # 팔 자세 확인 출력
    bpy.context.view_layer.update()
    lb_pose = arm_obj.pose.bones.get(BONE_L_FOREARM)
    rb_pose = arm_obj.pose.bones.get(BONE_R_FOREARM)
    print(f"\n[팔 IK 적용 확인]")
    if lb_pose:
        lf_tail_w = arm_obj.matrix_world @ lb_pose.tail
        print(f"  LEFT  ForeArm tail (= IK 손목): {lf_tail_w.x:.3f} {lf_tail_w.y:.3f} {lf_tail_w.z:.3f}")
        print(f"  LEFT  IK empty 목표           : {emp_lw.location.x:.3f} {emp_lw.location.y:.3f} {emp_lw.location.z:.3f}")
    if rb_pose:
        rf_tail_w = arm_obj.matrix_world @ rb_pose.tail
        print(f"  RIGHT ForeArm tail (= IK 손목): {rf_tail_w.x:.3f} {rf_tail_w.y:.3f} {rf_tail_w.z:.3f}")
        print(f"  RIGHT IK empty 목표           : {emp_rw.location.x:.3f} {emp_rw.location.y:.3f} {emp_rw.location.z:.3f}")

    # ── Rest Basis 출력 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("REST BASIS")
    print("="*60)
    rest_norm_l = get_rest_norm_and_bone_z(arm_obj, REST_BONES_L, "LEFT")
    rest_norm_r = get_rest_norm_and_bone_z(arm_obj, REST_BONES_R, "RIGHT")

    # auto-sign 판정용 dot product 사전 계산
    a_inv_global = arm_obj.matrix_world.to_3x3().inverted()
    tf_l, tl_l, tn_l_plus1 = target_hand_basis(l_pts21, pn_sign=+1.0)
    tf_r, tl_r, tn_r_plus1 = target_hand_basis(r_pts21, pn_sign=+1.0)

    def auto_pn_sign(tn_plus1_w, rest_norm_a, side_label):
        """target_norm_pn+1 (world) 와 rest_norm (armature-space) 내적으로 pn_sign 결정."""
        if rest_norm_a is None or tn_plus1_w is None:
            return 1.0
        tn_a = (a_inv_global @ tn_plus1_w).normalized()
        dot  = tn_a.dot(rest_norm_a)
        sign = 1.0 if dot >= 0 else -1.0
        print(f"  [{side_label} AUTO] tn·rest_norm dot={dot:+.3f}  → pn_sign={sign:+.0f}")
        return sign

    # ── 접근법 A/B/C/D 반복 ─────────────────────────────────────
    # (label, flip_palm_L, flip_palm_R, use_basis, auto_sign)
    approaches = [
        ('A', False, False, False, False),
        ('B', True,  True,  False, False),
        ('C', None,  None,  True,  False),   # pn_sign=+1 고정
        ('D', None,  None,  True,  True ),   # pn_sign=auto
    ]

    print(f"\n{'='*60}")
    print("NOTE: 손가락 = Rest Pose 유지 (비교 목적)")
    print("      팔 자세 = IK 적용 후 고정 (모든 접근법 동일)")
    print(f"{'='*60}")

    for approach_label, flip_l, flip_r, use_basis, do_auto in approaches:
        pn_sign_l = 1.0   # 기본값 (use_basis=False 분기 리포트에서 사용)
        pn_sign_r = 1.0

        print(f"\n{'='*60}")
        print(f"접근법 {approach_label}", end="")
        if not use_basis:
            print(f": set_hand_rotation  flip_L={flip_l} flip_R={flip_r}")
        else:
            sign_mode = "auto" if do_auto else "pn_sign=+1"
            print(f": set_hand_by_basis  pn_sign={sign_mode}")
        print("="*60)

        # 팔 자세는 변경하지 않고 Hand 뼈만 초기화 (Rest)
        for hand_bn in [BONE_L_HAND, BONE_R_HAND]:
            pb = arm_obj.pose.bones.get(hand_bn)
            if pb:
                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = mathutils.Quaternion()
        bpy.context.view_layer.update()   # IK 재평가 (팔 자세 재확인)

        # LEFT Hand 설정
        if tf_l is not None:
            if use_basis:
                if do_auto:
                    pn_sign_l = auto_pn_sign(tn_l_plus1, rest_norm_l, "LEFT")
                else:
                    pn_sign_l = 1.0
                tf_use_l, tl_use_l, tn_use_l = target_hand_basis(l_pts21, pn_sign=pn_sign_l)
                if tf_use_l is not None:
                    set_hand_by_basis(arm_obj, BONE_L_HAND, tf_use_l, tl_use_l, tn_use_l)
            else:
                set_hand_rotation_diag(arm_obj, BONE_L_HAND, pose_lw or lw3, lh9, l_pn, flip_l)

        # RIGHT Hand 설정
        if tf_r is not None:
            if use_basis:
                if do_auto:
                    pn_sign_r = auto_pn_sign(tn_r_plus1, rest_norm_r, "RIGHT")
                else:
                    pn_sign_r = 1.0
                tf_use_r, tl_use_r, tn_use_r = target_hand_basis(r_pts21, pn_sign=pn_sign_r)
                if tf_use_r is not None:
                    set_hand_by_basis(arm_obj, BONE_R_HAND, tf_use_r, tl_use_r, tn_use_r)
            else:
                set_hand_rotation_diag(arm_obj, BONE_R_HAND, pose_rw or rw3, rh9, r_pn, flip_r)

        bpy.context.view_layer.update()   # Hand 회전 반영

        # 리포트
        if tf_l is not None:
            tn_l = target_hand_basis(l_pts21, pn_sign=pn_sign_l if use_basis else 1.0)[2]
            tl_l_use = target_hand_basis(l_pts21, pn_sign=pn_sign_l if use_basis else 1.0)[1]
            if tn_l:
                report_hand(arm_obj, BONE_L_HAND, tf_l, tl_l_use, tn_l,
                            f"{approach_label} LEFT")

        if tf_r is not None:
            tn_r = target_hand_basis(r_pts21, pn_sign=pn_sign_r if use_basis else 1.0)[2]
            tl_r_use = target_hand_basis(r_pts21, pn_sign=pn_sign_r if use_basis else 1.0)[1]
            if tn_r:
                report_hand(arm_obj, BONE_R_HAND, tf_r, tl_r_use, tn_r,
                            f"{approach_label} RIGHT")

        # 액션 저장
        action_name = f"HandTest2_{approach_label}_WithArmPose"
        save_action_current_pose(arm_obj, action_name, frame=1)

    # ── 마무리 ───────────────────────────────────────────────────
    # 첫 액션(A)으로 복귀
    act_a = bpy.data.actions.get('HandTest2_A_WithArmPose')
    if act_a and arm_obj.animation_data:
        arm_obj.animation_data.action = act_a

    emp_lw.hide_viewport = True
    emp_rw.hide_viewport = True

    print(f"\n{'='*60}")
    print("진단 완료.")
    print("Blender 액션 드롭다운: HandTest2_A/B/C/D_WithArmPose")
    print()
    print("  A: set_hand_rotation  flip_L=False flip_R=False  (현재 코드)")
    print("  B: set_hand_rotation  flip_L=True  flip_R=True")
    print("  C: set_hand_by_basis  pn_sign=+1 고정")
    print("  D: set_hand_by_basis  pn_sign=auto (rest_norm 내적 판정)")
    print()
    print("확인 기준:")
    print("  - 팔 자세가 모든 액션에서 동일한지")
    print("  - 손목이 팔과 자연스럽게 이어지는 접근법")
    print("  - 손바닥이 영상의 해당 프레임과 일치하는 접근법")
    print(f"{'='*60}")


# ── pn_sign 변수 scope 안전 처리 ─────────────────────────────────
# approaches 루프 내 pn_sign_l/r 이 use_basis=False 분기에서 미정의될 수 있음
# → 1.0 기본값으로 초기화 보장
def _patch_local_vars():
    pass   # 아래 main() 에서 pn_sign_l/r 을 루프 시작 시 1.0 으로 초기화


main()
