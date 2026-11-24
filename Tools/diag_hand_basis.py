"""
Tools/diag_hand_basis.py
────────────────────────────────────────────────────────────────
Hand bone basis 진단 + 4가지 접근법(A/B/C/D) 단일 프레임 테스트

실행: Blender Python 콘솔에서
    exec(open(r"C:\\MyUnityProjects\\WordsInSilence\\Tools\\diag_hand_basis.py").read())

출력:
  - Rest basis (bone head_local 기반)
  - Target basis (MediaPipe 랜드마크 기반)
  - 각 접근법 적용 후 Forward / Normal / Lateral 오차
  - 기저 행렬 determinant
  - 4개 Blender 액션 저장: HandTest_A / HandTest_B / HandTest_C / HandTest_D

수정 금지:
  - SWAP_HAND_LABELS, MIRROR_X, depth conversion, 손가락 FK, 팔 IK
  - set_hand_rotation() 외부에서 호출되는 구조

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

# 분석할 프레임 (유효 손 프레임 중 n번째)
TEST_FRAME_IDX   = 0

# ================================================================
# MediaPipe 인덱스
# ================================================================
P_L_SHOULDER = 11; P_R_SHOULDER = 12

H_WRIST     = 0
H_INDEX_MCP = 5
H_MID_MCP   = 9
H_PINK_MCP  = 17

# ================================================================
# Mixamo 뼈 이름
# ================================================================
BONE_L_HAND   = 'mixamorig:LeftHand'
BONE_R_HAND   = 'mixamorig:RightHand'

# rest basis 계산에 사용할 자식 뼈 (head_local 위치 이용)
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


def convert_hand_pt(x, y, z, cx, cy, sw, origin, char_sw):
    sc = char_sw * BODY_SCALE
    xs = 1 if MIRROR_X else -1
    return origin + mathutils.Vector((
        xs * (x - cx) / sw * sc,
        z / sw * sc * HAND_DEPTH_GAIN - FORWARD_OFFSET,
        -(y - cy) / sw * sc,
    ))


def det3(c0, c1, c2):
    """3×3 행렬의 determinant. c0/c1/c2 = 열 벡터."""
    return c0.dot(c1.cross(c2))


def angle_deg(a, b):
    """두 벡터 사이 각도 (도)."""
    try:
        return math.degrees(a.angle(b))
    except Exception:
        return 999.0


# ================================================================
# Rest basis — bone head_local 위치에서 계산
# ================================================================
def rest_hand_basis(arm, bones_dict):
    """
    레스트 포즈 Hand 뼈의 기저 벡터를 bone head_local 위치에서 계산.
    모두 Armature-space (= bone local rest = matrix_local 기준).

    반환: (fwd, lat, norm) 각각 normalized Vector3
          fwd  = Middle1.head → Hand.head 방향의 역 (손목→중지 MCP)
          lat  = Index1.head  → Pinky1.head  방향의 역 (새끼→검지, i.e. 검지 쪽이 +)
          norm = fwd × lat_raw (오른손 좌표계 손바닥 법선 후보)
    """
    def hl(name):
        bd = arm.data.bones.get(name)
        return mathutils.Vector(bd.head_local) if bd else None

    h0 = hl(bones_dict['hand'])
    hi = hl(bones_dict['index1'])
    hm = hl(bones_dict['middle1'])
    hp = hl(bones_dict['pinky1'])

    if not all([h0, hi, hm, hp]):
        missing = [k for k, v in [('hand', h0), ('index1', hi), ('middle1', hm), ('pinky1', hp)] if not v]
        print(f"  [오류] rest_hand_basis: 뼈 없음 {missing}")
        return None, None, None

    # forward: 손목(Hand.head) → 중지 MCP(Middle1.head)
    fwd_raw = (hm - h0)
    if fwd_raw.length < 1e-6:
        print("  [오류] rest forward 길이 0 (Middle1 == Hand)")
        return None, None, None
    fwd = fwd_raw.normalized()

    # lateral_raw: 새끼 MCP(Pinky1) → 검지 MCP(Index1)  = 검지 쪽이 양수
    lat_raw = (hi - hp)
    if lat_raw.length < 1e-6:
        print("  [오류] rest lateral 길이 0 (Index1 == Pinky1)")
        return None, None, None

    # 손바닥 법선 후보: fwd × lat_raw (정규화 전)
    norm_raw = fwd.cross(lat_raw.normalized())
    if norm_raw.length < 1e-6:
        print("  [오류] rest norm 길이 0 (forward ∥ lateral)")
        return None, None, None
    norm = norm_raw.normalized()

    # lat 재직교화: norm × fwd
    lat = norm.cross(fwd).normalized()

    return fwd, lat, norm


# ================================================================
# Target basis — MediaPipe 랜드마크에서 계산 (Blender world)
# ================================================================
def target_hand_basis(pts21, pn_sign=1.0):
    """
    MediaPipe 21개 손 랜드마크(Blender world 좌표)에서 목표 기저 벡터 계산.

    pn_sign: +1 or -1 (손바닥 법선 방향 반전 여부)

    반환: (fwd, lat, norm) — Blender world 좌표계의 normalized 벡터
    """
    wrist     = pts21[H_WRIST]
    index_mcp = pts21[H_INDEX_MCP]
    mid_mcp   = pts21[H_MID_MCP]
    pink_mcp  = pts21[H_PINK_MCP]

    fwd_raw = mid_mcp - wrist
    if fwd_raw.length < 1e-6:
        return None, None, None
    fwd = fwd_raw.normalized()

    # 새끼 → 검지 방향 (검지 쪽이 양수)
    lat_raw = index_mcp - pink_mcp
    if lat_raw.length < 1e-6:
        return None, None, None

    # 손바닥 법선: fwd × lat_raw — pn_sign으로 방향 선택
    norm_raw = fwd.cross(lat_raw.normalized()) * pn_sign
    if norm_raw.length < 1e-6:
        return None, None, None
    norm = norm_raw.normalized()

    # lat 재직교화
    lat = norm.cross(fwd).normalized()

    return fwd, lat, norm


# ================================================================
# 접근법 A/B — set_hand_rotation (blend_animate.py 복사, flip_palm 파라미터)
# ================================================================
def set_hand_rotation_diag(arm_obj, bone_name, wrist_w, mid9_w, palm_n_w, flip_palm):
    """
    blend_animate.py set_hand_rotation() 그대로 복사.
    (수정 없음 — 진단 목적으로만 여기서 재정의)
    """
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
# 접근법 C/D — set_hand_by_basis (target basis 직접 매핑)
# ================================================================
def set_hand_by_basis(arm_obj, bone_name, fwd_w, lat_w, norm_w):
    """
    Hand 뼈를 target basis (world 좌표)로 직접 설정.
      bone Y = fwd_w  (손목 → 중지 MCP)
      bone Z = norm_w (손바닥 법선)
      bone X = lat_w  (새끼 → 검지)

    set_hand_rotation() 의 두 단계 회전 대신 기저 행렬을 직접 구성.
    """
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        return False

    a_inv = arm_obj.matrix_world.to_3x3().inverted()

    # World → Armature-space 변환
    ny = (a_inv @ fwd_w).normalized()
    nz = (a_inv @ norm_w)
    # nz: forward 성분 제거 후 재직교화
    nz = (nz - nz.dot(ny) * ny)
    if nz.length < 1e-6:
        return False
    nz = nz.normalized()
    nx = ny.cross(nz).normalized()   # X = Y × Z (오른손 좌표계: X×Y=Z → nx×ny=nz)
    # 검증용 재직교화 — nx 방향이 lat_w 와 맞는지는 report 에서 확인
    # (nx = ny×nz 이면 nx×ny=nz 가 보장됨)

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
# 진단 리포트 — 적용 후 오차 측정
# ================================================================
def report_hand(arm_obj, bone_name, target_fwd_w, target_lat_w, target_norm_w, label):
    """
    현재 Hand 뼈 회전을 읽어 target basis 와 오차 출력.
    World 좌표계에서 측정 (armature_world × bone_pose_matrix).
    """
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        print(f"  [{label}] 뼈 없음: {bone_name}")
        return

    # 현재 pose 뼈의 world-space 회전 matrix
    bone_w_mat = (arm_obj.matrix_world @ b.matrix).to_3x3()
    actual_x = bone_w_mat.col[0].normalized()   # bone X (lateral)
    actual_y = bone_w_mat.col[1].normalized()   # bone Y (forward)
    actual_z = bone_w_mat.col[2].normalized()   # bone Z (palm normal)

    # target 벡터 (이미 world 좌표)
    tf = target_fwd_w.normalized()
    tl = target_lat_w.normalized()
    tn = target_norm_w.normalized()

    fwd_err  = angle_deg(actual_y, tf)
    norm_err = angle_deg(actual_z, tn)
    lat_err  = angle_deg(actual_x, tl)

    # 현재 basis 의 determinant
    d = det3(actual_x, actual_y, actual_z)

    # target basis 의 determinant
    td = det3(tl, tf, tn)

    print(f"\n  [{label}] {bone_name}")
    print(f"    actual X (lateral): {actual_x.x:+.3f} {actual_x.y:+.3f} {actual_x.z:+.3f}")
    print(f"    actual Y (forward): {actual_y.x:+.3f} {actual_y.y:+.3f} {actual_y.z:+.3f}")
    print(f"    actual Z (normal) : {actual_z.x:+.3f} {actual_z.y:+.3f} {actual_z.z:+.3f}")
    print(f"    target fwd        : {tf.x:+.3f} {tf.y:+.3f} {tf.z:+.3f}")
    print(f"    target lat        : {tl.x:+.3f} {tl.y:+.3f} {tl.z:+.3f}")
    print(f"    target norm       : {tn.x:+.3f} {tn.y:+.3f} {tn.z:+.3f}")
    print(f"    ── 오차 ──────────────────────────────")
    print(f"    forward_err : {fwd_err:6.1f}°   (Y vs target_fwd)")
    print(f"    normal_err  : {norm_err:6.1f}°   (Z vs target_norm)  ← 핵심")
    print(f"    lateral_err : {lat_err:6.1f}°   (X vs target_lat)")
    print(f"    ── determinant ───────────────────────")
    print(f"    actual  det : {d:+.4f}  ({'RIGHT-HANDED' if d > 0 else 'LEFT-HANDED !!!'})")
    print(f"    target  det : {td:+.4f}  ({'RIGHT-HANDED' if td > 0 else 'LEFT-HANDED !!!'})")


def report_rest(arm, bones_dict, label):
    """Rest basis 정보 출력 (bone head_local 기반 + bone matrix_local col[2])."""
    fwd, lat, norm = rest_hand_basis(arm, bones_dict)
    if fwd is None:
        return

    hand_bn = bones_dict['hand']
    bd = arm.data.bones.get(hand_bn)
    rest_mat = bd.matrix_local.to_3x3() if bd else None
    rest_x = rest_mat.col[0].normalized() if rest_mat else mathutils.Vector((0, 0, 0))
    rest_y = rest_mat.col[1].normalized() if rest_mat else mathutils.Vector((0, 0, 0))
    rest_z = rest_mat.col[2].normalized() if rest_mat else mathutils.Vector((0, 0, 0))

    d_computed = det3(lat, fwd, norm)
    d_rest_mat = det3(rest_x, rest_y, rest_z)

    print(f"\n  [Rest Basis — {label}]")
    print(f"    (bone head_local 기반)")
    print(f"    rest fwd  (Middle1-Hand)  : {fwd.x:+.3f} {fwd.y:+.3f} {fwd.z:+.3f}")
    print(f"    rest lat  (Index1-Pinky1) : {lat.x:+.3f} {lat.y:+.3f} {lat.z:+.3f}")
    print(f"    rest norm (fwd × lat_raw) : {norm.x:+.3f} {norm.y:+.3f} {norm.z:+.3f}")
    print(f"    computed det(lat,fwd,norm): {d_computed:+.4f}")
    print(f"    (bone matrix_local)")
    print(f"    rest col[0]=X : {rest_x.x:+.3f} {rest_x.y:+.3f} {rest_x.z:+.3f}")
    print(f"    rest col[1]=Y : {rest_y.x:+.3f} {rest_y.y:+.3f} {rest_y.z:+.3f}")
    print(f"    rest col[2]=Z : {rest_z.x:+.3f} {rest_z.y:+.3f} {rest_z.z:+.3f}")
    print(f"    rest mat det  : {d_rest_mat:+.4f}")

    # rest norm vs bone Z 비교
    try:
        angle = math.degrees(norm.angle(rest_z))
    except Exception:
        angle = 999.0
    print(f"    computed_norm ∠ bone_Z : {angle:.1f}°  "
          f"({'동일 방향' if angle < 90 else '반대 방향 ← 법선 부호 반전 필요'})")
    return norm   # auto-sign 판정에 사용


# ================================================================
# 액션 저장 헬퍼
# ================================================================
def save_single_frame_action(arm_obj, action_name, frame=1):
    """
    현재 pose 상태를 frame 에 키프레임으로 저장한 Blender 액션 생성.
    기존 동명 액션은 삭제 후 재생성.
    """
    # 기존 액션 제거
    old = bpy.data.actions.get(action_name)
    if old:
        bpy.data.actions.remove(old)

    act = bpy.data.actions.new(action_name)
    arm_obj.animation_data_create()
    arm_obj.animation_data.action = act

    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.keyframe_insert('rotation_quaternion', frame=frame)

    print(f"  → 액션 저장: '{action_name}' (frame {frame})")
    return act


def reset_pose(arm_obj):
    """모든 뼈를 rest pose 로 초기화."""
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = mathutils.Quaternion()
        pb.location = mathutils.Vector((0, 0, 0))
        pb.scale    = mathutils.Vector((1, 1, 1))


# ================================================================
# 메인 진단
# ================================================================
def main():
    # ── 아마추어 준비 ────────────────────────────────────────────
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
    print(f"[JSON] 총 {len(frames_data)} 프레임")

    # ── 캐릭터 스케일 (어깨 기준) ─────────────────────────────
    lb = arm_obj.pose.bones.get('mixamorig:LeftArm')
    rb = arm_obj.pose.bones.get('mixamorig:RightArm')
    if lb and rb:
        l_sh_w  = arm_obj.matrix_world @ lb.head
        r_sh_w  = arm_obj.matrix_world @ rb.head
        char_origin = (l_sh_w + r_sh_w) / 2
        char_sw     = (l_sh_w - r_sh_w).length
    else:
        char_origin = mathutils.Vector((0, 0, 0)); char_sw = 1.5
    print(f"[캐릭터] 어깨 중심: {char_origin}  너비: {char_sw:.3f}")

    # ── 유효 프레임 수집 ─────────────────────────────────────────
    # l_frames: 캐릭터 LEFT 손이 있는 프레임 (pts21 + pn)
    # r_frames: 캐릭터 RIGHT 손이 있는 프레임

    l_frames = []
    r_frames = []

    for fd in frames_data:
        if not fd.get('HasPose', False):
            continue
        pose_pts  = fd.get('PosePoints') or []
        mp_l_pts  = fd.get('LeftHandPoints') or []
        mp_r_pts  = fd.get('RightHandPoints') or []
        has_mp_l  = fd.get('HasLeftHand', False)
        has_mp_r  = fd.get('HasRightHand', False)

        if SWAP_HAND_LABELS:
            ch_l_pts, has_ch_l = mp_r_pts, has_mp_r   # mp_right → 캐릭터 LEFT
            ch_r_pts, has_ch_r = mp_l_pts, has_mp_l   # mp_left  → 캐릭터 RIGHT
        else:
            ch_l_pts, has_ch_l = mp_l_pts, has_mp_l
            ch_r_pts, has_ch_r = mp_r_pts, has_mp_r

        ls = get_pt(pose_pts, P_L_SHOULDER)
        rs = get_pt(pose_pts, P_R_SHOULDER)
        if not ls or not rs:
            continue
        cx = (ls[0] + rs[0]) / 2
        cy = (ls[1] + rs[1]) / 2
        sw = math.sqrt((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2)
        if sw < 1e-4:
            continue

        def chp(idx, pts):
            pt = get_pt(pts, idx)
            return convert_hand_pt(pt[0], pt[1], pt[2], cx, cy, sw, char_origin, char_sw) if pt else None

        if has_ch_l:
            pts21 = [chp(i, ch_l_pts) for i in range(21)]
            if all(p is not None for p in pts21):
                wrist = pts21[H_WRIST]
                idx_m = pts21[H_INDEX_MCP]
                pnk_m = pts21[H_PINK_MCP]
                pn    = (idx_m - wrist).cross(pnk_m - wrist)
                if pn.length > 1e-6:
                    l_frames.append({'pts21': pts21, 'pn': pn.normalized()})

        if has_ch_r:
            pts21 = [chp(i, ch_r_pts) for i in range(21)]
            if all(p is not None for p in pts21):
                wrist = pts21[H_WRIST]
                idx_m = pts21[H_INDEX_MCP]
                pnk_m = pts21[H_PINK_MCP]
                pn    = (idx_m - wrist).cross(pnk_m - wrist)
                if pn.length > 1e-6:
                    r_frames.append({'pts21': pts21, 'pn': pn.normalized()})

    print(f"[유효] LEFT 손 프레임: {len(l_frames)}  RIGHT 손 프레임: {len(r_frames)}")

    if not l_frames and not r_frames:
        print("[오류] 유효한 손 프레임 없음"); return

    fidx = min(TEST_FRAME_IDX, max(len(l_frames) - 1, 0))
    print(f"[진단 프레임] TEST_FRAME_IDX={TEST_FRAME_IDX}  실제 사용={fidx}")

    # ── Rest Basis 출력 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REST BASIS (bone head_local)")
    print("=" * 60)
    rest_norm_l = report_rest(arm_obj, REST_BONES_L, "LEFT")
    rest_norm_r = report_rest(arm_obj, REST_BONES_R, "RIGHT")

    # ── 각 접근법 테스트 ─────────────────────────────────────────
    approaches = [
        ('A', False, False, False),  # (label, flip_palm, use_basis, auto_sign)
        ('B', True,  False, False),
        ('C', False, True,  False),
        ('D', False, True,  True ),
    ]

    for approach_label, flip_palm, use_basis, auto_sign in approaches:
        print(f"\n{'=' * 60}")
        print(f"접근법 {approach_label}:")
        if not use_basis:
            print(f"  set_hand_rotation()  flip_palm={flip_palm}")
        else:
            sign_mode = "auto (rest_norm 비교)" if auto_sign else "+1 고정"
            print(f"  set_hand_by_basis()  pn_sign={sign_mode}")
        print("=" * 60)

        reset_pose(arm_obj)
        bpy.context.view_layer.update()   # parent matrix 갱신 (LEFT b.matrix 신뢰성 확보)

        # LEFT 손 테스트
        if l_frames:
            fd_l  = l_frames[fidx if fidx < len(l_frames) else 0]
            pts21 = fd_l['pts21']
            pn_w  = fd_l['pn']
            wrist = pts21[H_WRIST]
            mid9  = pts21[H_MID_MCP]

            # target basis for report
            tf, tl, tn_plus1 = target_hand_basis(pts21, pn_sign=+1.0)
            _, _, tn_minus1  = target_hand_basis(pts21, pn_sign=-1.0)

            if use_basis:
                if auto_sign and rest_norm_l is not None and tf is not None:
                    # rest_norm_l 은 armature-space; tn_plus1 은 world-space.
                    # 직접 비교하려면 동일 공간으로 변환 필요.
                    # → armature-space 로 변환해서 비교
                    a_inv = arm_obj.matrix_world.to_3x3().inverted()
                    tn_a  = (a_inv @ tn_plus1).normalized()
                    pn_sign = 1.0 if tn_a.dot(rest_norm_l) >= 0 else -1.0
                    print(f"  [LEFT AUTO] tn·rest_norm dot={tn_a.dot(rest_norm_l):+.3f}  → pn_sign={pn_sign:+.0f}")
                else:
                    pn_sign = 1.0

                tf_use, tl_use, tn_use = target_hand_basis(pts21, pn_sign=pn_sign)
                if tf_use is not None:
                    set_hand_by_basis(arm_obj, BONE_L_HAND, tf_use, tl_use, tn_use)
                    # bpy.context.view_layer.update() 는 keyframe 삽입 없이도 pose 반영됨
                    bpy.context.view_layer.update()
                    report_hand(arm_obj, BONE_L_HAND, tf_use, tl_use, tn_use,
                                f"{approach_label} LEFT  pn_sign={pn_sign:+.0f}")
            else:
                set_hand_rotation_diag(arm_obj, BONE_L_HAND, wrist, mid9, pn_w, flip_palm)
                bpy.context.view_layer.update()
                # 리포트 용 target (flip_palm 이면 norm 반전)
                tn_report = tn_minus1 if flip_palm else tn_plus1
                tl_report = target_hand_basis(pts21, pn_sign=-1.0 if flip_palm else 1.0)[1]
                if tf is not None:
                    report_hand(arm_obj, BONE_L_HAND, tf, tl_report, tn_report,
                                f"{approach_label} LEFT  flip={flip_palm}")

        # RIGHT 손 테스트
        if r_frames:
            fd_r  = r_frames[fidx if fidx < len(r_frames) else 0]
            pts21 = fd_r['pts21']
            pn_w  = fd_r['pn']
            wrist = pts21[H_WRIST]
            mid9  = pts21[H_MID_MCP]

            tf, tl, tn_plus1 = target_hand_basis(pts21, pn_sign=+1.0)
            _, _, tn_minus1  = target_hand_basis(pts21, pn_sign=-1.0)

            if use_basis:
                if auto_sign and rest_norm_r is not None and tf is not None:
                    a_inv = arm_obj.matrix_world.to_3x3().inverted()
                    tn_a  = (a_inv @ tn_plus1).normalized()
                    pn_sign = 1.0 if tn_a.dot(rest_norm_r) >= 0 else -1.0
                    print(f"  [RIGHT AUTO] tn·rest_norm dot={tn_a.dot(rest_norm_r):+.3f}  → pn_sign={pn_sign:+.0f}")
                else:
                    pn_sign = 1.0

                tf_use, tl_use, tn_use = target_hand_basis(pts21, pn_sign=pn_sign)
                if tf_use is not None:
                    set_hand_by_basis(arm_obj, BONE_R_HAND, tf_use, tl_use, tn_use)
                    bpy.context.view_layer.update()
                    report_hand(arm_obj, BONE_R_HAND, tf_use, tl_use, tn_use,
                                f"{approach_label} RIGHT pn_sign={pn_sign:+.0f}")
            else:
                set_hand_rotation_diag(arm_obj, BONE_R_HAND, wrist, mid9, pn_w, flip_palm)
                bpy.context.view_layer.update()
                tn_report = tn_minus1 if flip_palm else tn_plus1
                tl_report = target_hand_basis(pts21, pn_sign=-1.0 if flip_palm else 1.0)[1]
                if tf is not None:
                    report_hand(arm_obj, BONE_R_HAND, tf, tl_report, tn_report,
                                f"{approach_label} RIGHT flip={flip_palm}")

        # 현재 pose 를 액션으로 저장
        action_name = f"HandTest_{approach_label}"
        save_single_frame_action(arm_obj, action_name, frame=1)

    # ── 마무리 ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("진단 완료.")
    print("Blender 타임라인 → 액션 목록에서 HandTest_A/B/C/D 를 선택해")
    print("  각 접근법의 손 자세를 시각적으로 비교하세요.")
    print()
    print("  A: set_hand_rotation (현재) flip=False")
    print("  B: set_hand_rotation         flip=True  (법선 반전)")
    print("  C: set_hand_by_basis         pn_sign=+1 (직접 기저 매핑)")
    print("  D: set_hand_by_basis         pn_sign=auto (rest_norm 자동 판정)")
    print()
    print("정상적인 손 자세 기준:")
    print("  forward_err ≈ 0°, normal_err ≈ 0°, lateral_err ≈ 0° or 180°")
    print("  (lateral 180° 는 Left/Right 뼈 방향 차이로 정상일 수 있음)")
    print("=" * 60)

    # 마지막 액션(D)이 활성화된 상태로 종료 — 필요하면 A로 변경 가능
    act_a = bpy.data.actions.get('HandTest_A')
    if act_a and arm_obj.animation_data:
        arm_obj.animation_data.action = act_a
        print("[정보] 현재 액션 → HandTest_A 로 전환 (A=현재 방식 기준선)")


main()
