"""
Tools/diag_palm_flip_sequence_safe_v2.py
------------------------------------------------------------
Palm flip 네 조합 시각 비교용 진단 스크립트 (안전 버전 v2).

v1 대비 변경 사항:
  1) 제약조건 mute/unmute 만 사용 — b.constraints.remove() 없음
  2) 팔 FK 베이킹 — IK 평가된 visual matrix 를 FK quaternion 으로 역산,
     Action 키프레임에 저장 → 진단 IK 제거 후에도 Action 단독 재현 가능
  3) 독립성 검증 — 각 Action 생성 후 진단 IK mute, 재평가, 행렬 차이 출력
     (< 1e-4 이어야 성공)
  4) 예외 안전 — 전체 변형을 try/except 로 래핑, 오류 시 diag_palm_restore() 자동 호출
  5) use_fake_user = True — Blender GC 로부터 Action 보호
  6) diag_palm_restore() — 스크립트 실행 후에도 전역에서 호출 가능
  7) 성공 후 상태 — 진단 IK 제거, 팔 자세는 Action 키, 원본 제약은 mute 유지,
     첫 번째 DiagPalm Action 활성, scene frame=1

실행:
  blender --background --python Tools/diag_palm_flip_sequence_safe_v2.py

생성 액션:
  DiagPalm_감사합니다_1033
  DiagPalm_빵_1033

성공 후 복원:
  (Blender Python 콘솔에서) diag_palm_restore()
------------------------------------------------------------
"""

import bpy, json, math, mathutils, traceback

# ================================================================
# 설정 (blend_animate.py 확정값과 동일)
# ================================================================
RECORDINGS_BASE = (
    r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\Recordings"
)
DIAG_TARGETS = [
    {
        'label':       '감사합니다',
        'json_path':   RECORDINGS_BASE + r'\감사합니다.json',
        'target_ts':   1033,
        'action_name': 'DiagPalm_감사합니다_1033',
    },
    {
        'label':       '빵',
        'json_path':   RECORDINGS_BASE + r'\빵.json',
        'target_ts':   1033,
        'action_name': 'DiagPalm_빵_1033',
    },
]

# flip 조합 (frame, l_flip, r_flip, label)
COMBOS = [
    ( 1, False, False, 'FF'),
    (11, False, True,  'FT_CURRENT'),
    (21, True,  False, 'TF_OPPOSITE'),
    (31, True,  True,  'TT'),
]

# blend_animate.py 확정값
SWAP_HAND_LABELS  = False
HAND_DEPTH_GAIN   = 1.0
BODY_SCALE        = 1.0
DEPTH_SCALE       = 0.0
FORWARD_OFFSET    = 0.15
MIRROR_X          = True
FLIP_FINGER_DIR_L = False
FLIP_FINGER_DIR_R = False

# MediaPipe 인덱스
P_L_SHOULDER = 11;  P_R_SHOULDER = 12
P_L_WRIST    = 15;  P_R_WRIST    = 16
H_WRIST      = 0
H_THUMB_CMC=1; H_THUMB_MCP=2; H_THUMB_IP=3;  H_THUMB_TIP=4
H_INDEX_MCP=5; H_INDEX_PIP=6; H_INDEX_DIP=7;  H_INDEX_TIP=8
H_MID_MCP=9;   H_MID_PIP=10;  H_MID_DIP=11;  H_MID_TIP=12
H_RING_MCP=13; H_RING_PIP=14; H_RING_DIP=15; H_RING_TIP=16
H_PINK_MCP=17; H_PINK_PIP=18; H_PINK_DIP=19; H_PINK_TIP=20

# Mixamo 뼈 이름
BONE_L_ARM     = 'mixamorig:LeftArm'
BONE_L_FOREARM = 'mixamorig:LeftForeArm'
BONE_L_HAND    = 'mixamorig:LeftHand'
BONE_R_ARM     = 'mixamorig:RightArm'
BONE_R_FOREARM = 'mixamorig:RightForeArm'
BONE_R_HAND    = 'mixamorig:RightHand'

BONE_L_SHOULDER = 'mixamorig:LeftShoulder'
BONE_R_SHOULDER = 'mixamorig:RightShoulder'

# 정적 fallback (런타임에 _collect_arm_bake_chain() 으로 교체됨)
ARM_BAKE_BONES = [BONE_L_SHOULDER, BONE_L_ARM, BONE_L_FOREARM,
                  BONE_R_SHOULDER, BONE_R_ARM, BONE_R_FOREARM]

# 진단용 IK 제약조건 이름 (production 제약과 구별)
CON_L_NAME = 'DiagPalm_IK_L'
CON_R_NAME = 'DiagPalm_IK_R'

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
# 전역 복원 상태 (스크립트 종료 후에도 diag_palm_restore() 에서 사용)
# ================================================================
_DIAG_SNAP    = {}
_diag_arm_obj = None
_diag_emp_lw  = None
_diag_emp_rw  = None


# ================================================================
# diag_palm_restore() — 전역 복원 함수 (언제든지 호출 가능)
# ================================================================
def diag_palm_restore():
    """
    스크립트 실행 이전 상태로 완전 복원.
    - 진단 IK 제약 제거
    - 모든 제약 mute 상태를 스냅샷 값으로 복원
    - rotation_mode 복원
    - 원본 Action 복원
    - 씬 frame/frame_start/frame_end 복원
    - Empty 숨김
    """
    global _DIAG_SNAP, _diag_arm_obj, _diag_emp_lw, _diag_emp_rw

    snap    = _DIAG_SNAP
    arm_obj = _diag_arm_obj

    if not snap or not arm_obj:
        print("[diag_palm_restore] 스냅샷 없음 - 복원할 상태가 없습니다.")
        return

    try:
        # 0) 원본 active object 복원 (mode 변경 전)
        orig_active_r = snap.get('orig_active')
        if orig_active_r:
            try:
                bpy.context.view_layer.objects.active = orig_active_r
            except Exception:
                pass

        bpy.context.view_layer.objects.active = arm_obj
        if bpy.context.mode != 'POSE':
            if bpy.context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.mode_set(mode='POSE')

        # 1) 진단 IK 제약 제거 (존재하면)
        for bn, cname in [(BONE_L_FOREARM, CON_L_NAME), (BONE_R_FOREARM, CON_R_NAME)]:
            pb = arm_obj.pose.bones.get(bn)
            if pb:
                for c in list(pb.constraints):
                    if c.name == cname:
                        pb.constraints.remove(c)

        # 2) 모든 제약 mute 상태 복원 (스냅샷 기준)
        muted_list = []
        for bn, cons_snap in snap.get('constraints', {}).items():
            pb = arm_obj.pose.bones.get(bn)
            if not pb:
                continue
            for (c_name, c_mute_orig) in cons_snap:
                c = next((x for x in pb.constraints if x.name == c_name), None)
                if c:
                    c.mute = c_mute_orig
                    if c_mute_orig:
                        muted_list.append(f"{bn}/{c_name}")

        # 3) rotation_mode 복원
        for bn, rmode in snap.get('rotation_modes', {}).items():
            pb = arm_obj.pose.bones.get(bn)
            if pb:
                pb.rotation_mode = rmode

        # 4) 원본 Action 복원
        if arm_obj.animation_data:
            arm_obj.animation_data.action = snap.get('action')

        # 5) 씬 프레임 복원
        bpy.context.scene.frame_start   = snap.get('frame_start', 1)
        bpy.context.scene.frame_end     = snap.get('frame_end', 250)
        bpy.context.scene.frame_current = snap.get('frame', 1)

        # 6) Empty 숨김
        if _diag_emp_lw:
            _diag_emp_lw.hide_viewport = True
        if _diag_emp_rw:
            _diag_emp_rw.hide_viewport = True

        # 6) 원본 mode 복원
        orig_mode_r = snap.get('orig_mode', 'OBJECT')
        target_mode = orig_mode_r if orig_mode_r in ('OBJECT', 'POSE') else 'OBJECT'
        bpy.ops.object.mode_set(mode=target_mode)

        # 원본 active object로 최종 복원 (있으면)
        orig_active_r = snap.get('orig_active')
        if orig_active_r and orig_active_r != arm_obj:
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.context.view_layer.objects.active = orig_active_r
            except Exception:
                pass

        print("[diag_palm_restore] 복원 완료")
        print(f"  원본 Action: {snap.get('action').name if snap.get('action') else 'None'}")
        print(f"  씬 frame: {snap.get('frame', 1)}")
        print(f"  원본 mode: {snap.get('orig_mode', 'unknown')}")

    except Exception:
        print("[diag_palm_restore] 복원 중 오류:")
        traceback.print_exc()


# ================================================================
# 스냅샷 — 실행 전 전체 상태 저장
# ================================================================
def _take_snapshot(arm_obj):
    """
    변형 시작 전 아마추어 전체 상태 스냅샷.
    저장 항목: active action, scene frame/range, mode,
              모든 pose bone 의 제약 mute 상태, rotation_mode.
    """
    snap = {
        'action':      arm_obj.animation_data.action if arm_obj.animation_data else None,
        'frame':       bpy.context.scene.frame_current,
        'frame_start': bpy.context.scene.frame_start,
        'frame_end':   bpy.context.scene.frame_end,
        'mode':        bpy.context.mode,
        'constraints': {},   # bone_name -> [(con_name, mute), ...]
        'rotation_modes': {},
    }
    for pb in arm_obj.pose.bones:
        snap['constraints'][pb.name]    = [(c.name, c.mute) for c in pb.constraints]
        snap['rotation_modes'][pb.name] = pb.rotation_mode
    return snap


# ================================================================
# 유틸 — blend_animate.py 와 동일한 수식
# ================================================================
def get_pt(pts, idx):
    if not pts or idx >= len(pts):
        return None
    p = pts[idx]
    return (p.get('x', 0), p.get('y', 0), p.get('z', 0)) if isinstance(p, dict) else None


def to_blender(x, y, z, cx, cy, sw, origin, char_sw):
    """Pose landmark -> Blender world (깊이 0, 손목 IK 타겟용)."""
    sc = char_sw * BODY_SCALE
    xs = 1 if MIRROR_X else -1
    return origin + mathutils.Vector((
        xs * (x - cx) / sw * sc,
        z * sc * DEPTH_SCALE - FORWARD_OFFSET,
        -(y - cy) / sw * sc,
    ))


def convert_hand_pt(x, y, z, cx, cy, sw, origin, char_sw):
    """Hand landmark -> Blender world (HAND_DEPTH_GAIN 적용)."""
    sc = char_sw * BODY_SCALE
    xs = 1 if MIRROR_X else -1
    return origin + mathutils.Vector((
        xs * (x - cx) / sw * sc,
        z / sw * sc * HAND_DEPTH_GAIN - FORWARD_OFFSET,
        -(y - cy) / sw * sc,
    ))


def bone_arma_mat(arm_obj, bone_name, parent_mat=None):
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


def get_parent_mat(arm_obj, bone_name, anchor_name, anchor_mat):
    b = arm_obj.pose.bones.get(bone_name)
    if not b or not b.parent or b.parent.name == anchor_name:
        return anchor_mat
    chain = []
    cur = b.parent
    while cur and cur.name != anchor_name:
        chain.append(cur.name)
        cur = cur.parent
    chain.reverse()
    mat = anchor_mat
    for inter in chain:
        mat = bone_arma_mat(arm_obj, inter, mat)
    return mat


def set_hand_rotation(arm_obj, bone_name, wrist_w, mid9_w, palm_n_w, flip_palm):
    """손뼈: 방향(wrist->mid9) + 손바닥 법선으로 roll 설정."""
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        return False
    hand_dir = (mid9_w - wrist_w)
    if hand_dir.length < 1e-6:
        return False
    hand_dir = hand_dir.normalized()
    a_inv  = arm_obj.matrix_world.to_3x3().inverted()
    dir_a  = (a_inv @ hand_dir).normalized()
    rest = bd.matrix_local.to_3x3()
    ry = rest.col[1].normalized()
    rx = rest.col[0].normalized()
    rz = rest.col[2].normalized()
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
        nz_perp = nz  - nz.dot(ny) * ny
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


def set_bone_dir(arm_obj, bone_name, from_w, to_w, parent_mat=None, roll_pn=None):
    """뼈를 from_w->to_w 방향으로 FK 설정. roll_pn: palm normal (world)."""
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        return False, parent_mat
    direction = (to_w - from_w)
    if direction.length < 1e-6:
        return False, parent_mat
    direction = direction.normalized()
    a_inv = arm_obj.matrix_world.to_3x3().inverted()
    dir_a = (a_inv @ direction).normalized()
    rest  = bd.matrix_local.to_3x3()
    ry    = rest.col[1].normalized()
    rx    = rest.col[0].normalized()
    rz    = rest.col[2].normalized()
    q  = ry.rotation_difference(dir_a)
    ny = dir_a
    nx = (q @ rx).normalized()
    nz = (q @ rz).normalized()
    if roll_pn is not None and roll_pn.length > 1e-5:
        pna = (a_inv @ roll_pn).normalized()
        pp  = pna - pna.dot(ny) * ny
        zp  = nz  - nz.dot(ny) * ny
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
    local_rot_mat = b.rotation_quaternion.to_matrix()
    if b.parent:
        p_mat_used = parent_mat if parent_mat is not None else b.parent.matrix.to_3x3()
        p_bd       = arm_obj.data.bones.get(b.parent.name)
        p_rest     = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        new_mat    = p_mat_used @ p_rest.inverted() @ rest @ local_rot_mat
    else:
        new_mat = rest @ local_rot_mat
    return True, new_mat


# ================================================================
# 데이터 추출
# ================================================================
def extract_frame_at_ts(json_path, target_ts, char_origin, char_sw):
    """JSON에서 TimestampMs~target_ts에 가장 가까운 프레임 데이터 반환."""
    with open(json_path, encoding='utf-8') as f:
        recording = json.load(f)

    best_diff = float('inf')
    best_fd   = None
    for fd in recording.get('Frames', []):
        if not fd.get('HasPose', False):
            continue
        ts = fd.get('TimestampMs')
        if ts is None:
            continue
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_fd   = fd

    if best_fd is None:
        return None, None

    fd       = best_fd
    pose_pts = fd.get('PosePoints') or []
    mp_l_pts = fd.get('LeftHandPoints') or []
    mp_r_pts = fd.get('RightHandPoints') or []
    has_mp_l = fd.get('HasLeftHand', False)
    has_mp_r = fd.get('HasRightHand', False)
    actual_ts = fd.get('TimestampMs')

    if SWAP_HAND_LABELS:
        ch_l_pts, has_ch_l = mp_r_pts, has_mp_r
        ch_r_pts, has_ch_r = mp_l_pts, has_mp_l
    else:
        ch_l_pts, has_ch_l = mp_l_pts, has_mp_l
        ch_r_pts, has_ch_r = mp_r_pts, has_mp_r

    ls = get_pt(pose_pts, P_L_SHOULDER)
    rs = get_pt(pose_pts, P_R_SHOULDER)
    if not ls or not rs:
        return None, actual_ts
    cx = (ls[0] + rs[0]) / 2
    cy = (ls[1] + rs[1]) / 2
    sw = math.sqrt((ls[0] - rs[0])**2 + (ls[1] - rs[1])**2)
    if sw < 1e-4:
        return None, actual_ts

    def bp(idx):
        pt = get_pt(pose_pts, idx)
        return to_blender(*pt, cx, cy, sw, char_origin, char_sw) if pt else None

    def chp(idx, pts):
        pt = get_pt(pts, idx)
        return convert_hand_pt(pt[0], pt[1], pt[2], cx, cy, sw, char_origin, char_sw) if pt else None

    lw_ik = bp(P_L_WRIST)
    rw_ik = bp(P_R_WRIST)

    l_pts_21 = [chp(i, ch_l_pts) if has_ch_l else None for i in range(21)]
    r_pts_21 = [chp(i, ch_r_pts) if has_ch_r else None for i in range(21)]

    lw3 = chp(H_WRIST,     ch_l_pts) if has_ch_l else None
    li3 = chp(H_INDEX_MCP, ch_l_pts) if has_ch_l else None
    lp3 = chp(H_PINK_MCP,  ch_l_pts) if has_ch_l else None
    if has_ch_l and lw3 and li3 and lp3:
        v = (li3 - lw3).cross(lp3 - lw3)
        l_pn_raw = v if v.length > 1e-6 else None
    else:
        l_pn_raw = None

    rw3 = chp(H_WRIST,     ch_r_pts) if has_ch_r else None
    ri3 = chp(H_INDEX_MCP, ch_r_pts) if has_ch_r else None
    rp3 = chp(H_PINK_MCP,  ch_r_pts) if has_ch_r else None
    if has_ch_r and rw3 and ri3 and rp3:
        v = (ri3 - rw3).cross(rp3 - rw3)
        r_pn_raw = v if v.length > 1e-6 else None
    else:
        r_pn_raw = None

    lh0 = chp(H_WRIST,   ch_l_pts) if has_ch_l else None
    lh9 = chp(H_MID_MCP, ch_l_pts) if has_ch_l else None
    rh0 = chp(H_WRIST,   ch_r_pts) if has_ch_r else None
    rh9 = chp(H_MID_MCP, ch_r_pts) if has_ch_r else None

    return {
        'ts':       actual_ts,
        'lw_ik':    lw_ik,   'rw_ik':    rw_ik,
        'l_pts':    l_pts_21, 'r_pts':    r_pts_21,
        'l_pn_raw': l_pn_raw, 'r_pn_raw': r_pn_raw,
        'lh0': lh0, 'lh9': lh9,
        'rh0': rh0, 'rh9': rh9,
        'has_l': has_ch_l, 'has_r': has_ch_r,
    }, actual_ts


# ================================================================
# 제약조건 관리 (mute/unmute 만 사용, remove 없음)
# ================================================================
def mute_existing_ik(arm_obj, bone_name, exclude_con_name):
    """named constraint 제외한 기존 IK 를 mute. muted 목록 반환."""
    b = arm_obj.pose.bones.get(bone_name)
    if not b:
        return []
    muted = []
    for c in b.constraints:
        if c.type == 'IK' and c.name != exclude_con_name and not c.mute:
            c.mute = True
            muted.append(c.name)
    return muted


def mute_all_constraints(arm_obj, bone_name):
    """지정 뼈의 모든 제약을 mute. 원래 False였던 제약 이름 목록 반환."""
    b = arm_obj.pose.bones.get(bone_name)
    if not b:
        return []
    muted = []
    for c in b.constraints:
        if not c.mute:
            c.mute = True
            muted.append(c.name)
    return muted


def add_named_ik(arm_obj, bone_name, target, chain_count, con_name):
    """진단 IK 추가. 이미 존재하면 건너뜀."""
    b = arm_obj.pose.bones.get(bone_name)
    if not b:
        return
    for c in b.constraints:
        if c.name == con_name:
            return
    c = b.constraints.new('IK')
    c.name        = con_name
    c.target      = target
    c.chain_count = chain_count
    c.use_rotation = False


def remove_named_ik(arm_obj, bone_name, con_name):
    """진단 IK 제약 제거 (진단 IK 전용, production 제약은 건드리지 않음)."""
    b = arm_obj.pose.bones.get(bone_name)
    if not b:
        return
    for c in list(b.constraints):
        if c.name == con_name:
            b.constraints.remove(c)
            return


def get_or_create_empty(name, col):
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
    else:
        obj = bpy.data.objects.new(name, None)
        obj.empty_display_size = 0.04
        col.objects.link(obj)
    obj.hide_viewport = False
    return obj


# ================================================================
# 팔 bake chain 동적 수집
# ================================================================
def _collect_arm_bake_chain(arm_obj):
    """
    LeftShoulder/RightShoulder의 LCA를 상체 anchor로 자동 탐색.
    bake 대상: [anchor, LeftShoulder, LeftArm, LeftForeArm,
                         RightShoulder, RightArm, RightForeArm]
    Hips/Spine/Spine1 등 하위 체인은 제외. root→child 순서.
    Mixamo 기본 리그: anchor = mixamorig:Spine2.
    """
    # LeftShoulder 계보 집합 구성 (Spine2, Spine1, Spine, Hips 포함)
    l_ancestors = set()
    pb = arm_obj.pose.bones.get(BONE_L_SHOULDER)
    while pb:
        l_ancestors.add(pb.name)
        pb = pb.parent

    # RightShoulder에서 올라가며 첫 번째 공통 조상(LCA) 탐색
    anchor = None
    pb = arm_obj.pose.bones.get(BONE_R_SHOULDER)
    while pb:
        if pb.name in l_ancestors and pb.name != BONE_L_SHOULDER:
            anchor = pb.name
            break
        pb = pb.parent

    if anchor is None:          # fallback: LeftShoulder 직계 부모
        pb_l = arm_obj.pose.bones.get(BONE_L_SHOULDER)
        anchor = pb_l.parent.name if (pb_l and pb_l.parent) else BONE_L_SHOULDER

    print(f"  [bake chain] 상체 anchor (LCA) = {anchor}")

    def _branch(tip_name):
        """tip_name から anchor 직전까지 child→parent 수집 후 reverse (parent→child)."""
        pb = arm_obj.pose.bones.get(tip_name)
        seg = []
        while pb and pb.name != anchor:
            seg.append(pb.name)
            pb = pb.parent
        seg.reverse()
        return seg

    return [anchor] + _branch(BONE_L_FOREARM) + _branch(BONE_R_FOREARM)


# ================================================================
# 팔 FK 베이킹 (요구사항 2)
# ================================================================
def _capture_arm_ref_mats(arm_obj, bake_chain):
    """
    진단 IK 활성 상태에서 팔 뼈의 평가된 armature-space 4x4 matrix 캡처.
    IK 를 제거한 후 비교 기준으로 사용.
    """
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    ref = {}
    for bn in bake_chain:
        pb = arm_obj.pose.bones.get(bn)
        if pb:
            ref[bn] = pb.matrix.copy()
    # anchor local pose 별도 저장 — anchor-relative 비교 시 anchor 자체 검증용
    anchor_bn = bake_chain[0] if bake_chain else None
    if anchor_bn:
        pb_anc = arm_obj.pose.bones.get(anchor_bn)
        if pb_anc:
            ref['__anchor_basis__'] = pb_anc.matrix_basis.copy()
    return ref


def _bake_arm_fk(arm_obj, action, ref_mats, frames, bake_chain):
    """
    IK 평가된 armature-space matrix 로부터 FK local quaternion 을 역산,
    Action 에 모든 combo frame 에 동일하게 삽입.
    bake_chain: root→child 순서. Spine/Shoulder/Arm/ForeArm 모두 포함.

    역산 공식 (3x3 회전 부분):
      local_rot = bone_rest^-1 @ p_rest @ p_eval^-1 @ bone_eval
    부모 없는 경우:
      local_rot = bone_rest^-1 @ bone_eval

    ref_mats 에서 부모 eval matrix 를 가져와 진단 IK 로 드리븐된 값을 사용.
    """
    baked_quats = {}

    for bn in bake_chain:
        if bn not in ref_mats:
            continue
        pb = arm_obj.pose.bones.get(bn)
        bd = arm_obj.data.bones.get(bn)
        if not pb or not bd:
            continue

        bone_eval = ref_mats[bn].to_3x3()
        bone_rest = bd.matrix_local.to_3x3()

        if pb.parent:
            p_name = pb.parent.name
            p_bd   = arm_obj.data.bones.get(p_name)
            # 부모의 IK 평가 matrix — ref_mats 에 있으면 사용, 없으면 현재 평가값
            if p_name in ref_mats:
                p_eval = ref_mats[p_name].to_3x3()
            else:
                p_pb   = arm_obj.pose.bones.get(p_name)
                p_eval = p_pb.matrix.to_3x3() if p_pb else mathutils.Matrix.Identity(3)
            p_rest = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
            # local_rot = bone_rest^-1 @ p_rest @ p_eval^-1 @ bone_eval
            local_rot_3x3 = bone_rest.inverted() @ p_rest @ p_eval.inverted() @ bone_eval
        else:
            local_rot_3x3 = bone_rest.inverted() @ bone_eval

        q = local_rot_3x3.to_quaternion()
        q.normalize()
        baked_quats[bn] = q

    # 키프레임 삽입 — 모든 combo frame 에 동일한 팔 자세
    keyed_bones = []
    for bn, q in baked_quats.items():
        pb = arm_obj.pose.bones.get(bn)
        if not pb:
            continue
        pb.rotation_mode      = 'QUATERNION'
        pb.rotation_quaternion = q
        for frame in frames:
            pb.keyframe_insert('rotation_quaternion', frame=frame)
        keyed_bones.append(bn)

    # --- 콘솔 출력 ---
    print(f"\n  [팔 FK 베이킹]")
    for bn in keyed_bones:
        q = baked_quats[bn]
        # fcurve 존재 확인
        dp = f'pose.bones["{bn}"].rotation_quaternion'
        has_fc = any(fc.data_path == dp for fc in action.fcurves)
        print(f"    {bn}: q=({q.w:+.4f},{q.x:+.4f},{q.y:+.4f},{q.z:+.4f})  "
              f"fcurve={'OK' if has_fc else 'MISSING'}")
    print(f"  [Action] use_fake_user = {action.use_fake_user}")

    return keyed_bones


# ================================================================
# 독립성 검증 (요구사항 3)
# ================================================================
def _verify_arm_independence(arm_obj, ref_mats, action_name, bake_chain=None):
    """
    진단 IK 를 mute 하고 Action 단독으로 팔/어깨/Spine 자세를 재현할 수 있는지 검증.
    bake_chain: None 이면 ref_mats.keys() 사용.
    성공 기준: 전체 max_diff < 1e-4
    검증 후 진단 IK 를 unmute (이미 제거된 경우 무시).

    반환: (max_diff: float, ok: bool)
    """
    chain = bake_chain if bake_chain else list(ref_mats.keys())

    # 진단 IK 일시 mute (이미 제거됐으면 con_l/r = None → 무시)
    pb_l  = arm_obj.pose.bones.get(BONE_L_FOREARM)
    pb_r  = arm_obj.pose.bones.get(BONE_R_FOREARM)
    con_l = next((c for c in (pb_l.constraints if pb_l else [])
                  if c.name == CON_L_NAME), None)
    con_r = next((c for c in (pb_r.constraints if pb_r else [])
                  if c.name == CON_R_NAME), None)
    if con_l: con_l.mute = True
    if con_r: con_r.mute = True

    anchor_bn   = chain[0] if chain else None
    ref_anc_inv = (ref_mats[anchor_bn].inverted()
                   if anchor_bn and anchor_bn in ref_mats
                   else mathutils.Matrix.Identity(4))
    ref_anc_basis = ref_mats.get('__anchor_basis__')   # anchor local pose 기준

    # frame 1, 11, 21, 31 모두 검사
    print(f"\n  [독립성 검증] {action_name} — 비교 (4프레임, {len(chain)}뼈):")
    bone_max = {bn: 0.0 for bn in chain if bn in ref_mats}
    max_diff = 0.0
    for (frame, _, _, label) in COMBOS:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        pb_anc      = arm_obj.pose.bones.get(anchor_bn) if anchor_bn else None
        cur_anc_inv = pb_anc.matrix.inverted() if pb_anc else mathutils.Matrix.Identity(4)
        for bn in chain:
            pb = arm_obj.pose.bones.get(bn)
            if not pb or bn not in ref_mats:
                continue
            if bn == anchor_bn and ref_anc_basis is not None:
                # anchor: local pose (matrix_basis) 직접 비교 — identity 함정 방지
                cur_b = pb.matrix_basis
                diff  = max(abs(cur_b[r][c] - ref_anc_basis[r][c])
                            for r in range(4) for c in range(4))
            else:
                # Shoulder/Arm/ForeArm: anchor 기준 상대 4×4 비교
                cur_rel = cur_anc_inv @ pb.matrix
                ref_rel = ref_anc_inv @ ref_mats[bn]
                diff    = max(abs(cur_rel[r][c] - ref_rel[r][c])
                              for r in range(4) for c in range(4))
            bone_max[bn] = max(bone_max[bn], diff)
            max_diff      = max(max_diff, diff)

    # 핵심 뼈만 출력 (Shoulder/Arm/ForeArm)
    key_bones = [BONE_L_SHOULDER, BONE_R_SHOULDER,
                 BONE_L_ARM, BONE_R_ARM, BONE_L_FOREARM, BONE_R_FOREARM]
    for bn in key_bones:
        if bn in bone_max:
            print(f"    {bn}: max(1/11/21/31) = {bone_max[bn]:.2e}")
    other_max = max((d for bn, d in bone_max.items() if bn not in key_bones), default=0.0)
    if other_max > 0:
        print(f"    [Spine 등 기타]: max = {other_max:.2e}")

    ok = max_diff < 1e-4
    status = "성공 (< 1e-4)" if ok else "실패 (>= 1e-4) WARNING"
    print(f"  [독립성 검증] 전체 max_diff = {max_diff:.2e}  [{status}]")

    # 진단 IK 복원 (다음 타겟 또는 최종 제거 전까지)
    if con_l: con_l.mute = False
    if con_r: con_r.mute = False

    return max_diff, ok


# ================================================================
# 손가락 FK 적용
# ================================================================
def apply_fingers(arm_obj, hand_bone, pts_21, eff_pn, chains, flip_dir, frame):
    hand_mat = bone_arma_mat(arm_obj, hand_bone)
    for chain in chains:
        first_pmat = get_parent_mat(arm_obj, chain[0][0], hand_bone, hand_mat)
        p_mat = first_pmat
        for (bn, fi, ti) in chain:
            fw = pts_21[fi]
            tw = pts_21[ti]
            if fw is None or tw is None:
                p_mat = bone_arma_mat(arm_obj, bn, p_mat)
                continue
            if flip_dir:
                fw, tw = tw, fw
            ok, new_mat = set_bone_dir(arm_obj, bn, fw, tw,
                                       parent_mat=p_mat, roll_pn=eff_pn)
            if ok:
                arm_obj.pose.bones[bn].keyframe_insert('rotation_quaternion', frame=frame)
                p_mat = new_mat
            else:
                p_mat = bone_arma_mat(arm_obj, bn, p_mat)


def set_constant_interpolation(action):
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'CONSTANT'
        fc.update()


# ================================================================
# DiagPalm Action 생성 (팔 FK 베이킹 포함)
# ================================================================
def build_diag_action(arm_obj, action_name, fd, bake_chain):
    """
    단일 동작에 대해 DiagPalm Action 생성.
    1) 진단 IK 활성 상태에서 Spine/Shoulder/Arm/ForeArm ref_mats 캡처
    2) Action 생성 (use_fake_user=True)
    3) 팔+어깨+Spine FK 역산 → 모든 combo frame 에 동일하게 키 삽입
    4) 각 combo frame 에 손/손가락 FK 적용 및 키 삽입
    5) CONSTANT 보간
    6) 독립성 검증 (IK mute 후 bake_chain 전체 행렬 비교)

    반환: (action, ok, ref_mats)
    """
    SEP = '-' * 62

    # ── 팔+어깨+Spine IK 참조 행렬 캡처 ──────────────────────────
    print(f"\n{SEP}")
    print(f"[DiagPalm] {action_name}  (ts={fd['ts']}ms)")
    ref_mats = _capture_arm_ref_mats(arm_obj, bake_chain)
    print(f"  bake chain ({len(bake_chain)}뼈): {bake_chain}")

    # ── 기존 DiagPalm Action 안전 제거 ───────────────────────────
    old = bpy.data.actions.get(action_name)
    if old:
        # DiagPalm_ 접두어 확인 — production Action 은 건드리지 않음
        if old.name.startswith('DiagPalm_'):
            if arm_obj.animation_data and arm_obj.animation_data.action is old:
                arm_obj.animation_data.action = None
            bpy.data.actions.remove(old)
            print(f"  기존 DiagPalm Action 제거: {action_name}")
        else:
            print(f"  [경고] 동일 이름 비-DiagPalm Action 존재: {action_name} — 덮어쓰지 않음")
            return None, False, ref_mats

    # ── Action 생성 ───────────────────────────────────────────────
    action = bpy.data.actions.new(name=action_name)
    action.use_fake_user = True          # GC 방지
    arm_obj.animation_data.action = action
    print(f"  Action 생성: {action_name}  use_fake_user={action.use_fake_user}")

    # ── 팔 FK 베이킹 (모든 combo frame 에 동일한 자세) ─────────────
    frames = [c[0] for c in COMBOS]
    keyed_bones = _bake_arm_fk(arm_obj, action, ref_mats, frames, bake_chain)

    # ── 각 combo frame 에 손/손가락 FK ────────────────────────────
    l_pts    = fd['l_pts']
    r_pts    = fd['r_pts']
    l_pn_raw = fd['l_pn_raw']
    r_pn_raw = fd['r_pn_raw']
    lh0 = fd['lh0']; lh9 = fd['lh9']
    rh0 = fd['rh0']; rh9 = fd['rh9']

    print(f"  SWAP_HAND_LABELS={SWAP_HAND_LABELS}  "
          f"has_L={fd['has_l']}  has_R={fd['has_r']}")

    if lh0 and lh9:
        fwd = (lh9 - lh0).normalized()
        print(f"  LEFT  wrist->mid9: ({fwd.x:+.3f},{fwd.y:+.3f},{fwd.z:+.3f})")
    if rh0 and rh9:
        fwd = (rh9 - rh0).normalized()
        print(f"  RIGHT wrist->mid9: ({fwd.x:+.3f},{fwd.y:+.3f},{fwd.z:+.3f})")
    if l_pn_raw:
        n = l_pn_raw.normalized()
        print(f"  LEFT  raw palm normal: ({n.x:+.3f},{n.y:+.3f},{n.z:+.3f})")
    if r_pn_raw:
        n = r_pn_raw.normalized()
        print(f"  RIGHT raw palm normal: ({n.x:+.3f},{n.y:+.3f},{n.z:+.3f})")

    for (frame, l_flip, r_flip, label) in COMBOS:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        # effective_pn — 부호만 변경
        eff_pn_l = (-l_pn_raw if l_flip else l_pn_raw) \
                   if (l_pn_raw and l_pn_raw.length > 1e-4) else None
        eff_pn_r = (-r_pn_raw if r_flip else r_pn_raw) \
                   if (r_pn_raw and r_pn_raw.length > 1e-4) else None

        print(f"\n  [{label}] frame={frame}  L_flip={l_flip}  R_flip={r_flip}")
        if eff_pn_l:
            en = eff_pn_l.normalized()
            print(f"    LEFT  eff_pn: ({en.x:+.3f},{en.y:+.3f},{en.z:+.3f})")
        if eff_pn_r:
            en = eff_pn_r.normalized()
            print(f"    RIGHT eff_pn: ({en.x:+.3f},{en.y:+.3f},{en.z:+.3f})")

        # LEFT hand + fingers (flip_palm=False: effective_pn 에 이미 부호 반영)
        if fd['has_l'] and lh0 and lh9 and eff_pn_l:
            set_hand_rotation(arm_obj, BONE_L_HAND, lh0, lh9, eff_pn_l, False)
            arm_obj.pose.bones[BONE_L_HAND].keyframe_insert(
                'rotation_quaternion', frame=frame)
            apply_fingers(arm_obj, BONE_L_HAND, l_pts, eff_pn_l,
                          FINGER_CHAINS_L, FLIP_FINGER_DIR_L, frame)

        # RIGHT hand + fingers
        if fd['has_r'] and rh0 and rh9 and eff_pn_r:
            set_hand_rotation(arm_obj, BONE_R_HAND, rh0, rh9, eff_pn_r, False)
            arm_obj.pose.bones[BONE_R_HAND].keyframe_insert(
                'rotation_quaternion', frame=frame)
            apply_fingers(arm_obj, BONE_R_HAND, r_pts, eff_pn_r,
                          FINGER_CHAINS_R, FLIP_FINGER_DIR_R, frame)

    # ── CONSTANT 보간 ─────────────────────────────────────────────
    set_constant_interpolation(action)
    print(f"\n  [보간] CONSTANT 설정 완료")

    # ── 키된 뼈 목록 출력 ────────────────────────────────────────
    keyed_all = sorted(set(fc.data_path.split('"')[1]
                           for fc in action.fcurves
                           if '"' in fc.data_path))
    print(f"  [키된 뼈] {len(keyed_all)}개: {keyed_all}")

    # ── 독립성 검증 (요구사항 3) ──────────────────────────────────
    max_diff, ok = _verify_arm_independence(arm_obj, ref_mats, action_name, bake_chain)

    return action, ok, ref_mats


# ================================================================
# mute된 제약 목록 출력 (요구사항 9)
# ================================================================
def _print_muted_constraints(arm_obj, snap):
    """스냅샷 대비 새로 mute 된 제약 목록 출력."""
    orig_muted = set()
    for bn, cons_snap in snap.get('constraints', {}).items():
        for (c_name, c_mute) in cons_snap:
            if c_mute:
                orig_muted.add(f"{bn}/{c_name}")

    now_muted = set()
    for pb in arm_obj.pose.bones:
        for c in pb.constraints:
            if c.mute:
                now_muted.add(f"{pb.name}/{c.name}")

    newly_muted = sorted(now_muted - orig_muted)
    print(f"\n  [mute된 제약] 새로 mute 된 {len(newly_muted)}개:")
    for entry in newly_muted:
        print(f"    {entry}")


# ================================================================
# 메인
# ================================================================
def main():
    global _DIAG_SNAP, _diag_arm_obj, _diag_emp_lw, _diag_emp_rw

    arm_obj = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if not arm_obj:
        print("[오류] 아마추어 없음")
        return
    print(f"[정보] 아마추어: {arm_obj.name}")

    # ── 원본 상태: mode/active 변경 전에 저장 ────────────────────
    orig_active = bpy.context.view_layer.objects.active
    orig_mode   = bpy.context.mode

    arm_obj.animation_data_create()
    bpy.context.view_layer.objects.active = arm_obj

    # ── 모드 정규화 ──────────────────────────────────────────────
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='POSE')

    # ── 스냅샷 (변형 전) ─────────────────────────────────────────
    _DIAG_SNAP               = _take_snapshot(arm_obj)
    _DIAG_SNAP['orig_active'] = orig_active  # mode 변경 전 원본 active object
    _DIAG_SNAP['orig_mode']   = orig_mode    # 원본 mode (OBJECT/POSE/…)
    _diag_arm_obj = arm_obj
    print(f"[스냅샷] 원본 Action: "
          f"{_DIAG_SNAP['action'].name if _DIAG_SNAP['action'] else 'None'}")
    print(f"[스냅샷] 씬 frame={_DIAG_SNAP['frame']}  "
          f"range={_DIAG_SNAP['frame_start']}-{_DIAG_SNAP['frame_end']}")
    total_cons = sum(len(v) for v in _DIAG_SNAP['constraints'].values())
    print(f"[스냅샷] 제약 총 {total_cons}개 저장")

    # ── 캐릭터 어깨 기준 ─────────────────────────────────────────
    lb = arm_obj.pose.bones.get(BONE_L_ARM)
    rb = arm_obj.pose.bones.get(BONE_R_ARM)
    if lb and rb:
        l_sh_w      = arm_obj.matrix_world @ lb.head
        r_sh_w      = arm_obj.matrix_world @ rb.head
        char_origin = (l_sh_w + r_sh_w) / 2
        char_sw     = (l_sh_w - r_sh_w).length
    else:
        char_origin = mathutils.Vector((0, 0, 0))
        char_sw     = 1.5
    print(f"[정보] 어깨 중심: {char_origin}  너비: {char_sw:.3f}")

    # ── 진단용 IK Empty ───────────────────────────────────────────
    bpy.ops.object.mode_set(mode='OBJECT')
    col = bpy.data.collections.get('DiagPalm_IK')
    if not col:
        col = bpy.data.collections.new('DiagPalm_IK')
        bpy.context.scene.collection.children.link(col)
    emp_lw = get_or_create_empty('DiagPalm_IK_L_Wrist', col)
    emp_rw = get_or_create_empty('DiagPalm_IK_R_Wrist', col)
    _diag_emp_lw = emp_lw
    _diag_emp_rw = emp_rw
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    # ── 팔 bake chain 수집 (POSE 모드에서 부모 체인 동적 조사) ────
    bake_chain = _collect_arm_bake_chain(arm_obj)
    print(f"[정보] bake_chain ({len(bake_chain)}뼈): {bake_chain}")

    # ── 변형 시작 (try/except 로 전체 래핑) ─────────────────────
    success_actions = []
    all_ok = True

    try:
        # 1) 손/손가락 뼈 모든 제약 mute (remove 아님), rotation_mode=QUATERNION
        hand_bones = (
            [BONE_L_HAND, BONE_R_HAND] +
            [bn for chain in FINGER_CHAINS_L for (bn, _, _) in chain] +
            [bn for chain in FINGER_CHAINS_R for (bn, _, _) in chain]
        )
        muted_hand_cons = []
        for bn in hand_bones:
            b = arm_obj.pose.bones.get(bn)
            if b:
                newly = mute_all_constraints(arm_obj, bn)
                muted_hand_cons.extend([f"{bn}/{c}" for c in newly])
                b.rotation_mode = 'QUATERNION'
        print(f"\n[준비] 손/손가락 제약 mute: {len(muted_hand_cons)}개")

        # 2) Forearm 기존 IK mute + 진단 IK 추가
        mute_existing_ik(arm_obj, BONE_L_FOREARM, CON_L_NAME)
        mute_existing_ik(arm_obj, BONE_R_FOREARM, CON_R_NAME)
        add_named_ik(arm_obj, BONE_L_FOREARM, emp_lw, 2, CON_L_NAME)
        add_named_ik(arm_obj, BONE_R_FOREARM, emp_rw, 2, CON_R_NAME)
        print(f"[준비] 진단 IK 추가: {CON_L_NAME}, {CON_R_NAME}")

        # 3) 씬 프레임 범위 설정
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end   = 31

        # 4) 각 타겟 처리
        all_ref_mats = []   # [(action_name, ref_mats)] — 최종 검증용
        for target in DIAG_TARGETS:
            label       = target['label']
            json_path   = target['json_path']
            target_ts   = target['target_ts']
            action_name = target['action_name']

            print(f"\n{'='*62}")
            print(f"[타겟] {label}")

            fd, actual_ts = extract_frame_at_ts(
                json_path, target_ts, char_origin, char_sw)
            if fd is None:
                print(f"  [오류] 유효 프레임 없음 또는 어깨 미검출")
                all_ok = False
                continue

            diff_ms = abs(actual_ts - target_ts) if actual_ts is not None else -1
            print(f"  TimestampMs~{target_ts} -> 실제 {actual_ts}ms  (diff={diff_ms}ms)")

            # IK Empty 를 Pose wrist 위치에 고정 (애니메이션 없음)
            if fd['lw_ik']:
                emp_lw.location = fd['lw_ik']
                try:
                    emp_lw.keyframe_delete('location')
                except Exception:
                    pass
            if fd['rw_ik']:
                emp_rw.location = fd['rw_ik']
                try:
                    emp_rw.keyframe_delete('location')
                except Exception:
                    pass

            # Action 생성 + 팔 베이킹 + 독립성 검증
            action, ok, ref_mats = build_diag_action(arm_obj, action_name, fd, bake_chain)

            if action is None:
                all_ok = False
                continue

            all_ref_mats.append((action_name, ref_mats))
            if ok:
                success_actions.append(action_name)
                print(f"\n[완료] {action_name} — 독립성 검증 성공")
            else:
                all_ok = False
                print(f"\n[경고] {action_name} — 독립성 검증 실패 (팔 자세 오차 큼)")

        # ── 성공 후 상태 (요구사항 5) ─────────────────────────────
        print(f"\n{'='*62}")
        print("[정리] 진단 IK 제거 (Action 단독으로 팔 자세 재현)")
        remove_named_ik(arm_obj, BONE_L_FOREARM, CON_L_NAME)
        remove_named_ik(arm_obj, BONE_R_FOREARM, CON_R_NAME)

        # muted 제약 목록 출력
        _print_muted_constraints(arm_obj, _DIAG_SNAP)

        # ── 최종 검증: IK 완전 제거 후 Action 단독 재현 확인 ────────
        print("\n[최종 검증] 진단 IK 완전 제거 후 Action 단독 재현:")
        for aname, rmats in all_ref_mats:
            act_obj = bpy.data.actions.get(aname)
            if not act_obj or not rmats:
                continue
            arm_obj.animation_data.action = act_obj
            _, ok_final = _verify_arm_independence(arm_obj, rmats, aname + ' [IK제거후]', bake_chain)
            if not ok_final:
                all_ok = False

        # ── Action 전환 재검증: A→B→A ─────────────────────────────
        if len(all_ref_mats) >= 2:
            print("\n[전환 검증] Action A→B→A 전환 후 독립성 재확인:")
            for aname_sw, rmats_sw in [all_ref_mats[0], all_ref_mats[1], all_ref_mats[0]]:
                act_sw = bpy.data.actions.get(aname_sw)
                if act_sw and rmats_sw:
                    arm_obj.animation_data.action = act_sw
                    bpy.context.scene.frame_set(1)
                    bpy.context.view_layer.update()
                    _, ok_sw = _verify_arm_independence(
                        arm_obj, rmats_sw, aname_sw + ' [전환검증]', bake_chain)
                    if not ok_sw:
                        all_ok = False

        # Empty 숨김
        bpy.ops.object.mode_set(mode='OBJECT')
        emp_lw.hide_viewport = True
        emp_rw.hide_viewport = True
        bpy.ops.object.mode_set(mode='POSE')

        # 첫 번째 DiagPalm Action 활성화
        first_action_name = DIAG_TARGETS[0]['action_name']
        first_action = bpy.data.actions.get(first_action_name)
        if first_action:
            arm_obj.animation_data.action = first_action
            print(f"[상태] 활성 Action: {first_action_name}")

        # 씬 frame=1
        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()

    except Exception:
        # 오류 시 자동 복원 (요구사항 6)
        print("\n[오류] 예외 발생 — 자동 복원 시작:")
        traceback.print_exc()
        diag_palm_restore()
        return

    # ── 최종 출력 ────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("[종료] 진단 완료")
    print(f"  성공 Action: {success_actions}")
    print(f"  전체 결과: {'OK' if all_ok else 'WARNING (일부 실패)'}")
    print()
    print("  [프레임 매핑]")
    for (fr, lf, rf, lbl) in COMBOS:
        print(f"    frame {fr:2d}: L_flip={lf}  R_flip={rf}  [{lbl}]")
    print()
    print("  [복원 방법] Blender Python 콘솔에서:")
    print("    diag_palm_restore()")
    print(f"{'='*62}")


main()
