"""
Tools/diag_palm_flip_sequence.py
------------------------------------------------------------
Palm flip 네 조합 시각 비교용 진단 스크립트.

대상:
  감사합니다.json @ TimestampMs~1033  -> DiagPalm_감사합니다_1033
  빵.json        @ TimestampMs~1033  -> DiagPalm_빵_1033

각 Action 프레임 레이아웃:
  frame  1 : L_PALM_FLIP=False R_PALM_FLIP=False  [FF]
  frame 11 : L_PALM_FLIP=False R_PALM_FLIP=True   [FT_CURRENT]
  frame 21 : L_PALM_FLIP=True  R_PALM_FLIP=False  [TF_OPPOSITE]
  frame 31 : L_PALM_FLIP=True  R_PALM_FLIP=True   [TT]

- 몸/상완/전완/손목 IK 위치는 네 프레임 모두 동일
- 변경되는 것은 LEFT/RIGHT effective_pn 부호뿐
- CONSTANT 보간 -> 상태 사이 중간 자세 없음
- production blend_animate.py 는 수정하지 않음

실행:
  blender --background --python Tools/diag_palm_flip_sequence.py

생성 액션:
  DiagPalm_감사합니다_1033
  DiagPalm_빵_1033
------------------------------------------------------------
"""

import bpy, json, math, mathutils

# ================================================================
# 설정
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

# blend_animate.py 확정 설정과 동일
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

# 진단용 IK 제약조건 이름 (production 제약과 구별)
CON_L_NAME = 'DiagPalm_IK_L'
CON_R_NAME = 'DiagPalm_IK_R'

# 상완/전완 일관성 검증 대상 뼈
ARM_CHECK_BONES = [BONE_L_ARM, BONE_L_FOREARM, BONE_R_ARM, BONE_R_FOREARM]


# ================================================================
# 유틸 - blend_animate.py 와 동일한 수식
# ================================================================
def get_pt(pts, idx):
    if not pts or idx >= len(pts): return None
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
    if not b or not bd: return False
    hand_dir = (mid9_w - wrist_w)
    if hand_dir.length < 1e-6: return False
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
        if flip_palm: pn_a = -pn_a
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
        if not fd.get('HasPose', False): continue
        ts = fd.get('TimestampMs')
        if ts is None: continue
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
    sw = math.sqrt((ls[0]-rs[0])**2 + (ls[1]-rs[1])**2)
    if sw < 1e-4:
        return None, actual_ts

    def bp(idx):
        pt = get_pt(pose_pts, idx)
        return to_blender(*pt, cx, cy, sw, char_origin, char_sw) if pt else None

    def chp(idx, pts):
        pt = get_pt(pts, idx)
        return convert_hand_pt(pt[0], pt[1], pt[2], cx, cy, sw, char_origin, char_sw) if pt else None

    # IK 타겟 = Pose wrist (blend_animate.py 동일)
    lw_ik = bp(P_L_WRIST)
    rw_ik = bp(P_R_WRIST)

    # Hand 21 landmarks (Blender world)
    l_pts_21 = [chp(i, ch_l_pts) if has_ch_l else None for i in range(21)]
    r_pts_21 = [chp(i, ch_r_pts) if has_ch_r else None for i in range(21)]

    # Palm normal raw: (pt5-pt0) x (pt17-pt0)
    lw3 = chp(H_WRIST, ch_l_pts)
    li3 = chp(H_INDEX_MCP, ch_l_pts)
    lp3 = chp(H_PINK_MCP,  ch_l_pts)
    if has_ch_l and lw3 and li3 and lp3:
        v = (li3 - lw3).cross(lp3 - lw3)
        l_pn_raw = v if v.length > 1e-6 else None
    else:
        l_pn_raw = None

    rw3 = chp(H_WRIST, ch_r_pts)
    ri3 = chp(H_INDEX_MCP, ch_r_pts)
    rp3 = chp(H_PINK_MCP,  ch_r_pts)
    if has_ch_r and rw3 and ri3 and rp3:
        v = (ri3 - rw3).cross(rp3 - rw3)
        r_pn_raw = v if v.length > 1e-6 else None
    else:
        r_pn_raw = None

    # Hand direction landmarks (pt0 -> pt9)
    lh0 = chp(H_WRIST,   ch_l_pts) if has_ch_l else None
    lh9 = chp(H_MID_MCP, ch_l_pts) if has_ch_l else None
    rh0 = chp(H_WRIST,   ch_r_pts) if has_ch_r else None
    rh9 = chp(H_MID_MCP, ch_r_pts) if has_ch_r else None

    return {
        'ts':      actual_ts,
        'lw_ik':   lw_ik,  'rw_ik': rw_ik,
        'l_pts':   l_pts_21, 'r_pts': r_pts_21,
        'l_pn_raw': l_pn_raw, 'r_pn_raw': r_pn_raw,
        'lh0': lh0, 'lh9': lh9,
        'rh0': rh0, 'rh9': rh9,
        'has_l': has_ch_l, 'has_r': has_ch_r,
    }, actual_ts


# ================================================================
# 제약조건 관리
# ================================================================
def mute_existing_ik(arm_obj, bone_name, exclude_con_name):
    """named constraint 제외한 기존 IK 제약을 mute. muted 목록 반환."""
    b = arm_obj.pose.bones.get(bone_name)
    if not b: return []
    muted = []
    for c in b.constraints:
        if c.type == 'IK' and c.name != exclude_con_name and not c.mute:
            c.mute = True
            muted.append(c.name)
    return muted


def unmute_constraints(arm_obj, bone_name, con_names):
    b = arm_obj.pose.bones.get(bone_name)
    if not b: return
    for c in b.constraints:
        if c.name in con_names:
            c.mute = False


def add_named_ik(arm_obj, bone_name, target, chain_count, con_name):
    b = arm_obj.pose.bones.get(bone_name)
    if not b: return
    for c in b.constraints:
        if c.name == con_name: return
    c = b.constraints.new('IK')
    c.name       = con_name
    c.target     = target
    c.chain_count = chain_count
    c.use_rotation = False


def remove_named_ik(arm_obj, bone_name, con_name):
    b = arm_obj.pose.bones.get(bone_name)
    if not b: return
    for c in list(b.constraints):
        if c.name == con_name:
            b.constraints.remove(c)


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
# 손가락 FK 적용 + 키프레임 삽입
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
# DiagPalm Action 생성
# ================================================================
def build_diag_action(arm_obj, action_name, fd):
    """
    단일 동작에 대해 DiagPalm Action 생성.
    fd: extract_frame_at_ts() 반환값 (frame_data dict)
    arm_obj: POSE 모드이어야 함.
    """
    old = bpy.data.actions.get(action_name)
    if old:
        bpy.data.actions.remove(old)
    action = bpy.data.actions.new(name=action_name)
    arm_obj.animation_data.action = action

    l_pts    = fd['l_pts']
    r_pts    = fd['r_pts']
    l_pn_raw = fd['l_pn_raw']
    r_pn_raw = fd['r_pn_raw']
    lh0 = fd['lh0']; lh9 = fd['lh9']
    rh0 = fd['rh0']; rh9 = fd['rh9']

    SEP = '-' * 60
    print(f"\n{SEP}")
    print(f"[DiagPalm] {action_name}  (ts={fd['ts']}ms)")
    print(f"  SWAP_HAND_LABELS={SWAP_HAND_LABELS}")
    print(f"  has_L={fd['has_l']}  has_R={fd['has_r']}")

    # forward 방향 출력
    if lh0 and lh9:
        fwd = (lh9 - lh0).normalized()
        print(f"  LEFT  wrist->mid9 (world): ({fwd.x:+.3f},{fwd.y:+.3f},{fwd.z:+.3f})")
    else:
        print(f"  LEFT  wrist->mid9: 없음")
    if rh0 and rh9:
        fwd = (rh9 - rh0).normalized()
        print(f"  RIGHT wrist->mid9 (world): ({fwd.x:+.3f},{fwd.y:+.3f},{fwd.z:+.3f})")
    else:
        print(f"  RIGHT wrist->mid9: 없음")

    # raw palm normal 출력
    if l_pn_raw:
        n = l_pn_raw.normalized()
        print(f"  LEFT  raw palm normal: ({n.x:+.3f},{n.y:+.3f},{n.z:+.3f})")
    else:
        print(f"  LEFT  raw palm normal: 없음")
    if r_pn_raw:
        n = r_pn_raw.normalized()
        print(f"  RIGHT raw palm normal: ({n.x:+.3f},{n.y:+.3f},{n.z:+.3f})")
    else:
        print(f"  RIGHT raw palm normal: 없음")

    # 네 조합 적용
    for (frame, l_flip, r_flip, label) in COMBOS:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()   # arm IK 평가

        # effective_pn (부호만 다름)
        if l_pn_raw and l_pn_raw.length > 1e-4:
            eff_pn_l = -l_pn_raw if l_flip else l_pn_raw
        else:
            eff_pn_l = None

        if r_pn_raw and r_pn_raw.length > 1e-4:
            eff_pn_r = -r_pn_raw if r_flip else r_pn_raw
        else:
            eff_pn_r = None

        print(f"\n  [{label}] frame={frame}  L_flip={l_flip}  R_flip={r_flip}")
        if eff_pn_l:
            en = eff_pn_l.normalized()
            print(f"    LEFT  eff_pn: ({en.x:+.3f},{en.y:+.3f},{en.z:+.3f})")
        if eff_pn_r:
            en = eff_pn_r.normalized()
            print(f"    RIGHT eff_pn: ({en.x:+.3f},{en.y:+.3f},{en.z:+.3f})")

        # LEFT hand + fingers (flip_palm=False — effective_pn 이미 적용)
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

    set_constant_interpolation(action)
    print(f"\n  [보간] {action_name} -> CONSTANT 완료")
    return action


# ================================================================
# 상완/전완 일관성 검증 (요구사항 10)
# ================================================================
def verify_arm_consistency(arm_obj, combo_frames, combo_labels):
    """
    네 프레임에서 상완/전완 pose matrix 가 동일한지 검증.
    IK empty가 고정이면 차이는 부동소수점 오차 수준이어야 함.
    """
    mats = {bn: [] for bn in ARM_CHECK_BONES}
    for frame in combo_frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for bn in ARM_CHECK_BONES:
            b = arm_obj.pose.bones.get(bn)
            mats[bn].append(b.matrix.copy() if b else None)

    ref_label = combo_labels[0]
    print(f"\n  [검증] 상완/전완 pose matrix 일관성 (기준: {ref_label})")
    max_global = 0.0
    for bn in ARM_CHECK_BONES:
        ref = mats[bn][0]
        if ref is None:
            print(f"    {bn}: 뼈 없음")
            continue
        max_diff = 0.0
        for i in range(1, len(combo_frames)):
            m = mats[bn][i]
            if m is None: continue
            diff = max(abs(m[r][c] - ref[r][c])
                       for r in range(4) for c in range(4))
            max_diff = max(max_diff, diff)
        max_global = max(max_global, max_diff)
        label_str = '/'.join(combo_labels[1:])
        print(f"    {bn}  vs [{label_str}]: max_diff = {max_diff:.2e}")
    print(f"    전체 최대 차이: {max_global:.2e}  "
          f"({'OK - 부동소수점 오차 수준' if max_global < 1e-4 else 'WARNING - 차이 있음'})")
    return max_global


# ================================================================
# 메인
# ================================================================
def main():
    arm_obj = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if not arm_obj:
        print("[오류] 아마추어 없음"); return
    print(f"[정보] 아마추어: {arm_obj.name}")

    # 원래 상태 저장
    arm_obj.animation_data_create()
    orig_action = arm_obj.animation_data.action

    # 캐릭터 어깨 기준
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    lb = arm_obj.pose.bones.get(BONE_L_ARM)
    rb = arm_obj.pose.bones.get(BONE_R_ARM)
    if lb and rb:
        l_sh_w    = arm_obj.matrix_world @ lb.head
        r_sh_w    = arm_obj.matrix_world @ rb.head
        char_origin = (l_sh_w + r_sh_w) / 2
        char_sw     = (l_sh_w - r_sh_w).length
    else:
        char_origin = mathutils.Vector((0, 0, 0))
        char_sw     = 1.5
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[정보] 어깨 중심: {char_origin}  너비: {char_sw:.3f}")

    # 진단용 IK empty
    col = bpy.data.collections.get("DiagPalm_IK")
    if not col:
        col = bpy.data.collections.new("DiagPalm_IK")
        bpy.context.scene.collection.children.link(col)
    emp_lw = get_or_create_empty("DiagPalm_IK_L_Wrist", col)
    emp_rw = get_or_create_empty("DiagPalm_IK_R_Wrist", col)

    # POSE 모드: 기존 IK mute + 진단 IK 추가 + 손/손가락 FK 전환
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    muted_l = mute_existing_ik(arm_obj, BONE_L_FOREARM, CON_L_NAME)
    muted_r = mute_existing_ik(arm_obj, BONE_R_FOREARM, CON_R_NAME)
    add_named_ik(arm_obj, BONE_L_FOREARM, emp_lw, 2, CON_L_NAME)
    add_named_ik(arm_obj, BONE_R_FOREARM, emp_rw, 2, CON_R_NAME)

    # 손/손가락 뼈 FK 전환 (기존 제약 제거, QUATERNION 모드)
    hand_bones = (
        [BONE_L_HAND, BONE_R_HAND] +
        [bn for chain in FINGER_CHAINS_L for (bn, _, _) in chain] +
        [bn for chain in FINGER_CHAINS_R for (bn, _, _) in chain]
    )
    for bn in hand_bones:
        b = arm_obj.pose.bones.get(bn)
        if b:
            for c in list(b.constraints): b.constraints.remove(c)
            b.rotation_mode = 'QUATERNION'

    bpy.ops.object.mode_set(mode='OBJECT')

    combo_frames = [c[0] for c in COMBOS]
    combo_labels = [c[3] for c in COMBOS]

    try:
        for target in DIAG_TARGETS:
            label       = target['label']
            json_path   = target['json_path']
            target_ts   = target['target_ts']
            action_name = target['action_name']

            print(f"\n{'='*62}")
            print(f"[타겟] {label}")

            fd, actual_ts = extract_frame_at_ts(json_path, target_ts, char_origin, char_sw)
            if fd is None:
                print(f"  [오류] 유효 프레임 없음 또는 어깨 미검출")
                continue
            print(f"  TimestampMs~{target_ts} -> 실제 {actual_ts}ms"
                  f"  (diff={abs(actual_ts - target_ts)}ms)")

            # IK empty를 Pose wrist 위치에 고정 (애니메이션 없음)
            if fd['lw_ik']:
                emp_lw.location = fd['lw_ik']
                emp_lw.keyframe_delete('location')  # 기존 키프레임 제거
            if fd['rw_ik']:
                emp_rw.location = fd['rw_ik']
                emp_rw.keyframe_delete('location')

            bpy.context.scene.frame_start = 1
            bpy.context.scene.frame_end   = 31

            bpy.context.view_layer.objects.active = arm_obj
            bpy.ops.object.mode_set(mode='POSE')

            build_diag_action(arm_obj, action_name, fd)
            verify_arm_consistency(arm_obj, combo_frames, combo_labels)

            bpy.ops.object.mode_set(mode='OBJECT')
            print(f"\n[완료] {action_name}")

    finally:
        # 정리: 진단 IK 제약 제거, 기존 IK 복원, empty 숨김, action 복원
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='POSE')
        remove_named_ik(arm_obj, BONE_L_FOREARM, CON_L_NAME)
        remove_named_ik(arm_obj, BONE_R_FOREARM, CON_R_NAME)
        unmute_constraints(arm_obj, BONE_L_FOREARM, muted_l)
        unmute_constraints(arm_obj, BONE_R_FOREARM, muted_r)
        bpy.ops.object.mode_set(mode='OBJECT')
        emp_lw.hide_viewport = True
        emp_rw.hide_viewport = True
        arm_obj.animation_data.action = orig_action

    # 최종 안내
    print(f"\n{'='*62}")
    print("[종료] 진단 완료")
    print("  생성 액션 (Dope Sheet > Action Editor에서 선택):")
    for t in DIAG_TARGETS:
        print(f"    {t['action_name']}")
    print("\n  프레임 매핑 (각 Action 공통):")
    for (fr, lf, rf, lbl) in COMBOS:
        print(f"    frame {fr:2d}: L_flip={lf}  R_flip={rf}  [{lbl}]")
    print(f"{'='*62}")


main()
