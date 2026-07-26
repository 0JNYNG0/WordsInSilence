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

# ================================================================
# 설정
# ================================================================
JSON_PATH        = r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\Recordings\감사합니다.json"
BODY_SCALE       = 1.0
DEPTH_SCALE      = 0.0
PALM_Z_SCALE     = 1.0
FORWARD_OFFSET   = 0.15
FPS              = 30
SMOOTH_WINDOW    = 5
JUMP_THRESHOLD   = 0.5
MIRROR_X         = True
SWAP_HAND_LABELS = False  # 전문가 비미러 영상: mp_right→캐릭터LEFT, mp_left→캐릭터RIGHT
L_PALM_FLIP      = False
R_PALM_FLIP      = True
FLIP_FINGER_DIR_L = False   # 왼손 손가락 방향이 반대면 True
FLIP_FINGER_DIR_R = False   # 오른손 손가락 방향이 반대면 True
HAND_DEPTH_GAIN  = 1.0      # Hand landmark 깊이 스케일 (0.25 / 0.5 / 1.0 비교)
INTRO_FRAMES     = 20

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
        bpy.data.objects[name].hide_viewport = True
        return bpy.data.objects[name]
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_size = 0.04
    obj.hide_viewport = True
    col.objects.link(obj)
    return obj


def add_ik(arm_obj, bone_name, target, chain_count):
    b = arm_obj.pose.bones.get(bone_name)
    if not b:
        print(f"[경고] 뼈 없음: {bone_name}"); return
    for c in b.constraints:
        if c.type == 'IK' and c.target == target: return
    c = b.constraints.new('IK')
    c.target = target; c.chain_count = chain_count; c.use_rotation = False


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
    bpy.context.scene.render.fps = FPS

    # 캐릭터 어깨 기준
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    lb = arm_obj.pose.bones.get(BONE_L_ARM)
    rb = arm_obj.pose.bones.get(BONE_R_ARM)
    if lb and rb:
        l_sh_w = arm_obj.matrix_world @ lb.head
        r_sh_w = arm_obj.matrix_world @ rb.head
        char_origin = (l_sh_w + r_sh_w) / 2
        char_sw     = (l_sh_w - r_sh_w).length
    else:
        char_origin = mathutils.Vector((0,0,0)); char_sw = 1.5
        l_sh_w = char_origin + mathutils.Vector((-char_sw/2,0,0))
        r_sh_w = char_origin + mathutils.Vector(( char_sw/2,0,0))
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[정보] 어깨 중심: {char_origin}  너비: {char_sw:.3f}")

    # ── 데이터 추출 ──────────────────────────────────────────────
    # l_p0..l_p20: 캐릭터 LEFT 손의 21개 랜드마크 Blender 월드 좌표
    # r_p0..r_p20: 캐릭터 RIGHT 손의 21개 랜드마크 Blender 월드 좌표
    raw = {
        'lw':[], 'rw':[],
        'lh9':[], 'rh9':[],
        'l_pn':[], 'r_pn':[],
        **{f'l_p{i}':[] for i in range(21)},
        **{f'r_p{i}':[] for i in range(21)},
        'l_valid':[], 'r_valid':[],
    }

    blender_frame = 0
    _DIAG_TS = 1033
    _diag_best_diff = float('inf')
    _diag_info = None
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
    smooth = {k: smooth_track(v) for k,v in raw.items() if k not in ('l_valid','r_valid')}

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
        idle = {'lw':idle_l,'rw':idle_r,'lh9':idle_l+hang,'rh9':idle_r+hang,
                'l_pn':mathutils.Vector((0,0,0)),'r_pn':mathutils.Vector((0,0,0)),
                **{f'l_p{i}':idle_l+hang for i in range(21)},
                **{f'r_p{i}':idle_r+hang for i in range(21)}}
        for k,v in idle.items():
            smooth[k] = [v.copy() for _ in range(INTRO_FRAMES)] + smooth[k]
        total += INTRO_FRAMES
        if l_first is not None: l_first += INTRO_FRAMES
        if r_first is not None: r_first += INTRO_FRAMES

    valid_frames = list(range(1, total+1))

    # ── IK Empty (팔 손목만) ─────────────────────────────────────
    col = bpy.data.collections.get("IK_Targets")
    if not col:
        col = bpy.data.collections.new("IK_Targets")
        bpy.context.scene.collection.children.link(col)
    emp_lw = make_empty("IK_L_Wrist", col)
    emp_rw = make_empty("IK_R_Wrist", col)

    # ── 컨스트레인트: 팔 IK만, 손/손가락은 모두 제거 ─────────────
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    add_ik(arm_obj, BONE_L_FOREARM, emp_lw, chain_count=2)
    add_ik(arm_obj, BONE_R_FOREARM, emp_rw, chain_count=2)

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
        emp_lw.location = lw
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
    animate_empty(emp_lw, smooth['lw'], valid_frames)
    animate_empty(emp_rw, smooth['rw'], valid_frames)

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

    for i, frame in enumerate(valid_frames):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()   # 팔 IK 평가

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

    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end   = total
    bpy.context.scene.frame_set(1)

    print(f"[완료] {total}프레임  액션: {action_name}")
    print("  손바닥 뒤집힘 → L_PALM_FLIP / R_PALM_FLIP = True")
    print("  손가락 방향 반대 → FLIP_FINGER_DIR_L / FLIP_FINGER_DIR_R = True")
    print("  손 반대쪽에 붙음 → SWAP_HAND_LABELS = not SWAP_HAND_LABELS")


main()
