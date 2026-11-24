"""
Tools/diag_hand_basis3_safe_v2.py
────────────────────────────────────────────────────────────────
Hand bone rotation 방향 진단 — 안전 버전 v2

v2 변경 사항 (diag_hand_basis3_safe.py 대비):
  [1] Hand/Finger constraint 삭제 → mute snapshot으로 교체
  [2] diag3_restore() 전역 함수 정의 (한 줄 원복)
  [3] Empty 제거 시 do_unlink=True
  [4] 실행 전 상태 snapshot (_DIAG3_STATE)
  [5] 예외 발생 시 자동 원복 (정상 완료 시에는 시각 비교 상태 유지)
  [6] DIAG3_ 잔여물 시작 시 정리 (do_unlink=True)

실행:
    exec(open(r"C:\\MyUnityProjects\\WordsInSilence\\Tools\\diag_hand_basis3_safe_v2.py").read())

원복:
    diag3_restore()
────────────────────────────────────────────────────────────────
"""

import bpy, json, math, mathutils, traceback

# ================================================================
# 설정
# ================================================================
JSON_PATH             = r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\Recordings\감사합니다.json"
BODY_SCALE            = 1.0
DEPTH_SCALE           = 0.0
FORWARD_OFFSET        = 0.15
HAND_DEPTH_GAIN       = 1.0
MIRROR_X              = True
SWAP_HAND_LABELS      = True       # 전문가 비미러 영상: mp_right→캐릭터LEFT
TEST_BOTH_VALID_INDEX = 19          # 양손 동시 유효 프레임 목록에서 몇 번째 사용
ARM_MATCH_TOL_DEG     = 1.0        # 팔 검증 허용 오차 (도)

P_L_SHOULDER = 11;  P_R_SHOULDER = 12
P_L_WRIST    = 15;  P_R_WRIST    = 16
H_WRIST      = 0;   H_INDEX_MCP  = 5
H_MID_MCP    = 9;   H_PINK_MCP   = 17

BONE_L_ARM     = 'mixamorig:LeftArm'
BONE_L_FOREARM = 'mixamorig:LeftForeArm'
BONE_L_HAND    = 'mixamorig:LeftHand'
BONE_R_ARM     = 'mixamorig:RightArm'
BONE_R_FOREARM = 'mixamorig:RightForeArm'
BONE_R_HAND    = 'mixamorig:RightHand'
ARM_BONES      = [BONE_L_ARM, BONE_L_FOREARM, BONE_R_ARM, BONE_R_FOREARM]

# 진단 전용 이름 — 기존 IK_L/R_Wrist, ForeArm IK constraint 불침
DIAG_EMPTY_L  = "DIAG3_IK_L_Wrist"
DIAG_EMPTY_R  = "DIAG3_IK_R_Wrist"
DIAG_CONSTR_L = "DIAG3_IK_LEFT"
DIAG_CONSTR_R = "DIAG3_IK_RIGHT"

REST_BONES_L = {'hand':    BONE_L_HAND,
                'index1':  'mixamorig:LeftHandIndex1',
                'middle1': 'mixamorig:LeftHandMiddle1',
                'pinky1':  'mixamorig:LeftHandPinky1'}
REST_BONES_R = {'hand':    BONE_R_HAND,
                'index1':  'mixamorig:RightHandIndex1',
                'middle1': 'mixamorig:RightHandMiddle1',
                'pinky1':  'mixamorig:RightHandPinky1'}


# ================================================================
# [수정4] 전역 상태 — diag3_restore() 접근용
# ================================================================
_DIAG3_STATE = {}


# ================================================================
# 유틸
# ================================================================
def get_pt(pts, idx):
    if not pts or idx >= len(pts): return None
    p = pts[idx]
    return (p.get('x', 0), p.get('y', 0), p.get('z', 0)) if isinstance(p, dict) else None


def to_blender(x, y, z, cx, cy, sw, origin, char_sw):
    sc = char_sw * BODY_SCALE; xs = 1 if MIRROR_X else -1
    return origin + mathutils.Vector((
        xs * (x - cx) / sw * sc,
        z * sc * DEPTH_SCALE - FORWARD_OFFSET,
        -(y - cy) / sw * sc,
    ))


def convert_hand_pt(x, y, z, cx, cy, sw, origin, char_sw):
    sc = char_sw * BODY_SCALE; xs = 1 if MIRROR_X else -1
    return origin + mathutils.Vector((
        xs * (x - cx) / sw * sc,
        z / sw * sc * HAND_DEPTH_GAIN - FORWARD_OFFSET,
        -(y - cy) / sw * sc,
    ))


def angle_deg(a, b):
    try:    return math.degrees(a.normalized().angle(b.normalized()))
    except: return 999.0


def mat3_angle_deg(m1, m2):
    return max(angle_deg(m1.col[i], m2.col[i]) for i in range(3))


def det3(x, y, z):
    return x.dot(y.cross(z))


# ================================================================
# [수정1] Hand/Finger constraint mute snapshot
# ================================================================
def suspend_hand_finger_constraints(arm_obj, bone_names):
    """
    지정 뼈의 기존 constraint mute 값을 저장 후 mute=True 로 설정.
    기존 constraint를 삭제하지 않음.
    반환: {(bone_name, constraint_name): orig_mute}
    """
    snapshot = {}
    for bone_name in bone_names:
        pb = arm_obj.pose.bones.get(bone_name)
        if not pb:
            continue
        for constraint in pb.constraints:
            snapshot[(bone_name, constraint.name)] = constraint.mute
            constraint.mute = True
    return snapshot


def restore_hand_finger_constraints(arm_obj, snapshot):
    """suspend_hand_finger_constraints 로 저장한 mute 값을 정확히 복원."""
    for (bone_name, constraint_name), old_mute in snapshot.items():
        pb = arm_obj.pose.bones.get(bone_name)
        if not pb:
            continue
        constraint = pb.constraints.get(constraint_name)
        if constraint:
            constraint.mute = old_mute


# ================================================================
# [수정2] 전역 원복 함수
# ================================================================
def diag3_restore():
    """
    diag_hand_basis3_safe_v2 실행 전 상태로 복원.
    정상 완료 후 수동 호출, 또는 예외 발생 시 자동 호출.

    사용:
        diag3_restore()
    """
    state = _DIAG3_STATE
    if not state:
        print("[diag3_restore] 상태 없음 — 이미 복원됐거나 실행 전입니다.")
        return

    arm_obj = state.get('arm_obj')

    if arm_obj and arm_obj.name in bpy.data.objects:
        # POSE 모드 전환
        try:
            bpy.context.view_layer.objects.active = arm_obj
            bpy.ops.object.mode_set(mode='POSE')
        except Exception as e:
            print(f"[diag3_restore] POSE 전환 실패: {e}")

        # 1. DIAG3_ IK constraint 제거
        for bn, cn in state.get('diag_constraints', []):
            pb = arm_obj.pose.bones.get(bn)
            if pb:
                for c in list(pb.constraints):
                    if c.name == cn:
                        pb.constraints.remove(c)
                        print(f"  [원복] {bn.replace('mixamorig:','')} '{cn}' 제거")

        # 2. 기존 ForeArm IK influence 복원
        for (bn, cn), inf in state.get('ik_snap', {}).items():
            pb = arm_obj.pose.bones.get(bn)
            if pb:
                for c in pb.constraints:
                    if c.name == cn:
                        c.influence = inf
                        print(f"  [원복] {bn.replace('mixamorig:','')} '{cn}' influence={inf:.2f}")

        # 3. Hand/Finger constraint mute 복원
        hf_snap = state.get('hf_snap', {})
        restore_hand_finger_constraints(arm_obj, hf_snap)
        if hf_snap:
            print(f"  [원복] Hand/Finger constraint mute 복원 ({len(hf_snap)}개)")

        # 4. 손/손가락 rotation_mode 복원
        for bn, mode in state.get('rot_mode_snap', {}).items():
            pb = arm_obj.pose.bones.get(bn)
            if pb:
                pb.rotation_mode = mode

        # 5. 원래 action 복원
        orig_action = state.get('original_action')
        arm_obj.animation_data_create()
        arm_obj.animation_data.action = orig_action

        # 6. 원래 frame 복원
        orig_frame = state.get('original_frame', 1)
        bpy.context.scene.frame_set(orig_frame)
        bpy.context.view_layer.update()

        # 7. 원래 mode 복원
        orig_mode = state.get('original_mode', 'OBJECT')
        if orig_mode not in {'OBJECT', 'POSE', 'EDIT'}:
            orig_mode = 'OBJECT'
        try:
            bpy.ops.object.mode_set(mode=orig_mode)
        except Exception as e:
            print(f"[diag3_restore] mode 복원 실패: {e}")

    # 8. DIAG3_ Empty 제거 (do_unlink=True)
    for ename in state.get('diag_empties', []):
        e = bpy.data.objects.get(ename)
        if e:
            bpy.data.objects.remove(e, do_unlink=True)
            print(f"  [원복] Empty '{ename}' 제거")

    _DIAG3_STATE.clear()
    print("[diag3_restore] 완료")


# ================================================================
# Empty 관리 — animation_data_clear() 미호출
# ================================================================
def make_diag_empty(name, col):
    """DIAG3_ 전용 Empty. 기존 프로젝트 Empty 불침, animation_data 불침."""
    if name in bpy.data.objects:
        o = bpy.data.objects[name]
        o.location      = mathutils.Vector((0, 0, 0))
        o.hide_viewport = False
        if name not in [x.name for x in col.objects]:
            col.objects.link(o)
        return o
    o = bpy.data.objects.new(name, None)
    o.empty_display_size = 0.04
    col.objects.link(o)
    return o


# ================================================================
# IK 관리 — 기존 constraint 불침
# ================================================================
def suspend_existing_ik(arm_obj):
    """기존(비DIAG3_) ForeArm IK constraint를 influence=0 으로 중단."""
    snap = {}
    for bn in [BONE_L_FOREARM, BONE_R_FOREARM]:
        pb = arm_obj.pose.bones.get(bn)
        if not pb: continue
        for c in pb.constraints:
            if c.type == 'IK' and not c.name.startswith('DIAG3_'):
                snap[(bn, c.name)] = c.influence
                c.influence = 0.0
    return snap


def add_diag_ik(arm_obj, bone_name, target, chain_count, cname):
    """DIAG3_ IK 추가. 이전 DIAG3_ 잔여물만 제거, 기존 constraint 불침."""
    b = arm_obj.pose.bones.get(bone_name)
    if not b: return None
    for c in list(b.constraints):
        if c.name == cname:
            b.constraints.remove(c)
    c = b.constraints.new('IK')
    c.name = cname; c.target = target
    c.chain_count = chain_count; c.use_rotation = False; c.influence = 1.0
    return c


def set_diag_ik_influence(arm_obj, inf):
    """DIAG3_ IK constraint만 influence 변경."""
    for bn, cn in [(BONE_L_FOREARM, DIAG_CONSTR_L), (BONE_R_FOREARM, DIAG_CONSTR_R)]:
        pb = arm_obj.pose.bones.get(bn)
        if pb:
            for c in pb.constraints:
                if c.name == cn:
                    c.influence = inf


# ================================================================
# FK bake
# ================================================================
def bake_ik_to_fk(arm_obj, saved_mats):
    result = {}
    for bn in ARM_BONES:
        pb = arm_obj.pose.bones.get(bn); bd = arm_obj.data.bones.get(bn)
        if not pb or not bd: continue
        rest = bd.matrix_local.to_3x3(); pose_a = saved_mats[bn].to_3x3()
        if pb.parent:
            pn = pb.parent.name
            p_pose_a = saved_mats.get(pn, pb.parent.matrix).to_3x3()
            p_bd     = arm_obj.data.bones.get(pn)
            p_rest   = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
            lr = rest.inverted() @ p_rest @ p_pose_a.inverted() @ pose_a
        else:
            lr = rest.inverted() @ pose_a
        result[bn] = lr.to_quaternion()
    return result


def apply_arm_fk(arm_obj, fk_dict):
    for bn, q in fk_dict.items():
        pb = arm_obj.pose.bones.get(bn)
        if pb:
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = q.copy()


def reset_hand_bones(arm_obj):
    for bn in [BONE_L_HAND, BONE_R_HAND]:
        pb = arm_obj.pose.bones.get(bn)
        if pb:
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = mathutils.Quaternion()


def reset_all_bones(arm_obj):
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = mathutils.Quaternion()
        pb.location = mathutils.Vector((0, 0, 0))
        pb.scale    = mathutils.Vector((1, 1, 1))


# ================================================================
# 팔 자세 검증
# ================================================================
def verify_arm(arm_obj, base_mats, label, fail_on_error=True):
    ok = True
    for bn in ARM_BONES:
        pb = arm_obj.pose.bones.get(bn)
        if not pb: continue
        d = mat3_angle_deg(pb.matrix.to_3x3(), base_mats[bn].to_3x3())
        if d > ARM_MATCH_TOL_DEG:
            tag = 'FAIL' if fail_on_error else 'WARN'
            print(f"  [{tag}] {bn.replace('mixamorig:',''):16s}  {d:.2f}° from base")
            ok = False
    if ok:
        print(f"  [OK] 팔 자세 일치 (tol={ARM_MATCH_TOL_DEG}°)")
    return ok


# ================================================================
# Rest basis (D auto_sign 용)
# ================================================================
def rest_hand_norm(arm, bd):
    def hl(n):
        b = arm.data.bones.get(n)
        return mathutils.Vector(b.head_local) if b else None
    h0 = hl(bd['hand']); hi = hl(bd['index1'])
    hm = hl(bd['middle1']); hp = hl(bd['pinky1'])
    if not all([h0, hi, hm, hp]): return None
    fwd = (hm - h0)
    if fwd.length < 1e-6: return None
    fwd = fwd.normalized()
    n = fwd.cross((hi - hp).normalized())
    return n.normalized() if n.length > 1e-6 else None


# ================================================================
# Target basis (오른손 좌표계 통일)
# ================================================================
def make_right_handed_target(fwd_w, effective_pn_w):
    """
    Y = fwd_w (normalized)
    Z = effective_pn_w 를 Y에 직교 투영 (normalized)
    X = Y × Z  →  det(X,Y,Z) = +1
    반환: (tx, ty, tz) 또는 실패 시 (None, None, None)
    """
    ty = fwd_w.normalized()
    pn_perp = effective_pn_w - effective_pn_w.dot(ty) * ty
    if pn_perp.length < 1e-6:
        return None, None, None
    tz = pn_perp.normalized()
    tx = ty.cross(tz).normalized()
    return tx, ty, tz


# ================================================================
# Hand FK 적용 — A/B: set_hand_AB
# ================================================================
def apply_hand_AB(arm_obj, bone_name, wrist_w, mid9_w, effective_pn_w):
    """
    Y = wrist→mid9, Z = effective_pn_w 투영, X = Y×Z.
    effective_pn_w: 이미 flip 적용된 법선.
    리포트 make_right_handed_target 과 동일한 effective_pn 사용.
    """
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd: return False
    fwd_w = (mid9_w - wrist_w)
    if fwd_w.length < 1e-6: return False
    fwd_w = fwd_w.normalized()
    a_inv = arm_obj.matrix_world.to_3x3().inverted()
    ny    = (a_inv @ fwd_w).normalized()
    rest  = bd.matrix_local.to_3x3()
    q0  = rest.col[1].normalized().rotation_difference(ny)
    nx0 = (q0 @ rest.col[0].normalized()).normalized()
    nz0 = (q0 @ rest.col[2].normalized()).normalized()
    pn_a = (a_inv @ effective_pn_w)
    if pn_a.length > 1e-4:
        pn_a = pn_a.normalized()
        pp   = pn_a - pn_a.dot(ny) * ny
        zp   = nz0  - nz0.dot(ny)  * ny
        if pp.length > 1e-4 and zp.length > 1e-4:
            pp = pp.normalized(); zp = zp.normalized()
            ra  = math.atan2(zp.cross(pp).dot(ny), max(-1., min(1., zp.dot(pp))))
            rq  = mathutils.Quaternion(ny, ra)
            nx0 = rq @ nx0; nz0 = rq @ nz0
    da = mathutils.Matrix([[nx0.x, ny.x, nz0.x],
                            [nx0.y, ny.y, nz0.y],
                            [nx0.z, ny.z, nz0.z]])
    if b.parent:
        p_bd   = arm_obj.data.bones.get(b.parent.name)
        p_rest = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        pm     = rest.inverted() @ p_rest @ b.parent.matrix.to_3x3().inverted() @ da
    else:
        pm = rest.inverted() @ da
    b.rotation_mode = 'QUATERNION'
    b.rotation_quaternion = pm.to_quaternion()
    return True


# ================================================================
# Hand FK 적용 — C/D: set_hand_CD
# ================================================================
def apply_hand_CD(arm_obj, bone_name, fwd_w, effective_norm_w):
    """Y = fwd_w, Z = effective_norm_w 투영, X = Y×Z."""
    b  = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd: return False
    a_inv = arm_obj.matrix_world.to_3x3().inverted()
    ny = (a_inv @ fwd_w).normalized()
    nz = a_inv @ effective_norm_w
    nz = nz - nz.dot(ny) * ny
    if nz.length < 1e-6: return False
    nz = nz.normalized(); nx = ny.cross(nz).normalized()
    da = mathutils.Matrix([[nx.x, ny.x, nz.x],
                            [nx.y, ny.y, nz.y],
                            [nx.z, ny.z, nz.z]])
    rest = bd.matrix_local.to_3x3()
    if b.parent:
        p_bd   = arm_obj.data.bones.get(b.parent.name)
        p_rest = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        pm     = rest.inverted() @ p_rest @ b.parent.matrix.to_3x3().inverted() @ da
    else:
        pm = rest.inverted() @ da
    b.rotation_mode = 'QUATERNION'
    b.rotation_quaternion = pm.to_quaternion()
    return True


# ================================================================
# 리포트 — 오른손 좌표계 통일
# ================================================================
def report_hand(arm_obj, bone_name, tx, ty, tz, label):
    """
    target: X=tx, Y=ty(forward), Z=tz(effective normal), X=Y×Z → det+1
    actual: bone world-space col[0/1/2]
    출력: target det, actual det, forward/normal/lateral error
    """
    bpy.context.view_layer.update()
    b = arm_obj.pose.bones.get(bone_name)
    if not b:
        print(f"  [{label}] 뼈 없음"); return
    wm = (arm_obj.matrix_world @ b.matrix).to_3x3()
    ax = wm.col[0].normalized(); ay = wm.col[1].normalized(); az = wm.col[2].normalized()
    tx = tx.normalized(); ty = ty.normalized(); tz = tz.normalized()
    t_det    = det3(tx, ty, tz); a_det    = det3(ax, ay, az)
    fwd_err  = angle_deg(ay, ty); norm_err = angle_deg(az, tz); lat_err  = angle_deg(ax, tx)
    print(f"\n  [{label}]")
    print(f"    target  X: {tx.x:+.3f} {tx.y:+.3f} {tx.z:+.3f}")
    print(f"    target  Y: {ty.x:+.3f} {ty.y:+.3f} {ty.z:+.3f}  (forward)")
    print(f"    target  Z: {tz.x:+.3f} {tz.y:+.3f} {tz.z:+.3f}  (effective normal)")
    print(f"    actual  X: {ax.x:+.3f} {ax.y:+.3f} {ax.z:+.3f}")
    print(f"    actual  Y: {ay.x:+.3f} {ay.y:+.3f} {ay.z:+.3f}")
    print(f"    actual  Z: {az.x:+.3f} {az.y:+.3f} {az.z:+.3f}")
    print(f"    target det : {t_det:+.4f}  ({'RIGHT-HANDED' if t_det > 0 else 'LEFT-HANDED !'})")
    print(f"    actual det : {a_det:+.4f}  ({'RIGHT-HANDED' if a_det > 0 else 'LEFT-HANDED !'})")
    print(f"    forward_err : {fwd_err:6.1f}°")
    print(f"    normal_err  : {norm_err:6.1f}°  <- 핵심")
    print(f"    lateral_err : {lat_err:6.1f}°")


# ================================================================
# 액션 저장 (즉시 action=None)
# ================================================================
def save_action(arm_obj, name, frame=1):
    old = bpy.data.actions.get(name)
    if old: bpy.data.actions.remove(old)
    act = bpy.data.actions.new(name)
    arm_obj.animation_data_create()
    arm_obj.animation_data.action = act
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.keyframe_insert('rotation_quaternion', frame=frame)
    arm_obj.animation_data.action = None
    print(f"  -> 저장: '{name}'  (action=None 복원)")
    return act


# ================================================================
# 메인 진단 — [수정5] 예외 시 분리를 위해 별도 함수
# ================================================================
def main_diagnostic(arm_obj):
    SEP = "=" * 62

    # ── [수정6] DIAG3_ 잔여물 정리 ──────────────────────────────
    print(f"\n[DIAG3_ 잔여물 정리]")
    for ename in [DIAG_EMPTY_L, DIAG_EMPTY_R]:
        e = bpy.data.objects.get(ename)
        if e:
            bpy.data.objects.remove(e, do_unlink=True)
            print(f"  이전 '{ename}' 제거")
    for bn, cn in [(BONE_L_FOREARM, DIAG_CONSTR_L), (BONE_R_FOREARM, DIAG_CONSTR_R)]:
        pb = arm_obj.pose.bones.get(bn)
        if pb:
            for c in list(pb.constraints):
                if c.name == cn:
                    pb.constraints.remove(c)
                    print(f"  이전 {bn.replace('mixamorig:','')} '{cn}' 제거")

    # ── JSON 로드 ─────────────────────────────────────────────────
    with open(JSON_PATH, encoding='utf-8') as f:
        rec = json.load(f)
    frames_data = rec.get('Frames', [])
    print(f"[JSON] {len(frames_data)}프레임")

    # ── 캐릭터 어깨 기준 ─────────────────────────────────────────
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

    # ── 양손 유효 프레임 수집 ────────────────────────────────────
    both_valid = []
    for fd_idx, fd in enumerate(frames_data):
        if not fd.get('HasPose', False): continue
        pp  = fd.get('PosePoints') or []
        mll = fd.get('LeftHandPoints') or []
        mrr = fd.get('RightHandPoints') or []
        hml = fd.get('HasLeftHand', False)
        hmr = fd.get('HasRightHand', False)
        ch_l, has_l = (mrr, hmr) if SWAP_HAND_LABELS else (mll, hml)
        ch_r, has_r = (mll, hml) if SWAP_HAND_LABELS else (mrr, hmr)
        if not (has_l and has_r): continue
        ls = get_pt(pp, P_L_SHOULDER); rs = get_pt(pp, P_R_SHOULDER)
        if not ls or not rs: continue
        cx = (ls[0] + rs[0]) / 2; cy = (ls[1] + rs[1]) / 2
        sw = math.sqrt((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2)
        if sw < 1e-4: continue
        both_valid.append((fd_idx, cx, cy, sw, ch_l, ch_r, pp, fd))

    n_both = len(both_valid)
    print(f"[탐색] 양손 유효 프레임: {n_both}개")
    if not both_valid:
        raise RuntimeError("양손 유효 프레임 없음")

    sel = min(TEST_BOTH_VALID_INDEX, n_both - 1)
    fd_idx, cx, cy, sw, ch_l_pts, ch_r_pts, pose_pts, fd_raw = both_valid[sel]
    ts_ms = fd_raw.get('TimestampMs', 'N/A')

    print(f"\n[선택 프레임]")
    print(f"  TEST_BOTH_VALID_INDEX : {TEST_BOTH_VALID_INDEX}  (실제 사용: {sel})")
    print(f"  원본 JSON frame index : {fd_idx}")
    print(f"  TimestampMs           : {ts_ms}")
    print(f"  전체 양손 유효 수     : {n_both}")
    print(f"  LEFT/RIGHT 동일 프레임: True (보장)")

    # ── 랜드마크 변환 ─────────────────────────────────────────────
    def chp(idx, pts):
        pt = get_pt(pts, idx)
        return convert_hand_pt(*pt, cx, cy, sw, char_origin, char_sw) if pt else None

    def bp(idx):
        pt = get_pt(pose_pts, idx)
        return to_blender(*pt, cx, cy, sw, char_origin, char_sw) if pt else None

    l_pts21 = [chp(i, ch_l_pts) for i in range(21)]
    r_pts21 = [chp(i, ch_r_pts) for i in range(21)]
    if any(p is None for p in l_pts21 + r_pts21):
        raise RuntimeError("랜드마크 변환 실패 — None 포함")

    pose_lw = bp(P_L_WRIST); pose_rw = bp(P_R_WRIST)
    lw3 = l_pts21[H_WRIST];   rw3 = r_pts21[H_WRIST]
    lh9 = l_pts21[H_MID_MCP]; rh9 = r_pts21[H_MID_MCP]

    l_pn_raw = (l_pts21[H_INDEX_MCP] - lw3).cross(l_pts21[H_PINK_MCP] - lw3)
    r_pn_raw = (r_pts21[H_INDEX_MCP] - rw3).cross(r_pts21[H_PINK_MCP] - rw3)
    if l_pn_raw.length < 1e-6 or r_pn_raw.length < 1e-6:
        raise RuntimeError("palm_normal 길이 0")
    l_pn = l_pn_raw.normalized(); r_pn = r_pn_raw.normalized()

    fwd_l = (lh9 - lw3).normalized(); fwd_r = (rh9 - rw3).normalized()
    lat_l = (l_pts21[H_INDEX_MCP] - l_pts21[H_PINK_MCP]).normalized()
    lat_r = (r_pts21[H_INDEX_MCP] - r_pts21[H_PINK_MCP]).normalized()
    cd_norm_l_p1 = fwd_l.cross(lat_l).normalized()
    cd_norm_r_p1 = fwd_r.cross(lat_r).normalized()

    print(f"\n[Pose vs Hand 손목 불일치]")
    if pose_lw: print(f"  LEFT  {(lw3-pose_lw).length:.4f} m  <- 진단은 Hand pt0 사용")
    if pose_rw: print(f"  RIGHT {(rw3-pose_rw).length:.4f} m  <- 진단은 Hand pt0 사용")

    # ── 기존 ForeArm IK 중단 ─────────────────────────────────────
    ik_snap = suspend_existing_ik(arm_obj)
    _DIAG3_STATE['ik_snap'] = ik_snap
    print(f"\n[기존 IK constraint 중단 — 원복: diag3_restore()]")
    if ik_snap:
        for (bn, cn), inf in sorted(ik_snap.items()):
            print(f"  {bn.replace('mixamorig:',''):16s} '{cn}'  orig={inf:.2f} -> 0.00")
    else:
        print("  (기존 IK 없음)")

    # ── DIAG3_ Empty 생성 ─────────────────────────────────────────
    col = bpy.data.collections.get("IK_Targets")
    if not col:
        col = bpy.data.collections.new("IK_Targets")
        bpy.context.scene.collection.children.link(col)
    emp_lw = make_diag_empty(DIAG_EMPTY_L, col)
    emp_rw = make_diag_empty(DIAG_EMPTY_R, col)

    # ── [수정1] Hand/Finger constraint mute + rotation_mode 저장 ─
    finger_bones = (
        [BONE_L_HAND, BONE_R_HAND] +
        [f"mixamorig:{s}{f}{n}"
         for s in ['Left', 'Right']
         for f in ['HandThumb', 'HandIndex', 'HandMiddle', 'HandRing', 'HandPinky']
         for n in [1, 2, 3]]
    )
    hf_snap = suspend_hand_finger_constraints(arm_obj, finger_bones)
    _DIAG3_STATE['hf_snap'] = hf_snap

    # rotation_mode snapshot (진단 후 원복)
    rot_mode_snap = {}
    for bn in finger_bones:
        pb = arm_obj.pose.bones.get(bn)
        if pb:
            rot_mode_snap[bn] = pb.rotation_mode
            pb.rotation_mode = 'QUATERNION'
    _DIAG3_STATE['rot_mode_snap'] = rot_mode_snap

    if hf_snap:
        print(f"[Hand/Finger constraint] mute 설정 ({len(hf_snap)}개) — 삭제 없음")
    else:
        print("[Hand/Finger constraint] 해당 없음")

    # ── DIAG3_ IK 추가 ────────────────────────────────────────────
    add_diag_ik(arm_obj, BONE_L_FOREARM, emp_lw, 2, DIAG_CONSTR_L)
    add_diag_ik(arm_obj, BONE_R_FOREARM, emp_rw, 2, DIAG_CONSTR_R)

    arm_obj.animation_data_create()
    arm_obj.animation_data.action = None

    # ── IK 평가 ──────────────────────────────────────────────────
    reset_all_bones(arm_obj)
    emp_lw.location = pose_lw if pose_lw else lw3
    emp_rw.location = pose_rw if pose_rw else rw3
    set_diag_ik_influence(arm_obj, 1.0)
    bpy.context.view_layer.update()

    print(f"\n[IK 평가 결과]")
    for bn in ARM_BONES:
        pb = arm_obj.pose.bones.get(bn)
        if pb:
            tail_w = arm_obj.matrix_world @ pb.tail
            print(f"  {bn.replace('mixamorig:',''):18s} tail: "
                  f"{tail_w.x:.3f} {tail_w.y:.3f} {tail_w.z:.3f}")

    # ── IK→FK bake ───────────────────────────────────────────────
    saved_mats = {}
    for bn in ARM_BONES:
        pb = arm_obj.pose.bones.get(bn)
        if pb:
            saved_mats[bn] = pb.matrix.copy()
            if pb.parent and pb.parent.name not in saved_mats:
                saved_mats[pb.parent.name] = pb.parent.matrix.copy()

    base_arm_fk = bake_ik_to_fk(arm_obj, saved_mats)

    # DIAG3_ IK=0 (기존 IK는 이미 0 — 복원 안 함 → 순수 FK 상태)
    set_diag_ik_influence(arm_obj, 0.0)
    apply_arm_fk(arm_obj, base_arm_fk)
    bpy.context.view_layer.update()

    print(f"\n[IK->FK bake 검증]")
    base_fk_mats = {}; bake_all_ok = True
    for bn in ARM_BONES:
        pb = arm_obj.pose.bones.get(bn)
        if not pb: continue
        base_fk_mats[bn] = pb.matrix.copy()
        d = mat3_angle_deg(pb.matrix.to_3x3(), saved_mats[bn].to_3x3())
        s = "OK" if d < ARM_MATCH_TOL_DEG else "FAIL"
        print(f"  [{s}] {bn.replace('mixamorig:',''):16s} diff={d:.3f}°")
        if d >= ARM_MATCH_TOL_DEG: bake_all_ok = False
    if not bake_all_ok:
        print("  [경고] bake 오차 — 결과 신뢰도 낮음")

    # ── Rest basis (D auto_sign 용) ──────────────────────────────
    a_inv_g     = arm_obj.matrix_world.to_3x3().inverted()
    rest_norm_l = rest_hand_norm(arm_obj, REST_BONES_L)
    rest_norm_r = rest_hand_norm(arm_obj, REST_BONES_R)

    def auto_sign(cd_norm_w, rest_norm_a, side):
        if rest_norm_a is None or cd_norm_w is None: return 1.0
        d = (a_inv_g @ cd_norm_w).normalized().dot(rest_norm_a)
        s = 1.0 if d >= 0 else -1.0
        print(f"  [{side} AUTO] rest vs cd_norm dot={d:+.3f} -> pn_sign={s:+.0f}")
        return s

    # ── 접근법 루프 ───────────────────────────────────────────────
    saved_actions     = {}
    approach_arm_mats = {}

    print(f"\n{SEP}")
    print("NOTE: 손가락=Rest Pose | 손목=Hand pt0 | DIAG3_ IK=0 (baked FK)")
    print(SEP)

    # ─ A, B ──────────────────────────────────────────────────────
    for lbl, flip_l, flip_r in [('A', False, False), ('B', True, True)]:
        eff_pn_l = -l_pn if flip_l else l_pn
        eff_pn_r = -r_pn if flip_r else r_pn
        print(f"\n{SEP}")
        print(f"접근법 {lbl}: set_hand_AB  flip_L={flip_l}  flip_R={flip_r}")
        print(f"  effective_pn_L = {'−l_pn' if flip_l else '+l_pn'}")
        print(f"  effective_pn_R = {'−r_pn' if flip_r else '+r_pn'}")
        print(SEP)
        arm_obj.animation_data.action = None
        apply_arm_fk(arm_obj, base_arm_fk); reset_hand_bones(arm_obj)
        bpy.context.view_layer.update()
        print("  [팔 검증] Hand 설정 전:"); verify_arm(arm_obj, base_fk_mats, lbl)
        apply_hand_AB(arm_obj, BONE_L_HAND, lw3, lh9, eff_pn_l)
        apply_hand_AB(arm_obj, BONE_R_HAND, rw3, rh9, eff_pn_r)
        bpy.context.view_layer.update()
        print("  [팔 검증] Hand 설정 후:"); verify_arm(arm_obj, base_fk_mats, lbl, False)
        txl, tyl, tzl = make_right_handed_target(lh9 - lw3, eff_pn_l)
        txr, tyr, tzr = make_right_handed_target(rh9 - rw3, eff_pn_r)
        if txl: report_hand(arm_obj, BONE_L_HAND, txl, tyl, tzl, f"{lbl} LEFT")
        if txr: report_hand(arm_obj, BONE_R_HAND, txr, tyr, tzr, f"{lbl} RIGHT")
        approach_arm_mats[lbl] = {
            bn: arm_obj.pose.bones.get(bn).matrix.copy()
            for bn in ARM_BONES if arm_obj.pose.bones.get(bn)
        }
        saved_actions[lbl] = save_action(arm_obj, f"HandTest3_{lbl}_ArmLocked")

    # ─ C, D ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[D auto_sign 계산]")
    pn_sign_l_d = auto_sign(cd_norm_l_p1, rest_norm_l, "LEFT")
    pn_sign_r_d = auto_sign(cd_norm_r_p1, rest_norm_r, "RIGHT")

    for lbl, pn_sign_l, pn_sign_r, desc in [
        ('C', 1.0,         1.0,         "pn_sign=+1"),
        ('D', pn_sign_l_d, pn_sign_r_d, "pn_sign=auto"),
    ]:
        cd_norm_l = (cd_norm_l_p1 * pn_sign_l).normalized()
        cd_norm_r = (cd_norm_r_p1 * pn_sign_r).normalized()
        print(f"\n{SEP}")
        print(f"접근법 {lbl}: set_hand_CD  {desc}")
        print(f"  Y = wrist->mid9,  Z = fwd x (Index-Pinky) x pn_sign")
        print(SEP)
        arm_obj.animation_data.action = None
        apply_arm_fk(arm_obj, base_arm_fk); reset_hand_bones(arm_obj)
        bpy.context.view_layer.update()
        print("  [팔 검증] Hand 설정 전:"); verify_arm(arm_obj, base_fk_mats, lbl)
        apply_hand_CD(arm_obj, BONE_L_HAND, fwd_l, cd_norm_l)
        apply_hand_CD(arm_obj, BONE_R_HAND, fwd_r, cd_norm_r)
        bpy.context.view_layer.update()
        print("  [팔 검증] Hand 설정 후:"); verify_arm(arm_obj, base_fk_mats, lbl, False)
        txl, tyl, tzl = make_right_handed_target(fwd_l, cd_norm_l)
        txr, tyr, tzr = make_right_handed_target(fwd_r, cd_norm_r)
        if txl: report_hand(arm_obj, BONE_L_HAND, txl, tyl, tzl, f"{lbl} LEFT")
        if txr: report_hand(arm_obj, BONE_R_HAND, txr, tyr, tzr, f"{lbl} RIGHT")
        approach_arm_mats[lbl] = {
            bn: arm_obj.pose.bones.get(bn).matrix.copy()
            for bn in ARM_BONES if arm_obj.pose.bones.get(bn)
        }
        saved_actions[lbl] = save_action(arm_obj, f"HandTest3_{lbl}_ArmLocked")

    # ── A 기준 팔 차이 ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("A 기준 팔 자세 차이 (Hand 설정 후 측정)")
    print(SEP)
    ref = approach_arm_mats.get('A', {})
    for lbl in ['B', 'C', 'D']:
        cur = approach_arm_mats.get(lbl, {})
        print(f"  A vs {lbl}:")
        for bn in ARM_BONES:
            if ref.get(bn) and cur.get(bn):
                d = mat3_angle_deg(ref[bn].to_3x3(), cur[bn].to_3x3())
                print(f"    {bn.replace('mixamorig:',''):16s} {d:.3f}°")

    # ── 저장 액션 전환 재검증 (DIAG3_ IK=0, 기존 IK=0) ──────────
    print(f"\n{SEP}")
    print("저장 액션 전환 후 팔 자세 재검증  (DIAG3_ IK=0, 기존 IK=0)")
    print(SEP)
    for lbl, act in saved_actions.items():
        arm_obj.animation_data.action = act
        bpy.context.scene.frame_set(1); bpy.context.view_layer.update()
        print(f"\n  HandTest3_{lbl}_ArmLocked 활성화 후:")
        verify_arm(arm_obj, base_fk_mats, lbl, fail_on_error=False)

    # A 액션으로 복귀 (IK=0 유지)
    act_a = saved_actions.get('A')
    if act_a: arm_obj.animation_data.action = act_a
    bpy.context.scene.frame_set(1); bpy.context.view_layer.update()

    emp_lw.hide_viewport = True; emp_rw.hide_viewport = True

    print(f"\n{SEP}")
    print("완료. Action Editor에서 HandTest3_A/B/C/D_ArmLocked 시각 비교:")
    print("  DIAG3_ IK = 0  |  기존 IK = 0  (순수 FK 상태)")
    print()
    print("  A: set_hand_AB  effective_pn = +l_pn  (flip=False)")
    print("  B: set_hand_AB  effective_pn = -l_pn  (flip=True)")
    print("  C: set_hand_CD  fwd x (Index-Pinky)  pn_sign=+1")
    print("  D: set_hand_CD  fwd x (Index-Pinky)  pn_sign=auto")
    print(SEP)


# ================================================================
# 진입점 — [수정5] try/except 래퍼
# ================================================================
def main():
    arm_obj = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if not arm_obj:
        print("[오류] 아마추어 없음"); return
    print(f"[아마추어] {arm_obj.name}")

    # [수정4] 실행 전 상태 snapshot
    bpy.context.view_layer.objects.active = arm_obj
    _DIAG3_STATE.clear()
    _DIAG3_STATE['arm_obj']          = arm_obj
    _DIAG3_STATE['original_action']  = (
        arm_obj.animation_data.action if arm_obj.animation_data else None
    )
    _DIAG3_STATE['original_frame']   = bpy.context.scene.frame_current
    _DIAG3_STATE['original_mode']    = arm_obj.mode
    _DIAG3_STATE['ik_snap']          = {}
    _DIAG3_STATE['hf_snap']          = {}
    _DIAG3_STATE['rot_mode_snap']    = {}
    _DIAG3_STATE['diag_empties']     = [DIAG_EMPTY_L, DIAG_EMPTY_R]
    _DIAG3_STATE['diag_constraints'] = [
        (BONE_L_FOREARM, DIAG_CONSTR_L),
        (BONE_R_FOREARM, DIAG_CONSTR_R),
    ]

    bpy.ops.object.mode_set(mode='POSE')

    # [수정5] 예외 시 자동 원복, 정상 완료 시 시각 비교 상태 유지
    try:
        main_diagnostic(arm_obj)
    except Exception:
        traceback.print_exc()
        print("\n[오류] 예외 발생 — 자동 원복 시도")
        diag3_restore()
        return

    print("\n시각 비교 완료 후 실행:")
    print("diag3_restore()")


main()
