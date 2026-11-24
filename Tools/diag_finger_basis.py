"""
Tools/diag_finger_basis.py
Blender Script Editor에 붙여넣고 실행.
blend_animate.py 실행(전체 애니메이션 생성) 직후에 사용.

출력 섹션:
  A. 현재 FK basis 방식 설명
  B. 검지 정적 basis 진단 (forward/bend/normal/roll 오차)
  B2. Current FK vs Palm-basis FK 비교
  C. 전체 프레임 연속성 이상 탐지
  D. 메시 스키닝 진단 + Index1 굽힘 테스트
"""
import bpy, json, math, mathutils

# ================================================================
# 설정 — blend_animate.py 와 동일하게 유지
# ================================================================
JSON_PATH        = r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\Recordings\감사합니다.json"
BODY_SCALE       = 1.0
DEPTH_SCALE      = 0.0
FORWARD_OFFSET   = 0.15
SMOOTH_WINDOW    = 5
JUMP_THRESHOLD   = 0.5
MIRROR_X         = True
SWAP_HAND_LABELS = True
HAND_DEPTH_GAIN  = 1.0
INTRO_FRAMES     = 20
TEST_FRAME_IDX   = 0       # 진단할 LEFT hand 검출 프레임 인덱스

P_L_SHOULDER = 11;  P_R_SHOULDER = 12
H_WRIST      = 0
H_INDEX_MCP  = 5;   H_INDEX_PIP = 6;  H_INDEX_DIP = 7;  H_INDEX_TIP = 8
H_MID_MCP    = 9
H_PINK_MCP   = 17

BONE_L_HAND    = 'mixamorig:LeftHand'
BONE_L_ARM     = 'mixamorig:LeftArm'
BONE_R_ARM     = 'mixamorig:RightArm'

INDEX_CHAIN_L = [
    ('mixamorig:LeftHandIndex1', H_INDEX_MCP, H_INDEX_PIP),
    ('mixamorig:LeftHandIndex2', H_INDEX_PIP, H_INDEX_DIP),
    ('mixamorig:LeftHandIndex3', H_INDEX_DIP, H_INDEX_TIP),
]

# 연속성 이상 탐지 임계값
QUAT_DOT_THRESH  = 0.90   # dot < 이 값 → 큰 회전 변화 의심
DELTA_DEG_THRESH = 45.0   # 도 — 프레임 간 회전 변화 임계

# ================================================================
# 유틸 (blend_animate.py 동일)
# ================================================================

def get_pt(pts, idx):
    if not pts or idx >= len(pts): return None
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


def smooth_track(vecs, window=SMOOTH_WINDOW, jump=JUMP_THRESHOLD):
    n = len(vecs)
    if n == 0: return []
    f = list(vecs)
    fi = next((i for i, v in enumerate(f) if v is not None), None)
    if fi is None: return [mathutils.Vector((0, 0, 0))] * n
    for i in range(fi): f[i] = f[fi].copy()
    li = next((i for i in range(n - 1, -1, -1) if f[i] is not None), 0)
    for i in range(li + 1, n): f[i] = f[li].copy()
    i = 0
    while i < n:
        if f[i] is None:
            a, b = i - 1, i
            while b < n and f[b] is None: b += 1
            for k in range(i, b): f[k] = f[a].lerp(f[b], (k - a) / (b - a))
            i = b
        else:
            i += 1
    for _ in range(2):
        for i in range(1, n - 1):
            avg = f[i - 1].lerp(f[i + 1], 0.5)
            if (f[i] - avg).length > jump: f[i] = avg
    half = window // 2
    result = []
    for i in range(n):
        chunk = f[max(0, i - half):min(n, i + half + 1)]
        result.append(mathutils.Vector((
            sum(v.x for v in chunk) / len(chunk),
            sum(v.y for v in chunk) / len(chunk),
            sum(v.z for v in chunk) / len(chunk),
        )))
    return result


def ang(a, b):
    """두 Vector 사이 각도 (도). 실패 시 999."""
    if a.length < 1e-6 or b.length < 1e-6: return 999.0
    try: return math.degrees(a.normalized().angle(b.normalized()))
    except: return 999.0


def bone_arma_mat(arm, bone_name, parent_mat=None):
    b  = arm.pose.bones.get(bone_name)
    bd = arm.data.bones.get(bone_name)
    if not b or not bd: return parent_mat or mathutils.Matrix.Identity(3)
    rest = bd.matrix_local.to_3x3()
    lr   = b.rotation_quaternion.to_matrix()
    if b.parent:
        pbd   = arm.data.bones.get(b.parent.name)
        prest = pbd.matrix_local.to_3x3() if pbd else mathutils.Matrix.Identity(3)
        pm    = parent_mat if parent_mat is not None else b.parent.matrix.to_3x3()
        return pm @ prest.inverted() @ rest @ lr
    return rest @ lr


def get_parent_mat(arm, bone_name, anchor_name, anchor_mat):
    b = arm.pose.bones.get(bone_name)
    if not b or not b.parent or b.parent.name == anchor_name:
        return anchor_mat
    chain = []
    cur = b.parent
    while cur and cur.name != anchor_name:
        chain.append(cur.name); cur = cur.parent
    chain.reverse()
    mat = anchor_mat
    for inter in chain:
        mat = bone_arma_mat(arm, inter, mat)
    return mat


def set_bone_dir(arm, bone_name, from_w, to_w, parent_mat=None, roll_pn=None):
    """
    뼈를 from_w→to_w 방향으로 FK 설정.
    roll_pn: palm_normal (world) — 지정 시 Z축 roll도 보정.
    반환: (성공, 이 뼈의 새 armature-space 3x3)
    """
    b  = arm.pose.bones.get(bone_name)
    bd = arm.data.bones.get(bone_name)
    if not b or not bd: return False, parent_mat

    d = (to_w - from_w)
    if d.length < 1e-6: return False, parent_mat
    d = d.normalized()

    ai   = arm.matrix_world.to_3x3().inverted()
    da   = (ai @ d).normalized()
    rest = bd.matrix_local.to_3x3()
    ry   = rest.col[1].normalized()
    rx   = rest.col[0].normalized()
    rz   = rest.col[2].normalized()

    q  = ry.rotation_difference(da)
    ny = da
    nx = (q @ rx).normalized()
    nz = (q @ rz).normalized()

    # palm_normal roll 보정 (optional)
    if roll_pn is not None and roll_pn.length > 1e-5:
        pna = (ai @ roll_pn).normalized()
        pp  = pna - pna.dot(ny) * ny
        zp  = nz  - nz.dot(ny)  * ny
        if pp.length > 1e-4 and zp.length > 1e-4:
            pp = pp.normalized(); zp = zp.normalized()
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
        pbd    = arm.data.bones.get(b.parent.name)
        prest  = pbd.matrix_local.to_3x3() if pbd else mathutils.Matrix.Identity(3)
        pm     = parent_mat if parent_mat is not None else b.parent.matrix.to_3x3()
        pose_m = rest.inverted() @ prest @ pm.inverted() @ desired_a
    else:
        pose_m = rest.inverted() @ desired_a

    b.rotation_mode = 'QUATERNION'
    b.rotation_quaternion = pose_m.to_quaternion()

    lr = b.rotation_quaternion.to_matrix()
    if b.parent:
        pm2   = parent_mat if parent_mat is not None else b.parent.matrix.to_3x3()
        pbd2  = arm.data.bones.get(b.parent.name)
        pr2   = pbd2.matrix_local.to_3x3() if pbd2 else mathutils.Matrix.Identity(3)
        nm    = pm2 @ pr2.inverted() @ rest @ lr
    else:
        nm = rest @ lr
    return True, nm


# ================================================================
# 진단 유틸
# ================================================================

def target_basis(from_w, to_w, palm_n_w):
    """
    MediaPipe 데이터로부터 손가락용 orthonormal basis (world space).
      forward  = from→to 방향
      bend     = 굽힘 축  = palm_normal_proj × forward
      normal   = forward × bend  (손등/손바닥 방향)
    반환: (forward, bend, normal) — 실패 시 (None, None, None)
    """
    fwd = (to_w - from_w)
    if fwd.length < 1e-6: return None, None, None
    fwd = fwd.normalized()

    if palm_n_w is not None and palm_n_w.length > 1e-6:
        pn = palm_n_w.normalized()
    else:
        pn = mathutils.Vector((0, 0, 1))  # fallback

    pp = pn - pn.dot(fwd) * fwd
    if pp.length < 1e-5:
        # palm normal이 forward와 평행 — 다른 기준 사용
        alt = mathutils.Vector((0, 1, 0))
        pp  = alt - alt.dot(fwd) * fwd
    if pp.length < 1e-5: return fwd, None, None
    pp   = pp.normalized()
    bend = pp.cross(fwd).normalized()
    norm = fwd.cross(bend).normalized()
    return fwd, bend, norm


# ================================================================
# 데이터 로드 (blend_animate.py 와 동일 경로·설정)
# ================================================================

def load_frame_data(arm):
    with open(JSON_PATH, encoding='utf-8') as f:
        rec = json.load(f)

    lb = arm.pose.bones.get(BONE_L_ARM); rb = arm.pose.bones.get(BONE_R_ARM)
    lsh = arm.matrix_world @ lb.head; rsh = arm.matrix_world @ rb.head
    origin  = (lsh + rsh) / 2
    char_sw = (lsh - rsh).length

    pts21_raw = []; pn_raw = []; valid_raw = []

    for fd in rec.get('Frames', []):
        if not fd.get('HasPose', False): continue
        pose_pts = fd.get('PosePoints') or []
        ml = fd.get('LeftHandPoints') or []; mr = fd.get('RightHandPoints') or []
        hl = fd.get('HasLeftHand', False); hr = fd.get('HasRightHand', False)
        if SWAP_HAND_LABELS:
            chl, has_chl = mr, hr
        else:
            chl, has_chl = ml, hl

        ls = get_pt(pose_pts, P_L_SHOULDER); rs = get_pt(pose_pts, P_R_SHOULDER)
        if not ls or not rs: continue
        cx = (ls[0] + rs[0]) / 2; cy = (ls[1] + rs[1]) / 2
        sw = math.sqrt((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2)
        if sw < 1e-4: continue

        # ── 손가락 21개 ──────────────────────────────────────────
        pts21 = []
        for i in range(21):
            if has_chl:
                pt = get_pt(chl, i)
                pts21.append(convert_hand_pt(pt[0], pt[1], pt[2], cx, cy, sw, origin, char_sw) if pt else None)
            else:
                pts21.append(None)
        pts21_raw.append(pts21)
        valid_raw.append(has_chl)

        # ── 손바닥 법선 ──────────────────────────────────────────
        def c(idx):
            pt = get_pt(chl, idx)
            return convert_hand_pt(pt[0], pt[1], pt[2], cx, cy, sw, origin, char_sw) if pt else None
        w3 = c(H_WRIST); i3 = c(H_INDEX_MCP); p3 = c(H_PINK_MCP)
        if has_chl and w3 and i3 and p3:
            pn = (i3 - w3).cross(p3 - w3)
            pn_raw.append(pn if pn.length > 1e-6 else None)
        else:
            pn_raw.append(None)

    total = len(pts21_raw)
    if total == 0:
        return None, None, None, None, 0

    # 스무딩
    sm_pts = [smooth_track([pts21_raw[i][j] for i in range(total)]) for j in range(21)]
    sm_pn  = smooth_track(pn_raw)

    # Intro padding
    hang = mathutils.Vector((0, 0, -0.15))
    if INTRO_FRAMES > 0:
        ref0 = sm_pts[0][0] if sm_pts[0] else mathutils.Vector((0, 0, 0))
        for j in range(21):
            pad_v = sm_pts[j][0] + hang if sm_pts[j] else ref0 + hang
            sm_pts[j] = [pad_v.copy() for _ in range(INTRO_FRAMES)] + sm_pts[j]
        sm_pn = [None] * INTRO_FRAMES + sm_pn

    # 진단 프레임 선택
    l_cands = [i for i, v in enumerate(valid_raw) if v]
    if not l_cands: return None, None, None, None, 0
    raw_idx = l_cands[min(TEST_FRAME_IDX, len(l_cands) - 1)]
    s_idx   = raw_idx + INTRO_FRAMES

    return sm_pts, sm_pn, s_idx, valid_raw, total + INTRO_FRAMES


# ================================================================
# 메인
# ================================================================

def main():
    arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if not arm: print("[오류] 아마추어 없음"); return

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')

    sm_pts, sm_pn, s_idx, valid_raw, total_frames = load_frame_data(arm)
    if sm_pts is None:
        print("[오류] 데이터 로드 실패 (JSON 경로 또는 유효 프레임 확인)"); return

    tframe = s_idx + 1
    l_pts  = [sm_pts[j][s_idx] for j in range(21)]
    l_pn   = sm_pn[s_idx]
    ai     = arm.matrix_world.to_3x3().inverted()

    def w2a(v):
        """world → armature space (normalized)."""
        return (ai @ v).normalized()

    SEP = "=" * 72

    # ═════════════════════════════════════════════════════════════════════
    # A. 현재 FK basis 방식
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print(f"diag_finger_basis — 진단 프레임 s_idx={s_idx}  blender_frame={tframe}")
    print(SEP)

    print("\n## A. 현재 FK basis 계산 방식 (blend_animate.py 기준)")
    print("  [손뼈]  set_hand_rotation: forward(wrist→mid9) + palm_normal → Z축 roll 보정")
    print("  [손가락] set_bone_dir:     rotation_difference(rest_Y, target_Y) + roll_pn=palm_normal")
    print("           → Y축(뼈 길이) 정렬 후 palm_normal로 Z축 roll 보정 (apply_hand 와 동일)")
    print("           → 각 마디 chain propagate (parent_mat 전달)")
    print("  ※ 섹션 B2의 'Current FK' = roll_pn 없음 (비교용), 'Palm-basis FK' = roll_pn 적용")

    # ═════════════════════════════════════════════════════════════════════
    # B. 검지 정적 basis 진단
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n## B. 검지 정적 basis 진단  (frame={tframe})")
    pn_str = f"({l_pn.normalized().x:+.3f}, {l_pn.normalized().y:+.3f}, {l_pn.normalized().z:+.3f})" \
             if l_pn and l_pn.length > 1e-6 else "N/A"
    print(f"  palm_normal (world): {pn_str}")

    # 현재 액션에서 pose 복원
    bpy.context.scene.frame_set(tframe)
    bpy.context.view_layer.update()

    prev_eval_z = None
    for (bn, fi, ti) in INDEX_CHAIN_L:
        b = arm.pose.bones.get(bn)
        if not b: print(f"\n  {bn}: 뼈 없음"); continue

        fwd_w, bend_w, norm_w = target_basis(l_pts[fi], l_pts[ti], l_pn)
        if fwd_w is None: print(f"\n  {bn}: 방향 계산 불가"); continue

        # target basis (armature space)
        tf = w2a(fwd_w)
        tb = w2a(bend_w) if bend_w else None
        tn = w2a(norm_w) if norm_w else None

        # Blender 평가 후 bone axes (armature space, pose_bone.matrix.col[])
        bm  = b.matrix.to_3x3()
        ef  = bm.col[1].normalized()   # Y = 뼈 길이 방향
        ex  = bm.col[0].normalized()   # X
        ez  = bm.col[2].normalized()   # Z

        fe  = ang(ef, tf)
        be  = ang(ex, tb) if tb else 999.0
        ne  = ang(ez, tn) if tn else 999.0

        # roll: eval_X를 forward에 직교 투영 후 target_bend와 각도
        if tb:
            xp  = ex - ex.dot(ef) * ef
            tbp = tb - tb.dot(ef) * ef
            roll = (math.degrees(math.acos(max(-1, min(1, xp.normalized().dot(tbp.normalized())))))
                    if xp.length > 1e-5 and tbp.length > 1e-5 else 999.0)
        else:
            roll = 999.0

        # parent→child normal 연속성
        twist = ang(ez, prev_eval_z) if prev_eval_z is not None else None
        prev_eval_z = ez.copy()

        print(f"\n  [{bn}]  MP {fi}→{ti}")
        print(f"    target fwd  (arma): {tf.x:+.3f} {tf.y:+.3f} {tf.z:+.3f}")
        if tb: print(f"    target bend (arma): {tb.x:+.3f} {tb.y:+.3f} {tb.z:+.3f}")
        if tn: print(f"    target norm (arma): {tn.x:+.3f} {tn.y:+.3f} {tn.z:+.3f}")
        print(f"    eval   Y    (arma): {ef.x:+.3f} {ef.y:+.3f} {ef.z:+.3f}")
        print(f"    eval   X    (arma): {ex.x:+.3f} {ex.y:+.3f} {ex.z:+.3f}")
        print(f"    eval   Z    (arma): {ez.x:+.3f} {ez.y:+.3f} {ez.z:+.3f}")
        print(f"    forward err : {fe:6.1f}도")
        print(f"    bend-ax err : {be:6.1f}도   (eval_X vs target_bend)")
        print(f"    normal  err : {ne:6.1f}도   (eval_Z vs target_norm)")
        print(f"    roll diff   : {roll:6.1f}도   (eval_X proj vs target_bend proj)")
        if twist is not None:
            print(f"    parent→child twist (Z-Z): {twist:6.1f}도")

    # ═════════════════════════════════════════════════════════════════════
    # B2. Current FK vs Palm-basis FK 비교
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n## B2. Current FK vs Palm-basis FK 비교  (frame={tframe})")
    print(f"  {'뼈':35s}  {'방식':14s}   fwd    bend   norm   roll")

    for use_palm, label in [(False, "Current FK"), (True, "Palm-basis FK")]:
        # 액션 복원 후 재적용
        bpy.context.scene.frame_set(tframe)
        bpy.context.view_layer.update()

        hm  = bone_arma_mat(arm, BONE_L_HAND)
        pm  = get_parent_mat(arm, INDEX_CHAIN_L[0][0], BONE_L_HAND, hm)
        for (bn, fi, ti) in INDEX_CHAIN_L:
            pn_arg = l_pn if use_palm else None
            ok, nm = set_bone_dir(arm, bn, l_pts[fi], l_pts[ti], parent_mat=pm, roll_pn=pn_arg)
            if ok: pm = nm
        bpy.context.view_layer.update()

        for (bn, fi, ti) in INDEX_CHAIN_L:
            b = arm.pose.bones.get(bn)
            if not b: continue
            fwd_w, bend_w, norm_w = target_basis(l_pts[fi], l_pts[ti], l_pn)
            if not fwd_w: continue
            tf = w2a(fwd_w); tb = w2a(bend_w) if bend_w else None; tn = w2a(norm_w) if norm_w else None
            bm  = b.matrix.to_3x3()
            ef  = bm.col[1].normalized(); ex = bm.col[0].normalized(); ez = bm.col[2].normalized()
            fe  = ang(ef, tf)
            be  = ang(ex, tb) if tb else 999.0
            ne  = ang(ez, tn) if tn else 999.0
            if tb:
                xp  = ex - ex.dot(ef) * ef; tbp = tb - tb.dot(ef) * ef
                roll = (math.degrees(math.acos(max(-1, min(1, xp.normalized().dot(tbp.normalized())))))
                        if xp.length > 1e-5 and tbp.length > 1e-5 else 999.0)
            else:
                roll = 999.0
            print(f"  {bn:35s}  {label:14s}  {fe:5.1f}  {be:5.1f}  {ne:5.1f}  {roll:5.1f}")

    # ═════════════════════════════════════════════════════════════════════
    # C. 전체 프레임 연속성 이상 탐지
    # ═════════════════════════════════════════════════════════════════════
    print("\n## C. 프레임 연속성 이상 탐지")

    action = arm.animation_data.action if arm.animation_data else None
    if not action:
        print("  [경고] 액션 없음 — blend_animate.py 먼저 실행 필요")
    else:
        monitored = [bn for (bn, _, _) in INDEX_CHAIN_L] + [BONE_L_HAND]
        fs = int(bpy.context.scene.frame_start)
        fe = int(bpy.context.scene.frame_end)

        prev_q = {}
        anomalies = []

        print(f"  frame {fs}~{fe} 스캔 중... (느릴 수 있음)")
        for fr in range(fs, fe + 1):
            bpy.context.scene.frame_set(fr)
            bpy.context.view_layer.update()
            for bn in monitored:
                b = arm.pose.bones.get(bn)
                if not b: continue
                q = b.rotation_quaternion.copy()
                if bn in prev_q:
                    dot = prev_q[bn].dot(q)
                    dq  = prev_q[bn].rotation_difference(q)
                    deg = math.degrees(2 * math.acos(max(-1.0, min(1.0, abs(dq.w)))))
                    if dot < 0:
                        anomalies.append((fr, bn, f"SIGN_FLIP  dot={dot:+.3f}  delta={deg:.0f}도"))
                    elif deg > DELTA_DEG_THRESH:
                        anomalies.append((fr, bn, f"LARGE_DELTA {deg:.0f}도"))
                prev_q[bn] = q

        # palm normal 반전 탐지
        for i in range(1, len(sm_pn)):
            a = sm_pn[i - 1]; b = sm_pn[i]
            if a and b and a.length > 1e-6 and b.length > 1e-6:
                if a.normalized().dot(b.normalized()) < 0:
                    anomalies.append((fs + i, 'palm_normal', 'REVERSAL'))

        if not anomalies:
            print("  이상 없음 — 전 프레임 정상")
        else:
            print(f"  총 {len(anomalies)}건 이상 탐지:")
            for (fr, bn, tag) in anomalies[:30]:
                print(f"    frame={fr:3d}  {bn}  {tag}")
            if len(anomalies) > 30:
                print(f"    ... 외 {len(anomalies) - 30}건 생략")

    # ═════════════════════════════════════════════════════════════════════
    # D. 메시 스키닝 진단
    # ═════════════════════════════════════════════════════════════════════
    print("\n## D. 메시 스키닝 진단")

    mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH' and o.find_armature() == arm]
    if not mesh_objs:
        print("  Armature에 연결된 메시 없음 (find_armature() 반환 None)")
    else:
        for mo in mesh_objs:
            print(f"\n  Mesh: {mo.name}  scale={tuple(round(v,4) for v in mo.scale)}")
            arm_mod = next((m for m in mo.modifiers if m.type == 'ARMATURE'), None)
            if arm_mod:
                print(f"  Armature modifier: '{arm_mod.name}'  "
                      f"target='{arm_mod.object.name if arm_mod.object else 'None'}'")
            else:
                print("  [경고] Armature modifier 없음")

            check_bones = [(bn, _, _) for (bn, _, _) in INDEX_CHAIN_L] + [(BONE_L_HAND, 0, 0)]
            for (bn, *_) in check_bones:
                vg = mo.vertex_groups.get(bn)
                if vg:
                    cnt = sum(1 for v in mo.data.vertices
                              if any(vge.group == vg.index for vge in v.groups))
                    print(f"    VG '{bn}': {cnt}개 vertex")
                else:
                    print(f"    VG '{bn}': 없음 — 이름 불일치 가능성")

        # ── Index1 굽힘 테스트 ────────────────────────────────────
        print("\n  [굽힘 테스트] Index1을 local-Z 20도 회전 → 영향 vertex 측정")
        bpy.context.scene.frame_set(tframe)
        bpy.context.view_layer.update()

        # 현재 쿼터니언 저장
        saved_q = {b.name: b.rotation_quaternion.copy() for b in arm.pose.bones}

        depsgraph = bpy.context.evaluated_depsgraph_get()
        for mo in mesh_objs:
            me0 = mo.evaluated_get(depsgraph)
            orig = [me0.data.vertices[i].co.copy() for i in range(len(mo.data.vertices))]

            b1  = arm.pose.bones.get('mixamorig:LeftHandIndex1')
            bd1 = arm.data.bones.get('mixamorig:LeftHandIndex1')
            if not b1 or not bd1:
                print("  [경고] mixamorig:LeftHandIndex1 없음"); continue

            # local Z축으로 20도 굽힘 (rest pose 기준)
            bend_axis_local = bd1.matrix_local.to_3x3().col[2].normalized()
            rq = mathutils.Quaternion(bend_axis_local, math.radians(20))
            base_q = saved_q.get('mixamorig:LeftHandIndex1', mathutils.Quaternion())
            b1.rotation_quaternion = (rq @ base_q).normalized()
            bpy.context.view_layer.update()

            depsgraph = bpy.context.evaluated_depsgraph_get()
            me1 = mo.evaluated_get(depsgraph)

            # vertex group별 최대 이동량
            vg_max = {}
            for v in mo.data.vertices:
                delta = (me1.data.vertices[v.index].co - orig[v.index]).length
                if delta < 1e-4: continue
                for vge in v.groups:
                    vgname = mo.vertex_groups[vge.group].name
                    if vgname not in vg_max or vg_max[vgname] < delta:
                        vg_max[vgname] = delta

            print(f"\n  mesh='{mo.name}'  Index1 20도 굽힘 → 영향 vertex group (max_delta>0.001):")
            FINGER_KWORDS = {'Index', 'Hand', 'Thumb', 'Middle', 'Ring', 'Pinky', 'Wrist'}
            for vgname, d in sorted(vg_max.items(), key=lambda x: -x[1]):
                if d < 0.001: continue
                is_expected = any(kw in vgname for kw in FINGER_KWORDS)
                flag = "" if is_expected else "  ← 비정상"
                print(f"    {vgname:45s}  max={d:.4f}{flag}")

            # 원래 자세 복원
            b1.rotation_quaternion = saved_q.get('mixamorig:LeftHandIndex1', mathutils.Quaternion())
            bpy.context.view_layer.update()

    # ═════════════════════════════════════════════════════════════════════
    # 완료
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("진단 완료. 섹션 B2의 bend/normal/roll 오차와 섹션 C 이상 건수를 보고해주세요.")
    print(SEP)
    bpy.ops.object.mode_set(mode='OBJECT')


main()
