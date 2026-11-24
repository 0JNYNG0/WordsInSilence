"""
research/scripts/fix_blender_retarget_mirror.py
────────────────────────────────────────────────
Phase 2 Fix: Blender retargeting with CORRECT mirroring & orientation.

Root cause of prior bugs:
  1. MIRROR BUG: Previous script applied MIRROR_X (negated X), which swapped
     left/right hands because MediaPipe world landmarks already use
     performer-centric coordinates (X+ = performer's left).
  2. FACING BUG: Character was rotated 180° because the Z→Y depth mapping
     was inverted, making the character face backward.

Correct coordinate transform (MediaPipe world → Blender):
  MediaPipe world: X+ = performer left, Y+ = down, Z+ = toward camera
  Blender:         X+ = right,          Z+ = up,   Y+ = forward (into screen)

  bx = +mp_x   (performer left  = character left  = Blender +X)
  by = -mp_z   (toward camera   = toward viewer   = Blender -Y)
  bz = -mp_y   (down            = down             = Blender -Z → inverted = up)

Hand binding: MediaPipe 'Left' → Character Left arm. NO swap.

Usage:
  blender.exe <rig.blend> --background --python research/scripts/fix_blender_retarget_mirror.py
"""

import bpy, json, math, mathutils, os, subprocess, csv

print("[SCRIPT] fix_blender_retarget_mirror.py — Corrected retarget (no mirror, correct facing)")

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
_PROJECT = r"C:\MyUnityProjects\WordsInSilence"
_TOOLS   = os.path.join(_PROJECT, "Tools")
OUT_V3A  = os.path.join(_TOOLS, "s04_out_v3a")
ANIM_OUT = os.path.join(_PROJECT, "research", "anim_out")
os.makedirs(ANIM_OUT, exist_ok=True)

LANDMARKS_PATH = os.path.join(OUT_V3A, "landmarks_raw_front.json")
BOUNDARIES_CSV = os.path.join(OUT_V3A, "boundaries_override_template.csv")
FBX_PATH       = os.path.join(ANIM_OUT, "Anim_SIGN001_Greeting.fbx")
BLEND_SAVE     = os.path.join(ANIM_OUT, "Anim_SIGN001_Greeting.blend")
REVIEW_DIR     = os.path.join(ANIM_OUT, "render_review")
REVIEW_MP4     = os.path.join(ANIM_OUT, "Anim_SIGN001_Greeting_review.mp4")
LOG_PATH       = os.path.join(ANIM_OUT, "retarget.log")

SYSTEM_PYTHON  = r"C:\Users\local-user\AppData\Local\Programs\Python\Python313\python.exe"
ENCODE_SCRIPT  = os.path.join(_TOOLS, "s04_encode_mp4.py")

FPS = 30

# ── Which rep to use ──
TARGET_REP = 1  # Rep 1: fi=787-849 (clean, 63 frames, 2.1s)

# ── Rest-pose padding ──
REST_PAD   = 18  # frames of idle before/after
EASE_FRAMES = 12  # smooth transition frames
FORWARD_OFFSET = 0.10  # slight forward offset for natural depth

# ── Hand settings (confirmed from CLAUDE.md / blend_animate.py) ──
L_PALM_FLIP = True
R_PALM_FLIP = False

SMOOTH_WINDOW = 3

# ═══════════════════════════════════════════════════════════════════════════
# Bone Names (Mixamo standard)
# ═══════════════════════════════════════════════════════════════════════════
BONE_L_ARM     = 'mixamorig:LeftArm'
BONE_R_ARM     = 'mixamorig:RightArm'
BONE_L_FOREARM = 'mixamorig:LeftForeArm'
BONE_R_FOREARM = 'mixamorig:RightForeArm'
BONE_L_HAND    = 'mixamorig:LeftHand'
BONE_R_HAND    = 'mixamorig:RightHand'

# MediaPipe Pose indices
P_L_SHOULDER, P_R_SHOULDER = 11, 12
P_L_ELBOW,    P_R_ELBOW    = 13, 14
P_L_WRIST,    P_R_WRIST    = 15, 16

# MediaPipe Hand indices
H_WRIST = 0
H_THUMB_CMC, H_THUMB_MCP, H_THUMB_IP, H_THUMB_TIP = 1, 2, 3, 4
H_INDEX_MCP, H_INDEX_PIP, H_INDEX_DIP, H_INDEX_TIP = 5, 6, 7, 8
H_MID_MCP,   H_MID_PIP,   H_MID_DIP,   H_MID_TIP   = 9, 10, 11, 12
H_RING_MCP,  H_RING_PIP,  H_RING_DIP,  H_RING_TIP  = 13, 14, 15, 16
H_PINK_MCP,  H_PINK_PIP,  H_PINK_DIP,  H_PINK_TIP  = 17, 18, 19, 20

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


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate Transform — THE FIX
# ═══════════════════════════════════════════════════════════════════════════

def mp_world_to_blender(p):
    """
    MediaPipe pose_world landmark [x, y, z, vis?] → Blender world Vector.

    MediaPipe world coordinates (hip-centered):
      X+ = performer's LEFT  (camera's right)
      Y+ = DOWN
      Z+ = TOWARD CAMERA (forward, out of screen)

    Blender coordinates:
      X+ = RIGHT
      Y+ = INTO SCREEN (away from viewer)
      Z+ = UP

    Character faces -Y (toward the viewer/camera).

    Transform:
      bx = +mp_x     (performer's left = character's left in Blender)
      by = -mp_z      (toward camera = toward viewer = -Y)
      bz = -mp_y      (down = -Z, so up = +Z)

    NO X-MIRROR. This was the bug — MediaPipe world is already
    performer-centric, not selfie-mirrored.
    """
    x, y, z = p[0], p[1], p[2]
    bx = x                      # performer left = +X
    by = -z - FORWARD_OFFSET    # toward camera = -Y (slight offset)
    bz = -y                     # up = +Z
    return mathutils.Vector((bx, by, bz))


def mp_hand_world_to_blender(hand_pt, pose_wrist_blender):
    """
    Hand world landmarks are WRIST-CENTERED (relative to hand wrist).
    Same axis convention as pose_world. Add to pose wrist position.
    """
    x, y, z = hand_pt[0], hand_pt[1], hand_pt[2]
    # Same transform, no offset (relative coords)
    bx = x
    by = -z
    bz = -y
    return pose_wrist_blender + mathutils.Vector((bx, by, bz))


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def smooth_vectors(vecs, window=SMOOTH_WINDOW):
    """Moving average for Vector tracks, skip None."""
    if window < 2:
        return vecs
    n = len(vecs)
    half = window // 2
    result = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        valid = [v for v in vecs[lo:hi] if v is not None]
        if valid:
            avg = mathutils.Vector((0, 0, 0))
            for v in valid:
                avg += v
            result.append(avg / len(valid))
        else:
            result.append(vecs[i])
    return result


def ease_smooth(t):
    """Cubic Hermite ease: t∈[0,1]→[0,1], zero velocity at endpoints."""
    return t * t * (3.0 - 2.0 * t)


def lerp_vec(a, b, t):
    return a * (1.0 - t) + b * t


def palm_normal_from_hand(hand_lm, wrist_b):
    """Palm normal in Blender world coords from hand landmarks."""
    p0 = mp_hand_world_to_blender(hand_lm[H_WRIST], wrist_b)
    p5 = mp_hand_world_to_blender(hand_lm[H_INDEX_MCP], wrist_b)
    p17 = mp_hand_world_to_blender(hand_lm[H_PINK_MCP], wrist_b)
    v1 = p5 - p0
    v2 = p17 - p0
    n = v1.cross(v2)
    return n.normalized() if n.length > 1e-6 else None


# ═══════════════════════════════════════════════════════════════════════════
# Blender Rig Functions (from blend_animate.py, cleaned)
# ═══════════════════════════════════════════════════════════════════════════

def make_empty(name, col):
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    emp = bpy.data.objects.new(name, None)
    emp.empty_display_type = 'PLAIN_AXES'
    emp.empty_display_size = 0.02
    col.objects.link(emp)
    return emp


def add_ik(arm_obj, bone_name, target, chain_count):
    b = arm_obj.pose.bones.get(bone_name)
    if not b:
        return None
    for c in list(b.constraints):
        if c.type == 'IK':
            b.constraints.remove(c)
    c = b.constraints.new('IK')
    c.target = target
    c.chain_count = chain_count
    c.use_tail = True
    c.use_rotation = False
    c.use_stretch = False
    c.influence = 1.0
    c.iterations = 500
    return c


def animate_empty(emp, positions, frames):
    emp.animation_data_create()
    if emp.animation_data.action is None:
        emp.animation_data.action = bpy.data.actions.new(emp.name + "_act")
    for frame, pos in zip(frames, positions):
        if pos is not None:
            emp.location = pos
            emp.keyframe_insert('location', frame=frame)


def bone_arma_mat(arm_obj, bone_name, parent_mat=None):
    b = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        return parent_mat or mathutils.Matrix.Identity(3)
    rest = bd.matrix_local.to_3x3()
    local_rot = b.rotation_quaternion.to_matrix()
    if b.parent:
        p_bd = arm_obj.data.bones.get(b.parent.name)
        p_rest = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        p_mat = parent_mat if parent_mat is not None else b.parent.matrix.to_3x3()
        return p_mat @ p_rest.inverted() @ rest @ local_rot
    return rest @ local_rot


def set_bone_dir(arm_obj, bone_name, from_w, to_w, parent_mat=None, roll_pn=None):
    b = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        return False, parent_mat
    direction = (to_w - from_w)
    if direction.length < 1e-6:
        return False, parent_mat
    direction = direction.normalized()
    a_inv = arm_obj.matrix_world.to_3x3().inverted()
    dir_a = (a_inv @ direction).normalized()
    rest = bd.matrix_local.to_3x3()
    ry, rx, rz = rest.col[1].normalized(), rest.col[0].normalized(), rest.col[2].normalized()
    q = ry.rotation_difference(dir_a)
    ny, nx, nz = dir_a, (q @ rx).normalized(), (q @ rz).normalized()
    if roll_pn is not None and roll_pn.length > 1e-5:
        pna = (a_inv @ roll_pn).normalized()
        pp = pna - pna.dot(ny) * ny
        zp = nz - nz.dot(ny) * ny
        if pp.length > 1e-4 and zp.length > 1e-4:
            pp, zp = pp.normalized(), zp.normalized()
            cr = max(-1.0, min(1.0, zp.dot(pp)))
            ra = math.atan2(zp.cross(pp).dot(ny), cr)
            rq = mathutils.Quaternion(ny, ra)
            nx, nz = rq @ nx, rq @ nz
    desired_a = mathutils.Matrix([[nx.x,ny.x,nz.x],[nx.y,ny.y,nz.y],[nx.z,ny.z,nz.z]])
    if b.parent:
        p_bd = arm_obj.data.bones.get(b.parent.name)
        p_rest = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        p_mat = parent_mat if parent_mat is not None else b.parent.matrix.to_3x3()
        pose_mat = rest.inverted() @ p_rest @ p_mat.inverted() @ desired_a
    else:
        pose_mat = rest.inverted() @ desired_a
    b.rotation_mode = 'QUATERNION'
    b.rotation_quaternion = pose_mat.to_quaternion()
    local_rot_mat = b.rotation_quaternion.to_matrix()
    if b.parent:
        p_mat_used = parent_mat if parent_mat is not None else b.parent.matrix.to_3x3()
        p_bd = arm_obj.data.bones.get(b.parent.name)
        p_rest = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        new_mat = p_mat_used @ p_rest.inverted() @ rest @ local_rot_mat
    else:
        new_mat = rest @ local_rot_mat
    return True, new_mat


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


def set_hand_rotation(arm_obj, bone_name, wrist_w, mid9_w, palm_n_w):
    b = arm_obj.pose.bones.get(bone_name)
    bd = arm_obj.data.bones.get(bone_name)
    if not b or not bd:
        return
    hand_dir = (mid9_w - wrist_w)
    if hand_dir.length < 1e-6:
        return
    hand_dir = hand_dir.normalized()
    a_inv = arm_obj.matrix_world.to_3x3().inverted()
    dir_a = (a_inv @ hand_dir).normalized()
    rest = bd.matrix_local.to_3x3()
    ry, rx, rz = rest.col[1].normalized(), rest.col[0].normalized(), rest.col[2].normalized()
    q = ry.rotation_difference(dir_a)
    ny, nx, nz = dir_a, (q @ rx).normalized(), (q @ rz).normalized()
    if palm_n_w is not None and palm_n_w.length > 1e-5:
        pna = (a_inv @ palm_n_w).normalized()
        pp = pna - pna.dot(ny) * ny
        zp = nz - nz.dot(ny) * ny
        if pp.length > 1e-4 and zp.length > 1e-4:
            pp, zp = pp.normalized(), zp.normalized()
            cr = max(-1.0, min(1.0, zp.dot(pp)))
            ra = math.atan2(zp.cross(pp).dot(ny), cr)
            rq = mathutils.Quaternion(ny, ra)
            nx, nz = rq @ nx, rq @ nz
    desired_a = mathutils.Matrix([[nx.x,ny.x,nz.x],[nx.y,ny.y,nz.y],[nx.z,ny.z,nz.z]])
    if b.parent:
        p_bd = arm_obj.data.bones.get(b.parent.name)
        p_rest = p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        p_mat = b.parent.matrix.to_3x3()
        pose_mat = rest.inverted() @ p_rest @ p_mat.inverted() @ desired_a
    else:
        pose_mat = rest.inverted() @ desired_a
    b.rotation_mode = 'QUATERNION'
    b.rotation_quaternion = pose_mat.to_quaternion()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    arm_obj = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if not arm_obj:
        print("[ERROR] No armature"); return
    print(f"[info] Armature: {arm_obj.name}")

    # ── Load rep boundaries ──
    rep_start, rep_end = None, None
    with open(BOUNDARIES_CSV, 'r') as f:
        for row in csv.DictReader(f):
            if int(row['rep_id']) == TARGET_REP:
                rep_start = int(row['front_start_fi'])
                rep_end = int(row['front_end_fi'])
                break
    if rep_start is None:
        print(f"[ERROR] Rep {TARGET_REP} not found"); return
    print(f"[info] Rep {TARGET_REP}: fi={rep_start}-{rep_end}")

    # ── Load landmarks ──
    print("[load] Reading landmarks...")
    with open(LANDMARKS_PATH, 'r', encoding='utf-8') as f:
        all_frames = json.load(f)
    if isinstance(all_frames, dict):
        all_frames = all_frames.get('frames', all_frames)

    src_frames = [fr for fr in all_frames if rep_start <= fr['fi'] <= rep_end]
    n_src = len(src_frames)
    print(f"[info] {n_src} source frames for rep {TARGET_REP}")

    # ── Character measurements ──
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    _mw = arm_obj.matrix_world
    lb = arm_obj.pose.bones.get(BONE_L_ARM)
    rb = arm_obj.pose.bones.get(BONE_R_ARM)
    lb_fa = arm_obj.pose.bones.get(BONE_L_FOREARM)
    rb_fa = arm_obj.pose.bones.get(BONE_R_FOREARM)

    def blen(pb):
        return (_mw @ pb.tail - _mw @ pb.head).length if pb else 0.0

    char_l_arm = blen(lb) + blen(lb_fa)
    char_r_arm = blen(rb) + blen(rb_fa)
    l_sh_w = _mw @ lb.head if lb else mathutils.Vector((-0.15, 0, 1.4))
    r_sh_w = _mw @ rb.head if rb else mathutils.Vector((0.15, 0, 1.4))
    char_origin = (l_sh_w + r_sh_w) / 2
    char_sw = (l_sh_w - r_sh_w).length
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[info] Char SW={char_sw:.4f}, arm L={char_l_arm:.4f} R={char_r_arm:.4f}")

    # Rest-pose idle positions (arms at sides)
    idle_l = l_sh_w + mathutils.Vector((0, -FORWARD_OFFSET, -0.40))
    idle_r = r_sh_w + mathutils.Vector((0, -FORWARD_OFFSET, -0.40))
    hang = mathutils.Vector((0, 0, -0.05))

    # ── Compute source shoulder width for scaling ──
    src_sws = []
    for fr in src_frames:
        pw = fr.get('pose_world')
        if pw and len(pw) > P_R_SHOULDER:
            ls = mp_world_to_blender(pw[P_L_SHOULDER])
            rs = mp_world_to_blender(pw[P_R_SHOULDER])
            src_sws.append((ls - rs).length)
    src_sw = sum(src_sws) / len(src_sws) if src_sws else 0.3
    scale = char_sw / src_sw if src_sw > 1e-4 else 1.0
    print(f"[info] Source SW={src_sw:.4f}, scale={scale:.4f}")

    # ── Extract raw tracks ──
    print("[extract] Building tracks...")
    lw_raw, rw_raw = [], []
    lh_pts_raw, rh_pts_raw = [], []
    l_pn_raw, r_pn_raw = [], []
    lh9_raw, rh9_raw = [], []
    l_valid, r_valid = [], []

    for fr in src_frames:
        pw = fr.get('pose_world')
        if pw is None or len(pw) < 25:
            for lst in [lw_raw, rw_raw, lh_pts_raw, rh_pts_raw,
                        l_pn_raw, r_pn_raw, lh9_raw, rh9_raw]:
                lst.append(None)
            l_valid.append(False); r_valid.append(False)
            continue

        # Source shoulder center for centering
        src_ls = mp_world_to_blender(pw[P_L_SHOULDER])
        src_rs = mp_world_to_blender(pw[P_R_SHOULDER])
        src_center = (src_ls + src_rs) / 2

        def to_char(v):
            """Center on source shoulders, scale to character, place at char origin."""
            return char_origin + (v - src_center) * scale if v else None

        lw_b = to_char(mp_world_to_blender(pw[P_L_WRIST]))
        rw_b = to_char(mp_world_to_blender(pw[P_R_WRIST]))
        lw_raw.append(lw_b)
        rw_raw.append(rw_b)

        # ── Hands ──
        # ANATOMICAL BINDING: mp 'Left' → character Left, mp 'Right' → character Right
        hands = fr.get('hands') or []
        left_lm, right_lm = None, None
        for h in hands:
            label = h.get('handedness', '')
            w = h.get('world')
            if w and len(w) >= 21:
                if label == 'Left' and left_lm is None:
                    left_lm = w
                elif label == 'Right' and right_lm is None:
                    right_lm = w

        # Process each hand
        for (side_lm, wrist_b, pts_list, pn_list, h9_list, valid_list) in [
            (left_lm,  lw_b, lh_pts_raw, l_pn_raw, lh9_raw, l_valid),
            (right_lm, rw_b, rh_pts_raw, r_pn_raw, rh9_raw, r_valid),
        ]:
            if side_lm and wrist_b:
                pts_b = []
                for i in range(21):
                    raw_pt = mp_hand_world_to_blender(side_lm[i], wrist_b)
                    # Scale relative to wrist
                    pts_b.append(wrist_b + (raw_pt - wrist_b) * scale)
                pts_list.append(pts_b)
                pn = palm_normal_from_hand(side_lm, wrist_b)
                pn_list.append(pn)
                h9_list.append(pts_b[H_MID_MCP] if pts_b[H_MID_MCP] else wrist_b)
                valid_list.append(True)
            else:
                pts_list.append(None)
                pn_list.append(None)
                h9_list.append(None)
                valid_list.append(False)

    print(f"[info] Valid: L={sum(l_valid)}/{n_src}, R={sum(r_valid)}/{n_src}")

    # ── Smooth ──
    lw_sm = smooth_vectors(lw_raw)
    rw_sm = smooth_vectors(rw_raw)
    lh9_sm = smooth_vectors(lh9_raw)
    rh9_sm = smooth_vectors(rh9_raw)

    # Per-landmark hand smoothing
    lh_pts_sm = [list(p) if p else None for p in lh_pts_raw]
    rh_pts_sm = [list(p) if p else None for p in rh_pts_raw]
    for pt_idx in range(21):
        for pts_sm in [lh_pts_sm, rh_pts_sm]:
            track = [pts_sm[i][pt_idx] if pts_sm[i] else None for i in range(n_src)]
            smoothed = smooth_vectors(track)
            for i in range(n_src):
                if pts_sm[i] and smoothed[i]:
                    pts_sm[i][pt_idx] = smoothed[i]

    # ── Build padded timeline ──
    # [REST_PAD] [EASE_IN] [MOTION] [EASE_OUT] [REST_PAD]
    total = REST_PAD + EASE_FRAMES + n_src + EASE_FRAMES + REST_PAD
    print(f"[info] Timeline: {REST_PAD}+{EASE_FRAMES}+{n_src}+{EASE_FRAMES}+{REST_PAD} = {total} frames")

    first_l = lw_sm[0] if lw_sm[0] else idle_l
    first_r = rw_sm[0] if rw_sm[0] else idle_r
    last_l = lw_sm[-1] if lw_sm[-1] else idle_l
    last_r = rw_sm[-1] if rw_sm[-1] else idle_r

    full_lw, full_rw = [], []
    full_lh, full_rh = [], []
    full_lpn, full_rpn = [], []
    full_lh9, full_rh9 = [], []
    full_lv, full_rv = [], []

    def add_idle(n):
        for _ in range(n):
            full_lw.append(idle_l.copy()); full_rw.append(idle_r.copy())
            full_lh.append(None); full_rh.append(None)
            full_lpn.append(None); full_rpn.append(None)
            full_lh9.append(idle_l + hang); full_rh9.append(idle_r + hang)
            full_lv.append(False); full_rv.append(False)

    # Rest pad
    add_idle(REST_PAD)

    # Ease-in
    for k in range(EASE_FRAMES):
        t = ease_smooth((k + 1) / (EASE_FRAMES + 1))
        full_lw.append(lerp_vec(idle_l, first_l, t))
        full_rw.append(lerp_vec(idle_r, first_r, t))
        full_lh.append(None); full_rh.append(None)
        full_lpn.append(None); full_rpn.append(None)
        full_lh9.append(lerp_vec(idle_l + hang, lh9_sm[0] if lh9_sm[0] else first_l, t))
        full_rh9.append(lerp_vec(idle_r + hang, rh9_sm[0] if rh9_sm[0] else first_r, t))
        full_lv.append(False); full_rv.append(False)

    # Motion
    for i in range(n_src):
        full_lw.append(lw_sm[i] if lw_sm[i] else idle_l)
        full_rw.append(rw_sm[i] if rw_sm[i] else idle_r)
        full_lh.append(lh_pts_sm[i])
        full_rh.append(rh_pts_sm[i])
        full_lpn.append(l_pn_raw[i])
        full_rpn.append(r_pn_raw[i])
        full_lh9.append(lh9_sm[i] if lh9_sm[i] else (lw_sm[i] if lw_sm[i] else idle_l))
        full_rh9.append(rh9_sm[i] if rh9_sm[i] else (rw_sm[i] if rw_sm[i] else idle_r))
        full_lv.append(l_valid[i])
        full_rv.append(r_valid[i])

    # Ease-out
    for k in range(EASE_FRAMES):
        t = ease_smooth((k + 1) / (EASE_FRAMES + 1))
        full_lw.append(lerp_vec(last_l, idle_l, t))
        full_rw.append(lerp_vec(last_r, idle_r, t))
        full_lh.append(None); full_rh.append(None)
        full_lpn.append(None); full_rpn.append(None)
        full_lh9.append(lerp_vec(lh9_sm[-1] if lh9_sm[-1] else last_l, idle_l + hang, t))
        full_rh9.append(lerp_vec(rh9_sm[-1] if rh9_sm[-1] else last_r, idle_r + hang, t))
        full_lv.append(False); full_rv.append(False)

    # Rest pad
    add_idle(REST_PAD)

    # Clamp reach
    for i in range(total):
        for wt, sh, al in [(full_lw, l_sh_w, char_l_arm), (full_rw, r_sh_w, char_r_arm)]:
            v = wt[i]
            if v:
                reach = (v - sh).length
                if reach > al * 0.95:
                    wt[i] = sh + (v - sh).normalized() * (al * 0.95)

    valid_frames = list(range(1, total + 1))

    # ══════════════════════════════════════════════════════════════════
    # PASS 1: Arm IK
    # ══════════════════════════════════════════════════════════════════
    print("[pass 1] Arm IK keyframes...")
    col = bpy.context.scene.collection
    emp_lw = make_empty('IK_LeftWrist', col)
    emp_rw = make_empty('IK_RightWrist', col)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    # Pre-bend forearms
    for fbn, sign in [(BONE_L_FOREARM, 1), (BONE_R_FOREARM, -1)]:
        fb = arm_obj.pose.bones.get(fbn)
        if fb:
            fb.rotation_mode = 'QUATERNION'
            fb.rotation_quaternion = mathutils.Quaternion(
                mathutils.Vector((1, 0, 0)), math.radians(5 * sign))
            fb.keyframe_insert('rotation_quaternion', frame=1)

    bpy.ops.object.mode_set(mode='OBJECT')
    add_ik(arm_obj, BONE_L_FOREARM, emp_lw, chain_count=2)
    add_ik(arm_obj, BONE_R_FOREARM, emp_rw, chain_count=2)

    animate_empty(emp_lw, full_lw, valid_frames)
    animate_empty(emp_rw, full_rw, valid_frames)

    # ══════════════════════════════════════════════════════════════════
    # PASS 2: Hand + Finger FK
    # ══════════════════════════════════════════════════════════════════
    print(f"[pass 2] Hand/Finger FK ({total} frames)...")
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    prev_q_l, prev_q_r = {}, {}

    def sign_norm(bn, prev_quats):
        b = arm_obj.pose.bones.get(bn)
        pq = prev_quats.get(bn)
        if b and pq and pq.dot(b.rotation_quaternion) < 0:
            b.rotation_quaternion.negate()
        if b:
            prev_quats[bn] = b.rotation_quaternion.copy()

    for i, frame in enumerate(valid_frames):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        for (bone, wrist, mid9, pn, flip, pts, chains, pq, is_v) in [
            (BONE_L_HAND, full_lw[i], full_lh9[i], full_lpn[i],
             L_PALM_FLIP, full_lh[i], FINGER_CHAINS_L, prev_q_l, full_lv[i]),
            (BONE_R_HAND, full_rw[i], full_rh9[i], full_rpn[i],
             R_PALM_FLIP, full_rh[i], FINGER_CHAINS_R, prev_q_r, full_rv[i]),
        ]:
            b = arm_obj.pose.bones.get(bone)
            if not is_v or pts is None:
                if b:
                    b.keyframe_insert('rotation_quaternion', frame=frame)
                continue

            eff_pn = (-pn if flip else pn) if (pn and pn.length > 1e-4) else None

            if eff_pn is not None and wrist and mid9:
                set_hand_rotation(arm_obj, bone, pts[H_WRIST], mid9, eff_pn)

            hand_mat = bone_arma_mat(arm_obj, bone)
            sign_norm(bone, pq)
            b.keyframe_insert('rotation_quaternion', frame=frame)

            for chain in chains:
                first_pm = get_parent_mat(arm_obj, chain[0][0], bone, hand_mat)
                pm = first_pm
                for (bn, fi, ti) in chain:
                    fw, tw = pts[fi], pts[ti]
                    if fw is None or tw is None:
                        pm = bone_arma_mat(arm_obj, bn, pm)
                        continue
                    ok, nm = set_bone_dir(arm_obj, bn, fw, tw, parent_mat=pm, roll_pn=eff_pn)
                    if ok:
                        sign_norm(bn, pq)
                        arm_obj.pose.bones[bn].keyframe_insert('rotation_quaternion', frame=frame)
                        pm = nm
                    else:
                        pm = bone_arma_mat(arm_obj, bn, pm)

    bpy.ops.object.mode_set(mode='OBJECT')

    # ── Set timeline ──
    bpy.context.scene.render.fps = FPS
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total
    bpy.context.scene.frame_set(1)

    # ── Diagnostic: verify handedness ──
    print("\n[DIAGNOSTIC] Handedness verification at gesture peak:")
    mid_frame_idx = REST_PAD + EASE_FRAMES + n_src // 2
    lw_pos = full_lw[mid_frame_idx]
    rw_pos = full_rw[mid_frame_idx]
    if lw_pos and rw_pos:
        print(f"  Mid-gesture (frame {mid_frame_idx+1}):")
        print(f"    L_wrist Blender: X={lw_pos.x:+.4f} Y={lw_pos.y:+.4f} Z={lw_pos.z:+.4f}")
        print(f"    R_wrist Blender: X={rw_pos.x:+.4f} Y={rw_pos.y:+.4f} Z={rw_pos.z:+.4f}")
        print(f"    L_shoulder:      X={l_sh_w.x:+.4f} Y={l_sh_w.y:+.4f} Z={l_sh_w.z:+.4f}")
        print(f"    R_shoulder:      X={r_sh_w.x:+.4f} Y={r_sh_w.y:+.4f} Z={r_sh_w.z:+.4f}")
        if lw_pos.x > 0 and rw_pos.x < 0:
            print("    [OK] L_wrist.x > 0, R_wrist.x < 0 — anatomically correct (no swap)")
        elif lw_pos.x < 0 and rw_pos.x > 0:
            print("    [WARN] Hands appear swapped! L_wrist.x < 0, R_wrist.x > 0")
        else:
            print("    [INFO] Wrists near center — check visually")

    # ── Save .blend ──
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_SAVE)
    print(f"[save] {BLEND_SAVE}")

    # ── Export FBX ──
    print("[export] FBX...")
    bpy.ops.export_scene.fbx(
        filepath=FBX_PATH,
        use_selection=False,
        object_types={'ARMATURE', 'MESH'},
        use_mesh_modifiers=True,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode='COPY',
        embed_textures=False,
        axis_forward='-Z',
        axis_up='Y',
    )
    print(f"[info] FBX: {FBX_PATH} ({os.path.getsize(FBX_PATH)//1024} KB)")

    # ── Render 3-view review ──
    print("[render] 3-view review...")
    os.makedirs(REVIEW_DIR, exist_ok=True)

    CAMERAS = {
        "front": ((0, -3.5, 1.2), (math.radians(80), 0, 0)),
        "side":  ((-3.5, 0, 1.2), (math.radians(80), 0, math.radians(-90))),
        "angle": ((2.5, -2.5, 1.2), (math.radians(80), 0, math.radians(45))),
    }
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    try:
        scene.eevee.taa_render_samples = 32
    except AttributeError:
        pass

    frame_dirs = {}
    for vn, (loc, rot) in CAMERAS.items():
        cn = f"ReviewCam_{vn}"
        if cn in bpy.data.objects:
            co = bpy.data.objects[cn]
        else:
            cd = bpy.data.cameras.new(cn)
            cd.lens = 35
            co = bpy.data.objects.new(cn, cd)
            scene.collection.objects.link(co)
        co.location = loc
        co.rotation_euler = rot
        scene.camera = co
        fd = os.path.join(REVIEW_DIR, vn)
        os.makedirs(fd, exist_ok=True)
        scene.render.filepath = os.path.join(fd, "")
        bpy.ops.render.render(animation=True)
        frame_dirs[vn] = fd

    # Stitch to MP4
    if os.path.exists(SYSTEM_PYTHON):
        stitch_py = os.path.join(ANIM_OUT, "_stitch.py")
        with open(stitch_py, 'w') as f:
            f.write(f"""
import cv2, os, numpy as np
views = {list(CAMERAS.keys())}
rd = r"{REVIEW_DIR}"
out = r"{REVIEW_MP4}"
w = None
for fi in range(1, {total + 1}):
    row = []
    for v in views:
        p = os.path.join(rd, v, f"{{fi:04d}}.png")
        img = cv2.imread(p)
        if img is None: img = np.zeros((480,640,3), dtype=np.uint8)
        cv2.putText(img, v, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(img, f"f={{fi}}/{total}", (10,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        row.append(img)
    canvas = np.hstack(row)
    if w is None:
        h2, w2 = canvas.shape[:2]
        w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), {FPS}, (w2, h2))
    w.write(canvas)
w.release()
print(f"Encoded: {{out}} ({{os.path.getsize(out)//1024}} KB)")
""")
        print("[encode] Stitching review MP4...")
        r = subprocess.run([SYSTEM_PYTHON, stitch_py],
                          capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.stdout: print(r.stdout.rstrip())
        if r.returncode != 0 and r.stderr: print(f"[warn] {r.stderr[-200:]}")

    print(f"\n{'='*60}")
    print(f"Phase 2 Complete — Mirror/Orientation Fix Applied")
    print(f"{'='*60}")
    print(f"  Coordinate transform:")
    print(f"    bx = +mp_x  (NO mirror — performer-centric)")
    print(f"    by = -mp_z - {FORWARD_OFFSET}  (depth → forward)")
    print(f"    bz = -mp_y  (height inverted)")
    print(f"  Hand binding: MP Left → Char Left, MP Right → Char Right")
    print(f"  Character facing: -Y (toward viewer)")
    print(f"  Timeline: {total} frames ({total/FPS:.1f}s)")
    print(f"  FBX: {FBX_PATH}")
    if os.path.exists(REVIEW_MP4):
        print(f"  Review: {REVIEW_MP4} ({os.path.getsize(REVIEW_MP4)//1024} KB)")


main()
