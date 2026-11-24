"""
Tools/s04_phase3a_retarget.py — Phase 3A: Single-View Blender Retargeting
Blender CLI: blender.exe <blend_file> --background --python Tools/s04_phase3a_retarget.py

Reads SIGN001_canonical_motion.json (MediaPipe world landmarks, 97 frames).
Adds rest-pose padding + ease-in/out + IK arms + FK hands/fingers.
Exports FBX + 3-angle review MP4.
"""

import bpy, json, math, mathutils, os, subprocess, sys

print("[SCRIPT] s04_phase3a_retarget.py — Phase 3A Single-View Retargeting")

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(_THIS_DIR, "s04_out_v3a")
JSON_PATH = os.path.join(OUT_DIR, "SIGN001_canonical_motion.json")
FBX_PATH  = os.path.join(OUT_DIR, "SIGN001_greeting.fbx")
RENDER_DIR = os.path.join(OUT_DIR, "render_review")
REVIEW_MP4 = os.path.join(OUT_DIR, "SIGN001_blender_render_review.mp4")
LOG_PATH   = os.path.join(OUT_DIR, "blender_retarget.log")

SYSTEM_PYTHON = r"C:\Users\local-user\AppData\Local\Programs\Python\Python313\python.exe"
ENCODE_SCRIPT = os.path.join(_THIS_DIR, "s04_encode_mp4.py")

FPS = 30
BODY_SCALE = 1.0
MIRROR_X = True  # MediaPipe world: +X = left, Blender: +X = right → mirror X
FORWARD_OFFSET = 0.10  # slight forward offset for depth

# Rest-pose padding
REST_PAD_FRAMES = 18  # frames of neutral rest pose before/after
EASE_FRAMES = 12      # frames for smooth transition (ease-in / ease-out)

# Confirmed settings from CLAUDE.md
L_PALM_FLIP = True
R_PALM_FLIP = False
FLIP_FINGER_DIR_L = False
FLIP_FINGER_DIR_R = False

SMOOTH_WINDOW = 3  # micro-smoothing window

# ── Bone Names ────────────────────────────────────────────────────────────
BONE_L_ARM     = 'mixamorig:LeftArm'
BONE_R_ARM     = 'mixamorig:RightArm'
BONE_L_FOREARM = 'mixamorig:LeftForeArm'
BONE_R_FOREARM = 'mixamorig:RightForeArm'
BONE_L_HAND    = 'mixamorig:LeftHand'
BONE_R_HAND    = 'mixamorig:RightHand'
BONE_HIPS      = 'mixamorig:Hips'
BONE_SPINE     = 'mixamorig:Spine'

# MediaPipe Pose indices
P_NOSE = 0
P_L_SHOULDER, P_R_SHOULDER = 11, 12
P_L_ELBOW, P_R_ELBOW = 13, 14
P_L_WRIST, P_R_WRIST = 15, 16
P_L_HIP, P_R_HIP = 23, 24

# MediaPipe Hand indices
H_WRIST = 0
H_THUMB_CMC, H_THUMB_MCP, H_THUMB_IP, H_THUMB_TIP = 1, 2, 3, 4
H_INDEX_MCP, H_INDEX_PIP, H_INDEX_DIP, H_INDEX_TIP = 5, 6, 7, 8
H_MID_MCP, H_MID_PIP, H_MID_DIP, H_MID_TIP = 9, 10, 11, 12
H_RING_MCP, H_RING_PIP, H_RING_DIP, H_RING_TIP = 13, 14, 15, 16
H_PINK_MCP, H_PINK_PIP, H_PINK_DIP, H_PINK_TIP = 17, 18, 19, 20

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
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def V(pts, idx):
    """MediaPipe world landmark → Blender Vector (Y-up, mirrored X)."""
    if pts is None or idx >= len(pts):
        return None
    p = pts[idx]
    if p is None:
        return None
    x, y, z = p[0], p[1], p[2]
    # MediaPipe world: X=right(camera), Y=down, Z=toward camera
    # Blender: X=right, Y=forward(into screen), Z=up
    bx = -x if MIRROR_X else x   # mirror for performer facing camera
    by = -z - FORWARD_OFFSET     # MP Z(toward camera) → Blender -Y(into screen)
    bz = -y                      # MP Y(down) → Blender Z(up)
    return mathutils.Vector((bx, by, bz))


def V_hand(hand_lm, idx, pose_wrist_blender):
    """Hand world landmark → Blender Vector.
    Hand world landmarks are wrist-centered → add pose wrist position."""
    if hand_lm is None or idx >= len(hand_lm):
        return None
    p = hand_lm[idx]
    if p is None:
        return None
    x, y, z = p[0], p[1], p[2]
    # Same coordinate transform as pose, but relative to wrist
    bx = -x if MIRROR_X else x
    by = -z
    bz = -y
    return pose_wrist_blender + mathutils.Vector((bx, by, bz))


def smooth_track(vectors, window=SMOOTH_WINDOW):
    """Moving average smoothing for Vector tracks, handling None."""
    if window < 2:
        return vectors
    result = []
    n = len(vectors)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        valid = [v for v in vectors[lo:hi] if v is not None]
        if not valid:
            result.append(vectors[i])
        else:
            avg = mathutils.Vector((0, 0, 0))
            for v in valid:
                avg += v
            avg /= len(valid)
            result.append(avg)
    return result


def smooth_scalar(values, window=SMOOTH_WINDOW):
    """Moving average for scalar lists."""
    if window < 2:
        return values
    result = []
    n = len(values)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


def ease_in_out(t):
    """Smooth ease-in-out function (cubic Hermite): t in [0,1] → [0,1]."""
    return t * t * (3 - 2 * t)


def lerp_vector(a, b, t):
    """Linear interpolation between two Vectors."""
    return a * (1 - t) + b * t


def palm_normal(hand_lm, wrist_b):
    """Compute palm normal from hand landmarks (in Blender world space)."""
    if hand_lm is None or len(hand_lm) < 18:
        return None
    p0 = V_hand(hand_lm, H_WRIST, wrist_b)
    p5 = V_hand(hand_lm, H_INDEX_MCP, wrist_b)
    p17 = V_hand(hand_lm, H_PINK_MCP, wrist_b)
    if p0 is None or p5 is None or p17 is None:
        return None
    v1 = p5 - p0
    v2 = p17 - p0
    n = v1.cross(v2)
    if n.length < 1e-6:
        return None
    return n.normalized()


# ═══════════════════════════════════════════════════════════════════════════
# Core retargeting functions (from blend_animate.py)
# ═══════════════════════════════════════════════════════════════════════════

def make_empty(name, col):
    emp = bpy.data.objects.get(name)
    if emp:
        bpy.data.objects.remove(emp, do_unlink=True)
    emp = bpy.data.objects.new(name, None)
    emp.empty_display_type = 'PLAIN_AXES'
    emp.empty_display_size = 0.02
    col.objects.link(emp)
    return emp


def add_ik(arm_obj, bone_name, target, chain_count, pole_target=None):
    b = arm_obj.pose.bones.get(bone_name)
    if not b:
        print(f"[warn] bone not found: {bone_name}")
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
    if pole_target:
        c.pole_target = pole_target
        c.pole_angle = 0
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
    ry = rest.col[1].normalized()
    rx = rest.col[0].normalized()
    rz = rest.col[2].normalized()

    q = ry.rotation_difference(dir_a)
    ny = dir_a
    nx = (q @ rx).normalized()
    nz = (q @ rz).normalized()

    if roll_pn is not None and roll_pn.length > 1e-5:
        pna = (a_inv @ roll_pn).normalized()
        pp = pna - pna.dot(ny) * ny
        zp = nz - nz.dot(ny) * ny
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


def set_hand_rotation(arm_obj, bone_name, wrist_w, mid9_w, palm_n_w, flip_palm):
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
    ry = rest.col[1].normalized()
    rx = rest.col[0].normalized()
    rz = rest.col[2].normalized()

    q = ry.rotation_difference(dir_a)
    ny = dir_a
    nx = (q @ rx).normalized()
    nz = (q @ rz).normalized()

    if palm_n_w is not None and palm_n_w.length > 1e-5:
        pna = (a_inv @ palm_n_w).normalized()
        pp = pna - pna.dot(ny) * ny
        zp = nz - nz.dot(ny) * ny
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
    # ── Find armature ──
    arm_obj = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if not arm_obj:
        print("[ERROR] No armature found!")
        return
    print(f"[info] Armature: {arm_obj.name}")

    # ── Load motion data ──
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        motion = json.load(f)
    src_frames = motion['frames']
    n_src = len(src_frames)
    print(f"[info] Loaded {n_src} source frames")

    # ── Character measurements ──
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    _mw = arm_obj.matrix_world

    lb = arm_obj.pose.bones.get(BONE_L_ARM)
    rb = arm_obj.pose.bones.get(BONE_R_ARM)
    lb_fa = arm_obj.pose.bones.get(BONE_L_FOREARM)
    rb_fa = arm_obj.pose.bones.get(BONE_R_FOREARM)

    def bone_len(pb):
        return (_mw @ pb.tail - _mw @ pb.head).length if pb else 0.0

    char_l_arm = bone_len(lb) + bone_len(lb_fa)
    char_r_arm = bone_len(rb) + bone_len(rb_fa)

    if lb and rb:
        l_sh_w = _mw @ lb.head
        r_sh_w = _mw @ rb.head
        char_origin = (l_sh_w + r_sh_w) / 2
        char_sw = (l_sh_w - r_sh_w).length
    else:
        char_origin = mathutils.Vector((0, 0, 0))
        char_sw = 0.3
        l_sh_w = char_origin + mathutils.Vector((-char_sw / 2, 0, 0))
        r_sh_w = char_origin + mathutils.Vector((char_sw / 2, 0, 0))

    # Rest-pose wrist positions (arms hanging at sides)
    idle_drop = mathutils.Vector((0, -FORWARD_OFFSET, -0.40))
    idle_l = l_sh_w + idle_drop
    idle_r = r_sh_w + idle_drop
    # Small hang offset for hand landmarks in rest pose
    hang = mathutils.Vector((0, 0, -0.05))

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[info] Char shoulder width: {char_sw:.4f}")
    print(f"[info] Char arm length L:{char_l_arm:.4f} R:{char_r_arm:.4f}")

    # ── Extract source landmark tracks ──
    print("[pass 0] Extracting landmark tracks...")

    # Compute source shoulder width for scaling
    src_sw_list = []
    for fr in src_frames:
        pw = fr.get('pose_world')
        if pw and len(pw) > max(P_L_SHOULDER, P_R_SHOULDER):
            ls = V(pw, P_L_SHOULDER)
            rs = V(pw, P_R_SHOULDER)
            if ls and rs:
                src_sw_list.append((ls - rs).length)
    src_sw_avg = sum(src_sw_list) / len(src_sw_list) if src_sw_list else 0.3
    scale = char_sw / src_sw_avg if src_sw_avg > 1e-4 else 1.0
    print(f"[info] Source shoulder width: {src_sw_avg:.4f}, scale: {scale:.4f}")

    # Raw tracks
    lw_raw = []  # left wrist IK targets
    rw_raw = []  # right wrist IK targets
    lh_pts_raw = []  # left hand 21 landmarks (Blender world)
    rh_pts_raw = []  # right hand 21 landmarks (Blender world)
    l_pn_raw = []  # left palm normal
    r_pn_raw = []  # right palm normal
    lh9_raw = []  # left mid9 (hand direction target)
    rh9_raw = []  # right mid9
    l_valid = []
    r_valid = []

    for fr in src_frames:
        pw = fr.get('pose_world')
        if pw is None or len(pw) < 25:
            lw_raw.append(None); rw_raw.append(None)
            lh_pts_raw.append(None); rh_pts_raw.append(None)
            l_pn_raw.append(None); r_pn_raw.append(None)
            lh9_raw.append(None); rh9_raw.append(None)
            l_valid.append(False); r_valid.append(False)
            continue

        # Source shoulder center for offset computation
        src_ls = V(pw, P_L_SHOULDER)
        src_rs = V(pw, P_R_SHOULDER)
        src_center = (src_ls + src_rs) / 2 if (src_ls and src_rs) else mathutils.Vector((0, 0, 0))

        def to_char(v):
            """Transform source world → character world (centered + scaled)."""
            if v is None:
                return None
            return char_origin + (v - src_center) * scale

        # Wrist positions
        lw_b = to_char(V(pw, P_L_WRIST))
        rw_b = to_char(V(pw, P_R_WRIST))
        lw_raw.append(lw_b)
        rw_raw.append(rw_b)

        # Hands
        hands = fr.get('hands') or []
        left_hand_lm = None
        right_hand_lm = None
        for h in hands:
            label = h.get('handedness', '')
            world = h.get('world')
            if world and len(world) >= 21:
                if label == 'Left' and left_hand_lm is None:
                    left_hand_lm = world
                elif label == 'Right' and right_hand_lm is None:
                    right_hand_lm = world

        # Process left hand
        if left_hand_lm and lw_b:
            pts_b = [V_hand(left_hand_lm, i, lw_b) for i in range(21)]
            # Scale hand landmarks relative to wrist
            pts_scaled = []
            for p in pts_b:
                if p is not None:
                    pts_scaled.append(lw_b + (p - lw_b) * scale)
                else:
                    pts_scaled.append(None)
            lh_pts_raw.append(pts_scaled)
            pn = palm_normal(left_hand_lm, lw_b)
            l_pn_raw.append(pn)
            mid9 = pts_scaled[H_MID_MCP] if pts_scaled[H_MID_MCP] else lw_b
            lh9_raw.append(mid9)
            l_valid.append(True)
        else:
            lh_pts_raw.append(None)
            l_pn_raw.append(None)
            lh9_raw.append(None)
            l_valid.append(False)

        # Process right hand
        if right_hand_lm and rw_b:
            pts_b = [V_hand(right_hand_lm, i, rw_b) for i in range(21)]
            pts_scaled = []
            for p in pts_b:
                if p is not None:
                    pts_scaled.append(rw_b + (p - rw_b) * scale)
                else:
                    pts_scaled.append(None)
            rh_pts_raw.append(pts_scaled)
            pn = palm_normal(right_hand_lm, rw_b)
            r_pn_raw.append(pn)
            mid9 = pts_scaled[H_MID_MCP] if pts_scaled[H_MID_MCP] else rw_b
            rh9_raw.append(mid9)
            r_valid.append(True)
        else:
            rh_pts_raw.append(None)
            r_pn_raw.append(None)
            rh9_raw.append(None)
            r_valid.append(False)

    print(f"[info] Valid frames: L={sum(l_valid)}/{n_src}, R={sum(r_valid)}/{n_src}")

    # ── Smooth tracks ──
    print("[pass 0] Smoothing...")
    lw_sm = smooth_track(lw_raw, SMOOTH_WINDOW)
    rw_sm = smooth_track(rw_raw, SMOOTH_WINDOW)
    lh9_sm = smooth_track(lh9_raw, SMOOTH_WINDOW)
    rh9_sm = smooth_track(rh9_raw, SMOOTH_WINDOW)

    # Smooth hand landmarks per point
    lh_pts_sm = []
    rh_pts_sm = []
    for i in range(n_src):
        if lh_pts_raw[i]:
            lh_pts_sm.append(lh_pts_raw[i])
        else:
            lh_pts_sm.append(None)
        if rh_pts_raw[i]:
            rh_pts_sm.append(rh_pts_raw[i])
        else:
            rh_pts_sm.append(None)

    # Per-landmark smoothing for hand points
    for pt_idx in range(21):
        l_track = [lh_pts_sm[i][pt_idx] if lh_pts_sm[i] else None for i in range(n_src)]
        r_track = [rh_pts_sm[i][pt_idx] if rh_pts_sm[i] else None for i in range(n_src)]
        l_smoothed = smooth_track(l_track, SMOOTH_WINDOW)
        r_smoothed = smooth_track(r_track, SMOOTH_WINDOW)
        for i in range(n_src):
            if lh_pts_sm[i] and l_smoothed[i]:
                lh_pts_sm[i][pt_idx] = l_smoothed[i]
            if rh_pts_sm[i] and r_smoothed[i]:
                rh_pts_sm[i][pt_idx] = r_smoothed[i]

    # ── Build full timeline with rest-pose padding ──
    # Structure: [REST_PAD] [EASE_IN] [SIGN_MOTION] [EASE_OUT] [REST_PAD]
    total_frames = REST_PAD_FRAMES + EASE_FRAMES + n_src + EASE_FRAMES + REST_PAD_FRAMES
    print(f"[info] Timeline: {REST_PAD_FRAMES} rest + {EASE_FRAMES} ease-in + "
          f"{n_src} motion + {EASE_FRAMES} ease-out + {REST_PAD_FRAMES} rest = {total_frames}")

    # Build full wrist IK tracks
    first_l = lw_sm[0] if lw_sm[0] else idle_l
    first_r = rw_sm[0] if rw_sm[0] else idle_r
    last_l = lw_sm[-1] if lw_sm[-1] else idle_l
    last_r = rw_sm[-1] if rw_sm[-1] else idle_r

    full_lw = []
    full_rw = []
    full_lh_pts = []
    full_rh_pts = []
    full_l_pn = []
    full_r_pn = []
    full_lh9 = []
    full_rh9 = []
    full_l_valid = []
    full_r_valid = []

    # Phase 1: Rest pad (arms at sides)
    for _ in range(REST_PAD_FRAMES):
        full_lw.append(idle_l.copy())
        full_rw.append(idle_r.copy())
        full_lh_pts.append(None)
        full_rh_pts.append(None)
        full_l_pn.append(None)
        full_r_pn.append(None)
        full_lh9.append(idle_l + hang)
        full_rh9.append(idle_r + hang)
        full_l_valid.append(False)
        full_r_valid.append(False)

    # Phase 2: Ease-in (rest → first sign frame)
    for k in range(EASE_FRAMES):
        t = ease_in_out((k + 1) / (EASE_FRAMES + 1))
        full_lw.append(lerp_vector(idle_l, first_l, t))
        full_rw.append(lerp_vector(idle_r, first_r, t))
        full_lh_pts.append(None)
        full_rh_pts.append(None)
        full_l_pn.append(None)
        full_r_pn.append(None)
        full_lh9.append(lerp_vector(idle_l + hang, lh9_sm[0] if lh9_sm[0] else first_l, t))
        full_rh9.append(lerp_vector(idle_r + hang, rh9_sm[0] if rh9_sm[0] else first_r, t))
        full_l_valid.append(False)
        full_r_valid.append(False)

    # Phase 3: Sign motion
    for i in range(n_src):
        full_lw.append(lw_sm[i] if lw_sm[i] else idle_l)
        full_rw.append(rw_sm[i] if rw_sm[i] else idle_r)
        full_lh_pts.append(lh_pts_sm[i])
        full_rh_pts.append(rh_pts_sm[i])
        full_l_pn.append(l_pn_raw[i])
        full_r_pn.append(r_pn_raw[i])
        full_lh9.append(lh9_sm[i] if lh9_sm[i] else (lw_sm[i] if lw_sm[i] else idle_l))
        full_rh9.append(rh9_sm[i] if rh9_sm[i] else (rw_sm[i] if rw_sm[i] else idle_r))
        full_l_valid.append(l_valid[i])
        full_r_valid.append(r_valid[i])

    # Phase 4: Ease-out (last sign frame → rest)
    for k in range(EASE_FRAMES):
        t = ease_in_out((k + 1) / (EASE_FRAMES + 1))
        full_lw.append(lerp_vector(last_l, idle_l, t))
        full_rw.append(lerp_vector(last_r, idle_r, t))
        full_lh_pts.append(None)
        full_rh_pts.append(None)
        full_l_pn.append(None)
        full_r_pn.append(None)
        full_lh9.append(lerp_vector(lh9_sm[-1] if lh9_sm[-1] else last_l, idle_l + hang, t))
        full_rh9.append(lerp_vector(rh9_sm[-1] if rh9_sm[-1] else last_r, idle_r + hang, t))
        full_l_valid.append(False)
        full_r_valid.append(False)

    # Phase 5: Rest pad (arms at sides)
    for _ in range(REST_PAD_FRAMES):
        full_lw.append(idle_l.copy())
        full_rw.append(idle_r.copy())
        full_lh_pts.append(None)
        full_rh_pts.append(None)
        full_l_pn.append(None)
        full_r_pn.append(None)
        full_lh9.append(idle_l + hang)
        full_rh9.append(idle_r + hang)
        full_l_valid.append(False)
        full_r_valid.append(False)

    # Clamp IK targets to arm reach
    for i in range(total_frames):
        for wrist_track, sh_w, arm_len in [
            (full_lw, l_sh_w, char_l_arm),
            (full_rw, r_sh_w, char_r_arm),
        ]:
            v = wrist_track[i]
            if v is None:
                continue
            reach = (v - sh_w).length
            max_reach = arm_len * 0.95
            if reach > max_reach:
                wrist_track[i] = sh_w + (v - sh_w).normalized() * max_reach

    valid_frames = list(range(1, total_frames + 1))  # 1-indexed

    # ── Setup IK empties ──
    print("[pass 1] Setting up IK...")
    col = bpy.context.scene.collection
    emp_lw = make_empty('IK_LeftWrist', col)
    emp_rw = make_empty('IK_RightWrist', col)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    # Pre-bend forearms to avoid IK singularity
    for fbn, sign in [(BONE_L_FOREARM, 1), (BONE_R_FOREARM, -1)]:
        fb = arm_obj.pose.bones.get(fbn)
        if fb:
            fb.rotation_mode = 'QUATERNION'
            fb.rotation_quaternion = mathutils.Quaternion(
                mathutils.Vector((1, 0, 0)), math.radians(5 * sign))
            fb.keyframe_insert('rotation_quaternion', frame=1)

    bpy.ops.object.mode_set(mode='OBJECT')

    # Add IK constraints
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    add_ik(arm_obj, BONE_L_FOREARM, emp_lw, chain_count=2)
    add_ik(arm_obj, BONE_R_FOREARM, emp_rw, chain_count=2)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Keyframe IK empties
    animate_empty(emp_lw, full_lw, valid_frames)
    animate_empty(emp_rw, full_rw, valid_frames)

    # ── Pass 2: Hand + Finger FK ──
    print(f"[pass 2] Hand/Finger FK ({total_frames} frames)...")
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    prev_quats_l = {}
    prev_quats_r = {}

    def sign_norm(bone_name, prev_quats):
        b = arm_obj.pose.bones.get(bone_name)
        pq = prev_quats.get(bone_name)
        if b and pq and pq.dot(b.rotation_quaternion) < 0:
            b.rotation_quaternion.negate()
        if b:
            prev_quats[bone_name] = b.rotation_quaternion.copy()

    for i, frame in enumerate(valid_frames):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        for (bone_name, wrist_pos, mid9, pn, flip, flip_dir, pts, chains,
             prev_quats, is_valid) in [
            (BONE_L_HAND, full_lw[i], full_lh9[i], full_l_pn[i],
             L_PALM_FLIP, FLIP_FINGER_DIR_L, full_lh_pts[i],
             FINGER_CHAINS_L, prev_quats_l, full_l_valid[i]),
            (BONE_R_HAND, full_rw[i], full_rh9[i], full_r_pn[i],
             R_PALM_FLIP, FLIP_FINGER_DIR_R, full_rh_pts[i],
             FINGER_CHAINS_R, prev_quats_r, full_r_valid[i]),
        ]:
            if not is_valid or pts is None:
                # During rest/ease phases, just keyframe current pose
                b = arm_obj.pose.bones.get(bone_name)
                if b:
                    b.keyframe_insert('rotation_quaternion', frame=frame)
                continue

            # effective_pn
            eff_pn = None
            if pn and pn.length > 1e-4:
                eff_pn = (-pn if flip else pn)

            # Hand rotation
            if eff_pn is not None and wrist_pos and mid9:
                set_hand_rotation(arm_obj, bone_name, pts[H_WRIST], mid9, eff_pn, False)

            hand_mat = bone_arma_mat(arm_obj, bone_name)
            sign_norm(bone_name, prev_quats)
            arm_obj.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=frame)

            # Finger chains
            for chain in chains:
                first_parent_mat = get_parent_mat(arm_obj, chain[0][0], bone_name, hand_mat)
                p_mat = first_parent_mat
                for (bn, from_idx, to_idx) in chain:
                    from_w = pts[from_idx]
                    to_w = pts[to_idx]
                    if from_w is None or to_w is None:
                        p_mat = bone_arma_mat(arm_obj, bn, p_mat)
                        continue
                    if flip_dir:
                        from_w, to_w = to_w, from_w
                    ok, new_mat = set_bone_dir(arm_obj, bn, from_w, to_w,
                                               parent_mat=p_mat, roll_pn=eff_pn)
                    if ok:
                        sign_norm(bn, prev_quats)
                        arm_obj.pose.bones[bn].keyframe_insert('rotation_quaternion', frame=frame)
                        p_mat = new_mat
                    else:
                        p_mat = bone_arma_mat(arm_obj, bn, p_mat)

    bpy.ops.object.mode_set(mode='OBJECT')

    # ── Set timeline ──
    bpy.context.scene.render.fps = FPS
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total_frames
    bpy.context.scene.frame_set(1)

    # ── Save .blend ──
    blend_path = os.path.join(OUT_DIR, "SIGN001_greeting.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[info] Saved: {blend_path}")

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
    fbx_size = os.path.getsize(FBX_PATH) // 1024
    print(f"[info] FBX exported: {FBX_PATH} ({fbx_size} KB)")

    # ── Render 3-angle review MP4 ──
    print("[render] 3-angle review video...")
    os.makedirs(RENDER_DIR, exist_ok=True)

    CAMERAS = {
        "front": {
            "location": (0.0, -3.5, 1.2),
            "rotation_euler": (math.radians(80), 0.0, 0.0),
        },
        "side": {
            "location": (-3.5, 0.0, 1.2),
            "rotation_euler": (math.radians(80), 0.0, math.radians(-90)),
        },
        "angle45": {
            "location": (2.5, -2.5, 1.2),
            "rotation_euler": (math.radians(80), 0.0, math.radians(45)),
        },
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
    for view_name, cam_cfg in CAMERAS.items():
        print(f"  rendering {view_name}...")
        cam_name = f"RenderCam_{view_name}"
        if cam_name in bpy.data.objects:
            cam_obj = bpy.data.objects[cam_name]
        else:
            cam_data = bpy.data.cameras.new(cam_name)
            cam_data.lens = 35
            cam_obj = bpy.data.objects.new(cam_name, cam_data)
            bpy.context.scene.collection.objects.link(cam_obj)
        cam_obj.location = cam_cfg["location"]
        cam_obj.rotation_euler = cam_cfg["rotation_euler"]
        scene.camera = cam_obj

        frame_dir = os.path.join(RENDER_DIR, view_name)
        os.makedirs(frame_dir, exist_ok=True)
        scene.render.filepath = os.path.join(frame_dir, "")
        bpy.ops.render.render(animation=True)
        frame_dirs[view_name] = frame_dir

    # ── Encode composite MP4 ──
    # Use system Python to stitch the 3 views side-by-side
    stitch_script = os.path.join(_THIS_DIR, "s04_stitch_review.py")
    stitch_code = f'''
import cv2, os, glob, numpy as np
views = {list(CAMERAS.keys())}
render_dir = r"{RENDER_DIR}"
out_path = r"{REVIEW_MP4}"
fps = {FPS}
total = {total_frames}
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = None
for fi in range(1, total + 1):
    row = []
    for view in views:
        pattern = os.path.join(render_dir, view, f"{{fi:04d}}.png")
        img = cv2.imread(pattern)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add view label
        cv2.putText(img, view, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(img, f"f={{fi}}/{{total}}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        row.append(img)
    canvas = np.hstack(row)
    if writer is None:
        h, w = canvas.shape[:2]
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    writer.write(canvas)
writer.release()
print(f"Encoded: {{out_path}} ({{os.path.getsize(out_path)//1024}} KB)")
'''
    with open(stitch_script, 'w', encoding='utf-8') as f:
        f.write(stitch_code)

    if os.path.exists(SYSTEM_PYTHON):
        print("[encode] Stitching 3-view review MP4...")
        result = subprocess.run(
            [SYSTEM_PYTHON, stitch_script],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode != 0 and result.stderr:
            print(f"[warn] encode error: {result.stderr[-300:]}")
    else:
        print(f"[warn] System Python not found: {SYSTEM_PYTHON}")

    print(f"\n=== Phase 3A Complete ===")
    print(f"  Timeline: {total_frames} frames ({total_frames/FPS:.1f}s)")
    print(f"  .blend: {blend_path}")
    print(f"  .fbx: {FBX_PATH}")
    if os.path.exists(REVIEW_MP4):
        print(f"  Review MP4: {REVIEW_MP4} ({os.path.getsize(REVIEW_MP4)//1024} KB)")


main()
