"""
Unified Blender retargeting + FBX export.
Loads filtered 3D JSON, applies arm + hand/finger retargeting, exports FBX.

Usage:
  blender <rig.blend> --background --python blender_retarget_and_export.py -- --sign-id SIGN001_THANKS
  blender <rig.blend> --background --python blender_retarget_and_export.py -- --sign-id SIGN003_LOVE
"""

import bpy
import json
import os
import sys
import math
import time
from mathutils import Vector, Quaternion, Matrix

# Parse args after "--"
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sign-id", required=True)
args = parser.parse_args(argv)

SIGN_ID = args.sign_id
PROJECT = r"C:\MyUnityProjects\WordsInSilence"
PIPELINE_DIR = os.path.join(PROJECT, "_multiview_rebuild_v3", f"pipeline_{SIGN_ID}")
FILTERED_JSON = os.path.join(PIPELINE_DIR, f"{SIGN_ID}_canonical_3d_filtered.json")
OUTPUT_BLEND = os.path.join(PIPELINE_DIR, f"{SIGN_ID}_retargeted.blend")
OUTPUT_FBX = os.path.join(PIPELINE_DIR, f"{SIGN_ID}.fbx")

PREFIX = "mixamorig:"
FPS = 30

print("=" * 60)
print(f"Blender Retarget + Export: {SIGN_ID}")
print("=" * 60)

# ===== Load data =====
print("\n1. Loading filtered 3D data...")
with open(FILTERED_JSON, "r") as f:
    data = json.load(f)
frames = data["frames"]
n_frames = len(frames)
print(f"  {n_frames} frames")

# ===== Setup scene =====
armature = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        armature = obj
        break
if not armature:
    raise SystemExit("No armature found!")

print(f"  Armature: {armature.name}")
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = n_frames
bpy.context.scene.render.fps = FPS
bpy.context.view_layer.objects.active = armature

def get_bone(name):
    return armature.pose.bones.get(PREFIX + name)

# ===== Measure rig =====
print("\n2. Measuring rig...")
l_arm_bone = get_bone("LeftArm")
r_arm_bone = get_bone("RightArm")

def bone_world_head(bone):
    return armature.matrix_world @ bone.bone.head_local

l_shoulder_pos = bone_world_head(l_arm_bone)
r_shoulder_pos = bone_world_head(r_arm_bone)
rig_shoulder_mid = (l_shoulder_pos + r_shoulder_pos) / 2
rig_shoulder_width = (l_shoulder_pos - r_shoulder_pos).length
rig_scale = rig_shoulder_width
print(f"  Shoulder width: {rig_shoulder_width:.4f}")

def src_to_rig(sx, sy, sz):
    return Vector((
        sx * rig_scale + rig_shoulder_mid.x,
        -sy * rig_scale + rig_shoulder_mid.y,  # character faces -Y
        sz * rig_scale + rig_shoulder_mid.z,
    ))

arm_rot = armature.matrix_world.to_quaternion()

# ===== FK Rotation computation =====
def compute_fk_rotation(bone, desired_dir_world):
    posed_head = armature.matrix_world @ bone.head
    posed_tail = armature.matrix_world @ bone.tail
    current_dir = (posed_tail - posed_head).normalized()
    target_dir = desired_dir_world.normalized()
    rot_world = current_dir.rotation_difference(target_dir)

    if bone.parent:
        parent_posed_rot = bone.parent.matrix.to_quaternion()
    else:
        parent_posed_rot = Quaternion()

    rest_mat = bone.bone.matrix_local.to_quaternion()
    parent_rest = bone.parent.bone.matrix_local.to_quaternion() if bone.parent else Quaternion()
    local_rest = parent_rest.inverted() @ rest_mat
    current_basis = arm_rot @ parent_posed_rot @ local_rest
    return current_basis.inverted() @ rot_world @ current_basis

# ===== Source joint extraction =====
def get_src_joint(fr, category, name):
    j = fr.get(category, {}).get(name)
    if j: return src_to_rig(j["x"], j["y"], j["z"])
    return None

# ===== S6: Arm retargeting =====
print("\n3. Arm retargeting...")
bpy.ops.object.mode_set(mode='POSE')

ARM_BONES = [
    ("LeftArm", "LeftForeArm", "left_shoulder", "left_elbow", "left_wrist"),
    ("RightArm", "RightForeArm", "right_shoulder", "right_elbow", "right_wrist"),
]

arm_count = 0
for fi_idx, fr in enumerate(frames):
    frame_num = fi_idx + 1
    bpy.context.scene.frame_set(frame_num)

    for upper_name, lower_name, src_shoulder, src_elbow, src_wrist in ARM_BONES:
        shoulder = get_src_joint(fr, "pose", src_shoulder)
        elbow = get_src_joint(fr, "pose", src_elbow)
        wrist = get_src_joint(fr, "pose", src_wrist)
        if not shoulder or not elbow or not wrist:
            continue

        upper_bone = get_bone(upper_name)
        lower_bone = get_bone(lower_name)
        if not upper_bone or not lower_bone:
            continue

        # Upper arm
        desired_dir_upper = (elbow - shoulder).normalized()
        upper_bone.rotation_quaternion = Quaternion()
        bpy.context.view_layer.update()
        rot_upper = compute_fk_rotation(upper_bone, desired_dir_upper)
        upper_bone.rotation_quaternion = rot_upper
        upper_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
        bpy.context.view_layer.update()

        # Forearm
        desired_dir_lower = (wrist - elbow).normalized()
        lower_bone.rotation_quaternion = Quaternion()
        bpy.context.view_layer.update()
        rot_lower = compute_fk_rotation(lower_bone, desired_dir_lower)
        lower_bone.rotation_quaternion = rot_lower
        lower_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
        bpy.context.view_layer.update()
        arm_count += 1

print(f"  {arm_count} arm poses applied")

# ===== S7: Hand + Finger retargeting =====
print("\n4. Hand retargeting...")

FINGER_MAP = {
    "Left": [
        (1,2,"LeftHandThumb1"),(2,3,"LeftHandThumb2"),(3,4,"LeftHandThumb3"),
        (5,6,"LeftHandIndex1"),(6,7,"LeftHandIndex2"),(7,8,"LeftHandIndex3"),
        (9,10,"LeftHandMiddle1"),(10,11,"LeftHandMiddle2"),(11,12,"LeftHandMiddle3"),
        (13,14,"LeftHandRing1"),(14,15,"LeftHandRing2"),(15,16,"LeftHandRing3"),
        (17,18,"LeftHandPinky1"),(18,19,"LeftHandPinky2"),(19,20,"LeftHandPinky3"),
    ],
    "Right": [
        (1,2,"RightHandThumb1"),(2,3,"RightHandThumb2"),(3,4,"RightHandThumb3"),
        (5,6,"RightHandIndex1"),(6,7,"RightHandIndex2"),(7,8,"RightHandIndex3"),
        (9,10,"RightHandMiddle1"),(10,11,"RightHandMiddle2"),(11,12,"RightHandMiddle3"),
        (13,14,"RightHandRing1"),(14,15,"RightHandRing2"),(15,16,"RightHandRing3"),
        (17,18,"RightHandPinky1"),(18,19,"RightHandPinky2"),(19,20,"RightHandPinky3"),
    ],
}

hand_count = 0
finger_count = 0

for fi_idx, fr in enumerate(frames):
    frame_num = fi_idx + 1
    bpy.context.scene.frame_set(frame_num)

    for side, side_lower, hand_key in [("Left","left","left_hand"), ("Right","right","right_hand")]:
        hand_data = fr.get(hand_key, {})
        lm0 = hand_data.get("lm0")
        lm9 = hand_data.get("lm9")
        if not lm0 or not lm9:
            continue

        # Hand bone direction: wrist (lm0) → middle MCP (lm9)
        wrist_pos = src_to_rig(lm0["x"], lm0["y"], lm0["z"])
        mid_pos = src_to_rig(lm9["x"], lm9["y"], lm9["z"])
        hand_dir = (mid_pos - wrist_pos).normalized()

        hand_bone = get_bone(f"{side}Hand")
        if hand_bone:
            hand_bone.rotation_quaternion = Quaternion()
            bpy.context.view_layer.update()
            rot = compute_fk_rotation(hand_bone, hand_dir)
            hand_bone.rotation_quaternion = rot
            hand_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
            bpy.context.view_layer.update()
            hand_count += 1

        # Fingers
        for parent_lm, child_lm, bone_name in FINGER_MAP[side]:
            plm = hand_data.get(f"lm{parent_lm}")
            clm = hand_data.get(f"lm{child_lm}")
            if not plm or not clm:
                continue
            p_pos = src_to_rig(plm["x"], plm["y"], plm["z"])
            c_pos = src_to_rig(clm["x"], clm["y"], clm["z"])
            finger_dir = (c_pos - p_pos).normalized()

            bone = get_bone(bone_name)
            if bone:
                bone.rotation_quaternion = Quaternion()
                bpy.context.view_layer.update()
                rot = compute_fk_rotation(bone, finger_dir)
                bone.rotation_quaternion = rot
                bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
                bpy.context.view_layer.update()
                finger_count += 1

    if (fi_idx + 1) % 20 == 0:
        print(f"  Frame {fi_idx+1}/{n_frames}")

print(f"  {hand_count} hand orientations, {finger_count} finger rotations")

# ===== S7.5: Finger smoothing =====
print("\n5. Finger quaternion smoothing...")

def slerp_smooth(quats, window=5):
    smoothed = list(quats)
    half = window // 2
    for i in range(len(quats)):
        start = max(0, i - half)
        end = min(len(quats), i + half + 1)
        avg = quats[start]
        count = 1
        for j in range(start + 1, end):
            q = quats[j]
            if avg.dot(q) < 0:
                q = q.copy(); q.negate()
            t = 1.0 / (count + 1)
            avg = avg.slerp(q, t)
            count += 1
        smoothed[i] = avg
    return smoothed

SMOOTH_BONES = []
for side in ["Left", "Right"]:
    SMOOTH_BONES.append(f"{side}Hand")
    for finger in ["Thumb","Index","Middle","Ring","Pinky"]:
        for i in range(1, 4):
            SMOOTH_BONES.append(f"{side}Hand{finger}{i}")

smooth_count = 0
for bone_name in SMOOTH_BONES:
    bone = get_bone(bone_name)
    if not bone: continue
    quats = []
    for fi in range(1, n_frames + 1):
        bpy.context.scene.frame_set(fi)
        quats.append(bone.rotation_quaternion.copy())
    smoothed = slerp_smooth(quats, window=5)
    for fi in range(1, n_frames + 1):
        bpy.context.scene.frame_set(fi)
        bone.rotation_quaternion = smoothed[fi - 1]
        bone.keyframe_insert(data_path="rotation_quaternion", frame=fi)
    smooth_count += 1
print(f"  Smoothed {smooth_count} bones")

# ===== Save .blend =====
print("\n6. Saving .blend...")
bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
print(f"  Saved: {OUTPUT_BLEND}")

# ===== Export FBX =====
print("\n7. Exporting FBX...")

# Unpack textures
tex_dir = os.path.join(PIPELINE_DIR, "textures")
os.makedirs(tex_dir, exist_ok=True)
for img in bpy.data.images:
    if img.packed_file is None: continue
    safe_name = img.name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not safe_name.endswith(".png"): safe_name += ".png"
    out_path = os.path.join(tex_dir, safe_name)
    img.filepath_raw = out_path
    img.save()

bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
for child in armature.children:
    if child.type == 'MESH': child.select_set(True)

bpy.ops.export_scene.fbx(
    filepath=OUTPUT_FBX,
    use_selection=True,
    apply_scale_options='FBX_SCALE_ALL',
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
    embed_textures=True,
    axis_forward='-Z',
    axis_up='Y',
)

size_mb = os.path.getsize(OUTPUT_FBX) / 1024 / 1024
print(f"  FBX saved: {OUTPUT_FBX} ({size_mb:.1f}MB)")

print("\n" + "=" * 60)
print(f"Retarget + Export complete: {SIGN_ID}")
print(f"  .blend: {OUTPUT_BLEND}")
print(f"  .fbx: {OUTPUT_FBX}")
print("=" * 60)
