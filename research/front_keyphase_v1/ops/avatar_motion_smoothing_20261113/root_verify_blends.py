"""Read-back verification of actual Blend assets; writes only this audit."""
from pathlib import Path
import bpy, hashlib, json
import numpy as np

U = Path(__file__).resolve().parent
P = U.parent / 'avatar_motion_pilot_20261113'
checks, results = [], {}
def check(name, condition, measured=None):
    checks.append(dict(name=name, passed=bool(condition), measured=measured))
def fingerprint(rig):
    h = hashlib.sha256()
    for bone in sorted(rig.data.bones, key=lambda b:b.name):
        h.update(bone.name.encode())
        h.update((bone.parent.name if bone.parent else '').encode())
        h.update(np.asarray(bone.matrix_local, dtype='<f8').tobytes())
        h.update(np.asarray([bone.head_local, bone.tail_local], dtype='<f8').tobytes())
    bones = h.hexdigest()
    h = hashlib.sha256()
    for mesh in sorted((o for o in bpy.data.objects if o.type == 'MESH'), key=lambda m:m.name):
        h.update(mesh.name.encode())
        for v in mesh.data.vertices:
            h.update(np.asarray(v.co, dtype='<f8').tobytes())
            h.update(json.dumps([(mesh.vertex_groups[g.group].name, float(g.weight)) for g in v.groups]).encode())
        for polygon in mesh.data.polygons:
            h.update(np.asarray(polygon.vertices, dtype='<i4').tobytes())
    return dict(bones=bones, mesh_weight_topology=h.hexdigest(), bone_count=len(rig.data.bones))

for label, path in [('static', P/'static_character.blend'),
                    ('baseline', U/'BASELINE_ORIGINAL_MATERIAL.blend'),
                    ('candidate', U/'CANDIDATE_ORIGINAL_MATERIAL.blend')]:
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False)
    rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    results[label] = fingerprint(rig)
    textures = {}
    for mesh in (o for o in bpy.data.objects if o.type == 'MESH'):
        for slot in mesh.material_slots:
            mat = slot.material
            if not mat or not mat.node_tree: continue
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image and 'Diffuse' in node.image.filepath:
                    check(label+'_diffuse_packed', bool(node.image.packed_file))
                    if node.image.packed_file:
                        textures[mesh.name] = hashlib.sha256(bytes(node.image.packed_file.data)).hexdigest()
    results[label]['diffuse_sha'] = textures
    if label != 'static':
        a = np.load(U/(label+'_poses.npz'), allow_pickle=False)
        indices = [0, 129, 194, 257, 353, 386, 504]
        matrix_error = 0.0
        for i in indices:
            f = float(a['source_frames'][i] + 1)
            bpy.context.scene.frame_set(int(f), subframe=f % 1)
            bpy.context.view_layer.update()
            actual = np.array([rig.pose.bones[name].matrix for name in a['names']])
            matrix_error = max(matrix_error, float(np.max(np.abs(actual - a['matrix'][i]))))
        check(label+'_saved_arrays_match_readback', matrix_error < 1e-6, matrix_error)
        modes, times_ok, curves = set(), True, 0
        for layer in rig.animation_data.action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        curves += 1
                        modes.update(p.interpolation for p in curve.keyframe_points)
                        times_ok &= np.array_equal([p.co.x for p in curve.keyframe_points], np.arange(1,128))
        check(label+'_144_curves_127_original_times', curves == 144 and times_ok)
        results[label]['interpolation'] = sorted(modes)
        if label == 'candidate': check('candidate_all_linear', modes == {'LINEAR'})
    else:
        check('static_no_animation', not bool(rig.animation_data and rig.animation_data.action))

for label in ['baseline','candidate']:
    for field in ['bones', 'mesh_weight_topology', 'bone_count', 'diffuse_sha']:
        check(label+'_original_'+field, results[label][field] == results['static'][field])
report = dict(status='PASS_BLEND_REREAD' if all(c['passed'] for c in checks) else 'FAIL',
              checks=checks, snapshots=results, verification_actor='root independent readback of actual files',
              human_sign_language_correctness_validated=False)
(U/'ROOT_BLEND_VERIFIED.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n','utf-8')
print('ROOT_BLEND_VERIFIED',report['status'],len(checks),[c for c in checks if not c['passed']])
assert all(c['passed'] for c in checks)
