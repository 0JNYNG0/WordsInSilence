"""SYNTHETIC ONLY: verify skinning, finger axes, fixed lengths, bake/reopen. No public sign animation."""
import bpy,json,sys,math,hashlib
import numpy as np
from pathlib import Path
from mathutils import Vector,Quaternion
U=Path(__file__).resolve().parent;sys.path.insert(0,str(U))
from hand_retarget_blender import apply_hand,rest_hand,bone_name,FINGERS
from geometry_core import hand_basis

def evaluated_vertices(mesh):
    dg=bpy.context.evaluated_depsgraph_get();ev=mesh.evaluated_get(dg);m=ev.to_mesh()
    p=np.array([list(ev.matrix_world@v.co)for v in m.vertices]);ev.to_mesh_clear();return p

bpy.ops.wm.open_mainfile(filepath=str(U/'static_character.blend'),load_ui=False,use_scripts=False)
assert len(bpy.data.actions)==0
rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');mesh=next(o for o in bpy.data.objects if o.type=='MESH')
base_vertices=evaluated_vertices(mesh)
rest={side:rest_hand(rig,side)for side in ('Left','Right')}
armnames=[bone_name(rig,s+a)for s in ('Left','Right')for a in ('Shoulder','Arm','ForeArm','Hand')]
armbaseline={n:np.array(rig.pose.bones[n].matrix)for n in armnames}
results=[];saved={}
scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=25;scene.render.fps=24
for frame in range(1,26):
    scene.frame_set(frame)
    for pb in rig.pose.bones:pb.matrix_basis.identity();pb.rotation_mode='QUATERNION'
    bpy.context.view_layer.update()
    angle=0.55*math.sin(math.pi*(frame-1)/24)
    for side in ('Left','Right'):
        p=rest[side].copy();basis=hand_basis(p)
        for finger,start in FINGERS:
            for k in range(3):
                v=rest[side][start+k+1]-rest[side][start+k]
                # The index-to-pinky palm frame has opposite anatomical normal signs
                # for left/right hands. Flexion goes toward the palmar side, not
                # a shared numeric +normal for both. These artificial angles are
                # a transport test, not calibrated physiological limits.
                flexion_sign=-1 if side=='Left' else 1
                q=Quaternion(Vector(basis[:,0]),flexion_sign*angle*(k+1)*(0.4 if finger=='Thumb' else 1))
                p[start+k+1]=p[start+k]+np.array(q@Vector(v))
        results.extend(apply_hand(rig,side,p))
        for finger,_ in FINGERS:
            for k in range(1,4):rig.pose.bones[bone_name(rig,side+'Hand'+finger+str(k))].keyframe_insert(data_path='rotation_quaternion',frame=frame)
    for n in armnames:assert np.max(np.abs(np.array(rig.pose.bones[n].matrix)-armbaseline[n]))<2e-4,(frame,n)
    if frame in (1,13,25):
        saved[str(frame)]={pb.name:[list(r)for r in pb.matrix]for pb in rig.pose.bones}
        if frame==13:bent_vertices=evaluated_vertices(mesh)
max_angle=max(x['angle_error_deg']for x in results);max_length=max(x['length_error']for x in results)
assert max_angle<.1,max_angle
assert max_length<1e-4,max_length
skin_delta=np.linalg.norm(bent_vertices-base_vertices,axis=1)
assert int((skin_delta>.001).sum())>20,'Skin binding not moving as expected'

# Neutral material and close hand camera are presentation-only. No copied private animation.
mat=bpy.data.materials.new('Synthetic preview navy');mat.diffuse_color=(.12,.2,.34,1)
mesh.data.materials.clear();mesh.data.materials.append(mat)
for poly in mesh.data.polygons:poly.material_index=0
scene.render.engine='BLENDER_WORKBENCH';scene.display.shading.light='STUDIO';scene.display.shading.color_type='MATERIAL'
scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True;scene.display.shading.background_type='WORLD'
scene.world.color=(.93,.95,.98);scene.render.resolution_x=900;scene.render.resolution_y=600;scene.render.resolution_percentage=100
camera_data=bpy.data.cameras.new('Synthetic hand inspection');cam=bpy.data.objects.new('Synthetic hand inspection',camera_data);scene.collection.objects.link(cam);scene.camera=cam
scene.frame_set(1);bpy.context.view_layer.update()
target=rig.matrix_world@Vector(rest['Left'].mean(axis=0));basis_world=rig.matrix_world.to_3x3().normalized()@Vector(hand_basis(rest['Left'])[:,2])
cam.location=target+basis_world*.35+Vector((0,-.12,.09));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();camera_data.type='ORTHO';camera_data.ortho_scale=.21
for frame in (1,13):
    scene.frame_set(frame);scene.render.filepath=str(U/('SYNTHETIC_hand_frame%02d.png'%frame));bpy.ops.render.render(write_still=True)
scene.frame_set(13)
for side in ('Left','Right'):
    hand=rest[side];basis=hand_basis(hand);normal=basis[:,2]*(-1 if side=='Left' else 1)
    target=rig.matrix_world@Vector(hand.mean(axis=0))
    worldnormal=rig.matrix_world.to_3x3().normalized()@Vector(normal)
    worldacross=rig.matrix_world.to_3x3().normalized()@Vector(basis[:,0])
    cam.location=target+worldnormal*.18+worldacross*.28
    cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(U/('SYNTHETIC_'+side.lower()+'_palmar_side.png'));bpy.ops.render.render(write_still=True)
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(U/'SYNTHETIC_finger_mapping.blend'))
bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);mesh.select_set(True);bpy.context.view_layer.objects.active=rig
bpy.ops.export_scene.fbx(filepath=str(U/'SYNTHETIC_finger_mapping.fbx'),use_selection=True,object_types={'ARMATURE','MESH'},add_leaf_bones=False,bake_anim=True,bake_anim_use_all_actions=False,bake_anim_use_nla_strips=False,bake_anim_simplify_factor=0.0,path_mode='AUTO')
bpy.ops.wm.open_mainfile(filepath=str(U/'SYNTHETIC_finger_mapping.blend'),load_ui=False,use_scripts=False)
rig=next(o for o in bpy.data.objects if o.type=='ARMATURE')
reopen=0.
for frame,pose in saved.items():
    bpy.context.scene.frame_set(int(frame));bpy.context.view_layer.update()
    for name,m in pose.items():reopen=max(reopen,float(np.max(np.abs(np.array(rig.pose.bones[name].matrix)-m))))
assert reopen<2e-4,reopen
out=dict(status='PASS_SYNTHETIC_CHARACTER_RETARGET_BAKE_REOPEN',meaning='Technical finger/skin pipeline only; artificial motion, not a Korean sign, no real source quality validation',
     driven_bones=30,frames=25,arms_fixed=True,palm_orientation_fixed=True,max_direction_error_deg=max_angle,max_length_error_rig_units=max_length,
     skin_vertices_moved_over_1mm=int((skin_delta>.001).sum()),reopen_max_matrix_error=reopen,
     untouched_terminal_extension_bones=10,blender_version=bpy.app.version_string,
     anatomical_direction='Left palm basis+Z is dorsal; right basis+Z is palmar. Synthetic flexion signs corrected to Left-1 Right+1. This is no guarantee of anatomical source accuracy.',
     camera_note='frame01/13 view is above the dorsal side of the left hand; additional left/right_palmar_side views show inward flexion',
     first_trial_preserved='initial_synthetic_diagnostic: first synthetic input bent left fingers dorsally; FBX import also showed a time offset. Curve simplification was initially suspected but disabling it did not fix the discrepancy.',
     source_asset_unchanged=hashlib.sha256(Path(json.loads((U/'STATIC_FBX_PIN.json').read_text())['source']).read_bytes()).hexdigest()==json.loads((U/'STATIC_FBX_PIN.json').read_text())['sha256'])
(U/'SYNTHETIC_CHARACTER_VERIFIED.json').write_text(json.dumps(out,indent=2)+'\n','utf-8')
print('SYNTHETIC_CHARACTER_VERIFIED',json.dumps(out))
