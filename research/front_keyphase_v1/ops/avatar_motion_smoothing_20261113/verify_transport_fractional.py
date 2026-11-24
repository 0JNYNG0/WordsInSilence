"""Check exported FBX between native keys as an additional transport diagnostic."""
import bpy,numpy as np,json
from pathlib import Path
U=Path(__file__).resolve().parent
z=np.load(U/'candidate_poses.npz');names=list(z['names']);mw=z['rig_world'];expected=z['head']@mw[:3,:3].T+mw[:3,3]
bpy.ops.wm.read_factory_settings(use_empty=True);bpy.context.scene.render.fps=30
bpy.ops.import_scene.fbx(filepath=str(U/'CANDIDATE_ORIGINAL_MATERIAL.fbx'),use_anim=True,use_image_search=False,anim_offset=0.)
rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');head=[];q=[]
for sf in z['source_frames']:
    bf=sf+1;bpy.context.scene.frame_set(int(bf),subframe=float(bf%1));bpy.context.view_layer.update()
    head.append([list(rig.matrix_world@rig.pose.bones[n].head)for n in names]);q.append([list(rig.pose.bones[n].matrix.to_quaternion())for n in names])
q=np.array(q);q/=np.linalg.norm(q,axis=-1,keepdims=True)
step=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q[1:]*q[:-1],axis=-1)),0,1)))
error=np.linalg.norm(np.array(head)-expected,axis=-1);mid=np.flatnonzero(z['source_frames']%1!=0)
np.savez_compressed(U/'fbx_quarter_poses.npz',source_frames=z['source_frames'],names=z['names'],head_world=head,global_quaternion=q)
out={'status':'FRACTIONAL_FBX_DIAGNOSTIC_COMPLETE','quarter_samples':505,'bones':65,'max_between_native_world_position_difference':float(error[mid].max()),'p95_between_native_world_position_difference':float(np.percentile(error[mid],95)),'max_quarter_world_rotation_step_deg':float(step.max()),'world_steps_over45':int((step>45).sum()),'world_steps_over90':int((step>90).sum()),'note':'Native FBX transport passed separately. Blender LINEAR quaternion-component interpolation and FBX exported rotation interpolation can differ between native frames; this diagnostic measures that difference, not source3D correctness.'}
(U/'FBX_FRACTIONAL_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2)+'\n','utf-8');print(out)
