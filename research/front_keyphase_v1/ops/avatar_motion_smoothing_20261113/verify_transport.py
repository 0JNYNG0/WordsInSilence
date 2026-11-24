import bpy,numpy as np,json,traceback
from pathlib import Path
U=Path(__file__).resolve().parent
try:
    d=np.load(U/'candidate_poses.npz');indices=d['native_indices'];expected=d['head'][indices];names=list(d['names']);mw=d['rig_world']
    expected=expected@mw[:3,:3].T+mw[:3,3]
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.context.scene.render.fps=30
    bpy.ops.import_scene.fbx(filepath=str(U/'CANDIDATE_ORIGINAL_MATERIAL.fbx'),use_anim=True,use_image_search=False,anim_offset=0.)
    rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');actual=[]
    for f in range(127):
        bpy.context.scene.frame_set(f+1);bpy.context.view_layer.update()
        actual.append([list(rig.matrix_world@rig.pose.bones[n].head)for n in names])
    error=np.linalg.norm(np.array(actual)-expected,axis=-1);assert np.max(error)<.001
    result={'status':'PASS_NATIVE_FBX_TRANSPORT_NOT_SOURCE_ACCURACY','frames':127,'bones':65,'positions':int(error.size),'max_world_error':float(error.max()),'p95_world_error':float(np.percentile(error,95)),'import_anim_offset':0,'interframe_note':'FBX baking/export can change between-key interpolation; this check certifies native keys only.'}
    (U/'FBX_TRANSPORT_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n','utf-8');print(result)
except Exception as e:
    (U/'FIRST_FBX_ERROR.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2)+'\n','utf-8');raise
