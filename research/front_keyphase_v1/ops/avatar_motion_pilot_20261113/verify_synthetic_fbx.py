"""Reimport only the newly created artificial animation; never an old source recording."""
import bpy,json,time
import numpy as np
from pathlib import Path
U=Path(__file__).resolve().parent
try:
    bpy.ops.wm.open_mainfile(filepath=str(U/'SYNTHETIC_finger_mapping.blend'),load_ui=False,use_scripts=False)
    rig=next(o for o in bpy.data.objects if o.type=='ARMATURE')
    expected={}
    for frame in (1,13,25):
        bpy.context.scene.frame_set(frame);bpy.context.view_layer.update()
        expected[frame]={p.name:np.array(rig.matrix_world@p.head)for p in rig.pose.bones}
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps=24
    bpy.ops.import_scene.fbx(filepath=str(U/'SYNTHETIC_finger_mapping.fbx'),use_anim=True,use_image_search=False,anim_offset=0.0)
    rig=next(o for o in bpy.data.objects if o.type=='ARMATURE')
    assert len(bpy.data.actions)==1,len(bpy.data.actions)
    errors=[];details=[]
    for frame in (1,13,25):
        bpy.context.scene.frame_set(frame);bpy.context.view_layer.update()
        for name,pos in expected[frame].items():
            assert name in rig.pose.bones,name
            actual=np.array(rig.matrix_world@rig.pose.bones[name].head)
            error=float(np.linalg.norm(actual-pos));errors.append(error)
            details.append(dict(frame=frame,bone=name,error=error,expected=pos.tolist(),actual=actual.tolist()))
    (U/'SYNTHETIC_FBX_DIAGNOSTIC.json').write_text(json.dumps(dict(max_error=max(errors),worst=sorted(details,key=lambda x:-x['error'])[:20],frame_max={f:max(x['error']for x in details if x['frame']==f)for f in expected}),indent=2)+'\n','utf-8')
    print('FBX_ERRORS',json.dumps(dict(max_error=max(errors),worst=sorted(details,key=lambda x:-x['error'])[:3],frame_max={f:max(x['error']for x in details if x['frame']==f)for f in expected})))
    assert max(errors)<.001,max(errors)
    out=dict(status='PASS_SYNTHETIC_FBX_REIMPORT',frames_checked=[1,13,25],bone_positions_checked=len(errors),
             max_world_position_error=max(errors),actions_imported=1,real_sign_animation=False,import_animation_frame_offset=0.0,
             meaning='Technical asset transport only; artificial finger animation, fixed arms, no source reconstruction')
    (U/'SYNTHETIC_FBX_VERIFIED.json').write_text(json.dumps(out,indent=2)+'\n','utf-8')
    print('SYNTHETIC_FBX_REIMPORT',json.dumps(out))
except Exception as e:
    p=U/'FIRST_SYNTHETIC_FBX_ERROR.json'
    if not p.exists():p.write_text(json.dumps(dict(error=repr(e),time=time.time()),indent=2)+'\n','utf-8')
    raise
