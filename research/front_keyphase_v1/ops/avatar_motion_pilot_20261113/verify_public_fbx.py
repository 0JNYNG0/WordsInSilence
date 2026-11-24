"""Verify transport of our newly derived public approximation, not source motion accuracy."""
import bpy,json,sys,numpy as np
from pathlib import Path
U=Path(__file__).resolve().parent;D=U/('public_approximate_preview_body_axis'if '--body-axis'in sys.argv else'public_approximate_preview')
try:
    bpy.ops.wm.open_mainfile(filepath=str(D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.blend'),load_ui=False,use_scripts=False)
    rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');expected={}
    for frame in [33,65,97,127]:
        bpy.context.scene.frame_set(frame);bpy.context.view_layer.update();expected[frame]={p.name:np.array(rig.matrix_world@p.head)for p in rig.pose.bones}
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.context.scene.render.fps=30
    bpy.ops.import_scene.fbx(filepath=str(D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.fbx'),use_anim=True,use_image_search=False,anim_offset=0.)
    rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');errors=[]
    for frame,pose in expected.items():
        bpy.context.scene.frame_set(frame);bpy.context.view_layer.update()
        for name,xyz in pose.items():errors.append(float(np.linalg.norm(np.array(rig.matrix_world@rig.pose.bones[name].head)-xyz)))
    assert max(errors)<.001,max(errors)
    (D/'FBX_TRANSPORT_VERIFIED.json').write_text(json.dumps(dict(status='PASS_PUBLIC_DERIVED_FBX_REIMPORT',source_frames=[32,64,96,126],bone_positions=len(errors),max_world_position_error=max(errors),source_geometry_accuracy_validated=False,import_anim_offset=0.),indent=2)+'\n')
    print('PUBLIC_FBX_TRANSPORT',max(errors))
except Exception as e:
    p=D/'FIRST_FBX_ERROR.json'
    if not p.exists():p.write_text(json.dumps(dict(error=repr(e)),indent=2)+'\n')
    raise
