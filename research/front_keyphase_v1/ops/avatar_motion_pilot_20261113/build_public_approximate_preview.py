"""One approximate public multiview reconstruction; no guessed keyframe poses or private motion."""
import bpy,json,sys,hashlib,time
import numpy as np
from pathlib import Path
from mathutils import Matrix,Vector
U=Path(__file__).resolve().parent;sys.path.insert(0,str(U));V=U/'video_only';BODY_AXIS='--body-axis' in sys.argv
D=U/('public_approximate_preview_body_axis'if BODY_AXIS else'public_approximate_preview');D.mkdir(exist_ok=True)
from hand_retarget_blender import rest_hand,apply_hand,bone_name,FINGERS
from geometry_core import hand_basis
from retarget_math import solve_arm,unit
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n','utf-8')
def basis(left,right,nose):
    origin=(left+right)/2;x=unit(left-right);up=nose-origin;y=unit(up-x*np.dot(x,up));z=unit(np.cross(x,y));return origin,np.column_stack([x,y,z])
def orient(pb,target):
    bpy.context.view_layer.update();q=(pb.tail-pb.head).normalized().rotation_difference(Vector(unit(target)));mat=(q@pb.matrix.to_quaternion()).to_matrix().to_4x4();mat.translation=pb.head;pb.matrix=mat;bpy.context.view_layer.update()
try:
    data=np.load(V/'approximate_xyz.npz');xyz=data['xyz'];joints=list(data['joints']);lookup={j:xyz[:,i]for i,j in enumerate(joints)}
    required=[0,5,6,7,8]+([11,12]if BODY_AXIS else[])+list(range(91,133));assert all(np.isfinite(lookup[j]).all()for j in required),'Missing required3D, no interpolation allowed'
    source_sha=hashlib.sha256((V/'approximate_xyz.npz').read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(U/'static_character.blend'),load_ui=False,use_scripts=False)
    assert len(bpy.data.actions)==0
    rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');mesh=next(o for o in bpy.data.objects if o.type=='MESH')
    scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=127;scene.render.fps=30
    sides={'Left':(5,7,91),'Right':(6,8,112)}
    rest={s:rest_hand(rig,s)for s in sides};bn=lambda n:rig.data.bones[bone_name(rig,n)]
    rig_center=(np.array(bn('LeftArm').head_local)+np.array(bn('RightArm').head_local))/2
    rig_up=rig_center-np.array(bn('Hips').head_local)
    rig_origin,rig_basis=basis(np.array(bn('LeftArm').head_local),np.array(bn('RightArm').head_local),rig_center+rig_up if BODY_AXIS else np.array(bn('Head').head_local))
    if BODY_AXIS:
        source_up=np.median((lookup[5]+lookup[6])/2-(lookup[11]+lookup[12])/2,axis=0)
    scales={};lengths={}
    for side,(s,e,h)in sides.items():
        upper=bn(side+'Arm').length;fore=bn(side+'ForeArm').length;lengths[side]=(upper,fore)
        src=np.linalg.norm(lookup[s]-lookup[e],axis=1)+np.linalg.norm(lookup[e]-lookup[h],axis=1)
        scales[side]=(upper+fore)/float(np.median(src))
    pose_records=[];allmetrics=[];saved={}
    for f in range(127):
        scene.frame_set(f+1)
        for pb in rig.pose.bones:pb.matrix_basis.identity();pb.rotation_mode='QUATERNION'
        bpy.context.view_layer.update()
        source_center=(lookup[5][f]+lookup[6][f])/2
        _,srcbasis=basis(lookup[5][f],lookup[6][f],source_center+source_up if BODY_AXIS else lookup[0][f]);mapping=rig_basis@srcbasis.T
        for side,(s,e,h)in sides.items():
            shoulder=np.array(bn(side+'Arm').head_local);elbow_hint=shoulder+mapping@(lookup[e][f]-lookup[s][f])*scales[side]
            wrist_target=shoulder+mapping@(lookup[h][f]-lookup[s][f])*scales[side]
            sol=solve_arm(shoulder,wrist_target,elbow_hint,*lengths[side]);upper=rig.pose.bones[bone_name(rig,side+'Arm')];fore=rig.pose.bones[bone_name(rig,side+'ForeArm')]
            orient(upper,sol['elbow']-shoulder);orient(fore,sol['wrist']-sol['elbow'])
            hand=np.array([lookup[h+i][f]for i in range(21)]);targetbasis=mapping@hand_basis(hand)
            pb=rig.pose.bones[bone_name(rig,side+'Hand')];restrot=bn(side+'Hand').matrix_local.to_3x3();rotation=targetbasis@hand_basis(rest[side]).T@np.array(restrot)
            matrix=Matrix(rotation).to_4x4();matrix.translation=pb.head;pb.matrix=matrix;bpy.context.view_layer.update()
            metrics=apply_hand(rig,side,hand,target_palm_basis=targetbasis);allmetrics.extend(metrics)
            for name in [side+'Arm',side+'ForeArm',side+'Hand']+[side+'Hand'+finger+str(k)for finger,_ in FINGERS for k in range(1,4)]:rig.pose.bones[bone_name(rig,name)].keyframe_insert(data_path='rotation_quaternion',frame=f+1)
            pose_records.append(dict(source_frame=f,side=side,unreachable=sol['unreachable'],reach_error_rig_units=sol['target_reach_error'],
                  source_hand_max_reprojection_px=float(np.nanmax(data['reprojection_error_px'][:,f,[joints.index(h+i)for i in range(21)]])),
                  hand_axis_det=float(np.linalg.det(targetbasis))))
        if f in [32,64,96,126]:saved[str(f+1)]={pb.name:[list(r)for r in pb.matrix]for pb in rig.pose.bones}
    assert max(x['length_error']for x in allmetrics)<1e-4
    assert max(x['angle_error_deg']for x in allmetrics)<.2
    # Presentation only; body and source data remain unaltered.
    mat=bpy.data.materials.new('Approximate character silver');mat.diffuse_color=(.48,.57,.67,1);mesh.data.materials.clear();mesh.data.materials.append(mat)
    for p in mesh.data.polygons:p.material_index=0
    scene.render.engine='BLENDER_WORKBENCH';scene.display.shading.light='STUDIO';scene.display.shading.color_type='MATERIAL';scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True;scene.display.shading.background_type='WORLD';scene.world.color=(.92,.94,.97)
    scene.render.resolution_x=720;scene.render.resolution_y=720;scene.render.resolution_percentage=100
    cd=bpy.data.cameras.new('Public approximate preview camera');cam=bpy.data.objects.new(cd.name,cd);scene.collection.objects.link(cam);scene.camera=cam;cd.type='ORTHO';cd.ortho_scale=1.25
    target=rig.matrix_world@Vector(rig_origin)+Vector((0,0,-.03));front=rig.matrix_world.to_3x3().normalized()@Vector(rig_basis[:,2]);left=rig.matrix_world.to_3x3().normalized()@Vector(rig_basis[:,0])
    for name,offset in [('front',front),('left_oblique',front*.65+left*.76),('right_oblique',front*.65-left*.76)]:
        cam.location=target+offset*3+Vector((0,0,.1));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
        for f in [32,64,96,126]:
            scene.frame_set(f+1);scene.render.filepath=str(D/(name+'_source%03d.png'%f));bpy.ops.render.render(write_still=True)
    cam.location=target+front*3+Vector((0,0,.1));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.blend'))
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);mesh.select_set(True);bpy.context.view_layer.objects.active=rig
    bpy.ops.export_scene.fbx(filepath=str(D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.fbx'),use_selection=True,object_types={'ARMATURE','MESH'},add_leaf_bones=False,bake_anim=True,bake_anim_use_all_actions=False,bake_anim_use_nla_strips=False,bake_anim_simplify_factor=0.0,path_mode='AUTO')
    write(D/'BAKE_RECORD.json',dict(status='APPROXIMATE_PUBLIC_ANIMATION_BAKED_PENDING_VISUAL_REVIEW',source_xyz_sha256=source_sha,source_clip='NIA_SL_WORD1567_REAL02',source_frames=127,fps=30,
          actual_calibration=False,source_units='arbitrary',torso_legs_face='static',manual_source_keyframe_corrections=0,missing_required_source_points=0,arm_scales=scales,
          body_up_definition='median shoulder-midpoint minus hip-midpoint, not face point'if BODY_AXIS else'nose minus shoulder-midpoint; known depth/height confound, retained first trial',
          wrist_unreachable_frames={s:sum(r['unreachable']for r in pose_records if r['side']==s)for s in sides},
          max_hand_direction_error_deg=max(x['angle_error_deg']for x in allmetrics),max_bone_length_error=max(x['length_error']for x in allmetrics),
          source_quality_limit='Estimated cameras and2D landmarks may distort handshape/depth. Reprojection does not establish true3D. Needs RGB/motion/anatomy/contact review.',
          skin_binding='Existing static NPC rig, no source/private motion imported',frame_records=pose_records))
    write(D/'EXPECTED_POSES.json',saved)
    print('PUBLIC_APPROXIMATE_BAKE_COMPLETE',str(D))
except Exception as e:
    import traceback
    p=D/'FIRST_ERROR.json'
    if not p.exists():write(p,dict(error=repr(e),traceback=traceback.format_exc()))
    raise
