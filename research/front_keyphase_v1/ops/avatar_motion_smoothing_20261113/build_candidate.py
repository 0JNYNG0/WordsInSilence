"""One frozen offline candidate; read-only public baseline, original skin and rig."""
import bpy,json,hashlib,traceback,re
import numpy as np
from pathlib import Path
from mathutils import Quaternion
U=Path(__file__).resolve().parent
P=U.parent/'avatar_motion_pilot_20261113'
BASE=P/'public_approximate_preview_body_axis/APPROXIMATE_PUBLIC_WORD1567_REAL02.blend'
def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n','utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalize(q):return q/np.linalg.norm(q,axis=-1,keepdims=True)
def angle(a,b):return np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(normalize(a)*normalize(b),axis=-1)),0,1)))
def rig_fingerprint(rig):
    meshes=[o for o in bpy.data.objects if o.type=='MESH']
    h=hashlib.sha256()
    for o in sorted(meshes,key=lambda x:x.name):
        for v in o.data.vertices:
            h.update(np.array(v.co,dtype='<f8').tobytes())
            h.update(json.dumps([(g.group,float(g.weight))for g in v.groups]).encode())
        for p in o.data.polygons:h.update(np.array(p.vertices,dtype='<i4').tobytes())
    return {'mesh_vertices_weights_topology_sha256':h.hexdigest(),'bones':[{'name':b.name,'head':list(b.head_local),'tail':list(b.tail_local),'parent':b.parent.name if b.parent else None,'matrix':[list(r)for r in b.matrix_local]}for b in rig.data.bones]}
def capture(rig,label):
    scene=bpy.context.scene;names=[b.name for b in rig.pose.bones]
    all_frames=np.arange(0,126.0001,.25,dtype=np.float64)
    head=[];tail=[];localq=[];globalq=[];mat=[]
    for sf in all_frames:
        bf=sf+1;scene.frame_set(int(np.floor(bf)),subframe=float(bf%1));bpy.context.view_layer.update()
        head.append([list(b.head)for b in rig.pose.bones]);tail.append([list(b.tail)for b in rig.pose.bones])
        localq.append([list(b.rotation_quaternion)for b in rig.pose.bones]);globalq.append([list(b.matrix.to_quaternion())for b in rig.pose.bones]);mat.append([[list(r)for r in b.matrix]for b in rig.pose.bones])
    np.savez_compressed(U/(label+'_poses.npz'),source_frames=all_frames,names=np.array(names),head=np.array(head),tail=np.array(tail),local_quaternion=np.array(localq),global_quaternion=np.array(globalq),matrix=np.array(mat),rig_world=np.array(rig.matrix_world),native_indices=np.arange(0,len(all_frames),4))
    return names
def material_restore():
    with bpy.data.libraries.load(str(P/'static_character.blend'),link=False)as (src,dst):dst.materials=['Ch29_Body']
    mat=dst.materials[0]
    diffuse=next(n for n in mat.node_tree.nodes if n.type=='TEX_IMAGE' and n.image and 'Diffuse' in n.image.filepath)
    for n in mat.node_tree.nodes:n.select=False
    diffuse.select=True;mat.node_tree.nodes.active=diffuse
    for o in bpy.data.objects:
        if o.type=='MESH':
            o.data.materials.clear();o.data.materials.append(mat)
            for poly in o.data.polygons:poly.material_index=0
    scene=bpy.context.scene;scene.display.shading.color_type='TEXTURE'
    return {'material':mat.name,'active_diffuse':diffuse.image.name,'packed':bool(diffuse.image.packed_file),'original_texture_path':diffuse.image.filepath}
try:
    frozen=json.loads((U/'FROZEN.json').read_text('utf-8'))
    assert frozen['condition_count']==1
    for p,h in frozen['pins'].items():assert sha(p)==h,('pin mismatch',p)
    bpy.ops.wm.open_mainfile(filepath=str(BASE),load_ui=False,use_scripts=False)
    rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');scene=bpy.context.scene
    before=rig_fingerprint(rig);write(U/'RIG_SKIN_BEFORE.json',before)
    appearance=material_restore();scene.frame_set(1)
    names=capture(rig,'baseline');scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(U/'BASELINE_ORIGINAL_MATERIAL.blend'))
    action=rig.animation_data.action;curves={}
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for f in bag.fcurves:
                    match=re.fullmatch(r'pose\.bones\["(.+)"\]\.rotation_quaternion',f.data_path)
                    assert match,'unexpected animation channel'
                    curves.setdefault(match.group(1),{})[f.array_index]=f
    assert len(curves)==36
    records=[]
    for name,axes in curves.items():
        assert sorted(axes)==[0,1,2,3]
        assert all(len(c.keyframe_points)==127 for c in axes.values())
        assert all(np.allclose([k.co.x for k in c.keyframe_points],np.arange(1,128))for c in axes.values())
        original=np.array([[axes[a].keyframe_points[f].co.y for a in range(4)]for f in range(127)],dtype=np.float64)
        assert np.isfinite(original).all()
        q=normalize(original);flipped=[]
        for f in range(1,127):
            if np.dot(q[f-1],q[f])<0:q[f]*=-1;flipped.append(f)
        candidate=q.copy();wrist=name.split(':')[-1]in ['LeftHand','RightHand'];skips=[];capped=[]
        if wrist:
            for f in range(1,126):
                neighbors=[float(angle(q[f],q[f-1])),float(angle(q[f],q[f+1]))]
                if max(neighbors)>15:skips.append(f);continue
                avg=normalize(.25*q[f-1]+.5*q[f]+.25*q[f+1]);delta=float(angle(q[f],avg))
                if delta>2:
                    avg=np.array(Quaternion(q[f]).slerp(Quaternion(avg),2/delta));capped.append(f)
                candidate[f]=normalize(avg)
        for f in range(1,127):
            if np.dot(candidate[f-1],candidate[f])<0:candidate[f]*=-1
        for a,c in axes.items():
            for f,k in enumerate(c.keyframe_points):k.co.y=float(candidate[f,a]);k.interpolation='LINEAR'
            c.update()
        records.append({'bone':name,'hemisphere_flipped_source_frames':flipped,'smoothing':wrist,'skip_source_frames_large_neighbor':skips,'capped_source_frames':capped,'native_rotation_delta_deg':angle(original,candidate).tolist()})
    names2=capture(rig,'candidate');assert names2==names
    after=rig_fingerprint(rig);assert after==before
    scene.frame_set(1);bpy.ops.wm.save_as_mainfile(filepath=str(U/'CANDIDATE_ORIGINAL_MATERIAL.blend'))
    meshes=[o for o in bpy.data.objects if o.type=='MESH']
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True)
    for o in meshes:o.select_set(True)
    bpy.context.view_layer.objects.active=rig
    bpy.ops.export_scene.fbx(filepath=str(U/'CANDIDATE_ORIGINAL_MATERIAL.fbx'),use_selection=True,object_types={'ARMATURE','MESH'},add_leaf_bones=False,bake_anim=True,bake_anim_use_all_actions=False,bake_anim_use_nla_strips=False,bake_anim_simplify_factor=0.0,path_mode='COPY',embed_textures=True)
    write(U/'CANDIDATE_BUILD.json',{'status':'BUILT_PENDING_NUMERIC_AND_VISUAL_REVIEW','frozen_sha256':sha(U/'FROZEN.json'),'appearance':appearance,'rig_skin_unchanged':True,'driven_bones':len(curves),'native_frames':127,'quarter_samples':505,'records':records,'outputs':{p.name:sha(p)for p in [U/'BASELINE_ORIGINAL_MATERIAL.blend',U/'CANDIDATE_ORIGINAL_MATERIAL.blend',U/'CANDIDATE_ORIGINAL_MATERIAL.fbx',U/'baseline_poses.npz',U/'candidate_poses.npz']}})
    scene.render.resolution_x=720;scene.render.resolution_y=720;scene.frame_set(65);scene.render.filepath=str(U/'CANDIDATE_TEXTURE_CHECK.png');bpy.ops.render.render(write_still=True)
    print('CANDIDATE_BUILD_COMPLETE')
except Exception as e:
    p=U/'FIRST_IMPLEMENTATION_ERROR.json'
    if not p.exists():write(p,{'error':repr(e),'traceback':traceback.format_exc()})
    raise
