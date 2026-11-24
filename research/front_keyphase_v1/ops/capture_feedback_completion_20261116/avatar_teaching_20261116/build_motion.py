"""Public multiview retarget derivative. Never changes model weights or source/Assets."""
from pathlib import Path
import bpy,sys,json,hashlib
import numpy as np
from mathutils import Matrix,Vector,Quaternion
U=Path(__file__).resolve().parent;O=U.parent.parent;P=O/'avatar_motion_pilot_20261113'
sys.path.insert(0,str(P))
from hand_retarget_blender import rest_hand,apply_hand,bone_name,FINGERS
from geometry_core import hand_basis
from retarget_math import solve_arm,unit
args=sys.argv[sys.argv.index('--')+1:];clip=args[0];geo=args[1] if len(args)>1 else 'video_only';revision=args[2]if len(args)>2 else 'baseline'
D=U/clip/revision;D.mkdir(exist_ok=True)
cfgpath=U/clip/'RETARGET_CONSTRAINTS.json';cfg=json.loads(cfgpath.read_text('utf-8'))if revision!='baseline'and cfgpath.exists()else {}
def write(n,x):(D/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def basis(left,right,up):
 x=unit(left-right);y=unit(up-x*np.dot(x,up));z=unit(np.cross(x,y));return np.column_stack([x,y,z])
def orient(pb,target):
 bpy.context.view_layer.update();q=(pb.tail-pb.head).normalized().rotation_difference(Vector(unit(target)));m=(q@pb.matrix.to_quaternion()).to_matrix().to_4x4();m.translation=pb.head;pb.matrix=m;bpy.context.view_layer.update()
data=np.load(U/clip/geo/'approximate_xyz.npz');xyz=data['xyz'];joints=list(data['joints']);lookup={j:xyz[:,i]for i,j in enumerate(joints)};nf=len(xyz)
assert all(np.isfinite(lookup[j]).all()for j in [0,5,6,7,8,11,12]+list(range(91,133)))
bpy.ops.wm.open_mainfile(filepath=str(P/'static_character.blend'),load_ui=False,use_scripts=False)
rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');meshes=[o for o in bpy.data.objects if o.type=='MESH'];scene=bpy.context.scene
assert not bpy.data.actions
scene.frame_start=1;scene.frame_end=nf;scene.render.fps=30
bn=lambda n:rig.data.bones[bone_name(rig,n)]
sides={'Left':(5,7,91),'Right':(6,8,112)};rest={s:rest_hand(rig,s)for s in sides}
rig_center=(np.array(bn('LeftArm').head_local)+np.array(bn('RightArm').head_local))/2
rig_up=rig_center-np.array(bn('Hips').head_local)
rig_basis=basis(np.array(bn('LeftArm').head_local),np.array(bn('RightArm').head_local),rig_up)
source_up=np.median((lookup[5]+lookup[6])/2-(lookup[11]+lookup[12])/2,axis=0)
scales={};lengths={}
for side,(s,e,h)in sides.items():
 lengths[side]=(bn(side+'Arm').length,bn(side+'ForeArm').length)
 scales[side]=sum(lengths[side])/float(np.median(np.linalg.norm(lookup[s]-lookup[e],axis=1)+np.linalg.norm(lookup[e]-lookup[h],axis=1)))
records=[];metrics=[];head=[];tail=[];rot=[];allnames=[b.name for b in rig.pose.bones];prev={}
def envelope(frame,span,ramp):
 a,b=span;v=min((frame-(a-ramp))/max(ramp,1),((b+ramp)-frame)/max(ramp,1),1);v=float(np.clip(v,0,1));return v*v*(3-2*v)
templates={}
for item in cfg.get('shape_holds',[]):
 side=item['side'];h=sides[side][2];samples=[]
 for f in item['source_frames']:
  p=np.array([lookup[h+i][f]for i in range(21)]);b=hand_basis(p);samples.append((p-p[0])@b/np.linalg.norm(p[9]-p[0]))
 templates[side]=(np.median(samples,axis=0),item)
baseline=np.load(U/clip/'baseline/poses.npz')if cfg.get('mouth_anchor')else None
for f in range(nf):
 scene.frame_set(f+1)
 for pb in rig.pose.bones:pb.matrix_basis.identity();pb.rotation_mode='QUATERNION'
 bpy.context.view_layer.update()
 mapping=rig_basis@basis(lookup[5][f],lookup[6][f],source_up).T
 for side,(s,e,h)in sides.items():
  shoulder=np.array(bn(side+'Arm').head_local);elbow_hint=shoulder+mapping@(lookup[e][f]-lookup[s][f])*scales[side];wrist=shoulder+mapping@(lookup[h][f]-lookup[s][f])*scales[side]
  if cfg.get('mouth_anchor') and side=='Right':
   item=cfg['mouth_anchor'];weight=envelope(f,item['span'],item['ramp']);srcmouth=np.mean([lookup[k][f]for k in [71,74,77,80]],axis=0)
   local_source=mapping@(lookup[h+8][f]-srcmouth)*scales[side]
   desired=np.array(item['rig_mouth_xyz'])+local_source
   idx=allnames.index(bone_name(rig,'RightHandIndex3'));before=baseline['tail'][f,idx]
   wrist+=weight*(desired-before)
  sol=solve_arm(shoulder,wrist,elbow_hint,*lengths[side]);orient(rig.pose.bones[bone_name(rig,side+'Arm')],sol['elbow']-shoulder);orient(rig.pose.bones[bone_name(rig,side+'ForeArm')],sol['wrist']-sol['elbow'])
  hand=np.array([lookup[h+i][f]for i in range(21)]);targetbasis=mapping@hand_basis(hand)
  if side in templates:
   template,item=templates[side];w=envelope(f,item['span'],item['ramp']);b=hand_basis(hand);scale=np.linalg.norm(hand[9]-hand[0]);held=(template*scale)@b.T+hand[0];hand=(1-w)*hand+w*held
  pb=rig.pose.bones[bone_name(rig,side+'Hand')];rotation=targetbasis@hand_basis(rest[side]).T@np.array(bn(side+'Hand').matrix_local.to_3x3());m=Matrix(rotation).to_4x4();m.translation=pb.head;pb.matrix=m;bpy.context.view_layer.update()
  metrics.extend(apply_hand(rig,side,hand,target_palm_basis=targetbasis))
  for name in [side+'Arm',side+'ForeArm',side+'Hand']+[side+'Hand'+finger+str(k)for finger,_ in FINGERS for k in range(1,4)]:
   pb=rig.pose.bones[bone_name(rig,name)];q=np.array(pb.rotation_quaternion)
   if name in prev and np.dot(q,prev[name])<0:q=-q;pb.rotation_quaternion=Quaternion(q)
   prev[name]=q;pb.keyframe_insert(data_path='rotation_quaternion',frame=f+1)
  records.append(dict(frame=f,side=side,unreachable=sol['unreachable'],reach_error=float(sol['target_reach_error'])))
 head.append([list(b.head)for b in rig.pose.bones]);tail.append([list(b.tail)for b in rig.pose.bones]);rot.append([list(b.rotation_quaternion)for b in rig.pose.bones])
for action in bpy.data.actions:
 for layer in action.layers:
  for strip in layer.strips:
   for bag in strip.channelbags:
    for curve in bag.fcurves:
     for key in curve.keyframe_points:key.interpolation='LINEAR'
np.savez_compressed(D/'poses.npz',head=head,tail=tail,local_quaternion=rot,names=allnames,rig_world=np.array(rig.matrix_world),rig_basis=rig_basis)
# Original Ch29 appearance, with packed local textures.
for mat in bpy.data.materials:
 if mat.use_nodes:
  diffuse=next((n for n in mat.node_tree.nodes if n.type=='TEX_IMAGE'and n.image and 'Diffuse'in n.image.filepath),None)
  if diffuse:
   for n in mat.node_tree.nodes:n.select=False
   diffuse.select=True;mat.node_tree.nodes.active=diffuse
for im in bpy.data.images:
 if im.type=='IMAGE'and im.size[0]>0:im.pack()
scene.render.engine='BLENDER_WORKBENCH';scene.display.shading.light='STUDIO';scene.display.shading.color_type='TEXTURE';scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True;scene.display.shading.background_type='WORLD';scene.world.color=(.92,.94,.97)
scene.render.resolution_x=640;scene.render.resolution_y=720;scene.render.resolution_percentage=100
cd=bpy.data.cameras.new('Source review camera');cam=bpy.data.objects.new(cd.name,cd);scene.collection.objects.link(cam);scene.camera=cam;cd.type='ORTHO';cd.ortho_scale=1.15
target=rig.matrix_world@Vector(rig_center)+Vector((0,0,-.05));front=rig.matrix_world.to_3x3().normalized()@Vector(rig_basis[:,2]);left=rig.matrix_world.to_3x3().normalized()@Vector(rig_basis[:,0])
anchors=[43,50,55,65,70,78]if '0099'in clip else[47,54,61,68,74,80]
for name,offset in [('front',front),('side',front*.65+left*.76)]:
 cam.location=target+offset*3;cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
 for f in anchors:
  scene.frame_set(f+1);scene.render.filepath=str(D/f'{name}_{f:03d}.png');bpy.ops.render.render(write_still=True)
cam.location=target+front*3;cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(D/'motion.blend'))
for im in bpy.data.images:
 if im.type=='IMAGE'and max(im.size)>1024:
  scale=1024/max(im.size);im.scale(round(im.size[0]*scale),round(im.size[1]*scale));im.pack()
bpy.ops.object.select_all(action='DESELECT')
rig.select_set(True)
for m in meshes:m.select_set(True)
bpy.context.view_layer.objects.active=rig
bpy.ops.export_scene.gltf(filepath=str(D/'motion.glb'),export_format='GLB',use_selection=True,export_animations=True,export_frame_range=True,export_animation_mode='SCENE',export_force_sampling=True,export_cameras=False,export_lights=False,export_yup=True,export_all_influences=True,export_image_format='AUTO',export_materials='EXPORT')
sys.path.insert(0,str(U))
from normalize_glb_time import normalize
normalize(D/'motion.glb')
write('BUILD.json',dict(status='BAKED_PENDING_SOURCE_AND_RIG_REVIEW',clip=clip,geometry=geo,frames=nf,fps=30,rig='Original Ch29 Mixamo',bones=len(rig.pose.bones),arm_scales=scales,unreachable={s:sum(r['unreachable']for r in records if r['side']==s)for s in sides},max_bone_length_error=max(x['length_error']for x in metrics),max_finger_direction_transfer_error=max(x['angle_error_deg']for x in metrics),source_xyz_sha256=hashlib.sha256((U/clip/geo/'approximate_xyz.npz').read_bytes()).hexdigest(),corrections=cfg,limitations=['Estimated geometry, not measured 3D','Static torso and facial expression','Source-to-character contact/shape validation pending']))
print('BUILD_COMPLETE',clip,revision,flush=True)
