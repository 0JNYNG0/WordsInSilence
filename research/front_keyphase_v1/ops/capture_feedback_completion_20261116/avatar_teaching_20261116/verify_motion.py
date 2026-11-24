"""Verify exported skin joint motion against baked Blender motion, every native frame."""
from pathlib import Path
import json,struct,hashlib
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.signal import find_peaks
U=Path(__file__).resolve().parent
results={}
for word,rev,geo in [('WORD0099','guided_v2','calibration_seed_v2'),('WORD1290','baseline','video_only')]:
 D=U/('NIA_SL_'+word+'_REAL01')/rev;b=(D/'motion.glb').read_bytes();length=struct.unpack_from('<I',b,12)[0];j=json.loads(b[20:20+length]);blob=b[28+length:];p=np.load(D/'poses.npz');names=list(p['names']);nf=len(p['head']);sizes={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16};types={5126:'<f4',5123:'<u2',5125:'<u4',5121:'u1'}
 def acc(i):
  a=j['accessors'][i];v=j['bufferViews'][a['bufferView']];dt=np.dtype(types[a['componentType']]);n=sizes[a['type']];offset=v.get('byteOffset',0)+a.get('byteOffset',0);stride=v.get('byteStride',n*dt.itemsize)
  return np.ndarray((a['count'],n),dtype=dt,buffer=blob,offset=offset,strides=(stride,dt.itemsize)).copy()
 nodes=j['nodes'];byname={n.get('name'):i for i,n in enumerate(nodes)};world=[]
 channels=j['animations'][0]['channels'];samplers=j['animations'][0]['samplers'];ct=[(c,acc(samplers[c['sampler']]['input']).ravel(),acc(samplers[c['sampler']]['output']))for c in channels]
 for f in range(nf):
  ts=[np.array(n.get('translation',[0,0,0]),float)for n in nodes];qs=[np.array(n.get('rotation',[0,0,0,1]),float)for n in nodes];ss=[np.array(n.get('scale',[1,1,1]),float)for n in nodes]
  for c,times,values in ct:
   t=np.clip(f/30,times[0],times[-1]);path=c['target']['path']
   value=values[0]if len(times)==1 else Slerp(times,Rotation.from_quat(values))([t]).as_quat()[0]if path=='rotation'else np.array([np.interp(t,times,values[:,k])for k in range(values.shape[1])])
   {'translation':ts,'rotation':qs,'scale':ss}[path][c['target']['node']]=value
  mats=[None]*len(nodes)
  def walk(i,parent):
   m=np.eye(4);m[:3,:3]=Rotation.from_quat(qs[i]).as_matrix()@np.diag(ss[i]);m[:3,3]=ts[i]
   if 'matrix'in nodes[i]:m=np.array(nodes[i]['matrix']).reshape(4,4).T
   mats[i]=parent@m
   for c in nodes[i].get('children',[]):walk(c,mats[i])
  for root in j['scenes'][j.get('scene',0)]['nodes']:walk(root,np.eye(4))
  world.append([mats[byname[n]][:3,3]for n in names])
 world=np.array(world);h=p['head'];src=(np.c_[h.reshape(-1,3),np.ones(nf*len(names))]@p['rig_world'].T)[:,:3].reshape(h.shape);expected=src[:,:,[0,2,1]]*np.array([1,1,-1]);err=np.linalg.norm(world-expected,axis=-1);assert np.isfinite(world).all();assert err.max()<1e-5,err.max()
 lengths=np.linalg.norm(p['tail']-h,axis=-1);quat=p['local_quaternion'];dots=np.abs((quat[1:]*quat[:-1]).sum(-1));angles=np.degrees(2*np.arccos(np.clip(dots,0,1)));idx=lambda suffix:next(i for i,n in enumerate(names)if n.endswith(suffix))
 out=dict(native_frames=nf,rig_bones=len(names),export_max_head_position_error_m=float(err.max()),max_length_change_rig_units=float(np.ptp(lengths,axis=0).max()),max_adjacent_bone_rotation_degrees=float(angles.max()),p99_adjacent_rotation_degrees=float(np.percentile(angles,99)),glb_sha256=hashlib.sha256(b).hexdigest())
 if word=='WORD1290':
  # Right hand edge relative to left hand, in character body-up coordinates.
  y=h[:,idx('RightHandPinky1'),1]-h[:,idx('LeftHandMiddle1'),1];low,_=find_peaks(-y[47:81],prominence=.3,distance=4);out['right_edge_downward_turn_frames']=(low+47).tolist();out['relative_height_samples']={str(f):round(float(y[f]),3)for f in [47,54,61,68,74,80]}
 else:
  tip=p['tail'][:,idx('RightHandIndex3')];out['mouth_sweep_right_index_tip_xyz']={str(f):tip[f].round(3).tolist()for f in [43,50,55]};out['wipe_right_hand_body_height']={str(f):round(float(h[f,idx('RightHand'),1]),3)for f in [65,70,74,78]}
 results[word]=out
(U/'MOTION_VERIFICATION.json').write_text(json.dumps(dict(status='PASS_EXPORTED_MOTION_MATCHES_BAKE',scope='Native 30 fps export consistency; not external motion-capture or linguistic certification.',results=results),ensure_ascii=False,indent=2),'utf-8');print(json.dumps(results,ensure_ascii=False,indent=2))
