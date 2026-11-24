import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
U=Path(__file__).resolve().parent
def norm(q):return q/np.linalg.norm(q,axis=-1,keepdims=True)
def angle(a,b):return np.degrees(2*np.arccos(np.clip(np.abs(np.sum(norm(a)*norm(b),axis=-1)),0,1)))
def stats(x):
    x=np.asarray(x,dtype=float)
    return {'mean':float(np.mean(x)),'p95':float(np.percentile(x,95)),'max':float(np.max(x))}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
A=np.load(U/'baseline_poses.npz');B=np.load(U/'candidate_poses.npz');assert np.array_equal(A['names'],B['names']);names=list(A['names']);ni=A['native_indices']
ix=lambda n:names.index('mixamorig1:'+n)
wr=[ix(s+'Hand')for s in ['Left','Right']];arms=[ix(s+k)for s in ['Left','Right']for k in ['Arm','ForeArm','Hand']]
fingers=[i for i,n in enumerate(names)if any('Hand'+f in n for f in ['Thumb','Index','Middle','Ring','Pinky'])]
driven=[names.index(r['bone'])for r in json.loads((U/'CANDIDATE_BUILD.json').read_text())['records']]
out={'status':'NUMERIC_COMPARISON_COMPLETE_NOT_KSL_VALIDATION','units':'pose positions rig local; multiplied by rig_world linear scale for rendered-world displacement','source_frames':127,'quarter_samples':505,'frozen_sha256':sha(U/'FROZEN.json'),'scope':'one approximate public clip; no anatomical/contact truth'}
native_delta=angle(A['local_quaternion'][ni],B['local_quaternion'][ni]);global_delta=angle(A['global_quaternion'][ni],B['global_quaternion'][ni])
pos_delta=(B['head'][ni]-A['head'][ni])@A['rig_world'][:3,:3].T
tail_delta=(B['tail'][ni]-A['tail'][ni])@A['rig_world'][:3,:3].T
out['native_protected']={'nonwrist_local_rotation_delta_deg':stats(native_delta[:,[i for i in range(65)if i not in wr]]),'wrist_local_rotation_delta_deg':stats(native_delta[:,wr]),'arm_wrist_head_world_displacement':stats(np.linalg.norm(pos_delta[:,arms],axis=-1)),'finger_local_rotation_delta_deg':stats(native_delta[:,fingers]),'finger_world_rotation_delta_deg':stats(global_delta[:,fingers]),'finger_head_world_displacement':stats(np.linalg.norm(pos_delta[:,fingers],axis=-1)),'finger_tail_world_displacement':stats(np.linalg.norm(tail_delta[:,fingers],axis=-1))}
out['bone_length']={label:stats(np.abs(np.linalg.norm(z['tail']-z['head'],axis=-1)-np.linalg.norm(z['tail'][0]-z['head'][0],axis=-1)))for label,z in [('baseline',A),('candidate',B)]}
out['quarter_path']={}
for label,z in [('baseline',A),('candidate',B)]:
    steps=angle(z['global_quaternion'][1:],z['global_quaternion'][:-1]);where=np.unravel_index(np.argmax(steps),steps.shape)
    local_steps=angle(z['local_quaternion'][1:],z['local_quaternion'][:-1])
    out['quarter_path'][label]={'all_bone_world_step_deg':stats(steps),'driven_local_step_deg':stats(local_steps[:,driven]),'world_steps_over_45deg':int((steps>45).sum()),'world_steps_over_90deg':int((steps>90).sum()),'largest_world_step':{'source_frame_start':float(z['source_frames'][where[0]]),'bone':names[where[1]],'degrees':float(steps[where])},'minimum_evaluated_local_quaternion_norm':float(np.min(np.linalg.norm(z['local_quaternion'],axis=-1))),'wrist_world_step_deg':stats(steps[:,wr])}
out['wrist_window']={}
for side,i in zip(['Left','Right'],wr):
    out['wrist_window'][side]={}
    for label,z in [('baseline',A),('candidate',B)]:
        q=norm(z['local_quaternion'][ni,i]);q=q[:,[1,2,3,0]]
        rot=Rotation.from_quat(q);rv=(rot[:-1].inv()*rot[1:]).as_rotvec()*180/np.pi
        out['wrist_window'][side][label]={'source_frame_88_to_96_step_deg':stats(np.linalg.norm(rv[88:96],axis=-1)),'source_frame_88_to_96_acceleration_deg_per_frame2':stats(np.linalg.norm(np.diff(rv[88:96],axis=0),axis=-1)),'all_native_step_deg':stats(np.linalg.norm(rv,axis=-1))}
out['dynamic_guard']={}
for label,z in [('baseline',A),('candidate',B)]:
    wrists=z['head'][ni][:,wr];order=np.sign(wrists[:,0,1]-wrists[:,1,1]);cross=np.flatnonzero(order[1:]!=order[:-1])+1
    tips=np.array([[z['tail'][ni,ix(s+'Hand'+f+'3')]for f in ['Thumb','Index','Middle','Ring','Pinky']]for s in ['Left','Right']]).transpose(2,0,1,3)
    distances=np.linalg.norm((tips[:,0,:,None,:]-tips[:,1,None,:,:])@z['rig_world'][:3,:3].T,axis=-1)
    closest=distances.reshape(127,25).min(axis=1);lowest=int(closest.argmin())
    out['dynamic_guard'][label]={'wrist_vertical_order_cross_source_frames':cross.tolist(),'source_48_to72_wrist_axis_amplitudes_rig_units':np.ptp(wrists[48:73],axis=0).tolist(),'nearest_twohand_fingertips_global_min_source_frame':lowest,'nearest_twohand_fingertips_min_world_distance':float(closest[lowest]),'per_fingerpair_minimum_source_frames':distances.argmin(axis=0).tolist()}
    np.savez_compressed(U/(label+'_distance_diagnostics.npz'),source_frames=np.arange(127),fingertip_distances_world=distances,nearest_tip_distance_world=closest,wrist_vertical_order=order)
out['notes']=['Same bone local finger rotations preserve driven joint shape; wrist smoothing still changes descendant world directions and positions.','No threshold-defined contact labels were invented. Closest-tip times are geometric diagnostics, not physical contact truth.','Centered offline smoothing has no deliberate frame shift; sample timestamps are unchanged, but no-lag cannot be inferred for every unseen motion.','Quaternion coordinate sign flips are not pose changes; sign continuity prevents interpolation across opposite quaternion representations.','Large quarter-frame values may reflect actual source jumps or unsmoothed finger/arm noise; smoother wrist does not validate source3D.']
(U/'MEASURED_COMPARISON.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n','utf-8')
print(json.dumps(out,indent=2))
