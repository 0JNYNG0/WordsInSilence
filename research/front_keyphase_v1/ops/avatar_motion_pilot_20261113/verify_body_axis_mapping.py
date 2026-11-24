"""Independent numeric source-to-rig wrist comparison; not correctness of reconstructed source3D."""
import bpy,json,numpy as np
from pathlib import Path
U=Path(__file__).resolve().parent
data=np.load(U/'video_only/approximate_xyz.npz');ids=list(data['joints']);p=data['xyz'];q=lambda j:p[:,ids.index(j)]
up=np.median((q(5)+q(6))/2-(q(11)+q(12))/2,axis=0)
def norm(v):return v/np.linalg.norm(v)
def basis(left,right,vertical):
    x=norm(left-right);y=norm(vertical-x*np.dot(vertical,x));return np.column_stack((x,y,np.cross(x,y)))
out={};examples=[]
for variant in ['public_approximate_preview','public_approximate_preview_body_axis']:
    D=U/variant;bake=json.loads((D/'BAKE_RECORD.json').read_text())
    bpy.ops.wm.open_mainfile(filepath=str(D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.blend'),load_ui=False,use_scripts=False)
    rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');bones={b.name.split(':')[-1]:b for b in rig.data.bones};poses={b.name.split(':')[-1]:b for b in rig.pose.bones}
    left=np.array(bones['LeftArm'].head_local);right=np.array(bones['RightArm'].head_local);center=(left+right)/2
    rb=basis(left,right,center-np.array(bones['Hips'].head_local));width=np.linalg.norm(left-right)
    rows=[]
    for f in range(127):
        bpy.context.scene.frame_set(f+1);bpy.context.view_layer.update();sb=basis(q(5)[f],q(6)[f],up)
        for side,s,h in [('Left',5,91),('Right',6,112)]:
            target=np.array(bones[side+'Arm'].head_local)+rb@sb.T@(q(h)[f]-q(s)[f])*bake['arm_scales'][side]
            actual=np.array(poses[side+'Hand'].head);targetlocal=(target-center)@rb/width;actual_local=(actual-center)@rb/width
            rows.append(dict(frame=f,side=side,target_body_xyz=targetlocal.tolist(),actual_body_xyz=actual_local.tolist(),height_error_shoulder_widths=float(actual_local[1]-targetlocal[1])))
            if f in [48,64,96]:examples.append(dict(variant=variant,**rows[-1]))
    err=np.array([r['height_error_shoulder_widths']for r in rows]);out[variant]=dict(median_absolute_wrist_height_error_shoulder_widths=float(np.median(abs(err))),max_absolute_error=float(np.max(abs(err))),rows=rows)
angles=[]
for f in range(127):
    a=basis(q(5)[f],q(6)[f],q(0)[f]-(q(5)[f]+q(6)[f])/2)[:,1];b=basis(q(5)[f],q(6)[f],up)[:,1]
    angles.append(float(np.degrees(np.arccos(np.clip(a@b,-1,1)))))
result=dict(status='BODY_AXIS_IMPLEMENTATION_COMPARISON',meaning='Checks expected same source shoulder-relative wrist targets, not true source calibration or sign teaching accuracy',
            original_nose_vs_torso_axis_angle_deg=dict(median=float(np.median(angles)),min=min(angles),max=max(angles)),variants=out,examples=examples,
            remaining='Cartoon head/arm/torso proportions differ. Preserved shoulder-relative wrist targets do not guarantee forehead/chest contact semantics match the human.')
(U/'BODY_AXIS_MAPPING_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:{kk:vv for kk,vv in v.items()if kk!='rows'}for k,v in out.items()}));print('AXIS_ANGLE',result['original_nose_vs_torso_axis_angle_deg'])
