"""Freeze label-free boxes and candidate checks before any HaMeR image inference."""
from pathlib import Path
import json, hashlib, importlib.util, sys, datetime
import numpy as np
U=Path(__file__).resolve().parent; O=U.parent
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
sys.path.insert(0,str(O/'word100_yolo_selective_wrist_20261025'))
from common import native_m1, rule

def main():
    prep=read(O/'word100_3d_hand_pilot_preflight_20261025/PREPARED.json')
    old={r['clip_id']:r for r in read(O/'word100_yolo_selective_wrist_20261025/FROZEN.json')['old_rows']}
    inputs=U/'inputs'; inputs.mkdir(exist_ok=True)
    manifest=[]
    for m in prep['manifest']:
        cid=m['clip_id']; row=old[cid]
        for k,h in [('arrays','arrays_sha256'),('rgb_path','rgb_sha256')]:assert sha(m[k])==m[h]
        rec=read(row['source_assets']['detections'])['frames']
        with np.load(m['arrays'],allow_pickle=False) as z: a={k:z[k] for k in z.files}
        n=len(a['frame_id']);full=np.full((n,2,21,2),np.nan,np.float32);valid=np.zeros((n,2,21),bool)
        for t in range(n):full[t],valid[t]=native_m1(a,t,rec[t])
        boxes=np.full((n,2,4),np.nan,np.float32);origin=np.zeros((n,2),np.uint8);trigger=np.zeros((n,2),bool)
        roi=np.asarray(m['fixed_roi']);pose=a['raw_pose_xy_px'];usable=a['raw_pose_usable']
        for t in range(n):
            good,scale=rule.context(pose[t],usable[t],1920,1080)
            if scale<=1:continue
            for s in (0,1):
                p=full[t,s];v=valid[t,s] & np.isfinite(p).all(-1)
                if v.sum()>=6:
                    lo=p[v].min(0);hi=p[v].max(0)
                    if min(hi-lo)>=8:boxes[t,s]=[*lo,*hi];origin[t,s]=1
                if not origin[t,s] and good[[13+s,15+s]].all():
                    # Current body wrist/index is only an image-search hint, never hand truth.
                    w=pose[t,15+s].copy(); finger=pose[t,19+s]
                    direction=finger-w if good[19+s] else (w-pose[t,13+s])*.3
                    center=w+direction*.5;size=max(scale*.35,np.linalg.norm(direction)*2.5)
                    boxes[t,s]=[*np.maximum(center-size/2,0),*np.minimum(center+size/2,[1919,1079])];origin[t,s]=2
                wrist=pose[t,15+s]
                inroi=good[15+s] and roi[0]<=wrist[0]<roi[2] and roi[1]<=wrist[1]<roi[3]
                trigger[t,s]=not valid[t,s].any() and bool(inroi) and origin[t,s]>0
        requested=trigger.copy()
        # Same frozen 2-frame temporal context for every candidate, also around review anchors/crossing controls.
        for t,s in np.argwhere(trigger):requested[max(0,t-2):min(n,t+3),s]=True
        for t in m['anchors']+m['crossing_frames']:requested[max(0,t-3):min(n,t+4),:]=True
        requested &= origin>0
        dest=inputs/(cid+'.npz')
        if dest.exists():raise FileExistsError(dest)
        np.savez_compressed(dest,full_xy=full,full_valid=valid,boxes=boxes,box_origin=origin,
                            trigger=trigger,requested=requested,pose=pose,pose_valid=usable,
                            frame_id=a['frame_id'],timestamp_us=a['timestamp_us'],roi_box=roi,
                            baseline_xy=a['redetected_roi_xy_px'][:,33:].reshape(n,2,21,2),
                            baseline_valid=a['redetected_roi_usable'][:,33:].reshape(n,2,21))
        manifest.append(dict(m,prepared=str(dest),prepared_sha256=sha(dest),inference_hands=int(requested.sum()),
                             absent_targets=int(trigger.sum())))
    protocol={'at_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
      'status':'FROZEN_BEFORE_HAMER_IMAGE_INFERENCE','conditions':['B_saved_MP_redetection','H_selective_HaMeR','HT_H_plus_3D_temporal_checks'],
      'manifest':manifest,'protected':prep['protected'],
      'box_policy':'Existing full-view MP hand bounds when available; otherwise current body wrist/index hint. Original RGB crop; official rescale_factor=2.0; fixed final ROI unchanged.',
      'identity_policy':'Reuse current visible arm rule .25 shoulder distance/.10 margin and existing duplicate checks; never screen-left/right identity; ambiguous stays pending.',
      'B_preservation':'Every existing point remains bitwise unchanged. Only entirely absent full-view hands can receive estimates. ROI applied last.',
      'H_policy':'Finite predicted MANO hand inside source, positive camera, same current-arm identity as requested, no duplicate/ambiguous overlap with baseline. Additional points remain model estimates, not directly observed or validated fingers.',
      'HT_policy':{'context_each_side_frames':2,'requires_both_temporal_neighbors':True,'max_pose_rotation_radians_per_neighbor':1.0471975512,'max_relative_bone_length_change':0.25,'max_wrist_residual_shoulder':0.15,'method':'Verify raw 3D hand pose/bone consistency and wrist linear-prediction residual. No smoothing, interpolation, or forced pose replacement; hold additional estimates on unsupported transitions.'},
      'fast_change_policy':'High changes are held estimates, not learner errors. Existing observations retained regardless.',
      'scale':'Original pixel coordinates; no changed TCN normalization, no direct 3D feed to TCN.',
      'ground_truth':'No independent 21-point/hidden 3D truth. Original-image qualitative review before inference recorded; anatomical accuracy cannot be quantified from detector agreement.',
      'decision':'No deployment if clear false fill or unverifiable hidden-finger reconstruction. No automatic clinical/anatomical claim.',
      'human_or_user_data':False,'training':False,'production_change':False,
      'pins':{str(p):sha(p) for p in [Path(__file__),U/'runtime_support.py',O/'word100_fullframe_hand_merge_20261020/candidate.py']}}
    save(U/'FROZEN_PROTOCOL.json',protocol)
    # Original-only visual notes: this is AI qualitative review, not expert landmark annotation.
    notes=[
      '후원 REAL01: 양손이 중앙에서 겹침. 세운 손의 손목/손바닥 일부가 다른 손에 가려짐. 가려진 손가락 전체 정답은 이 화면만으로 확정 불가.',
      '후원 REAL06: 동작 손은 화면 왼쪽, 반대손은 아래. 동작 손은 굽힌 손가락/사선 방향이며 손바닥 전체가 정면으로 보이지 않음.',
      '남아 REAL02: 두 손은 서로 떨어져 있고 주먹/올린 엄지의 윤곽이 보임. 두 손을 같은 위치에 겹쳐 복원하면 오류.',
      '멀다 REAL06: 화면 왼쪽의 동작 손은 가슴 앞에서 굽혀짐. 반대손은 아래. 모든 손가락 마디는 독립적으로 읽기 어려움.',
      '대지 REAL06: 두 손이 분리되어 몸 앞에 수평으로 나옴. 손등 쪽 윤곽을 볼 수 있으나 겹친 손가락별 정밀 주석은 없음.',
      '수어 REAL01: 양손이 가슴 앞에서 가까움. 기존 정상 관측/손 방향을 훼손하지 않아야 함.',
      '후원 REAL03: 중앙에서 양손 겹침. 한 손의 세운 손가락과 다른 손의 잡는 모양, 팔 연결을 함께 확인해야 함.'
    ]
    save(U/'ORIGINAL_REVIEW_BEFORE_INFERENCE.json',{'source':'Codex image inspection, not sign expert or precise ground truth','notes':notes,'new_model_outputs_seen':False})
    print('FROZEN',sum(x['frames'] for x in manifest),'frames',sum(x['inference_hands'] for x in manifest),'hand crops',sum(x['absent_targets'] for x in manifest),'missing targets')
if __name__=='__main__':main()
