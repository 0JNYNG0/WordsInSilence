"""One new public endpoint, using unchanged generic fullframe/crop merge.

All raw candidates, including side=-1 duplicates, participate in identity checks.
The original outputs are read-only. No posteriors are promoted using new masks.
"""
from pathlib import Path
import sys, importlib.util, json, hashlib, time, traceback, os
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(n,v):
    with (U/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def module(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    assert not (U/'ONE_RECOVERY_RESULT.json').exists() and not (U/'ONE_RECOVERY_STARTED.json').exists()
    old=O/'word100_observation_lineage_repair_20261101';p=old/'recovery.py'
    rec=module(p,'unchanged_public_recovery')
    f0=read(old/'RECOVERY_FROZEN.json')
    for v in [p,rec.MERGE_PATH,rec.OBSERVER_PATH,rec.MODEL,rec.CONTRACT]:assert sha(v)==f0['pins'][str(v)]
    row=read(O/'word100_action_feedback_casebook_20261101/INPUTS.json')['rows']['P065']
    record=read(row['record']);cap=record['native_capture']
    files=dict(native=row['observation'],raw=cap['raw'],assignment=str(Path(cap['candidate']).with_name('assignment.npz')),
               trace=str(Path(cap['candidate']).with_name('trace.json')),input=row['input'],video=row['source'])
    n=arrays(files['native']);raw=arrays(files['raw']);sides=arrays(files['assignment'])['sides'];base=arrays(files['input'])
    i=48;fid=48;pts=1600000
    for x in [n,raw,base]:assert int(x['frame_id'][i])==fid and int(x['timestamp_us'][i])==pts
    w,h=int(raw['width'][i]),int(raw['height'][i]);count=int(raw['raw_count'][i]);bs=sides[i,:count].tolist()
    eligible=rec.eligibility(n['observed'][i],raw['pose_xy_px'][i],raw['pose_usable'][i],row['box'],w,h,bs)
    assert [x['side'] for x in eligible if x['eligible']]==[0]
    assert not n['observed'][i,33:54].any() and n['observed'][i,54:75].all()
    prep=read(old/'RECOVERY_PREPARED.json');assert not any(x['alias']=='P065' for x in prep['clips'])
    assert row['clip_id'] not in [x['clip_id'] for x in read(rec.OLD/'FROZEN.json')['rows']]
    pins={str(v):sha(v) for v in [Path(__file__),p,rec.MERGE_PATH,rec.OBSERVER_PATH,rec.MODEL,rec.CONTRACT,*map(Path,files.values()),Path(row['record'])]}
    write('FROZEN_ONE_RECOVERY.json',dict(status='FROZEN_BEFORE_NEW_DETECTOR_CALL',alias='P065',frame_id=fid,timestamp_us=pts,
       scope='only existing EW01 reference focal; fix runtime missing observation, not labels or expected word',
       eligibility=eligible,baseline_raw_count=count,baseline_sides=bs,parameters=rec.rule.PARAMS,pins=pins,
       completed_cache_match=False,existing_hands_preserved=True,existing_posteriors_baseline_only=True,
       first_error_retained=True,new_classifier=0,new_training=0,new_dtw=0))
    write('ONE_RECOVERY_STARTED.json',dict(pid=os.getpid(),start=time.time()))
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    options=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(rec.MODEL)),running_mode=vision.RunningMode.IMAGE,
       num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    t0=time.perf_counter();lm=vision.HandLandmarker.create_from_options(options);init=time.perf_counter()-t0
    try:
        frame,ts,rgb=next(rec.selected_rgb(files['video'],{fid:pts}));assert rgb.shape[:2]==(h,w)
        full,fs=rec.detect(lm,rgb)
        boxes=rec.rule.crop_boxes(raw['pose_xy_px'][i],raw['pose_usable'][i],bs,w,h)
        crops=[];cs=0.;calls=0
        for side,box in boxes.items():
            if side!=0:continue
            got,t=rec.detect(lm,rgb[box[1]:box[3],box[0]:box[2]],box)
            for d in got:d['requested_side']=side
            crops.extend(got);cs+=t;calls+=1
        before={k:n[k][i] for k in ['xy_px','observed','usable','conf','hand_score']}
        merged,ledger=rec.merge_frame(before,raw['hand_xy_px'][i,:count],bs,raw['pose_xy_px'][i],raw['pose_usable'][i],w,h,row['box'],full,crops,
                                     read(rec.CONTRACT)['usable_rule']['hand_score_threshold'])
    finally:lm.close()
    cand={k:v.copy() for k,v in n.items()}
    for k,v in merged.items():cand[k][i]=v
    # Original confident pose/right hand and all other native frames unchanged.
    for k in ['xy_px','observed','usable','conf','hand_score']:
        assert np.array_equal(cand[k][:i],n[k][:i],equal_nan=True)
        assert np.array_equal(cand[k][i+1:],n[k][i+1:],equal_nan=True)
    cand['usable'] &= rec.inside_box(cand['xy_px'],row['box'])
    observer=module(rec.OBSERVER_PATH,'recovered_endpoint_observer')
    feat=observer.extract_features(cand)
    x,m,discarded=rec.lock_candidate_features(base['x'],base['mask'],np.nan_to_num(feat['x']).astype(np.float32),feat['mask'],np.array([i]))
    gained=m[i]&~base['mask'][i]
    with (U/'P065_f48_recovered.npz').open('xb') as f:
        np.savez_compressed(f,x=x[i:i+1],mask=m[i:i+1],baseline_x=base['x'][i:i+1],baseline_mask=base['mask'][i:i+1],
            frame_id=np.array([fid]),timestamp_us=np.array([pts]),xy_px=cand['xy_px'][i:i+1],observed=cand['observed'][i:i+1],usable=cand['usable'][i:i+1],
            original_xy_px=n['xy_px'][i:i+1],original_observed=n['observed'][i:i+1],original_usable=n['usable'][i:i+1])
    for pp,hh in pins.items():assert sha(pp)==hh
    write('ONE_RECOVERY_RESULT.json',dict(status='COMPLETE_NEW_INPUT_PENDING_VISUAL_VERIFICATION',alias='P065',frame_id=fid,timestamp_us=pts,
         source=row,eligibility=eligible,full=full,crops=crops,boxes=boxes,merge=ledger,
         added_feature_cells=int(gained.sum()),preserved_feature_cells=int(base['mask'][i].sum()),
         gained_feature_names=[v for j,v in enumerate(row['feature_names']) if gained[j]],out_of_scope_derivative_support_discarded=discarded,
         init_seconds=init,full_seconds=fs,crop_seconds=cs,full_calls=1,crop_calls=calls,
         total_wall_seconds=time.perf_counter()-t0,new_classifier=0,new_training=0,new_dtw=0,
         old_prediction_masks_not_promoted=True,output_sha256=sha(U/'P065_f48_recovered.npz'),landmark_accuracy=None))
    print(json.dumps(dict(added_features=int(gained.sum()),additions=ledger['additions'],decisions=[q['reason'] for q in ledger['decisions']],full_seconds=fs,crop_seconds=cs)))
if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'ONE_RECOVERY_FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
