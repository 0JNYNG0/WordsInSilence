"""Existing FIT206 point/role contract only; no new labels, model or threshold."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from collections import defaultdict
import hashlib
import json
import numpy as np

U = Path(__file__).resolve().parent
H = U.parent
T = H / 'typed_shape_transition_review'


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, float)
    return dict(n=len(a), minimum=float(a.min()), median=float(np.median(a)), maximum=float(a.max()))


def main():
    assert not (U/'BODY_POINT_CONTRACT.json').exists()
    dense = read(T/'REVIEW_COMPLETE.json')
    source = {r['clip_id']: r for r in read(H/'palm_preserving_controls/FIXTURE_FROZEN.json')['sources']}
    prior = read(U/'EVIDENCE.json')
    pins = {str(p): sha(p) for p in [Path(__file__), T/'REVIEW_COMPLETE.json', T/'PREPARED.json', U/'EVIDENCE.json']}
    frame_rows, segment_rows = [], []
    for row in dense['rows']:
        raw = source[row['clip_id']]
        assert sha(raw['raw_path']) == raw['raw_sha256']
        pins[raw['raw_path']] = raw['raw_sha256']
        with np.load(raw['raw_path'], allow_pickle=False) as z:
            xy = z['xy_px'].astype(float)
            fid, ts = z['frame_id'], z['timestamp_us']
            width, height = int(z['source_width']), int(z['source_height'])
        lookup = {int(f): i for i, f in enumerate(fid)}
        available = np.isfinite(xy).all(2) & (xy[...,0]>=0) & (xy[...,0]<width) & (xy[...,1]>=0) & (xy[...,1]<height)
        shoulders = available[:,5] & available[:,6]
        scale = float(np.median(np.linalg.norm(xy[shoulders,5]-xy[shoulders,6],axis=1)))
        centre = (xy[:,5]+xy[:,6])/2
        p = (xy-centre[:,None])/scale
        queries = dict(body_wrist=p[:,10], hand_root=p[:,112], stable_palm=p[:,[112,117,121,125,129]].mean(1))
        masks = dict(body_wrist=available[:,10]&shoulders, hand_root=available[:,112]&shoulders,
                     stable_palm=available[:,[112,117,121,125,129]].all(1)&shoulders)
        arm = p[:,10]-p[:,8]
        length = np.linalg.norm(arm,axis=1)
        unit = arm/np.maximum(length[:,None],1e-5)
        for record in row['frame_records']:
            i=lookup[record['frame']]
            frame_rows.append(dict(clip_id=row['clip_id'],frame=int(fid[i]),timestamp_us=int(ts[i]),
                existing_body_label=record['body_motion'], existing_hand_label=record['hand_state'],
                shoulder_origin_px=centre[i].tolist(), shoulder_scale_px=scale,
                query_xy={k:v[i].tolist() for k,v in queries.items()}, available={k:bool(v[i]) for k,v in masks.items()},
                body_elbow_xy=p[i,8].tolist(),body_forearm_unit=unit[i].tolist(),
                body_hand_root_separation=float(np.linalg.norm(p[i,10]-p[i,112]))))
        for segment in prior['dense_contiguous_segments']:
            if segment['clip_id']!=row['clip_id']:
                continue
            a,b=[lookup[f] for f in segment['source_frames']]
            q={}
            for key,values in queries.items():
                valid=masks[key][a:b+1]
                edges=valid[:-1]&valid[1:]
                steps=np.diff(values[a:b+1],axis=0)[edges]
                q[key]=dict(endpoint_available=bool(valid[0]and valid[-1]),observed=int(valid.sum()),
                    delta_xy=(values[b]-values[a]).tolist(),path=float(np.linalg.norm(steps,axis=1).sum()),
                    dy_positive_steps=int((steps[:,1]>0).sum()),dy_negative_steps=int((steps[:,1]<0).sum()),
                    adjacent_steps=len(steps))
            u,v=unit[a:b],unit[a+1:b+1]
            turns=np.degrees(np.arctan2(u[:,0]*v[:,1]-u[:,1]*v[:,0],(u*v).sum(1)))
            segment_rows.append(dict(clip_id=row['clip_id'],person=row['person'],source_frames=segment['source_frames'],
                existing_body_label=segment['existing_rgb_body_label'],queries=q,
                elbow_delta_xy=(p[b,8]-p[a,8]).tolist(),forearm_upright_delta=float(unit[a,1]-unit[b,1]),
                forearm_turn_degrees=float(turns.sum()),semantic_direction_or_correctness=None))
    summary=[]
    for label in ('wrist_lowering','forearm_raising','hold'):
        rows=[r for r in segment_rows if r['existing_body_label']==label]
        summary.append(dict(existing_label=label,segments=len(rows),
            dy={key:stats([r['queries'][key]['delta_xy'][1] for r in rows]) for key in ('body_wrist','hand_root','stable_palm')},
            forearm_upright_delta=stats([r['forearm_upright_delta']for r in rows])))
    assert len(frame_rows)==206
    for path,expected in pins.items():
        assert sha(path)==expected
    result=dict(status='FIT206_POINT_COMPARISON_COMPLETE_NO_RELABEL',frame_rows=frame_rows,segment_rows=segment_rows,summary=summary,
        measurement='Same source shoulder-centre origin and clip median shoulder-width scale; body6=RTMW10, handroot30=RTMW112, stablepalm=mean RTMW112/117/121/125/129. No fingertips.',
        signs='Image y increases downward. Report signed continuous changes, not thresholds/classes. Forearm upright=-unit_y of body wrist minus elbow.',
        no_exclusive_axis_negatives=True,labels_unchanged=True,no_model_inference=True,no_training=True,pins=pins)
    (U/'BODY_POINT_CONTRACT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    for row in segment_rows:
        if row['existing_body_label']!='hold':
            print(row['clip_id'],row['existing_body_label'],row['source_frames'],
                'dy body/root/palm',*[round(row['queries'][k]['delta_xy'][1],4)for k in ('body_wrist','hand_root','stable_palm')],
                'upright',round(row['forearm_upright_delta'],4))
    print('206 frames,',len(segment_rows),'existing-label runs; no new labels')


if __name__=='__main__':
    main()
