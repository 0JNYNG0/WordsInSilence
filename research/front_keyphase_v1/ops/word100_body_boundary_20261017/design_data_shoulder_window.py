"""TRAIN-only fixed median coefficient with the root's preregistered shoulder window.

No boundary/mask score is used to choose the coefficient. The coefficient is
the median of all eligible old stable-rest TRAIN clips, fixed before evaluating
the ten old failures. All observations are legacy; no detector or classifier.
"""
from pathlib import Path
import json
import sys
import math
import numpy as np
sys.dont_write_bytecode=True
import design_data_audit as prior
U=Path(__file__).resolve().parent

def shoulder_window(xy,valid,ts):
    end_prefix=int(np.searchsorted(ts, int(ts[0])+1_000_000, side='right'))
    inds=[11,12];ok=valid[:end_prefix,inds].all(1)
    for end in range(7,end_prefix):
        for start in range(end-6):
            if int(ts[end])-int(ts[start])<300_000:continue
            if not ok[start:end+1].all() or (np.diff(ts[start:end+1])>100_000).any():continue
            window=xy[start:end+1]
            scale=float(np.median(np.linalg.norm(window[:,11]-window[:,12],axis=1)))
            if not np.isfinite(scale) or scale<=0:continue
            med=np.median(window[:,inds],axis=0)
            movement=np.linalg.norm(window[:,inds]-med,axis=-1)
            if movement.max()>.05*scale:continue
            return dict(status='CALIBRATED_SHOULDERS',frame_indices=list(range(start,end+1)),
                        shoulder_width=scale,center=np.median((window[:,11]+window[:,12])*.5,axis=0).tolist(),
                        max_motion_px=float(movement.max()),duration_us=int(ts[end])-int(ts[start]))
    return dict(status='NOT_CALIBRATED_SHOULDERS')

def run_duration(mask,ts):
    ids=np.flatnonzero(mask)
    if not len(ids):return 0.
    groups=np.split(ids,np.where(np.diff(ids)!=1)[0]+1)
    dt=float(np.median(np.diff(ts))) if len(ts)>1 else 0.
    return max(float(ts[g[-1]]-ts[g[0]])+dt for g in groups)/1e6

def main():
    metadata=prior.read(prior.OPS/'word100_roi_training_20261017/root_audit_legacy_preflight_all_RESULT.json')
    old={r['clip_id']:r['calibration'] for r in metadata['results'] if r['role']=='train_candidate'}
    rows=[r for r in prior.read(prior.OPS/'word100_training_20261013/MANIFEST.json') if r['role']=='train_candidate']
    out=[];failure_arrays={}
    for row in rows:
        cid=row['clip_id'];rec=prior.read(prior.LEGACY/'records'/f'{cid}.json')
        with np.load(rec['observation'],allow_pickle=False) as z:
            xy=z['xy_px'].astype(float);valid=z['observed']&z['usable']&np.isfinite(xy).all(-1);ts=z['timestamp_us']
        sc=prior.read(Path(rec['observation']).with_suffix('.sidecar.json'));w,h=sc['width'],sc['height']
        # Root's requested observed+usable finite shoulder contract; no new thresholds.
        sw=shoulder_window(xy,valid,ts)
        entry={'clip_id':cid,'signer_id':row['signer_id'],'shoulder_window':sw,'old_status':old[cid]['status']}
        if old[cid]['status']=='CALIBRATED' and sw['status']=='CALIBRATED_SHOULDERS':
            entry['old_bottom']=old[cid]['bottom']
            entry['ratio']=(old[cid]['bottom']-sw['center'][1])/sw['shoulder_width']
        if old[cid]['status']!='CALIBRATED':failure_arrays[cid]=(row,xy,valid,ts,w,h)
        out.append(entry)
    coefficients=[r['ratio'] for r in out if 'ratio' in r]
    assert len(coefficients)==889
    k=float(np.median(coefficients))
    failure_results=[]
    for r in out:
        if r['old_status']=='CALIBRATED':continue
        row,xy,valid,ts,w,h=failure_arrays[r['clip_id']]
        sw=r['shoulder_window'];assert sw['status']=='CALIBRATED_SHOULDERS'
        bottom=math.floor(sw['center'][1]+k*sw['shoulder_width'])
        assert 0<bottom<=h
        mask=(xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<bottom)
        masked=valid&mask
        a,b=row['intervals'][0];core=(ts>=round(a*1e6))&(ts<=round(b*1e6))
        context=(ts>=round((a-.3)*1e6))&(ts<=round((b+.3)*1e6))
        oldany=valid[:,33:75].any(1);newany=masked[:,33:75].any(1)
        rec=dict(r,proposed_bottom=bottom,source_height=h,
                 core_frames=int(core.sum()),core_anyhand_before=int((core&oldany).sum()),
                 core_anyhand_after=int((core&newany).sum()),
                 core_both_hands_absent_after=int((core&~newany).sum()),
                 core_newly_all_absent_run_seconds=run_duration(core&oldany&~newany,ts),
                 context_observed_hand_frames=int((context&newany).sum()),
                 total_observed_hand_frames_after=int(newany.sum()),sides={})
        for name,lo in [('left',33),('right',54)]:
            sl=slice(lo,lo+21);oa=valid[:,sl].any(1);na=masked[:,sl].any(1);palm=lo+np.array([0,5,9,13,17])
            first=(ts-ts[0])<=500_000
            last=(ts[-1]-ts)<=500_000
            rec['sides'][name]={'core_newly_absent_run_seconds':run_duration(core&oa&~na,ts),
                'core_palm5_before':int(valid[core][:,palm].all(1).sum()),
                'core_palm5_after':int(masked[core][:,palm].all(1).sum()),
                'core_newly_absent_frame_ids':np.flatnonzero(core&oa&~na).tolist(),
                'first_half_second_visible_hand_points':int(masked[first,sl].sum()),
                'last_half_second_visible_hand_points':int(masked[last,sl].sum())}
        failure_results.append(rec)
    result={'status':'TRAIN_ONLY_PREREGISTERED_MEDIAN_BODY_FALLBACK_DIAGNOSTIC',
            'selection_rule':'Root chose median before this matching-window result. No score/label/dev tuning.',
            'coefficient':k,'coefficient_population':889,'coefficient_quantiles_descriptive':prior.quant(coefficients),
            'shoulder_window_available':sum(r['shoulder_window']['status']=='CALIBRATED_SHOULDERS' for r in out),
            'old_stable_preserved':889,'fallback_train_failures':len(failure_results),
            'training':0,'detector':0,'dev_arrays_read':0,'shoulder_rows':out,'fallback_diagnostics':failure_results,
            'limitations':'Legacy detector geometry only. Source rest positions and shoulders vary by performer. Fallback may retain resting hands or mask lower action; visual and actual-crop checks required. No classification improvement claim.'}
    with (U/'design_data_shoulder_window_RESULT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='shoulder_rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
