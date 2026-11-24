"""Read-only TRAIN-only geometric review prioritization; never a semantic PASS."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent
OUTPUT=U/'training_geometry_triage_RESULT.json'
# Predeclared review heuristics, NOT ROI changes, recognition thresholds, or acceptance gates.
RULES=dict(core='original TRAIN annotation [a,b], no extra context',
           long_absence_seconds=.30, minimum_reentry_support_frames=4,
           low_motion_p90_p10_shoulder_width=.20, mostly_absent_fraction=.90,
           native_visibility_min_fraction=.80, side_offset_min_shoulder_width=.40,
           near_boundary_band_shoulder_width=.35,
           movement_range_shoulder_width=.35, movement_speed_shoulder_width_per_second=.50,
           max_adjacent_gap_seconds=.10,
           categories_overlap=True,
           priority='simultaneous loss >=.30s; moving near boundary; lower/reentry; other; low-motion-below',
           linguistic_semantics='Not inferred. A low-motion support hand may be essential.')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def runs(mask,ts,fid):
    idx=np.flatnonzero(mask)
    if not len(idx):return []
    step=float(np.median(np.diff(ts))) if len(ts)>1 else 0.
    groups=np.split(idx,np.flatnonzero(np.diff(idx)>1)+1)
    return [dict(start_index=int(g[0]),end_index=int(g[-1]),
                 start_frame_id=int(fid[g[0]]),end_frame_id=int(fid[g[-1]]),
                 start_seconds=float(ts[g[0]])/1e6,end_seconds=float(ts[g[-1]])/1e6,
                 duration_seconds=float(ts[g[-1]]-ts[g[0]]+step)/1e6,
                 frames=int(len(g))) for g in groups]


def robust_range(x):
    if len(x)<4:return None
    return float(np.linalg.norm(np.percentile(x,90,axis=0)-np.percentile(x,10,axis=0)))


def percent(x): return 100*float(np.mean(x)) if len(x) else None


def hand_info(start,xy,old,new,core,ts,fid,bottom,shoulder_center,scale):
    ids=np.arange(start,start+21);palm=start+np.array([0,5,9,13,17])
    native=old[:,ids].any(1);kept=new[:,ids].any(1)
    lost=core&native&~kept
    native_palm=old[:,palm].all(1);kept_palm=new[:,palm].all(1)
    palm_drop=percent(native_palm[core])-percent(kept_palm[core])
    absence=runs(lost,ts,fid)
    center=np.full((len(ts),2),np.nan)
    for i in np.flatnonzero(core):
        valid=old[i,palm]
        if valid.sum()>=3:
            center[i]=np.median(xy[i,palm][valid],axis=0)
    valid=np.isfinite(center).all(1)&core
    rel=(center-shoulder_center)/scale
    excursion=robust_range(rel[valid])
    lost_excursion=robust_range(rel[valid&lost])
    near=valid&(np.abs(center[:,1]-bottom)<=RULES['near_boundary_band_shoulder_width']*scale)
    dt=np.diff(ts)/1e6
    consecutive=valid[1:]&valid[:-1]&(dt>0)&(dt<=RULES['max_adjacent_gap_seconds'])
    speed=np.linalg.norm(np.diff(rel,axis=0),axis=1)/np.maximum(dt,1e-9)
    near_pairs=consecutive&(near[1:]|near[:-1])
    near_speed=float(np.percentile(speed[near_pairs],75)) if near_pairs.any() else None
    lower_center_fraction=float(np.mean(center[valid,1]>=bottom)) if valid.any() else None
    side_offset=float(np.median(np.abs(rel[valid,0]))) if valid.any() else None
    observed_fraction=float(np.mean(native[core]));kept_fraction=float(np.mean(kept[core]))
    reentries=[]
    for gap in absence:
        a,b=gap['start_index'],gap['end_index']
        before=int(np.sum(core[:a]&kept[:a]));after=int(np.sum(core[b+1:]&kept[b+1:]))
        if gap['duration_seconds']>=RULES['long_absence_seconds'] and min(before,after)>=4:
            reentries.append({**gap,'kept_before_frames':before,'kept_after_frames':after})
    a_pattern=bool(observed_fraction>=.8 and kept_fraction<=.1 and excursion is not None and
                   excursion<=.2 and lower_center_fraction is not None and lower_center_fraction>=.9 and
                   side_offset>=.4)
    moving_near=bool(near.any() and (lost.any() or palm_drop>=10) and (
                    (excursion is not None and excursion>=.35) or
                    (near_speed is not None and near_speed>=.5)))
    evidence=[]
    if lost.any():
        candidates=np.flatnonzero(lost&valid)
        if len(candidates):
            # Choose maximum center displacement from its core median, not a classifier score.
            med=np.median(rel[valid],axis=0)
            i=int(candidates[np.argmax(np.linalg.norm(rel[candidates]-med,axis=1))])
            evidence.append(dict(frame_id=int(fid[i]),seconds=float(ts[i])/1e6,
                                 kind='native_palm_observed_but_entire_hand_removed',
                                 center_pixels=center[i].tolist(),bottom_pixels=bottom))
    if moving_near and near_pairs.any():
        candidates=np.flatnonzero(near_pairs)
        i=int(candidates[np.argmax(speed[candidates])]+1)
        evidence.append(dict(frame_id=int(fid[i]),seconds=float(ts[i])/1e6,
                             kind='moving_palm_near_fixed_boundary',center_pixels=center[i].tolist(),
                             speed_shoulder_width_per_second=float(speed[i-1]),bottom_pixels=bottom))
    return dict(native_core_visible_percent=100*observed_fraction,
                retained_core_visible_percent=100*kept_fraction,
                newly_absent_core_frames=int(lost.sum()),newly_absent_runs=absence,
                longest_new_absence_seconds=max([g['duration_seconds'] for g in absence],default=0.),
                palm5_drop_pp=palm_drop,valid_palm_center_frames=int(valid.sum()),
                robust_core_movement_shoulder_width=excursion,
                robust_lost_interval_movement_shoulder_width=lost_excursion,
                near_boundary_palm_frames=int(near.sum()),near_boundary_speed_p75=near_speed,
                palm_center_below_boundary_fraction=lower_center_fraction,
                median_absolute_side_offset_shoulder_width=side_offset,
                a_low_motion_side_hand_below_most_core=a_pattern,
                b_lowered_then_reentered=bool(reentries),reentry_runs=reentries,
                d_moving_hand_near_boundary=moving_near,evidence_frames=evidence),native,kept,lost


def main():
    started=time.perf_counter()
    assert not OUTPUT.exists(),'Preserve first triage; review before replacing'
    context_path=U/'PREP_CONTEXT.json';forecast_path=U/'LEGACY_PREFLIGHT.json'
    context=read(context_path);forecast=read(forecast_path)
    assert forecast['context_sha256']==sha(context_path)
    # Read role first; only TRAIN row fields/annotation and TRAIN observation files are accessed.
    train=[r for r in context['ordered_rows'] if r['role']=='train_candidate']
    forecasts={r['clip_id']:r for r in forecast['rows'] if r['role']=='train_candidate'}
    assert len(train)==899 and set(forecasts)=={r['clip_id'] for r in train}
    selected=[r for r in train if forecasts[r['clip_id']]['geometry_flags']]
    assert len(selected)==619
    pins={str(p.resolve()):sha(p) for p in (context_path,forecast_path,Path(__file__))}
    result=[]
    for row in selected:
        cid=row['clip_id'];assert 1<=row['signer_id']<=9
        f=forecasts[cid];assert f['calibration']=='CALIBRATED'
        rp=Path(row['legacy_record_path']);record=read(rp)
        assert record['clip_id']==cid and record['role']=='train_candidate'
        op=Path(record['observation']);assert sha(op)==record['observation_sha256']
        pins[str(rp.resolve())]=sha(rp);pins[str(op.resolve())]=record['observation_sha256']
        with np.load(op,allow_pickle=False) as z:
            xy=z['xy_px'];usable=z['usable'];observed=z['observed'];ts=z['timestamp_us'];fid=z['frame_id']
        assert np.all(np.diff(ts)>0)
        sidecar=read(op.with_suffix('.sidecar.json'))
        width=int(sidecar['width']);bottom=int(f['bottom'])
        old=usable&observed&np.isfinite(xy).all(-1)
        inside=(xy[:,:,0]>=0)&(xy[:,:,0]<width)&(xy[:,:,1]>=0)&(xy[:,:,1]<bottom)
        new=old&inside
        a,b=row['intervals'][0];core=(ts>=round(a*1e6))&(ts<=round(b*1e6))
        assert core.any()
        sv=core&old[:,11]&old[:,12]
        assert sv.any(),'No usable core shoulders: '+cid
        widths=np.linalg.norm(xy[sv,11]-xy[sv,12],axis=1)
        scale=float(np.median(widths));assert scale>0
        shoulder_center=np.median((xy[sv,11]+xy[sv,12])*.5,axis=0)
        hands={};native=[];kept=[];lost=[]
        for side,start in [('left',33),('right',54)]:
            h,n,k,l=hand_info(start,xy,old,new,core,ts,fid,bottom,shoulder_center,scale)
            hands[side]=h;native.append(n);kept.append(k);lost.append(l)
        simultaneous=core&lost[0]&lost[1]
        total_loss=core&(native[0]|native[1])&~kept[0]&~kept[1]
        simultaneous_runs=runs(simultaneous,ts,fid);total_loss_runs=runs(total_loss,ts,fid)
        maxsim=max([r['duration_seconds'] for r in simultaneous_runs],default=0.)
        maxtotal=max([r['duration_seconds'] for r in total_loss_runs],default=0.)
        tags=[]
        if any(h['a_low_motion_side_hand_below_most_core'] for h in hands.values()):tags.append('a_low_motion_side_hand_below_most_core')
        if any(h['b_lowered_then_reentered'] for h in hands.values()):tags.append('b_lowered_then_reentered')
        if simultaneous.any():tags.append('c_both_native_hands_simultaneously_removed')
        if any(h['d_moving_hand_near_boundary'] for h in hands.values()):tags.append('d_moving_hand_near_boundary')
        if maxtotal>=.3:priority=0;primary='full_hand_observation_gap_ge_0.30s_inside_core'
        elif 'd_moving_hand_near_boundary' in tags:priority=1;primary='moving_near_boundary_with_geometric_loss'
        elif 'b_lowered_then_reentered' in tags:priority=2;primary='lowered_then_reentered'
        elif any(not h['a_low_motion_side_hand_below_most_core'] and (h['newly_absent_core_frames'] or h['palm5_drop_pp']>=10) for h in hands.values()):priority=3;primary='other_partial_or_full_loss'
        else:priority=4;primary='low_motion_side_hand_below_most_core_only'
        out=dict(clip_id=cid,label=row['canonical_label'],signer_id=row['signer_id'],
                 method=f['method'],bottom_pixels=bottom,source_width=width,
                 core_seconds=[a,b],core_frames=int(core.sum()),shoulder_scale_pixels=scale,
                 flags_from_forecast=f['geometry_flags'],hands=hands,tags=tags,
                 simultaneous_loss_frames=int(simultaneous.sum()),simultaneous_loss_runs=simultaneous_runs,
                 longest_simultaneous_loss_seconds=maxsim,all_hand_loss_runs=total_loss_runs,
                 longest_all_hand_loss_seconds=maxtotal,priority_tier=priority,primary_review_reason=primary,
                 semantic_safety='UNDETERMINED_VISUAL_REVIEW_REQUIRED')
        result.append(out)
    result.sort(key=lambda r:(r['priority_tier'],-r['longest_all_hand_loss_seconds'],
                             -max(h['newly_absent_core_frames'] for h in r['hands'].values()),r['clip_id']))
    overlapping=Counter(tag for r in result for tag in r['tags'])
    exclusive=Counter(r['primary_review_reason'] for r in result)
    flaghands=Counter()
    for r in result:
        for flag in r['flags_from_forecast']:
            h=r['hands'][flag['side']]
            for k in ('a_low_motion_side_hand_below_most_core','b_lowered_then_reentered','d_moving_hand_near_boundary'):
                if h[k]:flaghands[k]+=1
    # The fixed100 source representatives and old20/fallback10 are already handled by root.
    root_review=set(context['fixed_reference_ids'])|set(context['prior20_ids'])|set(context.get('failure_train_ids',[]))
    additional=[r for r in result if r['clip_id'] not in root_review and r['priority_tier']<=1]
    summary=lambda r:{k:r[k] for k in ('clip_id','label','signer_id','method','bottom_pixels','priority_tier','primary_review_reason','longest_all_hand_loss_seconds','tags')}
    output=dict(status='TRAIN_LEGACY_GEOMETRY_TRIAGE_NOT_SEMANTIC_PASS',created_utc=datetime.now(timezone.utc).isoformat(),
                rules=RULES,source_scope='619 flagged TRAIN legacy observations; new detector outcomes may differ',
                train_total=899,train_flagged_clips=619,train_flagged_hand_conditions=633,
                train_unflagged_not_opened=280,development_observation_files_opened=0,
                development_annotation_values_used=0,detector_runs=0,training_runs=0,classifier_inference=0,
                overlapping_clip_categories=dict(overlapping),flagged_hand_categories=dict(flaghands),
                exclusive_review_priorities=dict(exclusive),
                both_hands_simultaneously_lost_ge_0_30s=sum(r['longest_simultaneous_loss_seconds']>=.3 for r in result),
                all_hands_lost_ge_0_30s=sum(r['longest_all_hand_loss_seconds']>=.3 for r in result),
                all_hands_lost_ge_0_60s=sum(r['longest_all_hand_loss_seconds']>=.6 for r in result),
                urgent_top30=[summary(r) for r in result[:30]],
                extra_priority_beyond_fixed_root_review_count=len(additional),
                extra_priority_beyond_fixed_root_review_top30=[summary(r) for r in additional[:30]],
                rows=result,source_pins=pins,seconds=time.perf_counter()-started,
                limitations=['Geometry alone cannot establish whether a lowered/static support hand is linguistically irrelevant.',
                             'Core annotations can include transitions; even>=0.30s absence is a review flag, not proven action truncation.',
                             'Legacy detector assignment errors or low confidence can affect this forecast; compare source RGB and fresh joint_v3.',
                             'No threshold, boundary, model, training sample, or per-word exception was changed.'],
                preliminary_inspection_errors=[{'type':'UnicodeDecodeError','cause':'Read-only ad hoc inspection used Windows default cp949',
                                               'recovery':'Explicit utf-8-sig; no input or result was modified.'}])
    for p,v in pins.items():assert sha(p)==v
    with OUTPUT.open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows','source_pins','urgent_top30','extra_priority_beyond_fixed_root_review_top30')},ensure_ascii=False),flush=True)
    print(json.dumps(output['extra_priority_beyond_fixed_root_review_top30'][:12],ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
