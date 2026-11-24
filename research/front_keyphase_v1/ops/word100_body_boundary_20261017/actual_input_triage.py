"""TRAIN-only actual-input vs legacy review triage; no detector or classifier.

The full audit requires all 1199 records and no active preparation lock. A smoke
audit explicitly reports PARTIAL and never serves as the final readiness gate.
Review flags do not assert loss of a linguistically necessary movement.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
sys.path.insert(0, str(U))
from training_geometry_triage import RULES, hand_info, runs, read, sha


def observe(path, expected, pins, all_outputs):
    path = Path(path)
    assert sha(path) == expected, 'Observation changed: ' + str(path)
    pins[str(path.resolve())] = expected
    sidecar = path.with_suffix('.sidecar.json')
    expected_sidecar = all_outputs[str(sidecar.resolve())]
    assert sha(sidecar) == expected_sidecar
    pins[str(sidecar.resolve())] = expected_sidecar
    with np.load(path, allow_pickle=False) as z:
        arrays = {k:z[k] for k in ('xy_px','observed','usable','timestamp_us','frame_id')}
    arrays['sidecar'] = read(sidecar)
    return arrays


def valid(obs):
    return obs['observed'] & obs['usable'] & np.isfinite(obs['xy_px']).all(-1)


def run_info(mask, ts, fid):
    values = runs(mask, ts, fid)
    return {'frames':int(mask.sum()), 'runs':values,
            'longest_seconds':max((r['duration_seconds'] for r in values),default=0.)}


def review_lineage(cid, context):
    paths = []
    for name in (f'evidence_{cid}.json', f'evidence_visual_triage_{cid}_v2.json'):
        if (U/name).exists(): paths.append(str(U/name))
    reports=[]
    if cid in context['early_ids']:reports.append(U/'EARLY_REVIEW_RESULT.json')
    if cid in context['fixed_reference_ids']:
        reports.extend([U/'training_reference_visual_review_RESULT.json',U/'training_reference_visual_review_CLARIFICATION.json'])
    if (U/f'evidence_{cid}.json').exists():reports.append(U/'evidence_VISUAL_REVIEW.json')
    if (U/f'evidence_visual_triage_{cid}_v2.json').exists():reports.append(U/'evidence_visual_triage_REVIEW.json')
    return {'in_early30':cid in context['early_ids'],
            'in_fixed100_train_representatives':cid in context['fixed_reference_ids'],
            'in_prior20_problem_clips':cid in context['prior20_ids'],
            'in_fallback10':cid in context['failure_train_ids'],
            'existing_source_visual_evidence':paths,
            'linked_review_reports':[str(p) for p in reports if p.exists()],
            'meaning':'Membership is not a semantic PASS; previous source evidence uses legacy joints unless explicitly stated otherwise.'}


def analyze(row, record, native, masked, sensor, forecast, legacy_triage, lineage):
    ts, fid = native['timestamp_us'], native['frame_id']
    assert len(ts)>1 and np.all(np.diff(ts)>0)
    for obs in (masked,sensor):
        assert np.array_equal(ts,obs['timestamp_us']) and np.array_equal(fid,obs['frame_id'])
    xy = native['xy_px']; old, new, sensed = valid(native), valid(masked), valid(sensor)
    box = np.asarray(record['calibration']['box_source_pixels'],float)
    x0,y0,x1,y1 = box
    inside = (xy[:,:,0]>=x0)&(xy[:,:,0]<x1)&(xy[:,:,1]>=y0)&(xy[:,:,1]<y1)&np.isfinite(xy).all(-1)
    assert np.array_equal(new,old&inside), 'Actual masked geometry differs from full native ROI'
    assert np.all(~sensed | ((sensor['xy_px'][:,:,0]>=x0)&(sensor['xy_px'][:,:,0]<x1)&
                            (sensor['xy_px'][:,:,1]>=y0)&(sensor['xy_px'][:,:,1]<y1)))
    a,b=row['intervals'][0]
    core=(ts>=round(a*1e6))&(ts<=round(b*1e6))
    assert core.any()
    shoulders=core&old[:,11]&old[:,12]
    assert shoulders.any(), 'No core shoulder scale for geometric interpretation'
    scale=float(np.median(np.linalg.norm(xy[shoulders,11]-xy[shoulders,12],axis=1)))
    assert scale>0
    center=np.median((xy[shoulders,11]+xy[shoulders,12])*.5,axis=0)
    hands={}; tags=[]
    for side,start,other in [('left',33,54),('right',54,33)]:
        geometry,native_any,mask_any,geom_lost=hand_info(start,xy,old,new,core,ts,fid,int(y1),center,scale)
        sensor_any=sensed[:,start:start+21].any(1)
        sensor_other=sensed[:,other:other+21].any(1)
        sensor_lost=core&mask_any&~sensor_any
        palm=start+np.array([0,5,9,13,17])
        mask_palm=new[:,palm].all(1); sensor_palm=sensed[:,palm].all(1)
        lost_palm=core&mask_palm&~sensor_palm
        geom_partial=core&(new[:,start:start+21].sum(1)>0)&(new[:,start:start+21].sum(1)<old[:,start:start+21].sum(1))
        sensor_stats=run_info(sensor_lost,ts,fid)
        sensor_stats.update(native_inside_roi_visible_core_percent=100*float(mask_any[core].mean()),
                            sensor_visible_core_percent=100*float(sensor_any[core].mean()),
                            sensor_only_visible_core_frames=int((core&sensor_any&~mask_any).sum()),
                            lost_while_other_sensor_hand_present_frames=int((sensor_lost&sensor_other).sum()),
                            complete_palm_lost=run_info(lost_palm,ts,fid),
                            caveat='Same-side observation disagreement: can reflect detection/assignment/framing, not proven physical hand absence.')
        previous=(legacy_triage or {}).get('hands',{}).get(side)
        comparison={'legacy_geometry_available':previous is not None}
        if previous:
            comparison.update(new_absence_frames_delta=geometry['newly_absent_core_frames']-previous['newly_absent_core_frames'],
                              longest_absence_seconds_delta=geometry['longest_new_absence_seconds']-previous['longest_new_absence_seconds'],
                              palm5_drop_pp_delta=geometry['palm5_drop_pp']-previous['palm5_drop_pp'],
                              newly_moving_near_boundary=geometry['d_moving_hand_near_boundary'] and not previous['d_moving_hand_near_boundary'])
        else:
            comparison['newly_moving_near_boundary']=geometry['d_moving_hand_near_boundary']
        hands[side]={'native_to_mask_geometry':geometry,'partial_geometry_loss':run_info(geom_partial,ts,fid),
                     'mask_to_sensor_observation':sensor_stats,'legacy_comparison':comparison}
        if geometry['d_moving_hand_near_boundary']:tags.append('moving_near_boundary_geometry')
        if geometry['b_lowered_then_reentered']:tags.append('geometry_lowered_then_reentered')
        if sensor_stats['longest_seconds']>=RULES['long_absence_seconds']:tags.append('inside_roi_hand_lost_by_sensor_ge_0_30s')
        if sensor_stats['complete_palm_lost']['longest_seconds']>=RULES['long_absence_seconds']:tags.append('inside_roi_complete_palm_lost_by_sensor_ge_0_30s')
    n=old[:,33:75].any(1);m=new[:,33:75].any(1);s=sensed[:,33:75].any(1)
    zero={
        'native_no_valid_hand':run_info(core&~n,ts,fid),
        'masked_no_valid_hand_total':run_info(core&~m,ts,fid),
        'native_has_hand_but_mask_has_none':run_info(core&n&~m,ts,fid),
        'sensor_no_valid_hand_total':run_info(core&~s,ts,fid),
        'mask_has_hand_but_sensor_has_none':run_info(core&m&~s,ts,fid),
        'native_no_hand_but_sensor_has_hand':run_info(core&~n&s,ts,fid)}
    previous_long=(legacy_triage or {}).get('longest_all_hand_loss_seconds',0.)
    current_long=zero['native_has_hand_but_mask_has_none']['longest_seconds']
    boundary_changed=int(y1)!=forecast['bottom'] or record['calibration']['method']!=forecast['method']
    if boundary_changed:tags.append('boundary_differs_from_legacy_forecast')
    if current_long>=RULES['long_absence_seconds']:tags.append('geometric_zero_hand_ge_0_30s')
    if zero['mask_has_hand_but_sensor_has_none']['longest_seconds']>=RULES['long_absence_seconds']:
        tags.append('sensor_zero_hand_while_mask_has_hand_ge_0_30s')
    if zero['native_no_valid_hand']['longest_seconds']>=RULES['long_absence_seconds']:
        tags.append('fullframe_native_zero_hand_ge_0_30s')
    newly_geometric=current_long>=RULES['long_absence_seconds'] and previous_long<RULES['long_absence_seconds']
    newly_moving=any(h['legacy_comparison']['newly_moving_near_boundary'] for h in hands.values())
    has_prior_visual=bool(lineage['existing_source_visual_evidence'])
    new_review=[]
    if boundary_changed:new_review.append('actual_boundary_changed_review_source_geometry')
    if newly_geometric:new_review.append('new_long_geometric_zero_hand_interval')
    if newly_moving:new_review.append('new_moving_near_boundary_flag')
    if not has_prior_visual and 'moving_near_boundary_geometry' in tags:new_review.append('moving_geometry_without_linked_source_visual_review')
    if 'sensor_zero_hand_while_mask_has_hand_ge_0_30s' in tags:new_review.append('actual_sensor_long_zero_hand_gap_despite_inside_roi_fullframe_hand')
    elif 'inside_roi_hand_lost_by_sensor_ge_0_30s' in tags:new_review.append('actual_sensor_long_same_side_gap_despite_inside_roi_fullframe_hand')
    return {'clip_id':row['clip_id'],'label':row['canonical_label'],'role':'train_candidate','signer_id':row['signer_id'],
            'core_seconds':[a,b],'core_frames':int(core.sum()),'boundary_method':record['calibration']['method'],
            'bottom_pixels':int(y1),'legacy_bottom_pixels':forecast['bottom'],
            'boundary_delta_pixels':int(y1)-forecast['bottom'],'boundary_changed':boundary_changed,
            'hands':hands,'zero_hand_intervals':zero,'legacy_longest_geometric_zero_hand_seconds':previous_long,
            'tags':sorted(set(tags)),'additional_review_reasons':new_review,'review_lineage':lineage,
            'semantic_status':'UNDETERMINED_REVIEW_ONLY',
            'actual_saved_visual_flags':record['visual_flags']}


def production_guard(context, smoke_limit):
    assert len(context['ordered_rows'])==1199
    expected={r['clip_id'] for r in context['ordered_rows']}
    present={p.stem for p in (U/'data/records').glob('*.json')}
    assert present<=expected, 'Unexpected record IDs'
    if smoke_limit is None:
        assert not (U/'PREP_RUNNING.json').exists(), 'Preparation still running; final audit forbidden'
        assert present==expected, f'Final audit requires all 1199 records; currently {len(present)}'
    else:
        assert smoke_limit>0
    return present


def main(output, smoke_limit=None):
    started=time.perf_counter()
    output=Path(output).resolve()
    assert output.parent==U and output.name.startswith('actual_input_triage'), 'Output must be this unit, actual_input_triage*'
    assert not output.exists(), 'First evidence must not be overwritten'
    paths=[U/'PREP_CONTEXT.json',U/'LEGACY_PREFLIGHT.json',U/'training_geometry_triage_RESULT.json',
           U/'training_geometry_triage.py',Path(__file__)]
    pins={str(p.resolve()):sha(p) for p in paths}
    context=read(paths[0]);present=production_guard(context,smoke_limit)
    context_hash=sha(paths[0]);fore=read(paths[1]);assert fore['context_sha256']==context_hash
    forecast={r['clip_id']:r for r in fore['rows'] if r['role']=='train_candidate'}
    oldtriage={r['clip_id']:r for r in read(paths[2])['rows']}
    train=[r for r in context['ordered_rows'] if r['role']=='train_candidate']
    assert len(train)==899
    selected=[r for r in train if r['clip_id'] in present]
    if smoke_limit is not None:selected=selected[:smoke_limit]
    assert selected
    results=[]
    for row in selected:
        cid=row['clip_id'];rp=U/'data/records'/f'{cid}.json'
        pins[str(rp.resolve())]=sha(rp);rec=read(rp)
        assert rec['clip_id']==cid and rec['role']=='train_candidate' and rec['context_sha256']==context_hash
        observations={}
        for mode in ('native','masked','sensor'):
            op=Path(rec['views'][mode]['observation']).resolve()
            observations[mode]=observe(op,rec['outputs'][str(op)],pins,rec['outputs'])
        lineage=review_lineage(cid,context)
        for p in lineage['existing_source_visual_evidence']+lineage['linked_review_reports']:pins[p]=sha(p)
        result=analyze(row,rec,observations['native'],observations['masked'],observations['sensor'],
                       forecast[cid],oldtriage.get(cid),lineage)
        result['record_path']=str(rp);result['record_sha256']=pins[str(rp.resolve())]
        results.append(result)
    # All files actually read are pinned before publishing results.
    for p,expected in pins.items():assert sha(p)==expected, 'Input changed during audit: '+p
    if smoke_limit is None:production_guard(context,None)
    priority=sorted((r for r in results if r['additional_review_reasons']),key=lambda r:(
        -r['zero_hand_intervals']['mask_has_hand_but_sensor_has_none']['longest_seconds'],
        -r['zero_hand_intervals']['native_has_hand_but_mask_has_none']['longest_seconds'],
        -max(h['mask_to_sensor_observation']['longest_seconds'] for h in r['hands'].values()),r['clip_id']))
    summary=lambda r:{k:r[k] for k in ('clip_id','label','bottom_pixels','boundary_delta_pixels',
                                     'additional_review_reasons','review_lineage')}
    result={'status':'PARTIAL_SMOKE_NOT_FINAL_GATE' if smoke_limit is not None else 'TRAIN_ACTUAL_INPUT_TRIAGE_REVIEW_REQUIRED',
            'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'TRAIN observations only',
            'available_record_count_at_start':len(present),'required_full_records':1199,'train_total':899,
            'analyzed_train_rows':len(results),'development_observations_opened':0,
            'rules_reused_from_legacy_triage':RULES,'context_sha256':context_hash,
            'boundary_changed_count':sum(r['boundary_changed'] for r in results),
            'tag_clip_counts':dict(Counter(t for r in results for t in r['tags'])),
            'additional_review_reason_counts':dict(Counter(t for r in results for t in r['additional_review_reasons'])),
            'additional_review_count':len(priority),'additional_review_priority':[summary(r) for r in priority],
            'comparison_meaning':{'native_to_mask':'Geometric removal after fullframe detection.',
                                  'mask_to_sensor':'Observed same-side disagreement with independent cropped detector; includes assignment and tracking effects.',
                                  'zero_hand':'No usable observed hand landmarks, not proof the RGB hand is physically absent.'},
            'limits':['No automatic semantic PASS and no recognition score or performance claim.',
                      'A stationary support hand may be necessary. Long/short absence alone does not define sign correctness.',
                      'Existing early30/fixed100/prior20 membership does not mean all actual sensor intervals were visually reviewed.',
                      'Legacy review of ending/rest must be rechecked if actual boundary or observations changed.',
                      'Missing-side flags can occur during a one-hand sign; they are review priorities, not automatic failures.'],
            'rows':results,'source_pins':pins,'seconds':time.perf_counter()-started,
            'new_detector_runs':0,'classifier_runs':0,'training_runs':0,'model_or_threshold_changes':0}
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_pins','additional_review_priority','rules_reused_from_legacy_triage')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(U/'actual_input_triage_RESULT.json'))
    parser.add_argument('--smoke-limit',type=int)
    args=parser.parse_args()
    main(args.output,args.smoke_limit)
