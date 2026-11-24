"""Inventory frozen TRAIN review flags and exact intervals; never approves READY.

Reads saved observations and existing visual-review JSON only. No source video,
detector, classifier, training, or score-dependent selection.
"""
import json
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import sys
import numpy as np
sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
OUT = U / 'review_flag_coverage_RESULT.json'
pins = {}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(p):
    p=Path(p); pins[str(p)]=sha(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))

def segments(mask,ts,fid):
    idx=np.flatnonzero(mask)
    if not len(idx):return []
    step=float(np.median(np.diff(ts))) if len(ts)>1 else 0.
    return [dict(start_index=int(g[0]),end_index=int(g[-1]),
                 start_frame_id=int(fid[g[0]]),end_frame_id=int(fid[g[-1]]),
                 frame_ids=fid[g].tolist(),start_seconds=float(ts[g[0]])/1e6,
                 end_seconds=float(ts[g[-1]])/1e6,
                 duration_seconds=float(ts[g[-1]]-ts[g[0]]+step)/1e6)
            for g in np.split(idx,np.flatnonzero(np.diff(idx)>1)+1)]

def frame_set(items):
    return set(int(x['frame_id'] if isinstance(x,dict) and 'frame_id' in x else
                   x['frame'] if isinstance(x,dict) else x) for x in items)

def main():
    assert not OUT.exists() and not (U/'PREP_RUNNING.json').exists()
    context=read(U/'PREP_CONTEXT.json')
    full=read(U/'FULL_INPUTS_INDEPENDENT.json')
    triage=read(U/'actual_input_triage_RESULT.json')
    assert full['prepared_records']==1199 and full['preparation_context_sha256']==sha(U/'PREP_CONTEXT.json')
    assert triage['context_sha256']==full['preparation_context_sha256']
    bytriage={r['clip_id']:r for r in triage['rows']}
    rows=[r for r in context['ordered_rows'] if r['role']=='train_candidate']
    assert len(rows)==899
    recs={r['clip_id']:read(U/'data/records'/f"{r['clip_id']}.json") for r in rows}
    assert all(pins[str(U/'data/records'/f'{cid}.json')]==full['record_sha256'][cid] for cid in recs)
    seen_source={};seen_actual={};evidence=[]

    def add(cid,frames,source_sha,bottom,report,kind='source_only'):
        if cid not in recs:return
        rec=recs[cid]
        assert source_sha==rec['source']['sha256'] and bottom==rec['calibration']['box_source_pixels'][3], (cid,'visual identity mismatch')
        ids=frame_set(frames)
        seen_source.setdefault(cid,set()).update(ids)
        if kind=='actual_four_view':seen_actual.setdefault(cid,set()).update(ids)
        evidence.append(dict(clip_id=cid,report=str(report),scope=kind,frame_ids=sorted(ids)))

    # Each frame list is credited only because a separate report says it was viewed.
    report=U/'training_reference_visual_review_RESULT.json'
    ref=read(report);montage=read(U/'training_reference_montage_RESULT.json')
    source={r['clip_id']:r for r in montage['rows']}
    for r in ref['rows']:
        m=source[r['clip_id']]
        add(r['clip_id'],r['inspected_selected_frames'],m['source_sha256'],m['bottom'],report)
    report=U/'evidence_VISUAL_REVIEW.json'
    for r in read(report)['rows']:
        add(r['clip_id'],r['selected_frames'],r['source_sha256'],r['bottom'],report)
    report=U/'evidence_visual_triage_REVIEW.json'
    for r in read(report)['rows']:
        m=read(r['metadata']);assert sha(r['metadata'])==r['metadata_sha256']
        add(r['clip_id'],m['overview_frames']+m['focus_frames'],m['source_sha256'],m['bottom'],report)
    # Root's EARLY review explicitly records both LOW_TRANSITION sheets viewed.
    report=U/'EARLY_REVIEW_RESULT.json';early=read(report)
    m=read(U/'evidence_visual_triage_LOW_TRANSITION.json')
    assert any('76-84' in s and '86-91' in s for s in early['observed'])
    add(m['clip_id'],m['frames'],m['source_sha256'],m['bottom'],report)

    outputs=[];condition_counts=Counter();clip_counts={};all_focus=0;all_windows=0.
    for row in rows:
        cid=row['clip_id'];rec=recs[cid];stored=rec['visual_flags']
        if not stored:continue
        obs={}
        for view in ('native','masked','sensor'):
            p=Path(rec['views'][view]['observation']);assert sha(p)==rec['outputs'][str(p)]
            pins[str(p)]=rec['outputs'][str(p)]
            with np.load(p,allow_pickle=False) as z:
                obs[view]={k:z[k] for k in ('usable','timestamp_us','frame_id')}
        ts=obs['native']['timestamp_us'];fid=obs['native']['frame_id']
        assert np.all(np.diff(ts)>0)
        for view in obs:
            assert np.array_equal(ts,obs[view]['timestamp_us']) and np.array_equal(fid,obs[view]['frame_id'])
        a,b=row['intervals'][0];core=(ts>=round(a*1e6))&(ts<=round(b*1e6))
        actual_flags=[];conditions=[];union=np.zeros(len(ts),bool)
        for comparison,v0,v1 in [('native_to_mask','native','masked'),('mask_to_sensor','masked','sensor')]:
            for side,start in [('left',33),('right',54)]:
                old=obs[v0]['usable'];new=obs[v1]['usable'];ids=np.arange(start,start+21);palm=start+np.array([0,5,9,13,17])
                absent=core&old[:,ids].any(1)&~new[:,ids].any(1)
                absruns=segments(absent,ts,fid);longest=max((g['duration_seconds'] for g in absruns),default=0.)
                op=old[:,palm].all(1);npalm=new[:,palm].all(1)
                drop=float(op[core].mean())-float(npalm[core].mean())
                if not(longest>=.3 or drop>=.1):continue
                actual_flags.append((comparison,side))
                saved=next(f for f in stored if (f['comparison'],f['side'])==(comparison,side))
                assert abs(saved['newly_absent_run_seconds']-longest)<1e-12
                assert abs(saved['palm5_drop_pp']-100*drop)<1e-10
                assert saved['newly_absent_frame_ids']==fid[absent].tolist()
                # All loss frames in a flagged condition remain explicit, including
                # short subruns; do not silently discard them after flagging.
                palmloss=core&op&~npalm
                focus=absent|palmloss
                assert focus.any();union|=focus
                condition_counts[comparison]+=1;clip_counts.setdefault(comparison,set()).add(cid)
                seen=seen_source.get(cid,set()) if comparison=='native_to_mask' else seen_actual.get(cid,set())
                conditions.append(dict(comparison=comparison,side=side,
                    trigger_long_absence=longest>=.3,trigger_net_palm_drop=drop>=.1,
                    max_newly_absent_seconds=longest,net_palm5_drop_pp=100*drop,
                    newly_absent_intervals=absruns,palm_loss_intervals=segments(palmloss,ts,fid),
                    focus_frame_ids=fid[focus].tolist(),
                    previously_seen_relevant_frame_ids=sorted(set(fid[focus].tolist())&seen),
                    not_previously_seen_frame_ids=sorted(set(fid[focus].tolist())-seen),
                    review_scope='source RGB geometry' if comparison=='native_to_mask' else 'actual native/masked/sensor correspondence',
                    complete_interval_semantic_review='NOT_CERTIFIED_BY_THIS_INVENTORY'))
        assert set(actual_flags)=={(f['comparison'],f['side']) for f in stored}
        window=np.zeros(len(ts),bool)
        for run in segments(union,ts,fid):
            window|=(ts>=ts[run['start_index']]-300000)&(ts<=ts[run['end_index']]+300000)
        merged=segments(window,ts,fid);all_focus+=int(union.sum());all_windows+=sum(x['duration_seconds'] for x in merged)
        t=bytriage[cid];flags=t['tags']
        # Priority changes review order only, never which flagged interval is reviewed.
        priority=(0 if any('zero_hand' in s for s in flags) else
                  1 if 'inside_roi_complete_palm_lost_by_sensor_ge_0_30s' in flags else
                  2 if 'moving_near_boundary_geometry' in flags else
                  3 if 'inside_roi_hand_lost_by_sensor_ge_0_30s' in flags else 4)
        outputs.append(dict(clip_id=cid,label=row['canonical_label'],signer_id=row['signer_id'],
            record_sha256=full['record_sha256'][cid],source=rec['source'],bottom_pixels=rec['calibration']['box_source_pixels'][3],
            core_seconds=[a,b],priority_tier=priority,triage_tags=flags,
            fixed100_membership_only=cid in context['fixed_reference_ids'],
            prior20_membership_only=cid in context['prior20_ids'],
            conditions=conditions,focus_frame_ids=fid[union].tolist(),
            merged_review_windows_with_300ms_context=merged,
            reviewed_source_frames_anywhere=sorted(seen_source.get(cid,set())),
            reviewed_actual_overlay_frames_anywhere=sorted(seen_actual.get(cid,set()))))
    outputs.sort(key=lambda r:(r['priority_tier'],r['clip_id']))
    assert all(sha(p)==v for p,v in pins.items()),'Read evidence changed'
    summary=dict(status='EXACT_FROZEN_TRAIN_FLAG_INTERVAL_INVENTORY_NOT_READY',
        created_utc=datetime.now(timezone.utc).isoformat(),training_allowed=False,
        train_rows_checked=899,flagged_train_clips=len(outputs),flagged_hand_comparison_conditions=sum(condition_counts.values()),
        conditions_by_comparison=dict(condition_counts),clips_by_comparison={k:len(v) for k,v in clip_counts.items()},
        unique_focus_frames_across_clips=all_focus,
        merged_review_window_count=sum(len(r['merged_review_windows_with_300ms_context']) for r in outputs),
        merged_review_window_total_seconds=all_windows,
        conditions_with_any_unseen_relevant_frame=sum(bool(c['not_previously_seen_frame_ids']) for r in outputs for c in r['conditions']),
        clips_with_any_unseen_relevant_frame=sum(any(c['not_previously_seen_frame_ids'] for c in r['conditions']) for r in outputs),
        source_visual_evidence_credited=len(evidence),actual_overlay_evidence_credited=0,
        snapshot_note='New actual geometry/sensor review reports were still being written. They are intentionally not credited until an explicit completed report and exact viewed frames can be linked.',
        coverage_meaning='Previously seen frames are source-viewing coverage only, not condition/interval semantic acceptance. Membership in early30/fixed100/prior20 does not credit a full video. Every flagged interval still needs an explicit review disposition; numeric flags are not errors.',
        interval_definition='For each frozen flagged hand/comparison, union every newly absent hand frame and source-complete/destination-incomplete palm5 frame within annotated TRAIN core. Retain all short subruns as well. Merge overlapping clip-level windows after adding 0.30s context. No detector or boundary thresholds changed.',
        evidence=evidence,rows=outputs,source_pins=pins,
        detector_runs=0,classifier_runs=0,training_runs=0,new_video_reads=0,
        verifier_sha256=sha(__file__))
    with OUT.open('x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','source_pins','evidence')},ensure_ascii=False))

if __name__=='__main__':
    main()
