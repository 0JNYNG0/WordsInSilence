from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse, hashlib, json, sys
import numpy as np
from evidence_route import describe_node, describe_comparison

U = Path(__file__).resolve().parent
OPS = U.parent
ROOT = OPS.parents[2]
OLD = OPS / 'word100_feedback_causal_pilot_20261101'
CASE = OPS / 'word100_action_feedback_casebook_20261101/INPUTS.json'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(name, value):
    p = U / name
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def freeze():
    rows = read(CASE)['rows']
    paths = [CASE, OLD/'POINT_DIAGNOSIS.json', OLD/'ONE_RECOVERY_RESULT.json',
             OLD/'COMBINED_CHECK.json', U/'audit.py', U/'evidence_route.py']
    for r in rows.values():
        paths += [Path(r['input']), Path(r['observation'])]
        assert sha(r['input']) == r['input_sha256']
    code = ['word100_signrep_temporal_head_20261029/prepare_inputs.py',
            'word100_stage_feedback_cycle_20261031/prepare_bundle.py',
            'word100_stage_feedback_cycle_20261031/learning.py',
            'word100_stage_feedback_cycle_20261031/feedback.py',
            'word100_semantic_feature_cycle_20261101/comparison.py',
            'word100_feedback_causal_pilot_20261101/diagnose.py']
    paths += [OPS / p for p in code]
    write('FROZEN.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': '2026-11-01 user: continue next work; audit gaps and image+joint reasoning, avoid requiring perfect coordinates',
        'scope': 'Existing public casebook 16 clips, all stored frames; saved model outputs only',
        'aliases': sorted(rows), 'primary_example': {'alias':'P065','frame_id':48},
        'action_scope': 'Existing metadata intervals, inclusive PTS samples; not human-verified exhaustive key phases',
        'missing_definition': 'No finite observed MediaPipe hand point in original full-frame observations; not proof of invisible hand',
        'gate_definition': 'Saved 2 wrist coordinate masks and at least 10 of 14 angle masks; existing proxy, not anatomical truth',
        'runs': 'Consecutive native frame IDs; frame count and first/last PTS; approximate exposure=count*median_native_period',
        'success': ['Original fields and denominators unchanged', 'Keep multimodal estimates even when geometry gate is false, as estimates only',
                    'No automatic correctness or new-recording verdict from a missing frame', 'Independent arithmetic from saved inputs'],
        'not_claimed': ['RGB-only calibrated accuracy', 'word recognition improvement', 'stage alignment accuracy', 'educational correctness'],
        'forbidden': ['new training/inference/detection/DTW', 'new labels or tuning', 'user/mentor data', 'web/model replacement',
                      'downloads/external transfers/Git', 'rewrite prior results'],
        'pins': {str(p):sha(p) for p in paths},
    })
    print('Frozen public audit:', len(rows), 'clips')

def runs(flags, ids, timestamps):
    out = []
    indices = np.flatnonzero(flags).tolist()
    groups = []
    for i in indices:
        if not groups or i != groups[-1][-1]+1 or ids[i] != ids[groups[-1][-1]]+1:
            groups.append([i])
        else:
            groups[-1].append(i)
    period = float(np.median(np.diff(timestamps)))/1e6
    for g in groups:
        out.append(dict(first_frame=int(ids[g[0]]), last_frame=int(ids[g[-1]]), frames=len(g),
                        start_seconds=float(timestamps[g[0]])/1e6,
                        last_sample_seconds=float(timestamps[g[-1]])/1e6,
                        approximate_exposure_seconds=len(g)*period))
    return out

def main():
    frozen = read(U/'FROZEN.json')
    for p,h in frozen['pins'].items():
        assert sha(p) == h, p
    rows = read(CASE)['rows']; out = {}; arrays = {}
    assert sorted(rows) == frozen['aliases']
    for alias in frozen['aliases']:
        r = rows[alias]
        with np.load(r['observation'], allow_pickle=False) as z:
            o = {k:z[k] for k in ['observed','xy_px','frame_id','timestamp_us']}
        with np.load(r['input'], allow_pickle=False) as z:
            a = {k:z[k] for k in ['mask','rgb','frame_id','timestamp_us']}
        arrays[alias] = a
        assert np.array_equal(o['frame_id'],a['frame_id'])
        assert np.array_equal(o['timestamp_us'],a['timestamp_us'])
        ids = o['frame_id']; ts=o['timestamp_us']; n=len(ids)
        assert n==r['frames'] and len(a['mask'])==n
        action = np.zeros(n,bool)
        for lo,hi in r['intervals']:
            action |= (ts>=round(lo*1e6)) & (ts<=round(hi*1e6))
        valid = o['observed'] & np.isfinite(o['xy_px']).all(-1)
        sides = {}
        for side,offset in [('L',33),('R',54)]:
            count = valid[:,offset:offset+21].sum(1)
            names=r['feature_names']; mask=a['mask']
            wrist=mask[:,names.index(side+'_hand_wrist_x')] & mask[:,names.index(side+'_hand_wrist_y')]
            ac=[i for i,name in enumerate(names) if name.startswith(side+'_') and '_angle' in name and 'elbow' not in name]
            assert len(ac)==14
            gate=wrist & (mask[:,ac].sum(1)>=10)
            scopes={}
            for scope, take in [('full',np.ones(n,bool)),('annotated_action',action)]:
                scopes[scope]=dict(frames=int(take.sum()),
                    no_observed_hand_frames=int(((count==0)&take).sum()),
                    some_observed_hand_frames=int(((count>0)&take).sum()),
                    all_21_observed_frames=int(((count==21)&take).sum()),
                    saved_shape_gate_frames=int((gate&take).sum()),
                    no_hand_runs=runs((count==0)&take,ids,ts),
                    shape_gate_failed_runs=runs(~gate&take,ids,ts))
            sides[side]=dict(scopes=scopes, observed_points=count.tolist(),
                             saved_shape_gate=gate.tolist(), valid_angle_count=mask[:,ac].sum(1).tolist())
        out[alias]=dict(alias=alias,clip_id=r['clip_id'],label=r['label'],recorded_person_id=r['person'],
                        role=r['role'],frames=n,annotation_intervals=r['intervals'],
                        frame_ids=ids.tolist(),timestamp_us=ts.tolist(),action_scope=action.tolist(),
                        rgb_features_present=np.isfinite(a['rgb']).all(1).tolist(),sides=sides)
    totals={}
    for scope in ['full','annotated_action']:
        totals[scope]={k:sum(row['sides'][s]['scopes'][scope][k] for row in out.values() for s in ['L','R'])
                       for k in ['frames','no_observed_hand_frames','some_observed_hand_frames','saved_shape_gate_frames']}
        totals[scope]['unit']='hand-frame; full/annotated absence includes legitimate rest/out-of-frame, not error rate'
    write('MISSINGNESS.json',dict(rows=out,totals=totals,clip_count=len(out),
        unique_frames=sum(r['frames'] for r in out.values()),new_inference=0,
        warning='Only stored original observations. Prior local recoveries do not change these counts. No new video truth labels.'))

    saved = read(OLD/'POINT_DIAGNOSIS.json')
    nodes={}
    for key,node in saved['nodes'].items():
        a=arrays[node['alias']]; ix=np.flatnonzero(a['frame_id']==node['frame_id'])
        assert len(ix)==1
        nodes[key]=describe_node(node,bool(np.isfinite(a['rgb'][ix[0]]).all()))
    comparisons=[]
    for record in saved['records']:
        target=record['source']['target']; ref=nodes[record['reference_node']]
        for qkey,previous in zip(record['query_nodes'],record['comparisons']):
            comparisons.append(dict(seed=record['seed'],opportunity=record['opportunity'],
                query_node=qkey,reference_node=record['reference_node'],target=target,
                evidence=describe_comparison(previous,nodes[qkey],ref,target)))
    summary=[]
    for seed in [42,43,44]:
        cc=[c for c in comparisons if c['seed']==seed]
        summary.append(dict(seed=seed,comparisons=len(cc),
            saved_comparison_states=dict(Counter(c['evidence']['saved_comparison']['status'] for c in cc)),
            model_advisory_states=dict(Counter(c['evidence']['advisory_status'] for c in cc)),
            unsupported_with_model_estimate=sum(c['evidence']['saved_comparison']['status']=='INSUFFICIENT_OBSERVATION'
                and c['evidence']['advisory_status']!='MODEL_OUTPUT_UNAVAILABLE' for c in cc)))
    write('EVIDENCE_CHANNELS.json',dict(nodes=nodes,comparisons=comparisons,summary=summary,
        new_predictions=0,changed_probabilities=0,support_promotions=0,educational_accuracy=None,
        note='Joint RGB+pose estimates, NOT independent RGB verdicts. Saved support, zero/positive semantics and sequence uncertainty retained.'))
    write('ARCHITECTURE.json',dict(
        encoder='Frozen SignRep on 16-frame RGB windows; outside fixed upper-body box masked, full frame resized 224x224',
        rgb='768 features per window, interpolated to native timestamps. Finite features do not verify finger visibility.',
        pose='69 geometric values + 69 masks, unavailable numeric values neutralized with explicit masks',
        temporal='Original 3 targets use saved native temporal head; new 4 targets use saved 13-frame center head',
        existing_gate='Detector-evidence proxy suppresses supported comparisons even if RGB+pose posterior exists',
        change='Only research output separates existing supported values from uncalibrated multimodal estimates; no gate relaxation',
        operating_web='C and T3-F preserved; this research route is not deployed',
        next='Evaluate image+time estimates on existing publicly labeled gaps and observed controls, split by person and main-action relevance before promoting to corrective feedback; no new detector sweep'))
    for p,h in frozen['pins'].items():
        assert sha(p)==h,p
    print(json.dumps({'P065':{s:out['P065']['sides'][s]['scopes'] for s in ['L','R']},'summary':summary},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['freeze','run']); args=parser.parse_args()
    freeze() if args.mode=='freeze' else main()
