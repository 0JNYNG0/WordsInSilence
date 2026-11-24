"""Final1199 numeric audit preparation. Never grants semantic/training approval.

Run only after preparation stops and exactly1199 committed records exist.
No detector, classifier, model training, source extraction, or source mutation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
sys.path.insert(0, str(U))
import verify_boundary as vb


def observation_group(obs):
    """Score-blind observation patterns; not linguistic one/two-hand labels."""
    ts = obs['timestamp_us']
    left = obs['usable'][:, 33:54].any(axis=1)
    right = obs['usable'][:, 54:75].any(axis=1)
    if (not left.any() and right.sum() >= 4) or (not right.any() and left.sum() >= 4):
        return 'one_hand_never_usable'
    step = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.
    for visible in (left, right):
        observed = np.flatnonzero(visible)
        for before, after in zip(observed[:-1], observed[1:]):
            if after > before + 1:
                duration = float(ts[after-1] - ts[before+1]) + step
                if duration >= 300000:
                    return 'internal_reentry'
    return 'other'


def group_selfcheck():
    def make(left, right):
        usable = np.zeros((len(left), 75), dtype=bool)
        usable[:, 33] = left; usable[:, 54] = right
        return {'usable': usable, 'timestamp_us': np.arange(len(left),dtype=np.int64)*100000}
    assert observation_group(make([False]*8, [True]*8)) == 'one_hand_never_usable'
    assert observation_group(make([True,False,False,False,True], [True]*5)) == 'internal_reentry'
    assert observation_group(make([False,False,False,True,True], [True]*5)) == 'other'
    assert observation_group(make([True,True,False,False,False], [True]*5)) == 'other'
    assert observation_group(make([False]*5, [False]*5)) == 'other'
    assert observation_group(make([True,False,False,True], [True]*4)) == 'other'


def run(output):
    output = Path(output)
    raw_audit_path = output.with_name(output.stem + '_CHANNELS.json')
    assert not output.exists() and not raw_audit_path.exists(), 'Preserve earlier audit evidence'
    assert not (U/'PREP_RUNNING.json').exists(), 'Wait for preparation lock to be released'
    context_path = U/'PREP_CONTEXT.json'
    context_hash = vb.sha(context_path)
    context = vb.read(context_path)
    manifest = vb.read(context['manifest_path'])
    io = vb.input_oracle_module()
    stats = io.manifest_audit(manifest)
    expected = {r['clip_id'] for r in manifest}
    paths = sorted((U/'data/records').glob('*.json'))
    assert len(paths)==1199 and {p.stem for p in paths}==expected, 'Exactly1199 committed records required'
    record_hashes = {p.stem: vb.sha(p) for p in paths}
    coefficient_path = U/'design_data_COEFFICIENT.json'
    coefficient_hash = vb.sha(coefficient_path)
    coef = vb.read(coefficient_path)
    assert coefficient_hash == vb.read(U/'BOUNDARY_FIT_INDEPENDENT.json')['coefficient_sha256']
    io.U = U
    io.records(raw_audit_path)
    numeric = vb.read(raw_audit_path)
    assert numeric['all_requested_prepared'] and numeric['prepared_records']==1199
    mod = vb.module()
    byid = {r['clip_id']: r for r in context['ordered_rows']}
    groups = {'masked': {}, 'sensor': {}}
    tokens = {}
    methods = Counter()
    source_rows = []
    for path in paths:
        rec = vb.read(path); cid = rec['clip_id']
        native = vb.Observation.load(rec['native_capture']['candidate'])
        old = vb.read(byid[cid]['legacy_record_path'])
        old_sidecar = vb.read(Path(old['observation']).with_suffix('.sidecar.json'))
        method = vb.assert_calibration(native, int(old_sidecar['width']), int(old_sidecar['height']), coef, rec['calibration'])
        methods[method] += 1
        assert rec['source']['sha256'] == old['source_sha256']
        box = rec['calibration']['box_source_pixels']
        for view in ('native_capture','sensor_capture'):
            capture = rec[view]
            meta = vb.read(Path(capture['baseline']).with_suffix('.sidecar.json'))
            token = meta.get('detector_instance') or meta.get('detector_instance_token') or capture.get('detector_instance')
            assert token and token not in tokens, ('Missing or reused detector instance',cid,view,token,tokens.get(token))
            tokens[token] = (cid, view)
            if view == 'sensor_capture':
                assert capture['box']==box and meta['source_bbox_px']==box
                assert [meta['width'],meta['height']]==[box[2]-box[0],box[3]-box[1]]
            else:
                assert [meta['width'],meta['height']]==[old_sidecar['width'],old_sidecar['height']]
        if rec['role']=='development_only':
            for view in groups:
                groups[view][cid] = observation_group(io.load(rec['views'][view]['observation']))
        source_rows.append({'clip_id':cid, 'role':rec['role'], 'method':method, 'bottom':box[3],
                            'visual_flags':rec['visual_flags']})
    assert all(len(g)==300 for g in groups.values())
    assert len(tokens)==2398
    assert not (U/'PREP_RUNNING.json').exists()
    assert vb.sha(context_path)==context_hash and vb.sha(coefficient_path)==coefficient_hash
    assert all(vb.sha(p)==record_hashes[p.stem] for p in paths)
    report = {'status':'PASS_FULL_NUMERIC_CONTRACT_PENDING_SEMANTIC_REVIEW', 'training_allowed':False,
              'manifest':stats, 'prepared_records':1199, 'boundary_methods':dict(methods),
              'matched_feature_values':numeric['matched_feature_values'],
              'hidden_points_perturbed':numeric['hidden_points_perturbed'],
              'context_pins_unchanged':numeric['context_pins_unchanged'],
              'independent_input_validation_passed':True,
              'fresh_sensor_detector_state_verified':True,
              'roi_hidden_coordinate_invariance_passed':True,
              'motion_preservation_review_passed':False,
              'remaining_gate':'Separate evidence-backed visual/semantic review; this audit never sets that gate true.',
              'preparation_context_sha256':context_hash, 'manifest_sha256':vb.sha(context['manifest_path']),
              'coefficient_sha256':coefficient_hash, 'record_sha256':record_hashes,
              'missingness_groups':groups, 'group_counts':{v:dict(Counter(g.values())) for v,g in groups.items()},
              'unique_native_sensor_detector_tokens':len(tokens),
              'fresh_state_evidence':'Distinct source/crop instance tokens, frozen capture function constructs a fresh landmarker and assigner each capture, source/crop bbox and source/record pins verified.',
              'numeric_audit_path':str(raw_audit_path), 'numeric_audit_sha256':vb.sha(raw_audit_path),
              'rows':source_rows, 'verifier_sha256':vb.sha(__file__),
              'boundary_oracle_sha256':vb.sha(vb.__file__), 'input_oracle_sha256':vb.sha(io.__file__),
              'detector_runs':0,'classifier_inference_runs':0,'training_runs':0}
    with output.open('x',encoding='utf-8') as file: json.dump(report,file,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','record_sha256','missingness_groups')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(U/'FULL_INPUTS_INDEPENDENT.json'))
    parser.add_argument('--selfcheck-only',action='store_true')
    args=parser.parse_args()
    group_selfcheck()
    if args.selfcheck_only:
        print('PASS six synthetic missingness grouping checks; full audit not run')
    else:
        run(args.output)
