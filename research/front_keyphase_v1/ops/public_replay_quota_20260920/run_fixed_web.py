"""Reuse the verified two-model evaluator for each fixed18 cohort; no training."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

OPS = Path(__file__).resolve().parent
WEB = OPS.parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('cohort', choices=['old', 'new'])
args = parser.parse_args()
path = WEB / 'ops/native_pair_web_development_20260919/replay_fixed_pair.py'
spec = importlib.util.spec_from_file_location('reused_verified_pair_replay', path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
replay.PLAN = OPS / ('web_' + args.cohort + '_plan_LOCAL_ONLY.json')
replay.RESULT = OPS / ('web_' + args.cohort + '_result_LOCAL_ONLY.json')
replay.execute()

if args.cohort == 'old':
    # Same accepted23 public conditions; no new transformation or user inputs.
    sys.path[:0] = [str(WEB), str(WEB / 'src'), str(WEB.parent / 'sign_model_pilot_v1/src')]
    import tools_prep_union93_candidate as u
    import tools_visibility_experiment as v
    from wis_lab.service import neural
    provenance = json.loads((WEB / 'runs/visibility_candidate_v1/provenance.json').read_text(encoding='utf-8'))
    reference_meta = Path(provenance['source_experiment']['directory']) / 'inputs.json'
    inputs = json.loads(reference_meta.read_text(encoding='utf-8'))
    items, eligible = [], {}
    for row in inputs['nikl']:
        native, present, timestamps = u._native_from_nikl(str(WEB / row['npz']), row['npz_sha256'])
        items.append({'clip_id': row['id'], 'label': row['label'], 'nat': native,
                      'present': present, 'ts': timestamps})
        if row.get('reviewed_non_sign_hand') in ('left', 'right'):
            eligible[row['id']] = {'side': row['reviewed_non_sign_hand']}
    assert len(items) == 18 and len(eligible) == 5
    clips = v.reference_clips(items, eligible)
    assert len(clips) == 23
    plan = json.loads(replay.PLAN.read_text(encoding='utf-8'))
    reports = {}
    for arm in ('control', 'treatment'):
        loaded = neural._load('offline_native_pair_' + arm)
        assert loaded['checkpoint_sha256'] == plan['models'][arm]['sha256']
        rows = v.reference_rows(loaded['model'], clips, loaded['classes'], sorted({it['label'] for it in items}))
        reports[arm] = {'rows': rows}
        for scope in ('project18', 'full93'):
            reports[arm][scope] = {
                condition: {'n': sum(r['condition'] == condition for r in rows),
                            'correct': sum(r['condition'] == condition and r[scope]['pred'] == r['true'] for r in rows),
                            'errors': [{'id': r['id'], 'true': r['true'], 'pred': r[scope]['pred']}
                                       for r in rows if r['condition'] == condition and r[scope]['pred'] != r['true']]}
                for condition in ('intact', 'reviewed_masked')}
    with (OPS / 'public_reference_result.json').open('x', encoding='utf-8') as f:
        json.dump({'status': 'EVALUATED_PENDING_RECOUNT', 'models': plan['models'],
                   'reference_meta': str(reference_meta),
                   'reference_meta_sha256': hashlib.sha256(reference_meta.read_bytes()).hexdigest(),
                   'eligible': eligible, 'results': reports, 'inference_calls': 46,
                   'training_steps': 0, 'limits': 'Public training originals and previously reviewed synthetic missingness conditions; not fresh people'},
                  f, ensure_ascii=False, indent=2)
    print(json.dumps({'public_reference': {a: {s: reports[a][s] for s in ('project18', 'full93')}
                                          for a in reports}}, ensure_ascii=False))
