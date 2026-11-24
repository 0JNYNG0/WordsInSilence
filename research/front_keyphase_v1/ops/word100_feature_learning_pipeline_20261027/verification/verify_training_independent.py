"""Saved-input/score arithmetic audit. Never imports producer or runs a model.

preflight does not read test logits or metrics. scores requires all 24 outputs.
This verifies contracts and stored arithmetic, not human annotation correctness.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import sys
import numpy as np

U = Path(__file__).resolve().parents[1]
V = U / 'verification'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_new(path, data):
    assert not path.exists(), f'Preserve prior audit: {path}'
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def run(stage):
    checks = []
    def check(name, condition, detail=None):
        checks.append({'name': name, 'passed': bool(condition), 'detail': detail})
        assert condition, name

    cfg = read(U / 'EXPERIMENT.json')
    rows = cfg['rows']; n = len(rows); classes = cfg['classes']; nc = len(classes)
    roles = {r: np.array([i for i, x in enumerate(rows) if x['role'] == r])
             for r in ('fit', 'validation', 'test')}
    labels = np.array([x['target'] for x in rows])
    contract_hash = sha(U / 'EXPERIMENT.json')
    inventory = read(U / 'data/DEEPHAND_INVENTORY.json')
    inventory_rows = inventory['rows']
    by_id = {r['id']: r for r in inventory_rows}
    check('all_3359_rows_and_original_45_classes', n == 3359 and nc == 45 and len(by_id) == 3359)
    check('original_class_targets_not_binary_remapped', all(
        x['label'] == by_id[x['id']]['label'] and classes[x['target']] == x['label'] for x in rows))
    check('fixed_signer_split', {r: len(i) for r, i in roles.items()} == {'fit':2693, 'validation':284, 'test':382}
          and all(x['role'] == ('fit' if int(x['signer'][-2:]) <= 5 else
                               'validation' if int(x['signer'][-2:]) <= 7 else 'test') for x in rows))
    for key in ('signer', 'clip', 'broadcast', 'pixel_sha256'):
        domains = defaultdict(set)
        for x in rows: domains[x[key]].add(x['role'])
        check('no_cross_role_' + key, all(len(v) == 1 for v in domains.values()))
    check('frozen_source_inventory_hash', sha(U / 'data/DEEPHAND_INVENTORY.json') == cfg['inventory_sha256'])
    lookup = {(r['clip'], r['original_frame_index']): i for i, r in enumerate(rows)}
    check('source_frame_identity_unique', len(lookup) == n)
    contexts = np.asarray([r['context_indices'] for r in rows])
    expected_contexts = np.asarray([[lookup.get((r['clip'], r['original_frame_index'] + d), -1)
                                   for d in (-2,-1,0,1,2)] for r in rows])
    check('exact_neighbor_indices_and_center_identity', np.array_equal(contexts, expected_contexts)
          and np.array_equal(contexts[:, 2], np.arange(n)))
    check('contexts_never_cross_roles', all(j < 0 or rows[j]['role'] == r['role']
          for r in rows for j in r['context_indices']))
    check('fixed_nominal_times_not_video_pts', cfg['temporal']['source_fps'] == 25
          and cfg['temporal']['relative_seconds'] == [-.08,-.04,0,.04,.08])
    features = {}
    pins = {'EXPERIMENT.json': contract_hash}
    for enc in cfg['encoder_candidates']:
        p = U / 'data' / (enc + '_features.npz'); z = np.load(p, allow_pickle=False)
        meta = read(U / 'data' / (enc + '_ENCODER.json'))
        check('encoder_cache_' + enc, list(z['ids']) == [r['id'] for r in rows]
              and z['features'].shape == (n, meta['feature_dim']) and np.isfinite(z['features']).all()
              and sha(p) == meta['features_sha256'] and meta['contract_sha256'] == contract_hash)
        features[enc] = {'dim': int(z['features'].shape[1]), 'sha256': sha(p)}
        pins[str(p.relative_to(U))] = sha(p)
    jp = U / 'data/PUBLIC_JOINTS.npz'; z = np.load(jp, allow_pickle=False)
    jm = read(U / 'data/PUBLIC_JOINTS_META.json')
    ji = {str(x): i for i, x in enumerate(z['image_ids'])}
    ix = np.array([ji[r['id']] for r in rows])
    raw = z['raw_xyz'][ix].astype(np.float32); mask = z['mask'][ix].astype(bool)
    check('joint_cache_source_alignment', len(ji) == n and raw.shape == (n,21,3)
          and mask.shape == (n,21) and sha(jp) == jm['npz_sha256'])
    rel = raw - raw[:, :1]
    scale = np.sqrt((rel[:, [5,9,13,17]] ** 2).sum(axis=2)).max(axis=1)
    seen = mask.all(axis=1) & (scale > 1e-6)
    rel /= np.maximum(scale[:,None,None], 1e-6); rel[~seen] = 0
    check('joint_normalization_missing_zero', np.isfinite(rel).all() and (rel[~seen] == 0).all()
          and int(seen.sum()) == 3195)
    pins[str(jp.relative_to(U))] = sha(jp)
    check('no_image_row_removed_for_missing_detector', sum(len(v) for v in roles.values()) == n)
    protected = jm['protected_before']
    check('protected_operating_files_unchanged', all(sha(Path(p)) == h for p,h in protected.items()), len(protected))
    check('fixed_fitting_budget_and_seeds', cfg['seeds'] == [27101,27102,27103]
          and cfg['probe']['epochs'] == 80 and cfg['temporal']['epochs'] == 60
          and cfg['probe']['batch'] == cfg['temporal']['batch'] == 64)
    add = read(U / 'MODEL_TRAINING_ADDENDUM.json')
    check('mask_baseline_and_capacity_addendum', add['recorded_before_training']
          and add['test_not_used'] and add['contract_sha256'] == contract_hash and 'MASK_T' in add['additional_diagnostic'])
    seeds = cfg['seeds']
    specs = {f'probe_{enc}_{seed}': {'epochs':80,'kind':'probe','name':enc,'seed':seed}
             for enc in cfg['encoder_candidates'] for seed in seeds}
    specs.update({f'observer_{condition}_{seed}': {'epochs':60,'kind':'observer','condition':condition,'seed':seed}
                  for condition in cfg['modalities'] + ['MASK_T'] for seed in seeds})
    freeze = read(U / 'TRAINING_FROZEN_BEFORE_TEST.json')
    check('all_24_expected_checkpoints_frozen', set(Path(x).stem for x in freeze['checkpoints']) == set(specs)
          and len(freeze['checkpoints']) == 24 and freeze['contract_sha256'] == contract_hash
          and freeze['encoder_selection_sha256'] == sha(U / 'ENCODER_SELECTION.json'))
    check('checkpoint_bytes_unchanged', all(sha(Path(p)) == h for p,h in freeze['checkpoints'].items()))
    selected = read(U / 'ENCODER_SELECTION.json')
    val_f1 = defaultdict(list); val_accuracy = defaultdict(list)

    def recompute(truth, score):
        assert score.shape == (len(truth), nc) and np.isfinite(score).all()
        pred = np.argmax(score, axis=1)
        cm = np.bincount(truth * nc + pred, minlength=nc * nc).reshape(nc,nc)
        support = cm.sum(1); predicted = cm.sum(0); tp = np.diag(cm)
        precision = np.divide(tp, predicted, out=np.zeros(nc,dtype=float), where=predicted>0)
        recall = np.divide(tp, support, out=np.zeros(nc,dtype=float), where=support>0)
        f1 = np.divide(2*tp, support+predicted, out=np.zeros(nc,dtype=float), where=(support+predicted)>0)
        return {'n':len(truth), 'correct':int(tp.sum()), 'accuracy':float(tp.sum()/len(truth)),
                'macro_f1_supported':float(f1[support>0].mean()), 'supported_classes':int((support>0).sum()),
                'confusion':cm.tolist(), 'per_class':[{'class':c,'support':int(support[i]),'predicted':int(predicted[i]),
                 'tp':int(tp[i]),'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])}
                 for i,c in enumerate(classes)]}

    def same(a, b):
        if isinstance(a, dict): return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list): return len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
        if isinstance(a, float): return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
        return a == b

    for stem, spec in specs.items():
        r = read(U / 'models' / (stem + '.json'))
        cp = U / 'models' / (stem + '.pt')
        trace = r['trace']
        check('fit_trace_' + stem, r['contract_sha256'] == contract_hash and sha(cp) == r['checkpoint_sha256']
              and r['test_used'] is False and len(trace) == spec['epochs']
              and all(t['epoch'] == e+1 and t['seen_count'] == 2693 and math.isfinite(t['train_loss'])
                      for e,t in enumerate(trace)))
        val = np.load(U / 'models' / (stem + '_validation.npz'), allow_pickle=False)
        independent = recompute(labels[roles['validation']], val['logits'])
        check('validation_saved_arithmetic_' + stem, np.array_equal(val['indices'],roles['validation'])
              and same(independent,r['validation']))
        if spec['kind'] == 'probe':
            val_f1[spec['name']].append(independent['macro_f1_supported'])
            val_accuracy[spec['name']].append(independent['accuracy'])
    speed = read(U / 'CPU_LATENCY.json')['models']
    means = {name: sum(v)/3 for name,v in val_f1.items()}
    best = max(means.values()); eligible = [name for name in cfg['encoder_candidates'] if means[name] >= best - .01]
    chosen = min(eligible, key=lambda name: speed[name]['p95_ms'])
    check('encoder_selection_validation_only_recomputed', chosen == selected['selected']
          and selected['test_used'] is False and selected['contract_sha256'] == contract_hash
          and {c['name'] for c in selected['eligible']} == set(eligible)
          and all(math.isclose(c['validation_macro_f1'],means[c['name']],abs_tol=1e-12)
                  for c in selected['candidates']))
    for name in ['train_observer.py','check_observer.py','MODEL_TRAINING_ADDENDUM.json','CPU_LATENCY.json',
                 'ENCODER_SELECTION.json','TRAINING_FROZEN_BEFORE_TEST.json']:
        pins[name] = sha(U / name)
    # Context labels define evaluation strata only; not an input feature.
    subgroups = {'detector_missing': ~seen[roles['test']], 'detector_present':seen[roles['test']]}
    for signer in ('signer08','signer09'):
        subgroups[signer] = np.array([rows[i]['signer'] == signer for i in roles['test']])
    subgroups['near_label_change'] = np.array([any(j>=0 and labels[j]!=labels[i] for j in contexts[i]) for i in roles['test']])
    check('heldout_reporting_denominators', int(subgroups['near_label_change'].sum()) == 19
          and sum(int(subgroups[k].sum()) for k in ['detector_missing','detector_present']) == 382
          and [int(subgroups[k].sum()) for k in ['signer08','signer09']] == [253,129])
    comparisons = []
    if stage == 'scores':
        check('all_24_test_pairs_present', all((U/'models'/(s+'_test.json')).exists()
              and (U/'models'/(s+'_test.npz')).exists() for s in specs))
        for stem in specs:
            p = U / 'models' / (stem + '_test.npz'); zt = np.load(p,allow_pickle=False)
            r = read(U / 'models' / (stem + '_test.json'))
            calc = recompute(labels[roles['test']],zt['logits'])
            check('test_saved_arithmetic_' + stem, np.array_equal(zt['indices'],roles['test'])
                  and same(calc,r['test']) and r['checkpoint_sha256'] == sha(U/'models'/(stem+'.pt'))
                  and r['freeze_sha256'] == sha(U/'TRAINING_FROZEN_BEFORE_TEST.json'))
            for key, sel in subgroups.items():
                check('subgroup_' + stem + '_' + key, same(recompute(labels[roles['test']][sel],zt['logits'][sel]),r['subgroups'][key]))
            comparisons.append({'stem':stem,'n':calc['n'],'correct':calc['correct'],
                                'accuracy':calc['accuracy'],'macro_f1_supported':calc['macro_f1_supported'],
                                'test_logits_sha256':sha(p)})
        allr = read(U/'COMPARISON.json')
        check('aggregate_preserves_all_24', len(allr['results']) == 24 and {r['stem'] for r in allr['results']} == set(specs)
              and allr['contract_sha256'] == contract_hash and allr['test_tuning'] is False
              and all(same(r,read(U/'models'/(r['stem']+'_test.json'))) for r in allr['results']))
    result = {'status':'PASS','at':datetime.now(timezone.utc).isoformat(),'stage':stage,
              'reviewer':'separate Codex agent with independent NumPy arithmetic; no producer imports or model execution',
              'checks':checks,'checked_count':len(checks),'pins_at_verification':pins,
              'rows':n,'roles':{k:len(v) for k,v in roles.items()},'original_classes':nc,
              'fit_classes':len(set(labels[roles['fit']])),'validation_classes':len(set(labels[roles['validation']])),
              'test_classes':len(set(labels[roles['test']])),'seen_landmarks':int(seen.sum()),
              'test_subgroup_denominators':{k:int(v.sum()) for k,v in subgroups.items()},
              'training_exposure_scope':'Source-reviewed fit-only sampling and stored epoch totals; no per-batch source-ID execution log',
              'test_scores_read':stage=='scores','model_inference_or_training_run_by_reviewer':False,
              'limitations':['External German-sign-language crop handshape classes, not KSL educational correctness.',
                             '45 labels retained but only 28 represented among 382 held-out crops.',
                             '19 near-change center crops overlap and represent only 6 changed adjacent edges in 5 source clips.',
                             'Test has no observed three-frame A-B-A pulse; short critical motion preservation remains unproved.',
                             'Existing C/T3F performance is not compared on this different task.',
                             'Preflight hashes recorded at review time are not falsely claimed as initial training-time code pins.'],
              'saved_score_checks':comparisons}
    write_new(V/('TRAINING_PREFLIGHT_INDEPENDENT.json' if stage=='preflight' else 'TRAINING_SCORES_INDEPENDENT.json'),result)
    print(json.dumps({'status':'PASS','stage':stage,'checks':len(checks),'test_scores_read':stage=='scores'},ensure_ascii=False))

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf8')
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','scores'],default='preflight')
    run(ap.parse_args().stage)
