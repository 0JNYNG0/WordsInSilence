"""Independent focused checks for the new quota sampler; synthetic inputs only."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys
import traceback

import numpy as np

OPS = Path(__file__).resolve().parent
WEB = OPS.parent.parent
sys.path[:0] = [str(WEB), str(WEB / 'src'), str(WEB.parent / 'sign_model_pilot_v1/src')]
import tools_head_only_native_pair as h
import tools_native_data_pair as base
import tools_public_replay_quota as subject

checks = []

def check(name, passed):
    checks.append({'name': name, 'passed': bool(passed)})
    if not passed:
        raise AssertionError(name)

def rejected(name, operation):
    try:
        operation()
    except (ValueError, RuntimeError, AssertionError, TypeError):
        check(name, True)
    else:
        check(name, False)

def main():
    public_ids = ['public_%02d' % i for i in range(18)]
    classes = ['class_%02d' % i for i in range(19)]
    gen = np.random.default_rng(1201)
    pool = [dict(sample_id='source_%02d' % i, class_index=i, label=classes[i],
                 origin='source', x=gen.normal(size=(4, 280)).astype(np.float32))
            for i in range(19)]
    pool += [dict(sample_id=sid, class_index=i, label=classes[i],
                  origin='nikl_original', x=gen.normal(size=(4, 280)).astype(np.float32))
             for i, sid in enumerate(public_ids)]
    groups, keys = h._group_by_class(pool)
    raw = np.random.default_rng(998).random(3 * 32 * 2).reshape(3, 32, 2)
    sampler, stats = subject.make_sampler(h._sample_batch, pool, public_ids)
    rng = base.UniformSchedule(raw.reshape(-1).tolist())
    original_bytes = {it['sample_id']: it['x'].tobytes() for it in pool}
    emitted = []
    for step in raw:
        actual = sampler(groups, keys, 32, rng)
        expected = []
        for j, (a, b) in enumerate(step):
            if j < 16:
                group = groups[keys[int(a * len(keys))]]
                expected.append(group[int(b * len(group))])
            else:
                expected.append(pool[19 + int(b * 18)])
        check('batch follows independent fixed quota and unchanged first-half draws',
              len(actual) == 32 and all(a is b for a, b in zip(actual, expected)))
        emitted.extend(actual)
    check('exactly two RNG values consumed for every sample', rng.consumed == 192)
    check('input arrays and metadata references are preserved',
          all(it['x'].tobytes() == original_bytes[it['sample_id']] for it in pool))
    sampler2, stats2 = subject.make_sampler(h._sample_batch, pool, public_ids)
    rng2 = base.UniformSchedule(raw.reshape(-1).tolist())
    replayed = [it for _ in range(3) for it in sampler2(groups, keys, 32, rng2)]
    check('same schedule emits same IDs and proof counters',
          [it['sample_id'] for it in emitted] == [it['sample_id'] for it in replayed]
          and stats == stats2)
    rejected('wrong batch is rejected', lambda: sampler(groups, keys, 31, rng))
    rejected('missing public original is rejected',
             lambda: subject.make_sampler(h._sample_batch, pool[:-1], public_ids))
    rejected('duplicate replay ID is rejected',
             lambda: subject.make_sampler(h._sample_batch, pool, public_ids[:-1] + public_ids[:1]))
    bad = [dict(it) for it in pool]
    bad[-1]['origin'] = 'participant_capture'
    rejected('non-public origin cannot enter replay quota',
             lambda: subject.make_sampler(h._sample_batch, bad, public_ids))
    bad = [dict(it) for it in pool]
    bad[-1]['class_index'] = bad[-2]['class_index']
    rejected('public class coverage cannot silently shrink',
             lambda: subject.make_sampler(h._sample_batch, bad, public_ids))
    bad = [dict(it) for it in pool]
    bad.append(dict(bad[0]))
    rejected('duplicate training ID is rejected',
             lambda: subject.make_sampler(h._sample_batch, bad, public_ids))
    sampler3, _ = subject.make_sampler(h._sample_batch, pool, public_ids)
    bad_groups = dict(groups)
    bad_groups[keys[0]] = []
    rejected('changed empty class grouping is rejected',
             lambda: sampler3(bad_groups, keys, 32, base.UniformSchedule(raw.reshape(-1))))

    # Exercise the actual old trainer through the new run hook, with synthetic model/data.
    import torch
    from pilot.model_masked_v1 import MaskedSignTCN
    torch.set_num_threads(2)
    torch.manual_seed(711)
    parent = MaskedSignTCN(width=8, dilations=(1,), dropout=0.5, n_classes=len(classes))
    plan = {'settings': {'batch_size': 32}, 'augmentation': False,
            'head_only': {'encoder_lr': 0.0, 'freeze_encoder_buffers': True,
                          'training_scope': 'head_only_frozen_encoder_and_buffers'},
            'public_replay': {'schema': 'fixed_public_original_quota_v1',
                              'sample_ids': public_ids, 'quota': 16, 'batch_size': 32,
                              'expected_origin': 'nikl_original', 'transforms': 'none',
                              'draw_rule': 'two draws per sample; explicit public quota'}}
    prepared = {'settings': {'encoder_lr': 0.0001},
                'nikl_items': [{k: it[k] for k in ('sample_id', 'class_index', 'label')}
                               for it in pool[19:]]}
    before_plan = copy.deepcopy(plan)
    before_prepared = copy.deepcopy(prepared)
    old_host, old_sampler, old_trainer = h._ORIGINAL_BASE_RUN, h._sample_batch, h.run_head_only_training
    old_slot, old_base_run = h.u.run_training, base.run
    host_calls = []
    trained = []
    configs = []
    def fake_host(p, prep, out):
        host_calls.append(True)
        for _ in range(2):
            model = copy.deepcopy(parent)
            config = {'optimizer_steps': 3, 'batch': {'total': 32, 'source': 32, 'nikl': 0},
                      'lr': {'encoder': 0.0, 'head': 0.001}, 'weight_decay': 0.001,
                      'seed': 998, 'batch_size': 32, 'cpu_threads': 2, 'augmentation': False}
            train_rng = base.UniformSchedule(raw.reshape(-1).tolist())
            result = h.u.run_training(model, pool, [], classes, config, train_rng, log=lambda _: None)
            trained.append((model, result, train_rng.consumed))
            configs.append(config)
        return 'SYNTHETIC_HOST_COMPLETED'
    try:
        h._ORIGINAL_BASE_RUN = fake_host
        check('new wrapper delegates through existing head-only training',
              subject.run(plan, prepared, str(OPS / 'unused_synthetic')) == 'SYNTHETIC_HOST_COMPLETED')
        check('both per-arm states are independent and consume correct RNG',
              len(trained) == 2 and all(t[2] == 192 for t in trained)
              and trained[0][1]['public_replay'] == trained[1][1]['public_replay'])
        check('effective sampling metadata recorded while optimizer unchanged',
              all(c['public_replay']['effective_quota_per_batch'] == 16 and c['lr'] == {'encoder': 0.0, 'head': 0.001}
                  and c['augmentation'] is False for c in configs))
        check('original plan and prepared metadata unchanged', plan == before_plan and prepared == before_prepared)
        check('real synthetic trainer preserved encoder and buffers',
              all(torch.equal(v, model.state_dict()[k]) for model, _, _ in trained
                  for k, v in parent.state_dict().items() if not k.startswith('head.')))
        check('classifier actually learned through wrapped path',
              all(not torch.equal(parent.head.weight, model.head.weight) for model, _, _ in trained))
        check('all temporary training hooks restored on success',
              h._sample_batch is old_sampler and h.run_head_only_training is old_trainer
              and h.u.run_training is old_slot and base.run is old_base_run)
        bad_plan = copy.deepcopy(plan)
        bad_plan['public_replay']['quota'] = True
        n = len(host_calls)
        rejected('invalid plan stops before host training',
                 lambda: subject.run(bad_plan, prepared, 'unused'))
        check('invalid plan performed no training call', len(host_calls) == n)
        def raising_host(*args):
            raise RuntimeError('deliberate synthetic host failure')
        h._ORIGINAL_BASE_RUN = raising_host
        rejected('host exception is propagated', lambda: subject.run(plan, prepared, 'unused'))
        check('all temporary hooks restored after host error',
              h._sample_batch is old_sampler and h.run_head_only_training is old_trainer
              and h.u.run_training is old_slot and base.run is old_base_run)
    finally:
        h._ORIGINAL_BASE_RUN = old_host
    return {'sampler_stats': stats, 'trainer_stats': trained[0][1]['public_replay']}

if __name__ == '__main__':
    details = {}
    error = None
    try:
        details = main()
    except Exception:
        error = traceback.format_exc()
    report = {'status': 'PASS_FOCUSED_QUOTA_WRAPPER' if error is None else 'FAIL',
              'passed': sum(c['passed'] for c in checks), 'total': len(checks),
              'checks': checks, 'error': error, 'details': details,
              'helper_sha256': hashlib.sha256((WEB / 'tools_public_replay_quota.py').read_bytes()).hexdigest(),
              'real_data_arrays_used': False, 'real_models_loaded': False}
    with (OPS / 'verification_v1.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ('status', 'passed', 'total', 'error')}))
    raise SystemExit(0 if error is None else 1)
