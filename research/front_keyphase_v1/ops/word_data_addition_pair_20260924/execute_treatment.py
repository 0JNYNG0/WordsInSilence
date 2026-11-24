"""One fixed head-only training run on the 250-row treatment context.

The learner is ops/partial_encoder_pair_20260922/generated_v1.py, reused verbatim.
run_pair() is the approved routine; its head-only arm IS this unit's candidate. The
routine's second arm (head + last block) is outside this unit's contract, is never
saved and is recorded as discarded.
"""
import copy, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from local_io import H, R, read, sha, write, plain, policy, parent_model, unpack
from generated_v1 import run_pair


def main():
    p = policy()
    ready = read(H / 'TREATMENT_PREFIX_READY.json')
    assert ready['status'] == 'PASS'
    assert sha(H / 'treatment_prefixes.npz') == ready['cache_sha256']
    assert sha(H / 'local_policy.json') == ready['policy_sha256']
    assert sha(p['paths']['learner']) == ready['learner_sha256']

    ck, m, classes = parent_model(p)
    with np.load(H / 'treatment_prefixes.npz', allow_pickle=False) as z:
        assert z['classes'].tolist() == classes
        base = unpack(z['base_prefix'], z['base_offsets'])
        context = unpack(z['context_prefix'], z['context_offsets'])
        by, cy = z['base_y'].copy(), z['context_y'].copy()
        teacher = z['teacher_base_logits'].copy()
        bi, ci = z['base_indices'].copy(), z['context_indices'].copy()
    assert len(base) == 2122 and len(context) == 250
    assert bi.shape == ci.shape == (660, 16)

    R.mkdir(exist_ok=False)
    pins = {str(H / n): sha(H / n) for n in
            ['local_policy.json', 'TREATMENT_PREFIX_READY.json', 'treatment_prefixes.npz',
             'local_io.py', 'prepare_treatment.py', 'execute_treatment.py']}
    pins[p['paths']['learner']] = sha(p['paths']['learner'])
    write(R / 'preflight.json', {
        'status': 'READY_FOR_ONE_HEAD_ONLY_RUN', 'pins': pins, 'parent': p['parent'],
        'control_reused_not_retrained': p['control'], 'train_clips': [len(base), len(context)],
        'train_signers': list(range(1, 10)), 'new_word_signer': 2,
        'user_training_records': 0, 'reserved_signer_records': 0, 'ksl_records': 0})

    started = time.time()

    def progress(arm, done, total):
        print({'arm': arm, 'steps': done, 'of': total, 'seconds': round(time.time() - started, 1)}, flush=True)

    kwargs = {k: p['settings'][k] for k in
              ['learning_rate', 'weight_decay', 'anchor_coefficient', 'output_coefficient']}
    out = run_pair(m, base, by, context, cy, teacher, bi, ci, progress=progress, **kwargs)

    # the head-only arm is this unit's candidate; the wider arm is discarded unsaved
    arm = out['control']
    discarded = {'arm_name_in_routine': 'treatment',
                 'trainable_names': list(out['treatment']['trainable_names']),
                 'saved': False,
                 'reason': 'Outside this unit contract (head.* only). Produced by the shared '
                           'approved routine in the same call and intentionally not persisted.'}

    state = arm['state_dict']
    allowed = {'head.weight', 'head.bias'}
    assert set(arm['trainable_names']) == allowed, arm['trainable_names']
    altered = [k for k in state if not torch.equal(state[k], ck['state_dict'][k])]
    assert altered and set(altered) <= allowed, altered
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
    assert arm['updates'] == 660 and np.isfinite(arm['losses']).all()
    assert arm['frozen_parameters_unchanged'] and arm['buffers_unchanged']

    control_ck = torch.load(p['control']['path'], map_location='cpu', weights_only=True)
    same_as_control = [k for k in state if torch.equal(state[k], control_ck['state_dict'][k])]
    encoder_identical_to_control = all(
        torch.equal(state[k], control_ck['state_dict'][k]) for k in state if not k.startswith('head.'))

    payload = copy.deepcopy(ck)
    payload['state_dict'] = state
    payload['arm'] = 'word_data_addition_head_only'
    payload['step'] = 660
    payload['produced_by'] = 'Approved Claude head-only learner (generated_v1.run_pair head arm), local execution'
    payload['parent_checkpoint'] = p['parent']['path']
    payload['parent_checkpoint_sha256'] = p['parent']['sha256']
    payload['settings'] = plain(out['settings'])
    payload['word_data_addition'] = {
        'policy_sha256': sha(H / 'local_policy.json'),
        'plan_sha256': sha(p['paths']['plan']),
        'mapping_manifest_sha256': sha(p['paths']['decisions']),
        'trainable_names': list(arm['trainable_names']),
        'context_rows': {'existing': 224, 'added': 26, 'total': 250},
        'new_recordings': 13, 'held_recordings_excluded': 3,
        'source_label_supervision_only': True,
        'approved_as_execution_reference': False,
        'augmentation_this_comparison': False,
        'all_buffers_fixed': True, 'encoder_fixed': True, 'last_block_fixed': True,
        'registered_as_default': False,
    }
    path = R / 'checkpoint_treatment_final.pt'
    torch.save(payload, path)

    with (R / 'training_arrays.npz').open('xb') as f:
        np.savez_compressed(f, base_indices=bi, context_indices=ci, treatment_losses=arm['losses'])

    for k, d in pins.items():
        assert sha(k) == d, k
    assert sha(p['parent']['path']) == p['parent']['sha256']
    assert sha(p['control']['path']) == p['control']['sha256']

    record = {
        'status': 'TRAINING_COMPLETE_CHECKPOINT_FROZEN_BEFORE_DEVELOPMENT_EVALUATION',
        'at_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'candidate': {'path': str(path), 'sha256': sha(path),
                      'changed_tensors': altered, 'trainable_names': list(arm['trainable_names'])},
        'discarded_arm': discarded,
        'parent': p['parent'], 'control_reused_not_retrained': p['control'],
        'encoder_identical_to_control': bool(encoder_identical_to_control),
        'tensors_identical_to_control': len(same_as_control),
        'settings': plain(out['settings']),
        'final_losses': {k: float(v) for k, v in zip(['ce', 'anchor', 'output', 'total'], arm['losses'][-1])},
        'seconds': round(time.time() - started, 2),
        'actual_training_runs_saved': 1, 'optimizer_updates_saved': 660,
        'retry_for_accuracy': 0, 'seed_or_coefficient_search': 0,
        'training_arrays_sha256': sha(R / 'training_arrays.npz'),
        'policy_sha256': sha(H / 'local_policy.json'),
        'user_arrays_read': 0, 'development_arrays_read': 0, 'reserved_arrays_read': 0, 'new_data': 0,
    }
    write(R / 'training.json', record)
    print('status', record['status'])
    print('changed tensors', altered)
    print('encoder identical to control:', encoder_identical_to_control)
    print('final losses', record['final_losses'])
    print('checkpoint sha256', record['candidate']['sha256'])
    print('seconds', record['seconds'])


if __name__ == '__main__':
    main()
