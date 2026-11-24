"""Preflight + prepare the treatment training cache.

Reuses the verified base(2122) and context(224) prefixes byte-for-byte and computes
frozen-encoder prefixes ONLY for the 26 new windows (13 accepted WORD recordings x 2 spans).
No training, no evaluation, no development/user/reserved array is read here.
"""
import time
import numpy as np
import torch
from local_io import H, L, R, PRIOR, read, sha, write, policy, parent_model, pack
from generated_v1 import extract_prefix, predict_prefix

HELD = {"NIA_SL_WORD1290_REAL02_F", "NIA_SL_WORD1544_REAL02_F", "NIA_SL_WORD2485_REAL02_F"}


def main():
    started = time.time()
    p = policy()
    assert not (H / 'TREATMENT_PREFIX_READY.json').exists(), 'already prepared'

    plan = read(p['paths']['plan'])
    decisions = read(p['paths']['decisions'])
    checks = {}

    # ---- contract identity -------------------------------------------------
    checks['plan_status_ready_not_executed'] = plan['status'] == 'COMPARISON_CONTRACT_READY_NOT_EXECUTED'
    checks['plan_mapping_manifest_matches_decisions'] = plan['mapping_manifest_sha256'] == sha(p['paths']['decisions'])
    checks['plan_parent_matches_policy'] = plan['parent']['sha256'] == p['parent']['sha256']
    checks['plan_control_is_reuse_not_retrain'] = plan['control']['retrain'] is False
    checks['plan_control_matches_policy'] = plan['control']['reuse']['sha256'] == p['control']['sha256']
    checks['plan_trainable_head_only'] = plan['trainable'].startswith('head.* only')
    for k, v in p['settings'].items():
        if k in plan['settings']:
            checks['setting_' + k] = plan['settings'][k] == v
    checks['plan_base_rows_2122_unchanged'] = plan['base']['rows'] == 2122 and plan['base']['unchanged'] is True
    checks['plan_base_metadata_hash'] = plan['base']['metadata_sha256'] == sha(p['paths']['metadata'])

    # ---- held clips must be absent ----------------------------------------
    added = plan['context']['added_rows']
    added_clips = sorted({r['clip_id'] for r in added})
    checks['added_rows_26'] = len(added) == 26 == plan['context']['new_windows']
    checks['added_recordings_13'] = len(added_clips) == 13 == plan['context']['new_recordings']
    checks['held_three_excluded'] = HELD.isdisjoint(added_clips)
    held_rows = [r for r in decisions['rows'] if r['decision'] == 'HOLD_SPECIFIC_UNRESOLVED_MAPPING']
    checks['decisions_hold_exactly_three'] = len(held_rows) == 3
    checks['decisions_hold_matches_held_set'] = {r['clip_id'] for r in held_rows} == HELD
    checks['decisions_hold_not_training_approved'] = all(r['training_approved'] is False for r in held_rows)
    accepted_rows = [r for r in decisions['rows'] if r['decision'] == 'ACCEPT_SOURCE_LABEL_FOR_CLASSIFICATION']
    checks['decisions_accept_thirteen'] = len(accepted_rows) == 13
    checks['decisions_accept_matches_added'] = {r['clip_id'] for r in accepted_rows} == set(added_clips)
    checks['decisions_accept_all_train_role'] = all(r['split_role'] == 'train' for r in accepted_rows)
    checks['decisions_accept_all_signer_02'] = all(r['signer_id'] == 2 for r in accepted_rows)
    checks['decisions_counts_16_13_3'] = (decisions['counts']['candidates'] == 16
                                          and decisions['counts']['accepted_for_limited_comparison'] == 13
                                          and decisions['counts']['held'] == 3)
    checks['no_new_person_added'] = decisions['counts']['new_people'] == 0

    # ---- parent model and class order -------------------------------------
    ck, m, classes = parent_model(p)
    checks['classes_93'] = len(classes) == 93
    with np.load(p['paths']['prior_prefixes'], allow_pickle=False) as z:
        prior_classes = z['classes'].tolist()
        base_prefix, base_off = z['base_prefix'].copy(), z['base_offsets'].copy()
        ctx_prefix, ctx_off = z['pair_context_prefix'].copy(), z['pair_context_offsets'].copy()
        base_y, ctx_y = z['base_y'].copy(), z['pair_context_y'].copy()
        teacher = z['teacher_base_logits'].copy()
        old_base_idx, old_ctx_idx = z['base_indices'].copy(), z['context_indices'].copy()
    checks['prior_cache_class_order_identical'] = prior_classes == classes
    checks['prior_cache_receipt_pass'] = read(p['paths']['prior_prefix_receipt'])['status'] == 'PASS'
    checks['prior_cache_counts'] = len(base_off) - 1 == 2122 and len(ctx_off) - 1 == 224
    checks['prior_cache_teacher_shape'] = teacher.shape == (2122, 93)

    # ---- schedule invariants ----------------------------------------------
    with np.load(p['paths']['schedule'], allow_pickle=False) as z:
        bi = z['base_indices'].copy()
        cci = z['control_context_indices'].copy()
        tci = z['treatment_context_indices'].copy()
        repl = z['replaced_context_slots'].copy()
    checks['schedule_shapes_660x16'] = bi.shape == cci.shape == tci.shape == repl.shape == (660, 16)
    checks['schedule_base_identical_to_old_control'] = np.array_equal(bi, old_base_idx)
    checks['schedule_control_context_identical_to_old_control'] = np.array_equal(cci, old_ctx_idx)
    checks['schedule_nonreplaced_slots_preserved'] = np.array_equal(tci[~repl], cci[~repl])
    checks['schedule_replaced_slots_point_to_new_rows'] = bool((tci[repl] >= 224).all() and (tci[repl] < 250).all())
    checks['schedule_nonreplaced_slots_point_to_old_rows'] = bool((tci[~repl] < 224).all())
    checks['schedule_replacement_count_1082'] = int(repl.sum()) == plan['schedule']['actual_new_window_draws'] == 1082
    checks['schedule_base_indices_in_range'] = bool((bi >= 0).all() and (bi < 2122).all())
    checks['schedule_covers_all_26_new_windows'] = len(np.unique(tci[repl])) == 26

    # ---- build prefixes for the 26 new windows ONLY ------------------------
    new_x, new_y, new_meta = [], [], []
    for row in added:
        assert row['clip_id'] not in HELD
        assert sha(row['path']) == row['npz_sha256'], row['clip_id']
        with np.load(row['path'], allow_pickle=False) as z:
            x = z[row['array_key']].copy()
        assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 280 and np.isfinite(x).all()
        assert row['signer_id'] == 2
        new_x.append(np.ascontiguousarray(x))
        new_y.append(classes.index(row['label']))
        new_meta.append({'clip_id': row['clip_id'], 'array_key': row['array_key'], 'label': row['label'],
                         'class_index': classes.index(row['label']), 'signer_id': row['signer_id'],
                         'frames': int(x.shape[0]), 'source_family_id': row['source_family_id'],
                         'npz_sha256': row['npz_sha256']})
    checks['new_windows_all_labels_in_class_order'] = all(0 <= i < 93 for i in new_y)
    checks['new_windows_two_spans_per_recording'] = all(
        sorted(r['array_key'] for r in new_meta if r['clip_id'] == c) == ['x_annotation', 'x_full']
        for c in added_clips)

    new_prefix = extract_prefix(m, new_x)
    checks['new_prefix_count_26'] = len(new_prefix) == 26
    checks['new_prefix_frames_match_inputs'] = all(len(a) == len(b) for a, b in zip(new_prefix, new_x))
    checks['new_prefix_width_64'] = all(a.shape[1] == 64 for a in new_prefix)
    checks['new_prefix_finite'] = all(bool(np.isfinite(a).all()) for a in new_prefix)

    # sanity: the frozen parent already produces a decision on these windows
    parent_out = predict_prefix(m, new_prefix)
    w = ck['state_dict']['head.weight'].numpy().astype(float)
    b = ck['state_dict']['head.bias'].numpy().astype(float)
    score = parent_out['pooled'].astype(float) @ w.T + b
    checks['new_prefix_affine_matches_forward'] = float(np.max(np.abs(score - parent_out['logits']))) < 1e-4
    parent_pred = [classes[i] for i in score.argmax(1)]
    parent_correct = int(sum(a == r['label'] for a, r in zip(parent_pred, new_meta)))

    # ---- assemble the 250-row treatment context ---------------------------
    prior_ctx = [ctx_prefix[a:b].copy() for a, b in zip(ctx_off[:-1], ctx_off[1:])]
    treatment_ctx = prior_ctx + new_prefix
    treatment_y = np.concatenate([ctx_y, np.array(new_y, dtype=np.int64)])
    checks['treatment_context_250'] = len(treatment_ctx) == 250 == plan['context']['treatment_rows']
    checks['existing_first224_order_preserved'] = all(
        np.array_equal(treatment_ctx[i], prior_ctx[i]) for i in range(224))
    checks['existing_first224_labels_preserved'] = np.array_equal(treatment_y[:224], ctx_y)

    packed_ctx, off_ctx = pack(treatment_ctx)
    arrays = {
        'base_prefix': base_prefix, 'base_offsets': base_off, 'base_y': base_y,
        'context_prefix': packed_ctx, 'context_offsets': off_ctx, 'context_y': treatment_y,
        'teacher_base_logits': teacher,
        'base_indices': bi, 'context_indices': tci,
        'control_context_indices': cci, 'replaced_context_slots': repl,
        'classes': np.array(classes),
    }
    failed = [k for k, v in checks.items() if not v]
    assert not failed, failed
    with (H / 'treatment_prefixes.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)

    result = {
        'status': 'PASS' if not failed else 'FAIL',
        'checks': checks, 'checks_total': len(checks), 'checks_failed': failed,
        'policy_sha256': sha(H / 'local_policy.json'),
        'learner_sha256': sha(p['paths']['learner']),
        'cache_sha256': sha(H / 'treatment_prefixes.npz'),
        'new_windows': new_meta,
        'new_window_encoder_runs': 26,
        'reused_prefix_clips': {'base': 2122, 'context_existing': 224},
        'recomputed_prefix_clips': 26,
        'parent_top1_on_new_windows': {'correct': parent_correct, 'of': 26,
                                       'note': 'frozen parent sanity read only; not a generalization figure'},
        'seconds': round(time.time() - started, 2),
        'training_runs': 0, 'development_arrays_read': 0, 'user_arrays_read': 0,
        'reserved_arrays_read': 0, 'new_extraction_from_video': 0, 'new_downloads': 0,
    }
    write(H / 'TREATMENT_PREFIX_READY.json', result)
    print('checks', len(checks), 'failed', failed)
    print('parent top1 on 26 new windows:', parent_correct, '/ 26')
    print('cache sha256', result['cache_sha256'])
    print('seconds', result['seconds'])


if __name__ == '__main__':
    main()
