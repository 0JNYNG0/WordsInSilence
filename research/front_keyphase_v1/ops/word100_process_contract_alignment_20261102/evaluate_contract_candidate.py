"""One fixed combined candidate evaluation. No old models or extraction run.

Whole, predicted nominal and labelled diagnostic intervals share the process
target. Net-motion diagnostics retain their separate truth and output branch.
Each exact input/query/index selection is forwarded at most once per branch.
"""
from pathlib import Path
from datetime import datetime, timezone
import sys
import json
import time
import traceback
import numpy as np
import torch

U = Path(__file__).resolve().parent
AB = U.parent / 'word100_action_completion_supervision_20261102'
sys.path.insert(0, str(AB))
sys.path.insert(0, str(U))
from process_head import ProcessEvidenceNet, QUERIES, batch
from process_data import read, sha, load_eval_dataset, bind_nominal_targets
from process_runtime_contract import qualify, known_seam_diagnostic
from process_qualification import propose_motion_intervals
from evaluate_process_head import score_summary
from evaluate_process_auto import summarize
from process_rgb_support import span_support


def write(p, value):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def pins_ok(contract):
    for p, h in contract['pins'].items():
        assert sha(p) == h, p


def main():
    out = U / 'EVALUATION.json'
    started = U / 'EVALUATION_STARTED.json'
    assert not out.exists() and not started.exists(), 'No silent repeated evaluation'
    c = read(U/'ALIGNMENT_FROZEN.json'); pins_ok(c)
    train = read(U/'TRAIN_RESULT.json')
    assert train['status'] == 'FIXED_CANDIDATE_TRAINED_NOT_EVALUATED'
    assert train['contract_sha256'] == sha(U/'ALIGNMENT_FROZEN.json')
    assert sha(train['checkpoint']) == train['checkpoint_sha256']
    assert sha(train['normalization']) == train['normalization_sha256']
    data = load_eval_dataset(read(train['normalization']), include_runtime=True)
    assert len(data['records']) == 357
    nominal_spec = c['nominal_span_manifest']
    assert nominal_spec['required'] is True
    assert sha(nominal_spec['path']) == nominal_spec['sha256']
    nominal_binding = bind_nominal_targets(data['records'], nominal_spec['path'], data['pins'], required=True)
    for p, h in data['pins'].items():
        assert c['pins'].get(p) == h, 'Unfrozen evaluation input: ' + p
    rgb_support = read(U/'RGB_SUPPORT_MANIFEST.json')
    rgb_verified = read(U/'RGB_SUPPORT_VERIFIED.json')
    assert rgb_verified['status'].startswith('PASS')
    assert rgb_verified['manifest_sha256'] == sha(U/'RGB_SUPPORT_MANIFEST.json')
    assert rgb_verified['helper_sha256'] == sha(U/'process_rgb_support.py')
    for path in (U/'RGB_SUPPORT_MANIFEST.json', U/'RGB_SUPPORT_VERIFIED.json', U/'process_rgb_support.py',
                 U/'process_runtime_contract.py', U/'process_data.py', U/'process_head.py', Path(__file__),
                 AB/'process_qualification.py', AB/'evaluate_process_head.py', AB/'evaluate_process_auto.py'):
        assert c['pins'].get(str(path.resolve())) == sha(path), 'Unfrozen evaluation dependency: ' + str(path)
    torch.set_num_threads(4)
    model = ProcessEvidenceNet()
    model.load_state_dict(torch.load(train['checkpoint'], map_location='cpu', weights_only=True))
    model.eval()
    write(started, dict(at=datetime.now(timezone.utc).isoformat(), queries=357,
                       checkpoint_sha256=train['checkpoint_sha256'],
                       contract_sha256=sha(U/'ALIGNMENT_FROZEN.json')))
    memo = {}; motion_memo = {}; forward_counts = {'process':0, 'netmotion':0}
    start = time.perf_counter(); rows = []

    def prediction(r, indices, branch='process'):
        # Input hash + source selection distinguish fresh bounded/full-native
        # observations of the same original frame. This is memoization only.
        ii = list(map(int, indices))
        key = (r['case_id'], tuple(ii))
        cache = memo if branch == 'process' else motion_memo
        if key not in cache:
            seq = data['sequences'][r['array_key']][ii]
            x, lengths = batch([seq])
            q = torch.tensor([QUERIES.index(r['physical_query'])])
            if branch == 'process':
                p = model(x, lengths, q)
                cache[key] = (float(p['process_logits'].sigmoid()[0]),
                              p['point_logits'].sigmoid()[0, :len(ii)].tolist())
            else:
                cache[key] = float(model.interval_logits(x, lengths, q).sigmoid()[0])
            forward_counts[branch] += 1
        return cache[key]

    # Same source groups and fixed queries as the frozen baseline. Targets are
    # used for reporting after inference, never to select runtime intervals.
    with torch.inference_mode():
        for r in data['records']:
            t0 = time.perf_counter(); n = len(r['axis'])
            p, points = prediction(r, range(n))
            available = [a['image_available'] for a in r['axis']]
            spec = propose_motion_intervals(points, available)
            local_scores = [prediction(r, v['output_indices'])[0] for v in spec['candidates']]
            q = qualify(p, points, available, local_scores)
            runtime_seconds = time.perf_counter() - t0
            # Assign source-time metadata only AFTER neural predictions.
            support_entry = rgb_support['inputs'][r['input_sha256']]
            assert support_entry['source_sha256'] == r['source_sha256']
            assert support_entry['role'] == r['role'] and support_entry['person'] == r['person']
            assert support_entry['clip_id'] == r['clip_id']
            support = support_entry['support']
            source_to_cache = {f:i for i,f in enumerate(support['source_frame_ids'])}
            cache_indices = [source_to_cache[a['source_frame']] for a in r['axis']]
            assert len(set(cache_indices)) == n
            assert [support['source_timestamp_us'][j] for j in cache_indices] == [a['source_timestamp_us'] for a in r['axis']]
            assert [support['image_available'][j] for j in cache_indices] == available
            q['global_rgb_context_support'] = span_support(support, cache_indices)
            for v in q['interval_candidates']:
                axis = [r['axis'][j] for j in v['output_indices']]
                v['source_frame_ids'] = [a['source_frame'] for a in axis]
                v['source_timestamps_us'] = [a['source_timestamp_us'] for a in axis]
                v['rgb_context_support'] = span_support(support, [cache_indices[j] for j in v['output_indices']])
            original = r['original_target']
            seams = original.get('seam_output_indices', [])
            point_scores = [dict(output_index=t['output_index'], source_frame=t['source_frame'],
                                 setup_truth=t['setup_relation'], terminal_truth=t['terminal_relation'],
                                 setup_probability=points[t['output_index']][0],
                                 terminal_probability=points[t['output_index']][1])
                            for t in r['sparse_relation_targets']]
            motion_scores = [dict(output_indices=t['output_indices'], source_frames=t['source_frames'],
                                  truth=t['label'], probability=prediction(r, t['output_indices'], 'netmotion'))
                             for t in r['fully_retained_motion_interval_targets']]
            nominal_scores = []
            for t in r['nominal_process_targets']:
                assert r['role'] == 'fit', 'Nominal labels here are FIT supervision diagnostics only'
                ii = t['output_indices']
                score = prediction(r, ii)[0]
                nominal_scores.append(dict(span_id=t['span_id'], output_indices=ii,
                    source_frames=t['source_frames'], source_timestamps_us=t['source_timestamps_us'],
                    image_available=t['image_available'], truth=t['observed_process_evidence'],
                    binding_kind=t['binding_kind'], direct_review=t['direct_review'],
                    probability=score, raw_display=1 if score>=.8 else 0 if score<=.2 else None,
                    rgb_context_support=span_support(support, [cache_indices[j] for j in ii]),
                    scope='FIT supervised nominal-span diagnostic; not independent DEV local accuracy',
                    whole_word_correctness=None, learner_error=None))
            truth = r['observed_process_evidence']
            rows.append(dict(case_id=r['case_id'], view_id=r['case_id'], query_id=r['query_id'],
                clip_id=r['clip_id'], person=r['person'], role=r['role'], origin=r['origin'],
                physical_query=r['physical_query'], input_sha256=r['input_sha256'],
                source_sha256=r['source_sha256'],
                kind=original.get('construction_kind_audit_only', 'automatic_runtime_proposal'),
                truth=truth, probability=p, raw_display=q['raw_display'],
                point_probabilities=points, sparse_point_scores=point_scores,
                interval_scores=motion_scores, nominal_process_scores=nominal_scores, qualification=q,
                oracle_known_seam_diagnostic=known_seam_diagnostic(q, seams),
                axis=r['axis'], frames=n, image_available=available,
                runtime_head_seconds=runtime_seconds,
                diagnostic_inclusive_seconds=time.perf_counter()-t0,
                recovered=truth==1 and q['qualified_display']==1,
                learned_selected_chain_exact_time_truth=None,
                additional_DTW=0, exact_boundary_accuracy_claim=False,
                end_to_end_latency_measured=False, whole_word_correctness=None, learner_error=None))
    summaries = {}
    for role in ('fit', 'development'):
        vv = [r for r in rows if r['role']==role and r['origin']=='reviewed_view']
        aa = [r for r in rows if r['role']==role and r['origin']=='runtime_proposal']
        per_query = {q:score_summary([r for r in vv if r['physical_query']==q]) for q in QUERIES}
        per_source = {qid:score_summary([r for r in vv if r['query_id']==qid]) for qid in sorted({r['query_id'] for r in vv})}
        ba = [v['raw_balanced_accuracy'] for v in per_query.values() if v['raw_balanced_accuracy'] is not None]
        source_ba = [v['raw_balanced_accuracy'] for v in per_source.values() if v['raw_balanced_accuracy'] is not None]
        summaries[role] = dict(raw=score_summary(vv), raw_per_query=per_query,
            raw_macro_query_balanced_accuracy=float(np.mean(ba)),
            raw_macro_source_query_balanced_accuracy=float(np.mean(source_ba)),
            qualification=summarize(vv), automatic=summarize(aa),
            automatic_per_query={q:summarize([r for r in aa if r['physical_query']==q]) for q in QUERIES},
            native_per_query={q:summarize([r for r in vv if r['kind']=='native' and r['physical_query']==q]) for q in QUERIES},
            by_kind={k:summarize([r for r in vv if r['kind']==k]) for k in sorted({r['kind'] for r in vv})})
    pins_ok(c)
    write(out, dict(status='ONE_FIXED_ALIGNED_CANDIDATE_EVALUATED_NOT_ADOPTION', rows=rows,
        summaries=summaries, seconds=time.perf_counter()-start, forward_counts=forward_counts,
        checkpoint_sha256=train['checkpoint_sha256'], contract_sha256=sha(U/'ALIGNMENT_FROZEN.json'),
        old_models_executed=0, extraction=0, new_DTW=0, whole_goal_complete=False,
        repeated_development=True, independent_human_expert=False,
        selected_span_local_truth_not_inferred=True, runtime_rgb_support_required_before_display=True,
        rgb_support_manifest_sha256=sha(U/'RGB_SUPPORT_MANIFEST.json'),
        rgb_support_verification_sha256=sha(U/'RGB_SUPPORT_VERIFIED.json'),
        nominal_span_manifest_sha256=nominal_spec['sha256'], nominal_binding=nominal_binding,
        nominal_diagnostic_is_fit_only=True,
        end_to_end_speed_improvement_claim=False))
    print(json.dumps(summaries['development'], ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        failure = U/'EVALUATION_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), encoding='utf8')
        raise
