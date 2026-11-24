"""FIT-only structural check of saved spans. No scores are recomputed."""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json

U = Path(__file__).resolve().parent
checks = 0


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(value, reason):
    global checks
    checks += 1
    if not value:
        raise AssertionError(reason)


def main():
    out = U/'PROPOSAL_SUPPORT_VERIFIED.json'
    check(not out.exists(), 'Never overwrite completed audit')
    audit = read(U/'PROPOSAL_SUPPORT_AUDIT.json')
    evaluation = read(U/'EVALUATION.json')
    labels = read(U/'NOMINAL_PROCESS_TARGETS.json')
    verified = read(U/'EVALUATION_VERIFIED.json')
    check(verified['evaluation_sha256'] == sha(U/'EVALUATION.json'), 'Evaluation lineage')
    for p, h in audit['pins'].items():
        check(sha(p) == h, 'Changed frozen audit dependency: '+p)
    # Non-FIT records are filtered before inspecting any probabilities/targets.
    fit = {r['case_id']: r for r in evaluation['rows'] if r['role'] == 'fit' and r['origin'] == 'runtime_proposal'}
    direct_list = [t for t in labels['rows'] if t['role'] == 'fit'
                   and t['binding_kind'] == 'explicit_direct_chain' and t['observed_process_evidence'] == 1]
    direct = {t['case_id']: t for t in direct_list}
    check(len(direct_list) == len(direct) == 44, 'Ambiguous or changed direct chain population')
    check(len(audit['rows']) == 44 and {r['case_id'] for r in audit['rows']} == set(direct), 'Audited scope differs')
    summary, supplemental, details = Counter(), Counter(), []
    for saved in audit['rows']:
        r, t = fit[saved['case_id']], direct[saved['case_id']]
        axis, av, points = r['axis'], r['image_available'], r['point_probabilities']
        source = [a['source_frame'] for a in axis]
        check(source == sorted(set(source)), 'Source axis not monotonic/unique')
        check(t['source_sha256'] == r['source_sha256'] and t['input_sha256'] == r['input_sha256'], 'Direct label input mismatch')
        check(t['physical_query'] == r['physical_query'] == saved['physical_query'], 'Physical query changed')
        check(t['source_frames'] == saved['known_direct_chain_source_frames'], 'Direct source span changed')
        check(set(t['source_frames']) <= set(source), 'Native proposal does not contain original reviewed chain')
        check([axis[j]['source_frame'] for j in t['output_indices']] == t['source_frames'], 'Direct local/native mapping')
        check([axis[j]['source_timestamp_us'] for j in t['output_indices']] == t['source_timestamps_us'], 'Direct native PTS')
        # Precompute the rightmost member of each positive terminal relation
        # run by scanning backwards, rather than repeating the audit while-loop.
        tails = {}
        for j in reversed(range(len(source))):
            if av[j] and points[j][1] >= .8:
                tails[j] = tails.get(j+1, j)
        observed = t['original_sequence']
        motion = set(range(observed['motion_interval'][0], observed['motion_interval'][1]+1))
        setup = set(observed['setup_frames'])
        terminal = set(observed['terminal_frames'])
        known = set(t['source_frames'])
        original_witness = set(t['required_source_frames'])
        check(setup <= original_witness and motion <= original_witness and terminal <= original_witness, 'Source roles no longer bound to chain')
        check([axis[j]['source_frame'] for j in t['explicit_chain_witness_output_indices']] == t['required_source_frames'], 'Sparse original witness mapping')
        candidates = r['qualification']['interval_candidates']
        check(len(candidates) == len(saved['candidates']), 'Candidate list changed')
        incumbent, envelope, old_motion, old_partial, rgb_motion, rgb_chain = [], [], [], [], [], []
        for c, a in zip(candidates, saved['candidates']):
            start, end = c['start'], c['end']
            check(end in tails and av[start] and points[start][0] >= .8, 'Candidate endpoint not valid observed relation')
            tail = tails[end]
            check(all(av[j] for j in range(start, tail+1)), 'Envelope bridges unavailable pixels')
            nominal = set(source[start:end+1])
            wider = set(source[start:tail+1])
            inc = known <= nominal
            env = known <= wider
            expected = dict(saved_start=source[start], saved_end=source[end], same_terminal_run_tail=source[tail],
                old_contains_reviewed_chain=inc, unscored_envelope_contains_reviewed_chain=env,
                appended_frames=tail-end, exact_span_truth=None,
                saved_local_probability=c['connected_process_probability'], unscored_envelope_probability=None)
            check(a == expected, 'Saved structural candidate differs')
            check((original_witness <= nominal) == inc, 'Dense span containment differs from sparse witnesses')
            check((original_witness <= wider) == env, 'Envelope sparse witness scope mismatch')
            incumbent.append(inc)
            envelope.append(env)
            old_motion.append(motion <= nominal)
            old_partial.append(bool(motion <= nominal and (setup & nominal) and
                                    ({z for z in terminal if z >= max(motion)} & nominal)))
            rgb = set(c['rgb_context_support']['observed_context_source_frames'])
            rgb_motion.append(motion <= rgb)
            rgb_chain.append(known <= rgb)
        expected_row = dict(incumbent_contains_chain=any(incumbent),
            unscored_run_envelope_contains_chain=any(envelope), no_point_candidate=not candidates,
            saved_confirmed=r['qualification']['qualified_display'] == 1)
        for k, v in expected_row.items():
            check(saved[k] == v, 'Saved case count differs: '+k)
            summary[k] += v
        check(saved['known_direct_chain_not_exact_event_boundary'] is True and saved['native_scope_contains_chain'] is True, 'Scope annotation')
        supplemental['old_nominal_contains_full_reviewed_motion_interval'] += any(old_motion)
        supplemental['old_nominal_contains_motion_and_some_setup_and_later_terminal'] += any(old_partial)
        supplemental['old_RGB_context_contains_reviewed_motion_interval'] += any(rgb_motion)
        supplemental['old_RGB_context_contains_whole_reviewed_chain_span'] += any(rgb_chain)
        if not any(incumbent):
            supplemental['missing_chain_cases'] += 1
            supplemental['missing_chain_also_missing_reviewed_motion_nominally'] += not any(old_motion)
            supplemental['missing_chain_but_RGB_has_entire_reviewed_motion'] += any(rgb_motion)
            supplemental['missing_chain_but_RGB_has_entire_reviewed_chain_span'] += any(rgb_chain)
        details.append(dict(case_id=saved['case_id'], physical_query=r['physical_query'],
            original_motion_interval=observed['motion_interval'], original_setup_frames=observed['setup_frames'],
            original_terminal_frames=observed['terminal_frames'],
            nominal_contains_reviewed_chain=any(incumbent), nominal_contains_reviewed_motion=any(old_motion),
            RGB_context_contains_reviewed_motion=any(rgb_motion), RGB_context_contains_reviewed_chain=any(rgb_chain),
            unscored_envelope_contains_reviewed_chain=any(envelope)))
    check(dict(summary) == audit['counts'], 'Aggregate counts differ')
    check(audit['directly_reviewed_fit_chains'] == 44 and audit['development_rows_used'] == 0, 'Split/count claim')
    check(audit['alternative_model_inference'] == audit['training'] == 0, 'Unexpected execution')
    check(audit['alternative_accuracy_claim'] is False and audit['adoption'] is False, 'Unsupported performance claim')
    report = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='PASS_FIT_SAVED_STRUCTURAL_AUDIT_WITH_SCOPE_LIMITS',
        audit_sha256=sha(U/'PROPOSAL_SUPPORT_AUDIT.json'), verifier_sha256=sha(__file__),
        checks=checks, directly_reviewed_FIT_chains=44, reconstructed_counts=dict(summary),
        supplemental_scope_counts=dict(supplemental), rows=details,
        interpretation=[
            '14/44 stored first-terminal candidate sets contain the entire already-reviewed physical chain span; unscored same-terminal-run envelopes contain44/44. This is structural inclusion only.',
            'All30 noncontained chains also lack some explicitly reviewed motion-interval frames in the nominal candidate. The discrepancy is not explained solely by requiring an unnecessarily late held terminal photograph.',
            'However29/30 of those cases still present the entire reviewed motion interval inside frozen RGB encoder context. Therefore the model was not necessarily blind to later image pixels: the nominal recurrent/geometry time span and RGB context disagree.',
            'The annotated motion interval is a reviewed directional-progress interval, not every-frame motion truth or an exact minimum extent. Containment is not proof that shorter spans are linguistically incomplete.',
            'Same-run extension may include long holds, return movements or repetition. Its false-acceptance and localization behavior were not evaluated, and no new probability is computed.',
            'The44 cases are known positive FIT chains selected by available direct supervision, not a representative test of all signs or an independent DEV improvement.'
        ],
        no_new_threshold_or_labels=True, new_model_inference=0, new_training=0,
        development_rows_used=0, alternative_evaluation_scores=0, exact_boundary_accuracy_claim=False,
        enough_or_complete_action_claim=False, adoption=False, independent_human_expert=False,
        old_artifacts_modified=False, whole_goal_complete=False)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], checks=checks, sha256=sha(out), counts=summary, supplemental=supplemental)))


if __name__ == '__main__':
    main()
