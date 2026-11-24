"""Record post-evaluation direct RGB scope audit; never alter labels/scores."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

AC = Path(__file__).resolve().parent
AB = AC.parent / 'word100_action_completion_supervision_20261102'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return {'path': str(path), 'sha256': sha(path)}


IDS = [
    'NIA_SL_WORD1567_REAL13_F_directed_53_60__middle_splice',
    'NIA_SL_WORD2523_REAL12_F_directed_57_67__middle_splice',
]
LARGE = [[52, 54, 56, 60], [55, 57, 59, 63]]
OBSERVATIONS = [
    {
        'original_visible_sequence': 'At f52 the anatomical right hand is below the left; f53-55 shows the hands exchanging vertical order, including projected overlap near f54. At f56 the right hand is above the left and f56-60 continues its upward movement. The later f64-72 sequence descends again.',
        'edited_visible_sequence': 'The edited view retains f41-52 and f56-72. Thus right-low and right-high configurations, further upward movement f56-60, and the later downward movement all remain. The three intermediate crossing images f53-55 are absent. This is not removal of the whole rising phase or a verified missing repetition.',
        'observed_fragment_relationship': 'partial_source_contiguous_low_to_high_chain',
        'retained_motion_source_frames': list(range(56, 61)),
        'source_endpoints': [52, 56],
        'diagnostic_limit': 'The skipped crossing produces a larger adjacent image change, but no matched real faster performance or semantic judgement establishes that this change is an invalid sign. Existing still images alone do not establish a physically impossible movement speed.',
    },
    {
        'original_visible_sequence': 'At f55 raised fingertips are projected near one another in an oblique triangular configuration. f56-58 changes the hand tilt and opens the pair. At f59 the hands are already separated and f59-63 continues outward and toward a more horizontal arrangement. Later separated images remain.',
        'edited_visible_sequence': 'The edited view retains f45-55 and f59-79. The near configuration, separated configuration, and outward tail f59-63 remain. Only f56-58 of the connecting movement is missing. No entire handshape state, complete repetition, or required semantic step is shown to have been removed by this audit.',
        'observed_fragment_relationship': 'partial_source_contiguous_near_to_apart_chain',
        'retained_motion_source_frames': list(range(59, 64)),
        'source_endpoints': [55, 59],
        'diagnostic_limit': 'The f55-to-f59 image change can also be interpreted as a faster opening/tilting movement in the edited sequence. The audit does not prove that it is normal, nor that a single frontal input can distinguish it reliably from an intact faster execution. Projected fingertip proximity is not verified 3D contact.',
    },
]


def main():
    output = AC / 'PARTIAL_ERROR_SCOPE_DIAG.json'
    verified = AC / 'PARTIAL_ERROR_SCOPE_DIAG_VERIFIED.json'
    assert not output.exists() and not verified.exists(), 'Do not overwrite completed diagnosis.'
    base_paths = [AC / 'EVALUATION.json', AC / 'DEV_VIEW_TARGETS_V2.json',
                  AB / 'DEV_PROCESS_REVIEW_MANIFEST.json', AC / 'process_head.py', AC / 'process_data.py']
    pins = [pin(p) for p in base_paths]
    e = {r['case_id']: r for r in read(base_paths[0])['rows']}
    t = {r['view_id']: r for r in read(base_paths[1])['rows']}
    native = {r['id']: r for r in read(base_paths[2])['rows']}
    rows = []
    checks = []

    def check(name, condition):
        checks.append({'name': name, 'pass': bool(condition)})
        assert condition, name

    for view_id, large_frames, observation in zip(IDS, LARGE, OBSERVATIONS):
        r, score = t[view_id], e[view_id]
        n = native[r['query_id']]
        native_frames = {f['frame']: f for f in n['frames']}
        axis = r['ordered_frame_map']
        src = [f['source_native_frame'] for f in axis]
        check(view_id + ': role', r['role'] == score['role'] == 'development')
        check(view_id + ': frozen label remains 0', r['observed_process_evidence'] == score['truth'] == 0)
        check(view_id + ': source pins', r['source_sha256'] == score['source_sha256'] == n['source_sha256'] == sha(r['source_path']))
        check(view_id + ': no unavailable pixels', all(f['image_available'] and f['pixel_operation'] == 'copy' for f in axis))
        check(view_id + ': all source maps match native', all(native_frames[f['source_native_frame']]['timestamp_us'] == f['source_native_timestamp_us'] for f in axis))
        edited_pages = r['artifact_pages']
        original_pages = n['pages']
        for page in edited_pages + original_pages:
            check(view_id + ': page ' + page['path'], sha(page['path']) == page['sha256'])
        check(view_id + ': edited page coverage', [j for p in edited_pages for j in p['output_indices']] == list(range(len(axis))))
        check(view_id + ': original page coverage', [j for p in original_pages for j in p['frames']] == list(native_frames))
        for f in n['frames']:
            check(view_id + ': source image ' + str(f['frame']), sha(f['path']) == f['sha256'])
        seam = [i for i in range(1, len(src)) if src[i] != src[i-1] + 1]
        check(view_id + ': one known source seam', seam == r['seam_output_indices'] and len(seam) == 1)
        j = seam[0]
        left, right = axis[j-1], axis[j]
        missing = list(range(src[j-1] + 1, src[j]))
        source_gap = right['source_native_timestamp_us'] - left['source_native_timestamp_us']
        input_gap = right['view_timestamp_us'] - left['view_timestamp_us']
        check(view_id + ': seam endpoints', [src[j-1], src[j]] == observation['source_endpoints'])
        check(view_id + ': three deleted samples', len(missing) == 3 and source_gap == 133334 and input_gap == 33333)
        motion = observation['retained_motion_source_frames']
        check(view_id + ': positive remaining motion explicitly reviewed', any(w['source_frames'] == motion and w.get('label') == 1 for w in r['direct_review']['motion_witnesses']))
        chosen = score['qualification']['witnessed_chain']
        check(view_id + ': saved qualified display', score['qualification']['qualified_display'] == 1)
        check(view_id + ': chosen span crosses source seam', j-1 in chosen['output_indices'] and j in chosen['output_indices'])
        for key in ['whole_word_correctness', 'learner_error', 'required_repetitions']:
            check(view_id + ': no semantic truth ' + key, r[key] is None)
        review_pin = r['review_source']
        check(view_id + ': previous direct annotation pin', sha(review_pin['path']) == review_pin['sha256'])
        pins.append(review_pin)
        rows.append({
            'view_id': view_id, 'query_id': r['query_id'], 'clip_id': r['clip_id'],
            'physical_query': r['physical_query'], 'recorded_person_id': r['recorded_person_id'],
            'role': 'development', 'source_path': r['source_path'], 'source_sha256': r['source_sha256'],
            'input_sha256': score['input_sha256'], 'pixel_view_key': r['pixel_view_key'],
            'reviewed_edited_pages': edited_pages, 'reviewed_original_pages': original_pages,
            'all_retained_images_reviewed': True, 'all_original_context_images_reviewed': True,
            'reviewed_output_indices': list(range(len(axis))),
            'reviewed_source_frames_in_original': list(native_frames),
            'large_original_images_actually_opened': [native_frames[f] for f in large_frames],
            'ordered_frame_map': axis,
            'known_edit_audit': {
                'before': left, 'after': right, 'deleted_source_frames': missing,
                'deleted_frame_records': [native_frames[f] for f in missing],
                'original_endpoint_gap_us': source_gap, 'edited_endpoint_gap_us': input_gap,
                'time_removed_between_endpoints_us': source_gap-input_gap,
                'source_seam_output_index': j,
                'native_gap_information_available_to_model': False,
            },
            'direct_visual_observations': observation,
            'remaining_direct_motion_witnesses': r['direct_review']['motion_witnesses'],
            'frozen_label': 0,
            'frozen_label_meaning': 'No fully connected setup-to-terminal process directly witnessed within one retained native-contiguous fragment. This is a physical observability/continuity criterion, not a sign-language correctness label and not a no-movement label.',
            'original_direct_label_reason': r['evidence_reason'],
            'original_review': review_pin,
            'saved_evaluation_unchanged': {
                'truth': score['truth'], 'qualified_display': score['qualification']['qualified_display'],
                'fixed_contract_false_positive': True,
                'witnessed_chain_output_indices': chosen['output_indices'],
                'witnessed_chain_source_frames': chosen['source_frame_ids'],
                'local_selected_span_truth': score['qualification'].get('selected_span_truth'),
                'rgb_context_outside_nominal_source_frames': chosen['rgb_context_support']['outside_nominal_observed_source_frames'],
            },
            'input_only_identification_of_edit_vs_natural_fast_motion': 'not_established',
            'important_semantic_action_omission_established': None,
            'whole_word_correctness': None, 'learner_error': None, 'required_repetitions': None,
            'training_label': None, 'use_for_training': False,
        })
    # Static source verification only: no import of a neural model or forward call.
    code = (AC / 'process_head.py').read_text(encoding='utf-8')
    check('make_input contains observed geometry and RGB concat', 'np.concatenate([geometry,observed.astype(np.float32),video,video_good[:,None]]' in code)
    for p in pins:
        check('immutable pin ' + p['path'], sha(p['path']) == p['sha256'])
    report = {
        'status': 'POST_EVALUATION_DIRECT_SCOPE_AUDIT_NOT_RESCORING',
        'at': datetime.now(timezone.utc).isoformat(),
        'selection': 'Exactly the two parent-specified frozen DEV qualified false-positive cases; no search or score-based relabelling.',
        'rows': rows, 'pins': pins,
        'review_scope': {
            'edited_views': 2, 'public_sources': 2, 'actual_sheet_pages': 8,
            'actual_large_images': 8, 'edited_frame_slots': 61, 'original_frame_slots': 67,
            'method': 'Directly opened every original and edited contact-sheet page and eight existing full-size JPEGs. Sequential images were reviewed; real-time video playback and a blind natural-speed discrimination test were not performed.',
            'reviewer': 'Codex direct physical observation; same research team, exposed to saved outcomes.',
            'independent_human_or_KSL_expert': False, 'semantic_correctness_gold_created': False,
        },
        'runtime_information_contract': {
            'actually_used_neural_inputs': ['59 body-relative geometric values', '59 observation masks', '768-dimensional RGB features', 'RGB availability', 'physical-query embedding'],
            'not_neural_inputs': ['native source frame index', 'source PTS gap', 'known splice seam', 'edit kind', 'person ID', 'word ID'],
            'important_limit': 'The model may detect image discontinuity or gesture evidence, but native continuity cannot be guaranteed from unavailable source provenance. RGB features may use wider local context than a selected nominal interval; that context is not direct local process truth.',
        },
        'conclusion_ko': '두 사례는 중간 3프레임 약 0.10초가 빠졌지만 앞뒤 손 배치와 뒤쪽 움직임은 남아 있다. 현재 0은 원본의 연결 과정이 편집 없이 직접 보였는지에 대한 기준이다. 이를 중요한 수어 동작을 생략한 실제 오류 2건이라고 해석하면 과장이다. 일반적인 빠른 동작과 입력만으로 구별 가능하다는 근거도 아직 확보되지 않았다. 기존 실험의 false-positive 2건 기록과 채택 실패 판정은 변경하지 않는다.',
        'prospective_next_step': [
            'Keep the frozen benchmark/report intact and report its observability criterion separately from educational error detection.',
            'Before another learning run, define observable action evidence, observation insufficiency, and semantic omission as separate targets; do not make all cuts semantic negatives.',
            'For a future prospectively fixed evaluation, include matched intact faster actions and independently reviewable missing-state/direction/order cases. Synthetic controls remain synthetic and do not substitute for expert semantic acceptance.',
            'Do not feed the hidden edit/source seam as a deployable answer feature or tune thresholds to rescue these repeated DEV examples.',
        ],
        'unsupported_claims': ['the edited signs are definitely incorrect', 'the edited signs are definitely acceptable KSL', 'every lost intermediate frame is important', 'fast motion and edited motion are provably indistinguishable', 'this scope audit improves model performance'],
        'fixed_labels_modified': 0, 'fixed_scores_modified': 0, 'new_inference': 0,
        'new_training': 0, 'new_frame_extraction': 0, 'private_sources_accessed': 0,
        'routine_inspection_errors_preserved': [
            'One initial guessed sheet filename was absent; exact manifest path then used successfully. No source was modified.',
            'A read-only console metadata print hit cp949 UnicodeEncodeError; stdout UTF-8 was set and read-only inspection continued. No inference or data write occurred in that failed command.',
        ],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    verified.write_text(json.dumps({
        'status': 'PASS', 'artifact': pin(output), 'checks': checks,
        'check_count': len(checks), 'scope': 'Saved provenance, mappings, fixed-label preservation, and actual sheet/image pins only. Not independent semantic validation.',
        'new_inference': 0, 'new_training': 0,
    }, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'checks': len(checks), 'artifact': pin(output), 'verification': pin(verified)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
