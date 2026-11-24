"""Research-only source-linked descriptors. No linguistic correctness classifier.

The sign ID selects candidate descriptions, never fills missing observations.
Dictionary text mentions and initial-handshape metadata are not token labels.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = 'wis_source_linked_feature_profile_v1'
# Official Korean dictionary notation; NOT MediaPipe landmark numbers.
DICTIONARY_FINGERS = {'1': 'index', '2': 'middle', '3': 'ring', '4': 'little', '5': 'thumb'}
FAMILIES = {
    'handshape': ['주먹', '손바닥', '펴서', '구부', '벌려', '1지', '2지', '3지', '4지', '5지'],
    'body_relation': ['이마', '코', '턱', '목', '가슴', '얼굴', '어깨', '입', '팔'],
    'orientation': ['바닥이', '등이', '향하게', '마주 보게'],
    'contact_description': ['댄다', '댔다가', '대고', '닿', '접촉'],
    'movement': ['올린', '내린', '내려', '돌린', '돌리', '흔든', '쓸', '움직', '당긴', '벌린', '뻗'],
    'order': ['다음', '댔다가', '동시에'],
    'repetition_description': ['두 번', '세 번', '반복'],
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def source_claims(entry):
    """Return exact, traceable text mentions. Do not infer compulsory rules."""
    out = []
    source = {k: entry.get(k) for k in ('source_id', 'entry_id', 'url', 'content_sha256')}
    text = entry.get('description_text') or ''
    for family, terms in FAMILIES.items():
        spans = []
        for term in terms:
            begin = 0
            while (i := text.find(term, begin)) >= 0:
                spans.append({'term': term, 'start': i, 'end': i + len(term)})
                begin = i + len(term)
        if spans:
            out.append(dict(family=family, kind='text_mention_candidate', source=source,
                            spans=spans, text=text, temporal_scope='source_sentence_order_not_frame_aligned',
                            requiredness='unknown', tolerance=None, video_variant_verified=False))
    typed = entry.get('typed_metadata', {})
    for name in ('signHandList', 'signHandPositionView', 'signHandCountView', 'signHandShapeView'):
        rows = typed.get(name) or []
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            if row.get('handId') is None:
                continue
            out.append(dict(family={'signHandList': 'handshape', 'signHandPositionView': 'body_relation',
                                    'signHandCountView': 'hand_count', 'signHandShapeView': 'two_hand_relation'}[name],
                            kind='dictionary_metadata', source=source, source_field=name,
                            source_value={k: row.get(k) for k in ('handId', 'handNm', 'handCode')},
                            temporal_scope='initial_handshape_only' if name == 'signHandList' else 'dictionary_metadata_scope_unmapped',
                            requiredness='unknown', tolerance=None, video_variant_verified=False))
    return out


def describe_comparison(query_value, reference_values, *, observed, references_observed,
                        variant_verified=False, sequence_alignment_verified=False,
                        source_claim=None, expected_from_reference=True):
    """One scalar measurement, reference executions kept separate.

    A source-confirmed inactive component may be marked not-planned; query absence
    alone must never change expected_from_reference. It does not declare optionality.
    """
    refs = list(reference_values)
    valid_ref = list(references_observed)
    if len(refs) != len(valid_ref) or len(refs) < 2:
        raise ValueError('Two or more separate reference executions required')
    qvalid = bool(observed and query_value is not None and math.isfinite(query_value))
    valid_ref = [bool(v and x is not None and math.isfinite(x)) for x, v in zip(refs, valid_ref)]
    differences = [float(query_value - x) if qvalid and valid_ref[i] else None for i, x in enumerate(refs)]
    status = 'measured_difference' if qvalid and any(valid_ref) else 'insufficient_observation'
    return {
        'planned_units': int(expected_from_reference),
        'query_observed': qvalid, 'reference_observed': valid_ref,
        'reference_values': [x if v else None for x, v in zip(refs, valid_ref)],
        'differences_per_reference': differences,
        'observed_pair_units': sum(x is not None for x in differences) if expected_from_reference else 0,
        'measurement_status': status if expected_from_reference else 'reference_inactive_not_planned',
        'sequence_status': 'verified_alignment' if sequence_alignment_verified else 'alignment_unverified',
        'source_link_status': 'verified_variant' if variant_verified else 'lexical_candidate_only',
        'source_claim': source_claim,
        'requiredness': 'unknown', 'linguistic_tolerance': None,
        'educational_judgment': None,
        'educational_status': 'no_validated_acceptability_rule',
    }


def check_reference_identity(query, references):
    if len(references) < 2:
        raise ValueError('Two eligible references required')
    for ref in references:
        if query['recorded_id'] == ref['recorded_id'] or query['content_sha256'] == ref['content_sha256']:
            raise ValueError('Self or duplicate-content reference')
    if len({r['recorded_id'] for r in references}) != len(references):
        raise ValueError('Repeated reference recorded ID')
    if len({r['content_sha256'] for r in references}) != len(references):
        raise ValueError('Duplicate reference content')


def build():
    acquisition = ROOT / 'source_audit'
    source_rows = read(acquisition / 'CURRENT100_SOURCE_CANDIDATES.json')['rows']
    details = read(acquisition / 'SOURCE_ENTRY_DETAILS.json')['entries']
    entry_index = {(x['source_id'], str(x['entry_id'])): x for x in details}
    profiles, queue = [], []
    for row in source_rows:
        if not row['word_id'].startswith('WORD'):
            continue
        candidates = []
        for system, sid, key in [('legacy', 'NIKL_legacy', 'exact_headword_token_match'),
                                 ('nuri', 'NIKL_nuri', 'exact_korean_expression_match')]:
            for candidate in row[system]['returned_candidates']:
                if not candidate.get(key, False):
                    continue
                entry = entry_index.get((sid, str(candidate['entry_id'])))
                candidates.append({'source_id': sid, 'entry_id': str(candidate['entry_id']),
                                   'entry_label': candidate['label'], 'url': candidate['url'],
                                   'link_status': 'lexical_candidate_variant_unverified',
                                   'description': entry.get('description_text') if entry else None,
                                   'claims': source_claims(entry) if entry else [],
                                   'source_variants': (entry.get('typed_metadata', {}).get('signVariantList') or []) if entry else [],
                                   'detail_available': entry is not None})
        profile = {'schema': SCHEMA, 'word_id': row['word_id'], 'display_label': row['label'],
                   'candidate_entries': candidates, 'verified_variant_id': None,
                   'required_features': None, 'acceptable_bounds': None,
                   'training_eligible': False, 'training_blocker': 'video_variant_and_token_feature_labels_unverified',
                   'no_search_candidate_does_not_mean_absent_from_dictionary': True}
        profiles.append(profile)
        if candidates:
            queue.append({'word_id': row['word_id'], 'label': row['label'],
                          'candidate_entry_ids': [x['source_id'] + ':' + x['entry_id'] for x in candidates],
                          'questions': ['Which source sense/variant matches the public execution?',
                                        'Which described feature is visible and at what time?',
                                        'Only for educational correction: which variation is accepted and what constitutes an error?'],
                          'need_review': 'variant_mapping_then_selected_ambiguous_rules_not_all_frames',
                          'status': 'unresolved'})
    # A lexical index is source material for later annotation, not a visual training target.
    summary = {'profiles': len(profiles), 'with_lexical_candidates': sum(bool(p['candidate_entries']) for p in profiles),
               'candidate_entry_links': sum(len(p['candidate_entries']) for p in profiles),
               'source_claims': sum(len(e['claims']) for p in profiles for e in p['candidate_entries']),
               'verified_variants': 0, 'token_feature_labels': 0, 'verified_error_examples': 0,
               'source_schema': SCHEMA, 'initial_handshape_is_not_entire_video': True,
               'dictionary_finger_notation': DICTIONARY_FINGERS,
               'claim_limit': 'Text tags are discovery candidates, not full semantic parsing or frame labels.'}
    dump(ROOT / 'FEATURE_PROFILES.json', {'summary': summary, 'profiles': profiles})
    dump(ROOT / 'VARIANT_REVIEW_QUEUE.json', {'rows': queue, 'scope': 'candidate labels only; no per-frame review demand'})
    dump(ROOT / 'TRAINING_READINESS.json', {
        'decision': 'NOT_ELIGIBLE_NO_VALIDATED_TOKEN_LABELS', 'authorized': True,
        'reason': 'The limiting factor is missing valid targets and evaluation truth, not missing authorization.',
        'descriptive_feature_training_gate': {'pass': False, 'verified_video_variant_links': 0, 'independent_token_labels': 0},
        'pedagogical_correctness_training_gate': {'pass': False, 'verified_acceptable_variant_labels': 0, 'verified_real_errors': 0},
        'trained_model_count': 0, 'gradient_updates': 0, 'new_TCN_accuracy': None, 'feedback_accuracy': None,
        'next_valid_artifact': 'Small token-feature annotation bundle linked to actual public frames and source variants, with person-separated validation; selected acceptable/error examples for correction.',
        'no_linguistic_ground_truth_from': ['sign names', 'low DTW distance', 'two-reference extrema', 'synthetic perturbations', 'MediaPipe confidence'],
    })
    return summary


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False))
