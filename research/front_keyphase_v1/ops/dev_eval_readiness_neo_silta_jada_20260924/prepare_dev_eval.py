"""Freeze the development-evaluation input list for 너 / 싫다 / 자다.

Reads ONLY already-audited local evidence. No archive is opened, no directory is
scanned, no network call, no download, no training, no model load. Absences that
earlier units already established are read from their records, not re-searched.

What it decides, per (word, development signer 11-13, front view):
  READY_PROVIDED_COORDINATES  keypoints are local and usable -> written into the list
  NEED_KEYPOINTS              label/interval known, coordinates not local
  NEED_SOURCE_VIDEO           additionally required for any MediaPipe web-input evaluation
  HOLD_SEMANTIC               the word-number mapping is unresolved, so it cannot be ground truth
"""
import csv, hashlib, json, pathlib, sys

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))

PINS = {
    'evidence/word_archive_mapping_20260918/coverage_256.csv': '80a5dd0fdb25bc879cb8d07564d5aa018f5dd0fb1eb042c3ab8e363dce80f5e0',
    'evidence/sentence_target_catalog_20260919/coverage_18.json': '5f87e2f7df5a8f332ffa8f2a2b3a588879a2ae1d4af08950d43bdf871143b46d',
    'ops/word_keypoint_archive_proposal_20260922/INSPECTION_RESULT.json': '93acb17167a0f143bdfe6f811165ecba0f8107507eef6720442af4114dd2ec7a',
    'ops/word_keypoint_archive_proposal_20260922/TARGET_COVERAGE.json': '4ae88fc6191162c3f44cc2f1e176f1570571dda429ccb1f1e382d9793ab8b2fd',
    'ops/word_native_readiness_20260922/SOURCE_OVERLAP.json': '4b6fe8391d56cdbaaaf2b52c5a5107f01cbf5941fd3c0c0d0e238eb11d38b667',
    'ops/word_semantic_review_20260922/SEMANTIC_DECISIONS.json': '08c4d5f153884ce538e60b9760f72e1ae21df3be534834f2cbf91853075e43fc',
    'ops/context_training_pair_20260921/feature_metadata.json': '815dff33a2d084b3fdf88e275b9d8b500dda1c70e363d8c219bb3df325b62525',
    'config/source_manifest.json': 'f5f03b0c9abdbf51b54358252d2434e934eb64eceaa9d1d52ef6275659ffd337',
}

TARGETS = {'너': 'NEO', '싫다': 'SILTA', '자다': 'JADA'}
FAMILIES = {                      # from SEMANTIC_DECISIONS / TARGET_COVERAGE, not re-derived
    '1351': '너',
    '1278': '싫다', '1385': '싫다',
    '1245': '자다', '1377': '자다', '1378': '자다', '1544': '자다',
}
DEV_SIGNERS = ['11', '12', '13']
TRAIN_SIGNERS = [str(i).zfill(2) for i in range(1, 10)]
RESERVED_SIGNERS = ['14', '15', '16']


def main():
    checks = {}
    for rel, digest in PINS.items():
        checks['pin_' + rel.split('/')[-1]] = sha(L / rel) == digest
    assert all(checks.values()), [k for k, v in checks.items() if not v]

    decisions = read(L / 'ops/word_semantic_review_20260922/SEMANTIC_DECISIONS.json')
    held_numbers = {r['word_number'] for r in decisions['rows']
                    if r['decision'] == 'HOLD_SPECIFIC_UNRESOLVED_MAPPING'}
    checks['held_numbers_read_from_record'] = held_numbers == {'1290', '1544', '2485'}

    # --- 1. local keypoint archive: which signers does it actually contain? -----
    insp = read(L / 'ops/word_keypoint_archive_proposal_20260922/INSPECTION_RESULT.json')
    local_word_signers = sorted(insp['clip_groups_by_signer'])
    checks['local_word_archive_is_signer_02_only'] = local_word_signers == ['2']
    checks['local_word_archive_has_no_dev_signer'] = not (set(local_word_signers) & {'11', '12', '13'})

    cov = read(L / 'ops/word_keypoint_archive_proposal_20260922/TARGET_COVERAGE.json')
    checks['no_family_reports_a_dev_signer'] = all(not f['dev_signers_found'] for f in cov['families'])

    # --- 2. sentence corpus: already-recorded absence for these three ----------
    cat = read(L / 'evidence/sentence_target_catalog_20260919/coverage_18.json')
    sent = {r['word_id']: r for r in cat['rows']}
    sentence_rows = {w: sent[TARGETS[w]]['sentence_candidate_annotation_rows'] for w in TARGETS}
    checks['sentence_corpus_has_no_case_for_the_three'] = all(v == 0 for v in sentence_rows.values())

    # --- 3. prepared development groups really lack these three ---------------
    meta = read(L / 'ops/context_training_pair_20260921/feature_metadata.json')
    dev_labels = {r.get('label', r.get('true')) for g in ('annotation_dev', 'whole_dev') for r in meta[g]}
    checks['prepared_dev_groups_lack_the_three'] = not (set(TARGETS) & dev_labels)
    dev_signers_in_use = sorted({r['signer_id'] for g in ('annotation_dev', 'whole_dev') for r in meta[g]})
    checks['dev_groups_are_signers_11_12_13'] = dev_signers_in_use == [11, 12, 13]

    # --- 4. training rows that must never be reused as development ------------
    train_ids = {r['id'] for g in ('base', 'pair_context') for r in meta[g]}
    added = {r['clip_id'] for r in read(
        L / 'ops/word_semantic_review_20260922/COMPARISON_PLAN.json')['context']['added_rows']}
    forbidden_as_dev = train_ids | added
    checks['thirteen_added_recordings_are_training_rows'] = len(added) == 13

    # --- 5. requirement rows from the audited 256-row ledger -------------------
    rows = list(csv.DictReader((L / 'evidence/word_archive_mapping_20260918/coverage_256.csv')
                               .open(encoding='utf-8-sig')))
    checks['ledger_has_256_rows'] = len(rows) == 256

    ready, required, held, seen = [], [], [], set()
    for r in rows:
        wn, sg = r['word_number'], r['signer_id']
        if wn not in FAMILIES or sg not in DEV_SIGNERS or r['view'] != 'F':
            continue
        word = FAMILIES[wn]
        clip_id = pathlib.Path(r['video_basename']).stem
        seen.add((wn, sg))
        assert clip_id not in forbidden_as_dev, clip_id
        assert r['split_role'] == 'development_only', (clip_id, r['split_role'])
        item = {
            'word': word, 'word_id': TARGETS[word], 'word_number': wn,
            'raw_label': r['raw_label'], 'signer_id': int(sg), 'view': 'F',
            'clip_id': clip_id, 'video_basename': r['video_basename'],
            'label_member': r['label_member'],
            'annotation_start_s': float(r['annotation_start_s']),
            'annotation_end_s': float(r['annotation_end_s']),
            'split_role': r['split_role'],
            'keypoints_local': r['prior_media_status'] != 'NOT_IN_DOWNLOADED_WORD_ARCHIVE',
            'source_video_local': r['local_file_exists_now'] == 'True',
        }
        if wn in held_numbers:
            item['state'] = 'HOLD_SEMANTIC_MAPPING_UNRESOLVED'
            item['blocker'] = 'word-number to target mapping is unresolved; cannot serve as ground truth'
            held.append(item)
        elif item['keypoints_local']:
            item['state'] = 'READY_PROVIDED_COORDINATES'
            ready.append(item)
        else:
            item['state'] = 'NEED_KEYPOINTS'
            item['needs'] = ['provided keypoint JSON for this clip (AIHub 103 REAL_WORD keypoint, signer '
                             + sg + ')']
            if not item['source_video_local']:
                item['needs'].append('source MP4 for this clip (only required for MediaPipe web-input '
                                     'evaluation, not for provided-coordinate evaluation)')
            required.append(item)

    checks['every_family_x_dev_signer_accounted'] = seen == {(wn, sg) for wn in FAMILIES for sg in DEV_SIGNERS}
    checks['no_train_or_reserved_signer_in_list'] = all(
        i['signer_id'] in (11, 12, 13) for i in ready + required + held)
    checks['ready_list_contains_no_signer_02'] = all(i['signer_id'] != 2 for i in ready)

    per_word = {}
    for w in TARGETS:
        per_word[w] = {
            'ready_provided_coordinates': sum(i['word'] == w for i in ready),
            'need_keypoints': sum(i['word'] == w for i in required),
            'hold_semantic': sum(i['word'] == w for i in held),
            'sentence_corpus_rows': sentence_rows[w],
            'existing_public_development_cases': 0,
        }

    failed = [k for k, v in checks.items() if not v]
    result = {
        'schema': 'wis_dev_eval_readiness_v1',
        'at_kst': '2026-09-24',
        'status': 'FROZEN_NO_USABLE_DEVELOPMENT_INPUT_AVAILABLE' if not ready else 'FROZEN_PARTIAL_INPUT_PREPARED',
        'scope': 'Development-evaluation input readiness for 너 / 싫다 / 자다, development signers 11-13, front view',
        'checks': checks, 'checks_total': len(checks), 'checks_failed': failed,
        'pins': PINS,
        'evaluation_list_frozen': ready,
        'evaluation_list_size': len(ready),
        'acquisition_required': required,
        'held_semantic': held,
        'per_word': per_word,
        'why_nothing_is_preparable_now': [
            'The local WORD keypoint archive contains signer 02 only (15,000 clip groups, '
            'clip_groups_by_signer {"2": 15000}); no development signer is inside it.',
            'The sentence corpus has zero annotation rows for these three words (coverage_18.json), '
            'so the existing annotation_dev 699 / whole_dev 74 cannot contain them.',
            'Label text and annotation intervals for signers 11-13 DO exist in the audited 256-row '
            'ledger, but label metadata is not coordinates and cannot be evaluated.',
        ],
        'evaluation_kinds': {
            'provided_coordinates': 'needs the keypoint JSON only. Preparable the moment those files exist.',
            'mediapipe_web_input': 'additionally needs the source MP4. Cannot be prepared from keypoints.',
        },
        'not_done': {
            'archive_opened': 0, 'directory_scanned': 0, 'downloads': 0, 'network_calls': 0,
            'training_runs': 0, 'model_loads': 0, 'user_arrays_read': 0, 'reserved_arrays_read': 0,
            'signer_02_or_reserved_moved_into_development': 0,
            'repeated_search_for_already_recorded_absence': 0,
        },
        'limits': [
            'A recorded absence is an absence in the audited local inventory, not proof of corpus-wide absence.',
            'Label rows are metadata; acquiring them would not by itself produce an evaluation input.',
            'File locations and archive sizes are not estimated here; only names already present in the ledger are listed.',
        ],
    }
    out = H / 'DEV_EVAL_READINESS.json'
    with out.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('checks', len(checks), 'failed', failed)
    print('frozen evaluation list size:', len(ready))
    print('acquisition required rows:', len(required), '| held-semantic rows:', len(held))
    for w, v in per_word.items():
        print(f"  {w}: ready={v['ready_provided_coordinates']} need={v['need_keypoints']} "
              f"hold={v['hold_semantic']} sentence_rows={v['sentence_corpus_rows']}")
    print('written', out.name, sha(out)[:16])
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
