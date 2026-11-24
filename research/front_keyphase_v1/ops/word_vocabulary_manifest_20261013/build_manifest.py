"""Training-person-only vocabulary inventory. No video, detector, or model access."""
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

U = Path(__file__).resolve().parent
OPS = U.parent
JOIN = OPS / 'word_dataset_plan_20261013/FRONT_VIDEO_JOIN.jsonl'
ORIGINAL = OPS / 'word_acquisition_inventory_20261011/FRONT_MANIFEST.jsonl'
TRAIN_IDS = set(range(1, 10))
REASONS = {
    'training_coverage_incomplete': '학습 9명 자료 연결 미완료',
    'training_label_disagreement': '학습 표기 혼재',
    'training_label_shared_across_codes': '같은 이름을 여러 코드가 사용',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    (U / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def signer(row):
    return int(re.search(r'_REAL(\d+)_F', row['clip_id']).group(1))


def is_train(row):
    return row['role'] == 'train_candidate' and row['metadata_video_join_eligible']


def classify_training(rows):
    """Only consult training rows, regardless of non-training metadata supplied."""
    groups, label_codes = defaultdict(list), defaultdict(set)
    for row in rows:
        if is_train(row):
            assert signer(row) in TRAIN_IDS
            assert len(row['labels']) == 1 and row['labels'][0].strip()
            groups[row['word_code']].append(row)
            label_codes[row['labels'][0]].add(row['word_code'])
    decisions = {}
    for code, group in sorted(groups.items()):
        ids = sorted(signer(r) for r in group)
        names = Counter(r['labels'][0] for r in group)
        reasons = []
        if len(group) != 9 or set(ids) != TRAIN_IDS:
            reasons.append('training_coverage_incomplete')
        if len(names) != 1:
            reasons.append('training_label_disagreement')
        collision = {name: sorted(label_codes[name] - {code}) for name in names if label_codes[name] - {code}}
        if collision:
            reasons.append('training_label_shared_across_codes')
        decisions[code] = dict(
            code=code, metadata_candidate=not reasons, review_reasons=reasons,
            training_people=len(set(ids)), training_clips=len(group), training_signer_ids=ids,
            training_labels=dict(names), related_label_codes=collision,
        )
    return decisions


def main():
    assert not (U / 'SUMMARY.json').exists(), 'Reuse completed inventory; do not overwrite.'
    sources = [JOIN, ORIGINAL, U / 'SCOPE.json']
    pins = {str(p): sha(p) for p in sources}
    rows = read_rows(JOIN)
    old = {r['clip_id']: r for r in read_rows(ORIGINAL)}
    assert len(rows) == 54000 == len(old)
    assert len({r['clip_id'] for r in rows}) == 54000
    groups = defaultdict(list)
    for row in rows:
        original = old[row['clip_id']]
        assert row['role'] == original['role'] and row['labels'] == original['labels']
        assert row['prior_metadata_clean'] == original['metadata_clean']
        groups[row['word_code']].append(row)
    assert len(groups) == 3000
    decisions = classify_training(rows)
    assert set(decisions) == set(groups)
    candidate_roles = Counter()
    selected_ids = set()
    label_conflicts, retained_bad, output_rows = 0, 0, 0
    candidate_bytes = Counter()
    review_list = []
    with (U / 'CANDIDATE_ROLES.jsonl').open('x', encoding='utf-8') as output:
        for code, decision in decisions.items():
            group = groups[code]
            inventory = {}
            for role in sorted({r['role'] for r in group}):
                role_rows = [r for r in group if r['role'] == role]
                eligible = [r for r in role_rows if r['metadata_video_join_eligible']]
                inventory[role] = dict(
                    annotation_rows=len(role_rows), eligible_front_rows=len(eligible),
                    eligible_signer_ids=sorted({signer(r) for r in eligible}),
                    exact_labels=dict(Counter(l for r in eligible for l in r['labels'])),
                )
            decision['inventory_only_roles'] = inventory
            decision['semantic_form_review'] = 'not_verified'
            decision['decoded_video_review'] = 'not_run_in_this_unit'
            if not decision['metadata_candidate']:
                review_list.append(decision)
                continue
            target = next(iter(decision['training_labels']))
            for r in group:
                selected_ids.add(r['clip_id'])
                aligned = r['labels'] == [target]
                bad = not r['metadata_video_join_eligible']
                assessment = 'pending_media_integrity_and_input_checks'
                if bad:
                    assessment = 'unresolved_original_metadata_or_video_join'
                    retained_bad += 1
                elif not aligned:
                    assessment = 'evaluation_label_disagrees_requires_adjudication'
                    label_conflicts += 1
                candidate_roles[(r['role'], 'all_rows')] += 1
                candidate_roles[(r['role'], 'eligible_rows')] += int(not bad)
                if not bad:
                    assert len(r['video_entries']) == 1
                    entry = r['video_entries'][0]
                    assert entry['filename_signer_id'] == signer(r)
                    candidate_bytes[r['role']] += entry['bytes']
                record = dict(r, metadata_vocabulary_candidate=True,
                              training_label=target, raw_label_matches_training_label=aligned,
                              readiness=assessment,
                              source_annotation_sha256=old[r['clip_id']]['source_payload_sha256'])
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
                output_rows += 1
    counts = Counter('candidate' if d['metadata_candidate'] else 'review' for d in decisions.values())
    train_people = Counter(d['training_people'] for d in decisions.values())
    reason_counts = Counter(reason for d in decisions.values() for reason in d['review_reasons'])
    priority_counts = Counter(d['review_reasons'][0] for d in decisions.values() if d['review_reasons'])
    targets = json.loads((OPS / 'word_acquisition_inventory_20261011/TARGET18_COVERAGE.json').read_text(encoding='utf-8'))
    target_rows = []
    for target in targets:
        name = target['target']
        exact = [d['code'] for d in decisions.values() if name in d['training_labels']]
        target_rows.append(dict(target=name, exact_training_codes=exact,
                                metadata_candidate_codes=[c for c in exact if decisions[c]['metadata_candidate']],
                                interpretation='Exact training spelling only; no synonym/form equivalence decision.'))
    summary = dict(
        status='COMPLETE_METADATA_VOCABULARY_CANDIDATES_NOT_TRAINING_READY',
        at_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), source_hashes=pins,
        raw_annotation_rows=len(rows), eligible_front_rows=sum(r['metadata_video_join_eligible'] for r in rows),
        total_codes=len(groups), all_codes_have_18_annotation_rows=all(len(g) == 18 for g in groups.values()),
        training_people_histogram={str(k): v for k, v in sorted(train_people.items(), reverse=True)},
        eligible_training_rows=sum(d['training_clips'] for d in decisions.values()),
        consistent_training_label_codes=sum(len(d['training_labels']) == 1 for d in decisions.values()),
        mixed_training_label_codes=sum(len(d['training_labels']) != 1 for d in decisions.values()),
        metadata_candidate_codes=counts['candidate'], review_codes=counts['review'],
        review_reason_counts_overlap=dict(reason_counts), review_priority_counts_disjoint=dict(priority_counts),
        candidate_role_counts={f'{k[0]}:{k[1]}': v for k, v in sorted(candidate_roles.items())},
        candidate_role_manifest_rows=output_rows, candidate_unique_clip_ids=len(selected_ids),
        candidate_eligible_video_bytes_by_role=dict(candidate_bytes),
        candidate_retained_bad_metadata_rows=retained_bad,
        candidate_eligible_nontraining_label_conflicts=label_conflicts,
        selected_via_nontraining_quality_or_scores=False, vocabulary_limit=None,
        verification='pending separate SQL and non-training invariance check',
        actual_sign_form_classes_verified=False, new_model_training=0, video_decode=0,
        person_id_interpretation='Dataset performer identifiers, not biometric identity verification. One F recording per code and person; five camera views are not five independent people/repeats.',
        next='Keep broad candidates; inspect specific training label/form conflicts and freeze temporal/extraction and comparison contract before bulk preparation or training. Retain unresolved evaluation rows and exposure history.',
    )
    write_json('VOCABULARY.json', dict(decision_source='eligible training rows only', codes=list(decisions.values()), old_target_mapping=target_rows))
    write_json('REVIEW_QUEUE.json', review_list)
    write_json('SUMMARY.json', summary)
    write_json('RESEARCH_NOTES.json', dict(
        checked_date='2026-10-13', sources=[
            dict(url='https://arxiv.org/html/1910.11006v2', section='4.4.2 and 4.4.3',
                 finding='Vocabulary size and samples per class both affect results. Larger-vocabulary performance cannot be inferred from small-vocabulary accuracy; adding classes is not guaranteed to help.',
                 limitation='ASL benchmark and its architectures; not a controlled guarantee for our KSL/TCN setting.'),
            dict(url='https://arxiv.org/abs/2007.12131',
                 finding='Large-scale BSL data provided useful transfer pretraining in the reported experiments.',
                 limitation='Supports possibility of useful shared representations, not the claim that all extra classes improve our task.')],
        proposal='Prepare all metadata-valid candidates without a top100 cap. Distinguish broad recognition vocabulary from game lesson vocabulary. No data-specific superiority claim without matched evaluation.'))
    esc = lambda x: html.escape(str(x), quote=True)
    html_rows = []
    for d in decisions.values():
        role_count = lambda role: d['inventory_only_roles'].get(role, {}).get('eligible_front_rows', 0)
        html_rows.append('<tr data-kind="' + ('candidate' if d['metadata_candidate'] else 'review') + '">' + ''.join(
            '<td>' + esc(v) + '</td>' for v in [d['code'], ' / '.join(f'{k} ({v})' for k, v in d['training_labels'].items()),
            '후보' if d['metadata_candidate'] else '검토', d['training_people'], role_count('development_only'),
            role_count('reserved_metadata_only'), role_count('official_validation_metadata_only'),
            role_count('unassigned'), ', '.join(REASONS[r] for r in d['review_reasons']) or '영상·표현 확인 전',
            '; '.join(f'{k}: {", ".join(v)}' for k, v in d['related_label_codes'].items())]) + '</tr>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Hub 단어 학습 구성표</title><style>body{font-family:'Malgun Gothic',sans-serif;margin:30px auto;max-width:1500px;padding:0 24px;color:#172536;line-height:1.65}h1{font-size:28px}.cards{display:flex;gap:14px;flex-wrap:wrap}.card{background:#edf4f8;padding:16px 24px;flex:1;min-width:180px}.card b{font-size:27px;display:block}.notice{background:#fff3dc;border-left:5px solid #bf6600;padding:14px 20px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #d7dfe7;padding:9px;text-align:left;vertical-align:top}th{background:#e7edf4;position:sticky;top:0}td:nth-child(2){min-width:160px}tr[data-kind=review]{background:#fff9f0}input,select{font:inherit;padding:8px;border:1px solid #879aaa;border-radius:4px;margin:8px}a{color:#1854a0}.scroll{overflow-x:auto}button{font:inherit}footer{margin:25px 0}</style>
<h1>AI Hub 정면 단어 자료 · 학습 구성표</h1><p>2026-10-13 · 전체 3,000코드 확인 · 선택 기준은 학습자 01~09의 자료만 사용</p>
<div class="cards"><div class="card">학습 9명 연결 완료<b>PEOPLE9개 코드</b></div><div class="card">이름·코드 조건 통과<b>CORE개 후보</b></div><div class="card">복구·표현 검토 필요<b>REVIEW개 코드</b></div></div>
<p>단어 종류를 100개로 제한하지 않습니다. 후보는 학습 9명·정상 연결·코드 내 동일 표기·다른 학습 코드와 표기 미중복을 만족합니다. 한 사람이 같은 단어를 9번 한 자료가 아니라, 서로 다른 촬영자 ID 9개에서 정면 영상 1개씩입니다.</p>
<div class="notice"><b>후보 = 학습 완료/정확도 검증이 아닙니다.</b> 실제 영상의 표현·파일 무결성·관절 검출·학습 구간은 후속 확인 대상입니다. 같은 이름의 다른 코드를 자동 합치거나 서로 다른 수어라고 단정하지 않았습니다. 8명만 정상 연결된 코드는 폐기하지 않고 복구 대상으로 남겼습니다.</div>
<p>전체 자료는 코드당 18개 촬영자 ID, 5개 시점입니다. 아래 숫자는 <b>정면 영상과 주석의 정상 연결 건수</b>입니다. 미배정 ID10을 학습에 추가하지 않았습니다. 평가 주석이 다른 경우도 후보를 취소하거나 몰래 제외하지 않고 별도 보류 기록을 남깁니다.</p>
<p><a href="RESULT_KO.txt">설명 보고서</a> · <a href="SUMMARY.json">집계 근거</a> · <a href="INDEPENDENT_CHECK.json">별도 검산</a></p>
<label>검색<input id="q" type="search" placeholder="단어 또는 WORD 코드"></label><label>구분<select id="kind"><option value="all">전체</option><option value="candidate">후보</option><option value="review">검토 필요</option></select></label><span id="count"></span>
<div class="scroll"><table><thead><tr><th>코드</th><th>학습 자료의 원문 표기 (건수)</th><th>상태</th><th>학습 9명</th><th>개발 3명</th><th>보류 3명</th><th>공식 검증 2명</th><th>미배정 1명</th><th>검토 사유</th><th>같은 표기의 다른 코드</th></tr></thead><tbody>ROWS</tbody></table></div>
<footer>개발·보류·공식 검증의 숫자/표기는 목록 확인에만 사용했습니다. 수어 분류모델 학습, 영상 재생·관절 추출, 사용자 자료 사용은 하지 않았습니다.</footer>
<script>const trs=[...document.querySelectorAll('tbody tr')];function filter(){const q=document.querySelector('#q').value.toLocaleLowerCase(),k=document.querySelector('#kind').value;let n=0;for(const tr of trs){const yes=(k==='all'||tr.dataset.kind===k)&&tr.textContent.toLocaleLowerCase().includes(q);tr.hidden=!yes;if(yes)n++;}document.querySelector('#count').textContent=n.toLocaleString()+'개 표시';}document.querySelector('#q').addEventListener('input',filter);document.querySelector('#kind').addEventListener('change',filter);filter();</script></html>'''
    page = page.replace('PEOPLE9', str(train_people[9])).replace('CORE', str(counts['candidate'])).replace('REVIEW', str(counts['review'])).replace('ROWS', '\n'.join(html_rows))
    (U / '단어_학습구성표.html').write_text(page, encoding='utf-8')
    assert pins == {str(p): sha(p) for p in sources}, 'Source modified'
    print(json.dumps({k: summary[k] for k in ['status', 'training_people_histogram', 'metadata_candidate_codes', 'review_codes', 'review_priority_counts_disjoint', 'candidate_role_counts', 'candidate_retained_bad_metadata_rows', 'candidate_eligible_nontraining_label_conflicts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
