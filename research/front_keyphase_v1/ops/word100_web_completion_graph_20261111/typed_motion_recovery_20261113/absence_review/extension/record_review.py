"""Record completed direct RGB inspection; no model execution or inferred truth."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from collections import Counter

P = Path(__file__).resolve().parent
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

# Authored after viewing every contact sheet individually with view_image.
# Each sheet contains every original frame between the endpoints, not a sample.
NOTES = {
 'NIA_SL_WORD0739_REAL04_F': '세운 오른팔과 어깨 부근 오른주먹 자세가 계속 유지된다. 작은 몸통 안정화와 별개로 추가로 팔을 세우는 진행은 보이지 않는다.',
 'NIA_SL_WORD1133_REAL04_F': '오른손이 반대팔 손목 부근에 머문다. 팔을 따라 쓸어가는 추가 진행 없이 두 손의 관계를 유지한다.',
 'NIA_SL_WORD1567_REAL04_F': '양손의 위아래 배치가 유지되고 어느 손이 위에 있는지 바뀌지 않는다. 추가 교대·회전 진행은 보이지 않는다.',
 'NIA_SL_WORD2523_REAL04_F': '좌우로 떨어진 양손 배치를 유지한다. 팔의 작은 안정화가 있으나 양손 간격을 더 벌리는 진행은 보이지 않는다.',
 'NIA_SL_WORD0739_REAL05_F': '세운 오른팔과 주먹 위치를 유지한다. 미세한 이완을 별도의 팔 세우기 동작으로 판단하지 않는다.',
 'NIA_SL_WORD1133_REAL05_F': '오른손이 반대 손목 부근에 머물며 쓸기 동작의 추가 진행은 보이지 않는다.',
 'NIA_SL_WORD1567_REAL05_F': '양손이 위아래 배치를 유지한 채 약간 함께 아래로 안정화된다. 91~92의 작은 하강을 새 교대 동작으로 보지 않는다. 관절 전체 정지가 아니라 해당 교대·회전 진행의 부재를 기록한다.',
 'NIA_SL_WORD2523_REAL05_F': '떨어져 있는 양손 배치를 유지한다. 몸·머리의 작은 움직임과 별개로 양손을 더 벌리는 진행은 보이지 않는다.',
 'NIA_SL_WORD0739_REAL06_F': '오른주먹과 세운 전완 자세가 유지된다. 104 부근의 미세한 하강을 새로운 팔 세우기 동작으로 보지 않는다.',
 'NIA_SL_WORD1133_REAL06_F': '오른손과 반대 손목의 상대 관계가 안정적으로 유지된다. 팔을 따라 추가로 쓸어가는 진행은 보이지 않는다.',
 'NIA_SL_WORD1567_REAL06_F': '양손의 위아래 배치가 계속 유지된다. 위쪽 손이 약간 아래로 안정화되지만 양손의 교대·반전 진행은 보이지 않는다.',
 'NIA_SL_WORD2523_REAL06_F': '양손이 서로 떨어진 배치로 유지되며 더 벌리는 진행은 보이지 않는다.',
 'NIA_SL_WORD0739_REAL07_F': '오른주먹을 상체 앞에 두고 전완을 세운 자세를 유지한다. 정확한 팔꿈치 각도나 교육적 정오를 판단하지 않으며 추가로 세우는 진행은 보이지 않는다.',
 'NIA_SL_WORD1133_REAL07_F': '반대 손목 가까이에 놓인 손의 관계를 유지하며 추가 쓸기 진행은 보이지 않는다.',
 'NIA_SL_WORD1567_REAL07_F': '얼굴 부근 위쪽 손과 가슴 부근 아래쪽 손의 배치가 유지된다. 추가 교대·회전 진행은 보이지 않는다.',
 'NIA_SL_WORD2523_REAL07_F': '떨어진 양손 배치가 유지된다. 손 기울기의 작은 변화로 양손 벌리기 진행이 발생했다고 보지 않는다.',
 'NIA_SL_WORD0739_REAL08_F': '오른손을 뺨 부근에 두고 전완을 세운 자세를 유지한다. 추가로 팔을 세우는 진행은 보이지 않는다.',
 'NIA_SL_WORD1133_REAL08_F': '오른손이 반대 손목 가까이 머물며 추가 쓸기 진행은 보이지 않는다.',
 'NIA_SL_WORD1567_REAL08_F': '위아래에 놓인 양손 관계를 유지하며 추가 교대·회전 진행은 보이지 않는다.',
 'NIA_SL_WORD2523_REAL08_F': '좌우로 떨어진 양손 배치가 유지되며 더 벌리는 진행은 보이지 않는다.',
 'NIA_SL_WORD0739_REAL09_F': '오른주먹을 어깨 부근에 두고 전완을 세운 모습이 전체 구간에서 안정적으로 유지된다. 추가로 팔을 세우는 진행은 보이지 않는다.',
 'NIA_SL_WORD1133_REAL09_F': '오른손이 반대 손목 부근에 머무르는 관계가 전체 구간에서 유지된다. 추가 쓸기 진행은 보이지 않는다.',
 'NIA_SL_WORD1567_REAL09_F': '양손의 위아래 배치가 전체 구간에서 유지된다. 추가 교대·회전 진행은 보이지 않는다.',
 'NIA_SL_WORD2523_REAL09_F': '양손이 좌우로 떨어진 배치가 유지되며 더 벌리는 진행은 보이지 않는다.',
}

def main():
    assert not (P / 'REVIEW_COMPLETE.json').exists(), 'Preserve completed review'
    prep = read(P / 'PREPARED.json')
    old = P.parent / 'REVIEW_COMPLETE.json'
    old_hash = '385b453171a1d8cc9c5fd405e1f5658b42d9e765919e536c06a8d3e82acbaa28'
    assert sha(old) == old_hash, 'Original ID02/03 review changed'
    assert set(NOTES) == {r['clip_id'] for r in prep['rows']}
    checked_paths = {}
    checks = 0
    def check(path, expected):
        nonlocal checks
        actual = sha(path)
        assert actual == expected, str(path)
        checked_paths[str(path)] = actual
        checks += 1
    rows = []
    for r in prep['rows']:
        check(r['source'], r['source_sha256'])
        check(r['source_annotation'], r['source_annotation_sha256'])
        check(r['contact_sheet'], r['contact_sheet_sha256'])
        frames = r['frames']
        assert frames == list(range(min(frames), max(frames)+1))
        assert [f['frame'] for f in r['frame_records']] == frames
        for f in r['frame_records']:
            check(f['path'], f['sha256'])
        assert r['family'] in [3,4,5,7]
        expected_split = 'train' if r['person'] in [4,5,6,7] else 'probe'
        assert r['head_split'] == expected_split and r['person'] in [4,5,6,7,8,9]
        assert r['upstream_exposed'] is True
        out = dict(r)
        out['visual_inspection_completed'] = True
        out['review'] = dict(
            reviewer='/root/absence_contract_review_1113 (Codex agent)',
            method='Direct view_image inspection of each of 24 source-RGB contact sheets, separately. Every original frame in each local interval was displayed and inspected.',
            all_frames_inspected=True,
            inspected_frames=frames,
            hold_span=[min(frames),max(frames)],
            hold_frames=frames,
            uncertain_frames=[],
            progression_frames=[],
            target_family=r['family'],
            state='terminal_hold_of_queried_family',
            observed_note_ko=NOTES[r['clip_id']],
            labels_scope='No continuing movement of the SAME queried family inside this local interval. Small settling is allowed. No other-family absence, whole-video absence, sign error, omitted action, or semantic completion label.',
            semantic_completion=None,
            educational_correctness=None,
            mandatory_repetitions=None,
            independent_human_or_ksl_expert=False,
            extra_full_frame_inspection=[])
        rows.append(out)
    summary = dict(clips=len(rows),source_frames_inspected=sum(len(r['frames']) for r in rows),
        hold_frames=sum(len(r['review']['hold_frames']) for r in rows),uncertain_frames=0,progression_frames=0,
        train_clips=sum(r['head_split']=='train' for r in rows),
        probe_clips=sum(r['head_split']=='probe' for r in rows),
        train_people=[4,5,6,7],probe_people=[8,9],
        train_frames=sum(len(r['frames']) for r in rows if r['head_split']=='train'),
        probe_frames=sum(len(r['frames']) for r in rows if r['head_split']=='probe'),
        target_families=[3,4,5,7],family2_hold_evidence_not_provided=True,
        additional_continuous_regions_outside_reviewed_ranges_not_acquired=True)
    assert summary['clips']==24 and summary['source_frames_inspected']==198
    assert summary['train_clips']==16 and summary['probe_clips']==8
    report = dict(at=datetime.now(timezone.utc).isoformat(),status='AGENT_RGB_REVIEW_COMPLETE_PENDING_ROOT_VERIFICATION',
        prepared_path=str(P/'PREPARED.json'),prepared_sha256=sha(P/'PREPARED.json'),
        script_sha256=sha(__file__),previous_review_path=str(old),previous_review_sha256=old_hash,
        frozen_scope_sha256=sha(P/'FROZEN_REVIEW_SCOPE.json'),summary=summary,rows=rows,
        policy=[
            'Public source IDs04..09 only. Original sources and previous ID02/03 review untouched.',
            'ID04..07 candidate-training and ID08/09 person-separated diagnostic split fixed BEFORE annotation. ID08/09 have upstream exposure and are NOT independent final evaluation.',
            'All original frames inside each selected terminal hold interval were inspected; no sparse labels were silently expanded.',
            'Small bodily settling does not make the action wrong. This label concerns progression of the SAME queried movement, not complete joint immobility.',
            'No family2 hold evidence. Never transfer hold absence labels to any other family.',
            'A null-candidate bag is supported only when its entire search domain and candidates lie inside an approved hold span with SAME target family.',
            'Whole source videos contain the preceding action and must not be labeled as absent or incorrect.',
            'No completion, minimum meaningful duration, obligatory repetition count, or educational correctness is inferred.',
            'Agent visual review, not independent human or KSL expert gold truth; root verification pending.',
            'No private/user/mentor/final8 input access, model training, prediction, or new model evaluation.',
            'First decoder-FPS assertion failure preserved in FIRST_FAILURE.txt; exact frame indices retained, decoded timestamps recorded without resampling.'
        ])
    output = P/'REVIEW_COMPLETE.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    verification=dict(status='PASS_FILE_HASH_FRAME_COVERAGE_SPLIT_CHECKS',
        review_sha256=sha(output),previous_review_unchanged=sha(old)==old_hash,
        path_hash_checks=checks,unique_paths_checked=len(checked_paths),
        all_frame_ranges_contiguous=True,all_frames_have_matching_saved_hashes=True,
        person_split_disjoint=True,upstream_exposure_disclosed=True,
        visual_review_is_agent_annotation_not_independent_human_gold=True,
        model_training_or_evaluation_run=False,summary=summary,checked_paths=checked_paths)
    (P/'VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['공개 FIT 마지막 유지 구간 연속 RGB 검토: ID04~09 확장','',
        '4표현 × ID04~09 24원본에서 사전 지정된 마지막 유지 표본의 최소~최대 구간, 총198원본 프레임을 빠짐없이 실제 이미지로 확인했다.',
        '24개 연락표를 한 장씩 확인했다. 해당 범위는 모두 같은 움직임의 추가 진행이 없는 유지로 기록했다. 약한 몸통 흔들림·팔의 안정화는 오답이나 새 주요동작으로 만들지 않았다.',
        f"사전 분리: ID04~07 후보 학습16영상/{summary['train_frames']}프레임, ID08/09 사람 분리 진단8영상/{summary['probe_frames']}프레임. ID08/09는 상류 노출이 있으므로 독립 최종평가가 아니다.",
        'family3/4/5/7만 해당한다. family2 유지 정답은 없으며 다른 움직임 부재로 자동 확장할 수 없다. 추가 연속 구간은 확보하지 않았다.',
        '이 국소 구간 안에서 같은 family의 진행이 없다는 근거다. 영상 전체의 행동 누락, 의미상 완료, 수어 정오, 반복 완료, 손가락 정오는 판단하지 않았다.',
        '일부 원본의 decoder FPS가 약30.10~30.13이어서 첫 준비가 실패한 기록은 보존했다. 프레임 번호는 바꾸지 않고 원본 전체 프레임을 추출하며 실제 decoder PTS를 함께 기록하도록 복구했다.',
        '기존 ID02/03 REVIEW_COMPLETE.json 해시 보존 확인. 학습·추론·모델 평가 실행 없음. Codex 에이전트 시각 주석으로 root 별도검수 대기.','']
    for r in rows:
        lines.append(f"{r['clip_id']} {r['head_split']} family{r['family']} 유지 {r['review']['hold_span']}: {r['review']['observed_note_ko']}")
    (P/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=summary,hash_checks=checks,unique_paths=len(checked_paths),review_sha256=sha(output)),ensure_ascii=False))

if __name__=='__main__':
    main()
