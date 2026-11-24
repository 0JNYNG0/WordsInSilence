"""Record the actual, score-blind visual review performed by progress_probe_review."""
from pathlib import Path
import datetime
import hashlib
import json

U = Path(__file__).resolve().parent
BP = U.parent
G = BP.parent.parent
SOURCE = G / 'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json'
META = G / 'candidate_description_supervision/source_rank_learning/INPUTS_COMPLETE.json'


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


NOTES = {
    ('WORD0739', 8): '얼굴 앞 손의 하강 뒤 전완을 세우는 변화가 보인다. 84는 마지막 offset band 끝점이라 보류. 86/88/92/96의 세운 전완 자세는 유지되며, 100 이후 귀환은 포함하지 않는다.',
    ('WORD1133', 8): '열린 오른손이 반대팔 위쪽에서 손목 쪽으로 이동한다. 92는 offset band에 포함되어 보류. 94/96/100에서 손목 부근의 몸 상대 자세를 유지한다. 104 이후 귀환은 포함하지 않는다.',
    ('WORD1567', 8): '양손 높이가 번갈아 바뀌는 여러 진행이 보이고 마지막 왼손 높은 자세가 유지된다. 104/106은 마지막 offset band에 있어 보류. 108/112만 sparse terminal body hold로 인정한다.',
    ('WORD2523', 8): '중앙에 모인 양손이 옆/아래로 벌어진다. 64는 offset band 끝점이라 보류. 66/68/72에서 벌어진 양손 몸 자세가 유지된다. 76 이후 귀환과 구분한다.',
    ('WORD0739', 9): '얼굴 앞 손을 내린 뒤 전완을 세우는 변화가 보인다. 76은 offset band 끝점이라 보류. 78/80은 세운 전완 유지, 84 이후 귀환은 포함하지 않는다.',
    ('WORD1133', 9): '오른손이 반대팔 위쪽에서 손목으로 내려간다. 60은 offset band 끝점이라 보류. 62/64/68에서 손목 부근 몸 자세가 유지되며 72 이후 귀환은 제외한다.',
    ('WORD1567', 9): '오른손 높음에서 양손이 교대로 움직여 왼손 높은 자세로 끝난다. 88/90/92/96/100은 모두 마지막 inclusive offset band 이후이며 유지 자세가 직접 보인다. 104 이후 귀환은 포함하지 않는다.',
    ('WORD2523', 9): '중앙에 모인 양손이 옆/아래로 벌어지고 유지된다. 76은 offset band 끝점이라 보류. 78/80만 포함하고 84 이후 귀환은 포함하지 않는다.',
}


def main():
    assert not (U / 'RGB_REVIEW.json').exists()
    source = read(SOURCE)
    assert source['training_allowed'] is False
    originals = [r for r in source['rows'] if r['person'] in [8, 9]]
    probe = [r for r in read(META)['sequences'] if r['split'] == 'probe']
    assert len(originals) == 8 and len(probe) == 48
    assert {r['clip_id'] for r in originals} == {r['clip_id'] for r in probe}
    pins = {str(SOURCE): sha(SOURCE), str(META): sha(META)}
    rows = []
    for r in originals:
        assert sha(r['source']) == r['source_sha256']
        pins[r['source']] = r['source_sha256']
        visible = set()
        sheets = []
        for s in r['sheets']:
            assert sha(s['path']) == s['sha256']
            pins[s['path']] = s['sha256']
            visible.update(s['frames'])
            sheets.append(dict(path=s['path'], sha256=s['sha256'], frames=s['frames'], actual_viewed_this_review=True))
        assert len(sheets) == 2
        bands = [(a, b) for e in r['events'] for a, b in [e['onset'], e['offset']]]
        qualified, excluded = [], []
        for f in r['hold_sample_frames']:
            assert f in visible and any(a <= f <= b for a, b in r['holds'])
            (excluded if any(a <= f <= b for a, b in bands) else qualified).append(f)
        last = max(r['groups'], key=lambda g: g['offset'][1])['family']
        rows.append(dict(clip_id=r['clip_id'], person=r['person'], original_role=r['role'], current_split='probe',
                         source=r['source'], source_sha256=r['source_sha256'], sheets=sheets,
                         original_hold_sample_frames=r['hold_sample_frames'], qualified_hold_sample_frames=qualified,
                         excluded_inclusive_boundary_hold_frames=excluded, all_event_boundary_bands=bands,
                         terminal_family=last, source_families=sorted({g['family'] for g in r['groups']}),
                         observation_ko=NOTES[(r['word_id'], r['person'])], unseen_frame_labels_added=False,
                         old_positive_contract='Preserve all existing y/mask motion positives including historically admitted FOREARM_REVIEW frames77/79 for0739_REAL08. This review adds no motion-positive point.'))
    result = dict(at=datetime.datetime.now().astimezone().isoformat(), status='QUALIFIED_SPARSE_BODY_PROGRESS_PROBE_RGB_REVIEW',
                  reviewer='Codex subagent progress_probe_review: actual16 original RGB overview/dense sheets visually inspected before any score/model/prediction access',
                  scope='evaluation_only_sparse_body_movement_vs_maintained_posture', training_allowed=False,
                  original_annotations_preserved=True, original_training_allowed=False, no_model_score_prediction_access_before_review=True,
                  probe_membership='Current source_rank_learning split=probe for persons08/09, despite historical source role=fit. These people are excluded from BP training by clip/person/source hash.',
                  hold_semantics='A negative target means no local body-family progression at this observed point. It is not incorrect sign, absent attempt, mandatory hold, semantic completion or hand articulation truth.',
                  limitations=['8 public clips, 2 performers, 4 words and 5 body families only.',
                               'Historically exposed research probe, including upstream common100; not fresh final test or globally unseen performers.',
                               'Codex visual review is not independent human or KSL-expert gold.',
                               'Sparse viewed points only; no dense interval filling or unseen-frame labels.',
                               'Original band uncertainty remains; inclusive onset/offset hold points withheld.',
                               'No model/epoch/hyperparameter/candidate interval selection or retraining permitted from this probe.'],
                  rows=rows, pins=pins)
    with (U / 'RGB_REVIEW.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(clips=len(rows), qualified_hold_points=sum(len(r['qualified_hold_sample_frames']) for r in rows),
                          withheld_boundary_points=sum(len(r['excluded_inclusive_boundary_hold_frames']) for r in rows))))


if __name__ == '__main__':
    main()
