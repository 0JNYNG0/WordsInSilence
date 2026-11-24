"""Serialize manual one-pass sheet observations; no model or video processing."""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

# Entered only after actual view_image review of all consecutive tiles in
# each fixed sheet. These are local occurrence decisions, never sign errors.
MANUAL = {
  1: (0, '44~54의 두 손은 목 아래 양쪽에 떨어진 배치를 유지한다. 중간 48~51과 끝 54까지 비교해도 서로 멀어지는 뚜렷한 진행이 없고, 작은 위치 정리만 보인다.', [44,48,51,54]),
  2: (0, '66~78에서 양손은 어깨 안쪽 앞의 비슷한 간격을 유지한다. 70~74의 중간 모습도 같은 배치이며 손가락과 높이의 작은 변화가 양손을 벌리는 이동으로 이어지지 않는다.', [66,70,74,78]),
  3: (0, '77~81의 다섯 연속 화면에서 양손과 팔꿈치의 배치가 유지된다. 이미 떨어진 두 손이 더 밖으로 이동하는 과정은 보이지 않는다.', [77,78,79,80,81]),
  4: (0, '97~108에서 두 손은 양어깨 앞의 떨어진 배치를 유지한다. 100~105 사이에도 간격을 더 벌리는 진행 없이 약간의 높이 정리만 보인다.', [97,100,103,105,108]),
  5: (0, '72~80의 양손은 목 아래에서 일정하게 떨어져 있다. 75~77과 마지막 80의 작은 손·팔 위치 변화는 추가로 벌리는 뚜렷한 동작으로 보이지 않는다.', [72,75,77,80]),
  6: (0, '75~81의 두 손 간격이 유지되고, 82~84에서는 두 손이 조금 내려오고 안쪽으로 정리된다. 이 구간에는 두 손이 서로 더 멀어지는 뚜렷한 이동이 없다.', [75,78,81,82,83,84]),
  7: (0, '53~58의 여섯 연속 화면에서 두 손은 목 아래 양쪽의 배치를 유지한다. 중간 55~56을 포함해 손 사이 간격을 더 벌리는 진행은 보이지 않는다.', [53,54,55,56,57,58]),
  8: (0, '63~72에서 두 손은 각각 어깨 가까이에 머문다. 팔꿈치와 손 높이가 조금 정리되지만 손들이 서로 멀어지는 뚜렷한 추가 이동은 관찰되지 않는다.', [63,66,68,70,72]),
  9: (0, '77~81에서 양손은 쇄골 부근 양쪽의 떨어진 자세를 유지한다. 78~80의 중간 화면에서도 간격 확대 과정이 보이지 않는다.', [77,78,79,80,81]),
  11: (0, '67~74에서 두 손의 떨어진 배치가 유지되고, 75~76에서는 두 손이 조금 내려온다. 위아래의 작은 정리와 달리 서로 벌어지는 뚜렷한 이동은 없다.', [67,70,73,74,75,76]),
  12: (0, '67~76의 두 손은 어깨 앞에 떨어진 상태를 유지한다. 중간 70~73과 끝 76에서도 손 사이 간격을 뚜렷하게 늘리는 진행 없이 작은 자세 정리만 보인다.', [67,70,72,73,76]),
  13: (0, '74~80의 두 손은 가슴 위쪽 양옆의 배치를 유지한다. 76~79의 중간 모습까지 확인했으나 두 손이 서로 멀어지는 추가 진행은 보이지 않는다.', [74,76,77,79,80]),
}

def main():
    queue_path = HERE / 'WORD2523_REVIEW_QUEUE.json'
    scope_path = HERE / 'REVIEW_SCOPE.json'
    queue = load(queue_path)
    assert len(queue['rows']) == 12
    assert {r['person'] for r in queue['rows']} == set(MANUAL)
    rows = []
    for q in queue['rows']:
        presence, reason, witnesses = MANUAL[q['person']]
        assert q['word_id'] == 'WORD2523'
        assert [v['index'] for v in q['queries']] == [1]
        assert q['native_frames'] == list(range(q['frames'][0], q['frames'][1]+1))
        assert len(q['native_frames']) == len(q['timestamp_us'])
        assert sha(q['sheet']['path']) == q['sheet']['sha256']
        assert set(witnesses).issubset(q['native_frames'])
        row = {k:q[k] for k in ['clip_id','word_id','role','person','source','source_sha256','input','input_sha256','frames','sheet','native_frames','timestamp_us']}
        row.update({
            'source_sha':q['source_sha256'], 'input_sha':q['input_sha256'],
            'queries':[{
                'target_index':1, 'definition':q['queries'][0]['definition'],
                'presence':presence, 'reason_ko':reason,
                'witness_frames':witnesses,
                'witness_timestamp_us':[q['timestamp_us'][q['native_frames'].index(f)] for f in witnesses],
                'observability':'both_hands_and_forearms_visible_all_fixed_native_frames',
            }],
            'whole_clause_truth':None, 'educational_error':None,
            'reviewer':'Codex /root/egypt_source_binding_review',
            'actually_viewed':True, 'sheet_view_count':1,
            'all_native_frames_actually_viewed':True,
            'judgment_scope':'Fixed-window substantial hands-apart movement occurrence only; holding an already separated pose is not movement occurrence.',
        })
        rows.append(row)
    result = {
        'status':'DIRECT_FIXED_WINDOW_OCCURRENCE_REVIEW_COMPLETE',
        'method':'One actual view_image opening per fixed chronological sheet, inspecting every consecutive native frame. No old y, model scores, or previous review contents read in this review.',
        'reviewer':'Codex /root/egypt_source_binding_review',
        'not_independent_KSL_expert_truth':True,
        'historical_exposure_possible':True,
        'model_predictions_read':False,
        'old_labels_read':False,
        'new_extraction_or_inference_or_training':False,
        'fixed_windows_changed':False,
        'queue':{'path':str(queue_path),'sha256':sha(queue_path)},
        'scope':{'path':str(scope_path),'sha256':sha(scope_path)},
        'counts':{'rows':len(rows),'sheets_actually_viewed':12,'native_frames_actually_viewed':sum(len(r['native_frames']) for r in rows),'fit':9,'development':3,'presence_1':0,'presence_0':12,'presence_unknown':0},
        'limitations':[
            'Local absence of this movement is not a wrong sign, omitted required action, or whole-clause correctness judgment.',
            'Single frontal 2D sheet observation does not prove depth, contact, exact movement boundaries, or all finger details.',
            'All labels apply only within the fixed reviewed window; no claim that the movement is absent elsewhere in the source video.',
            'Development recorded IDs 11-13 are repeated development material, not a new independent evaluation.',
            'Same Codex manual physical observations; not independent human or Korean Sign Language educator ground truth.',
        ],
        'rows':rows,
    }
    out = HERE / 'WORD2523_REVIEW.json'
    assert not out.exists(), 'Preserve existing review; refuse overwrite'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(out),'sha256':sha(out),'counts':result['counts']},ensure_ascii=False))

if __name__ == '__main__':
    main()
