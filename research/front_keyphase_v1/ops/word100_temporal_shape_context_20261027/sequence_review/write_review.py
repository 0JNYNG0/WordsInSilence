import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

U = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FROZEN = U / 'FROZEN.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# Observations below were written after visually opening all 18 ordered sheets.
# They are sequence safety observations, not static handshape ground truth.
observations = [
 ('T01', None, True, '가슴 앞에서 두 손이 겹쳐 있고, 화면 왼쪽 팔에서 이어지는 목표 오른손의 연결은 짧은 구간 내내 이어져 보인다. 손 묶음의 위치는 조금 움직이지만 목표 손끝 일부는 반대 손에 계속 가려져 있다.', '앞뒤 영상도 같은 가림을 유지한다. 정지에 가까워 보인다는 이유로 중심 손가락 모양을 확정할 수 없다.'),
 ('T02', None, True, '두 손이 가슴 앞에서 겹친 상태로 조금 내려간다. 목표 오른손의 팔 연결은 이어져 보이지만 손가락 끝부분은 겹침 때문에 분리해서 보기 어렵다.', '손 위치가 내려가는 것과 손가락 모양이 바뀌는 것을 구분해야 한다. 앞뒤 정보만으로 가려진 중심 모양을 정답 처리하지 않는다.'),
 ('T03', None, True, '목표 오른손과 세워진 반대 손이 가슴 앞에서 겹쳐 있다. 팔의 연결과 전체 움직임은 이어져 보이고 반대 손의 손가락 윤곽은 조금 달라지지만 목표 손의 일부 손가락은 계속 가려진다.', '반대 손에서 보이는 손가락 변화는 목표 오른손의 변화 근거가 아니다. 목표 손의 중심 모양은 확정하지 않는다.'),
 ('T04', None, True, '목표 오른손이 입 앞에서 세워져 있고 팔과 손의 연결은 계속 보인다. 끝부분에서 손이 약간 회전하지만 손가락들이 옆면으로 겹쳐 보여 개별 굽힘 변화는 확실하지 않다.', '마지막 K07 확대도 확인했다. 회전으로 윤곽이 달라지는 것과 손가락이 실제 굽혀지는 것을 구분하기 어려우므로 변화 여부를 보류한다.'),
 ('T05', True, False, '입 앞에 세운 목표 오른손이 후반부에 아래쪽으로 이동하면서 손가락 배열도 달라진다. 마지막 여러 장면에서는 앞부분보다 일부 손가락이 접힌 형태가 명확히 보인다. 팔과 손 연결은 이어진다.', '이 0.4초 구간 안에 실제 모양 전환이 있으므로 시간상 가까운 모양을 중심에 그대로 복사하면 짧은 변화를 지울 수 있다. 정확한 중심 라벨이나 전환 시점의 정답을 만들지는 않는다.'),
 ('T06', None, True, '입 앞의 목표 오른손이 후반에 바깥쪽으로 회전하며 내려온다. 팔에서 손까지의 연결은 계속 보인다. 마지막에는 손가락들이 더 분리되어 보이지만 처음은 옆면으로 겹쳐 있다.', '첫 K04와 마지막 K07 원크기 보기용 확대를 추가 확인했다. 회전에 따른 가시성 변화와 실제 굽힘 변화를 확실히 나누기 어려워 변화 여부를 보류한다.'),
 ('T07', False, False, '왼팔이 몸 옆으로 내려와 있고 목표 왼손은 짧은 구간 동안 거의 같은 위치와 윤곽을 유지한다. 팔에서 손까지 연결이 끊기거나 다른 손으로 바뀌어 보이는 장면은 없다.', '이 구간에서 내려놓은 손이 안정적으로 보인다는 관찰이다. 전체 단어에서 이 손이 필수인지, 화면 밖일 때도 같은 상태인지 판단하는 근거는 아니다.'),
 ('T08', False, False, '몸 옆에 내려놓은 목표 왼손과 팔의 연결이 계속 보인다. 약간의 몸 움직임은 있으나 손가락 윤곽의 뚜렷한 전환은 관찰되지 않는다.', '내려놓은 손을 활동 손처럼 강제로 분류하거나 변화가 없다는 이유로 오류라고 판단하지 않는다. 세부 손가락 관절의 정확성을 뜻하지는 않는다.'),
 ('T09', False, False, '목표 왼손이 몸 옆 아래에 머물고 팔과 손의 연결이 이어진다. 관찰한 13장에서는 뚜렷한 손모양 전환이나 다른 손과 뒤바뀌는 징후가 보이지 않는다.', '보이는 윤곽의 안정성만 기록한다. 이 짧은 휴식 장면을 모든 시간대의 휴식 정답으로 확대하지 않는다.'),
 ('T10', False, False, '편 목표 오른손이 반대 손바닥 쪽으로 내려갔다가 다시 간격이 벌어진다. 겹치는 순간에도 목표 오른팔과 손의 연결을 따라갈 수 있고 편 손가락 윤곽은 대체로 유지된다.', '정상적인 양손 겹침을 손 정체성 오류나 손모양 오류로 간주하지 않는다. RGB에서 따라갈 수 있다는 관찰이며 자동 추적기가 실제로 옳았다는 증명은 아니다.'),
 ('T11', False, False, '목표 오른손이 반대 손 위로 접근한 뒤 다시 떨어지는 움직임이 보인다. 팔 연결과 편 손 윤곽은 이어지고 두 손이 가까워지는 때도 어느 팔의 손인지 크게 바뀌어 보이지 않는다.', '양손의 간격 변화와 손가락 모양 전환을 분리했다. 이 구간에서 뚜렷한 모양 전환은 보이지 않지만 손가락 각도 정답을 보증하지 않는다.'),
 ('T12', False, False, '목표 오른손이 반대 손 위에서 가까워졌다 멀어지며 움직인다. 겹친 구간과 떨어진 구간 모두 오른팔 연결이 보이고 펼친 손 윤곽은 유지된다.', '정상적인 겹침과 반복적인 위치 움직임은 보존해야 한다. 손이 가까워졌다는 이유만으로 좌우를 바꾸거나 모양을 보류할 근거는 아니다.'),
 ('T13', True, False, '얼굴 앞 위쪽의 목표 오른손에서 손가락을 모았다가 펼치고 다시 모으는 변화가 뚜렷하다. 전완에서 손까지 연결이 이어지며 손 위치도 조금 이동한다.', '짧은 구간 안의 펼침과 다시 모음이 실제로 보인다. 앞뒤 모양이 같아 보여도 중간 변화를 없애면 안 된다. 중심 정답을 시간 추정으로 만들지 않는다.'),
 ('T14', True, False, '목표 오른손이 얼굴 위쪽에서 이동하면서 모인 손가락 사이가 잠시 벌어졌다가 다시 좁아진다. 같은 팔의 손을 이어서 볼 수 있다.', '변화 크기가 다른 사례보다 작아도 중간 펼침 윤곽이 보인다. 부드럽게 만든다는 이유로 이 변화를 평평하게 만들지 않아야 한다.'),
 ('T15', True, False, '목표 오른손이 얼굴 앞 위에서 손가락을 펼쳤다가 다시 모은다. 팔의 연결과 손 위치 이동은 연속적으로 보이고 손가락 변화도 여러 장면에서 확인된다.', '실제 손모양 전환이 있는 구간이다. 같은 손을 추적하는 것과 모양을 같게 유지하는 것은 다르다. 정답 전환 시점을 자동 부여하지 않는다.'),
 ('T16', True, False, '목표 오른손이 얼굴 앞에 머물다가 아래로 내려오면서 손가락이 접히는 변화가 후반부에 보인다. 손과 팔 연결은 계속 이어져 보인다.', '위치 내림과 손가락 접힘이 함께 나타난다. 이 구간 전체를 고정 손모양으로 처리하면 전환을 놓친다. 중심 순간의 모양은 별도 직접 관측으로 판단해야 한다.'),
 ('T17', True, False, '목표 오른손이 얼굴 앞에서 아래로 내려가며 펼쳐져 보이던 손가락들이 점차 모이고 접힌다. 손목과 팔의 연결은 유지된다.', '후반부의 손모양 변화가 분명하다. 시간상 이웃이라는 이유로 앞의 편 모양이나 뒤의 접힌 모양을 중심에 강제 복사하면 안 된다.'),
 ('T18', True, False, '목표 오른손이 얼굴 앞에 있다가 턱 아래 방향으로 내려오며 손가락 윤곽이 열린 상태에서 모이고 접힌 상태로 바뀐다. 같은 팔의 손이 계속 움직이는 것으로 보인다.', '이 짧은 구간에서는 정체성보다 모양 전환 보존이 중요하다. 중심의 불확실한 부분은 불확실한 채로 남겨야 한다.'),
]

if (OUT / 'SEQUENCE_REVIEW.json').exists():
    raise RuntimeError('Existing result must not be overwritten')
frozen = json.loads(FROZEN.read_text(encoding='utf-8-sig'))
side_map = {r['case_id']: r['side'] for r in frozen['rows']}
extra = {'T04':['K07_focus.png'], 'T06':['K04_focus.png','K07_focus.png']}
pins = {str(FROZEN): sha(FROZEN)}
rows = []
for case_id, change, ambiguous, observation, notes in observations:
    primary = U / 'visual_inputs' / case_id / 'ordered_contact.png'
    seen = [primary] + [primary.parent / x for x in extra.get(case_id, [])]
    for p in seen:
        pins[str(p)] = sha(p)
    rows.append({
        'case_id':case_id, 'target_side':side_map[case_id],
        'actual_viewed_path':str(primary), 'viewed_sha256':sha(primary),
        'viewed_files':[{'path':str(p),'sha256':sha(p)} for p in seen],
        'observation_ko':observation,
        'identity_concern':False,
        'visible_shape_change':change,
        'target_shape_unresolved_across_some_before_after_views':ambiguous,
        'central_truth_assigned':False,
        'notes':notes,
    })
assert len(rows) == 18 and len({r['case_id'] for r in rows}) == 18
assert set(side_map) == {r['case_id'] for r in rows}
result = {
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'reviewer':'Codex child agent /root/local_feature_inventory',
    'review_type':'Actual RGB ordered-contact visual safety review; no static feature labels assigned',
    'review_input':'18 frozen windows, 13 consecutive RGB frames each, center +/- 6; approximately 0.4 seconds per window',
    'blinding_limit':'Reviewer prepared the exports and has seen frozen source metadata; this is not a blinded independent human truth review. No other reviewers\' labels were read.',
    'field_definitions':{
        'identity_concern':'true means a concern was seen in RGB; false only means no identity discontinuity was noticed in this window. It does not validate a detector or prove identity.',
        'visible_shape_change':'true: visible target-hand finger-configuration change somewhere in the window; false: no clear change observed in sufficiently visible outline; null: viewpoint/overlap prevents distinguishing shape change. It is not a feature label or a center-frame truth.',
        'target_shape_unresolved_across_some_before_after_views':'Occlusion/edge-on views remain ambiguous; continuity must not be promoted to certain central truth.'
    },
    'restrictions':['No inference of invisible finger truth','No acceptance/error labels for sign performance','No model training, inference, alignment or source edits','No generalization from 0.4-second examples to whole signs','No claim of independent human ground truth'],
    'rows':rows,
    'summary':{
        'reviewed_windows':len(rows),
        'reviewed_ordered_frames':len(rows)*13,
        'identity_concern_true':sum(r['identity_concern'] for r in rows),
        'visible_shape_change_true':sum(r['visible_shape_change'] is True for r in rows),
        'visible_shape_change_false':sum(r['visible_shape_change'] is False for r in rows),
        'visible_shape_change_unknown':sum(r['visible_shape_change'] is None for r in rows),
        'central_truth_labels_created':0,
    }
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'SOURCE_PINS_BEFORE.json').write_text(json.dumps(pins,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'SEQUENCE_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
after = {p:sha(p) for p in pins}
assert pins == after
(OUT/'SOURCE_PINS_AFTER.json').write_text(json.dumps(after,ensure_ascii=False,indent=2),encoding='utf-8')
verify = {'status':'PASS','reviewed_windows':18,'viewed_files':sum(len(r['viewed_files']) for r in rows),'source_pins_unchanged':len(pins),'review_sha256':sha(OUT/'SEQUENCE_REVIEW.json'),'labels_from_other_reviews_read':False,'no_center_truth_labels':True}
(OUT/'VERIFY.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'summary':result['summary'],'verify':verify},ensure_ascii=False))
