from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
    with (U/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
v=read(U/'CONTEXT_VERIFIED.json');t=read(U/'NATIVE_TIME_AUDIT.json');n=read(U/'NEXT_ACTION_DECISION.json')
assert v['checks']==3004 and v['context_queries']==129
assert sha(U/'CONTEXT_LABELS.json')==v['context_labels_sha256']
assert t['clips']==12 and all(r['cached_axis_matches_media'] and r['queue_axis_matches_cache'] for r in t['rows'])
for p in n['reviewed_input_pins']:assert sha(p['path'])==p['sha256']
report='''공개 영상의 동작 맥락 직접 검토 결과

공개 한국수어 4표현 × 기록 ID01·04·07의12영상에서 기존 시간순 시트369칸을 실제 확인했다. 정확한129질문(고유 영상-프레임86개)에 해당 맥락52, 다른 맥락12, 불확실65를 기록했다. 이는 인식률이나 수어 정오 점수가 아니다. 같은 Codex의 별도 작업자가 저장 자료3004항목을 검산했으며, 독립된 수어 전문가의 정답 검증은 아니다.

중요한 발견: 손모양을 검사할 때 팔·몸의 배치로 비교할 맥락을 정할 수 있는 경우가 있었다. 그러나 쓸기·교대·내림 같은 움직임을 검사하기 전에 그 움직임 자체가 보이는 시점만 선택하면, 움직임을 생략한 경우를 비교에서 빼게 된다. 불확실65개를 검출 실패65개라고 해석하면 안 된다. 따라서 넓은 팔 작업 범위를 먼저 잡고 그 안에 실제 필요한 움직임이 있었는지를 별도로 묻도록 다음 구현을 정했다.

기존 음성 구간 일부에는 직접 관측한 움직임의 끝부분이 겹쳤다. 해당 옛 라벨을 고치거나 재채점하지 않고, 새 시험의 정지 정답으로 재사용하지 않기로 했다. 다음 고정 late3구간의 연속 원프레임14개를 별도로 확인한다.

시간 표시도 확인했다. 이전 시트2영상에서 평균FPS로 계산한 자막 시간이 실제 재생 시각과 최대0.011921초 달랐다. 실제12영상의 프레임 PTS와 저장 timestamp_us는 모두 일치한다. 원래 시간·관절·라벨을 바꿀 필요가 없으며, 이후 표시에는 검증된 원시각을 사용한다.

최초 통합 오류는 검토 파일의 추가 메타데이터까지 딕셔너리 전체 일치로 요구한 것이었다. 최초 코드를 남기고 필수 경로·해시·실제 파일을 검증하도록 수정했다. 라벨 내용은 바꾸지 않았다.

이번 단위 새학습0·모델추론0·기존평가재채점0·웹적용0. C·T3-F와 기존 데모 보존, 사용자·멘토 자료 학습·튜닝·선택 없음. 명확한 성능 향상 주장이 없으므로 HTML은 만들지 않았다.

다음: word100_context_episode_query_20261103에서 기존 고정 움직임 신경망을 사용해 독립적인 팔 작업 범위3개와 late3구간을 한번 비교한다. 기존 완료 판정부를 다시 채점하거나 임계값을 바꾸지 않는다. 이 FIT 진단은 새 사용자 일반화나 전체동작 완료를 증명하지 않는다. 실제 사용자 입력을 학습된 기준 흐름에 연결하는 전체 목표는 계속 진행 중이다.
'''
with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as f:f.write(report)
save('FINAL.json',dict(at=datetime.now(timezone.utc).isoformat(),status='DIRECT_EVENT_CONTEXT_REVIEW_COMPLETE',
     clips=12,claims=39,queries=129,unique_clip_frames=86,sheet_cells=369,context_counts=v['context_counts'],
     saved_checks=3004,source_native_time_verified_clips=12,display_only_time_issue=True,
     training=0,model_inference=0,web_changes=0,performance_improvement_claim=False,
     next_unit='word100_context_episode_query_20261103',whole_goal_complete=False,
     pins={str(U/k):sha(U/k) for k in ['CONTEXT_LABELS.json','CONTEXT_VERIFIED.json','NATIVE_TIME_AUDIT.json','NEXT_ACTION_DECISION.json','RESULT_KO.txt']}))
print('AF direct context pilot finalized; goal remains incomplete')
