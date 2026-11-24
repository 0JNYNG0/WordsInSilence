"""Architecture clarification from existing code/logs. No inference or training."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json,hashlib
R=Path(r'C:\MyUnityProjects\WordsInSilence');O=R/'research/front_keyphase_v1/ops'
U=Path(__file__).resolve().parent;B=O/'word100_multipeople_live_feedback_20261101';D=O/'word100_seated_web_framing_20261102'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
code=(B/'engine.py').read_text(encoding='utf-8-sig')
assert 'comparisons=[compare_one(a,arrays[r[\'clip_id\']],r) for r in rows]' in code
assert "selected=sorted(comparisons,key=lambda x:x['distance'])[:2]" in code
assert "linguistic_correctness=None" in code
manifest=read(D/'CODE_MANIFEST.json')
assert all(sha(Path(p))==h for p,h in manifest['base_pins'].items())
v=read(D/'LIVE_UI_VERIFICATION.json');timings=[]
for r in v['results']:
    total=r['request_timing']['analysis_seconds'];adapter=r['latency']['total_adapter_seconds']
    timings.append(dict(public_clip=r['clip'],request_seconds=total,input_observer_seconds=adapter,
                        comparison_guard_serialization_other_seconds=total-adapter,
                        note='Residual is not an isolated DTW microbenchmark; public crop test, not user timing.'))
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
summary=('사용자가 최종 목표를 재확인했다. 여러 수행자의 공개 한국수어 영상과 동작 설명으로 '
         '선택한 단어의 중요한 손모양·관계·움직임·순서를 학습하고, 실행 시 사용자 영상과 목표 단어로 '
         '특징별 차이·허용 변화·관측 부족을 판단하는 모델이 목표다. 현재 8804는 사용자 특징을 한 번 추출한 뒤 '
         '저장된 9개 참고와 각각 DTW 비교하고 같은 두 전체 참고 및 FIT 통계로 안내하는 연구 비교 도구다. '
         '학습된 7특징 판정부가 있지만 목표 단어의 교육적 수행 규칙을 배운 최종 판정부는 아직 없다. '
         '이 둘을 완성 수준이 같은 것으로 설명하지 않는다. 9인 추가 학습 후보도 좁은 특징 실험이며 미채택이었다.')
next_action=('현재 9참고 비교 도구의 세부 보완을 최종 목표의 중심으로 계속 확대하지 않는다. '
             '다음은 목표 단어를 조건으로 받는 시간별 특징·단계 판정부와 최종 차이 판단부의 입력/출력 및 학습 정답을 먼저 명세한다. '
             '공개 다인 영상·출처 설명·기존 시간 주석에서 실제 지원되는 항목을 재사용하고, 정상 변화와 중요한 차이의 정답 부족을 구분한다. '
             '학습 시 DTW를 단계 대응 보조로 쓸 수 있으나 DTW 대응을 진짜 단계 정답으로 자동 승격하지 않는다. '
             '제품 실행 경로는 사용자 영상+목표 단어로 판단하고, 9개 원영상 실시간 대조를 필수로 두지 않는다. '
             '대표 참고 재생/DTW는 선택적 설명·검토 용도다. 9참고 데모는 비교 기준/회귀 도구로 보존한다. '
             '같은 학습 단어의 다른 사람과 확인된 차이 대조로 평가하고, 사용자 자료·DEV 피드백을 학습/튜닝/선택에 사용하지 않는다. '
             '정상 영상만으로 모든 교육적 허용 경계가 저절로 학습된다고 주장하지 않는다. '
             '기존 C/T3F 및 현재 데모는 보존. 이번은 구조 확인·목표 명세이며 새 학습/추론/다운로드는 하지 않았다.')
report=f'''현재 구현과 사용자 목표의 구분 — {now}

확인 결과
{summary}

현재 구현
1. 사용자 영상을 한 번 관측한다. MediaPipe 관절과 동결 SignRep RGB 특징, 학습된 작은 시간별 판정부를 사용한다.
2. 작은 판정부는 오른손 검지만 펴짐/오른손 네 손가락 펴짐/왼손 네 손가락 펴짐/양손 각각 네 손가락 접힘/양손 높이 관계의 총7항목을 추정한다.
3. 선택한 단어는 참고를 고르는 조건이다. 현재 작은 판정부 입력에는 목표 단어의 설명이나 단어 ID를 넣지 않는다.
4. 저장된 해당 단어의9참고 특징과 각각 비교·시간 정렬한다. 참고 영상을9번 새로 검출/신경망 추론하는 것은 아니다.
5. 가까운 두 전체 참고를 선택하고 FIT 거리 범위·관측 지원 기준으로 유사/차이 후보/근거 부족을 만든다. 팔 대응을 추가 확인한다.
6. 출처 동작 설명은 단계에 연결돼 표시되지만, 그 문장을 읽고 교육적 정오를 판정하는 신경망은 아니다.

화면 의미
유사=비교 가능한 특징이 현재 참고 수치 범위에 있음. 차이 후보=측정 차이가 범위를 벗어남. 근거 부족=관측/대응/참고 합의가 부족함.
어느 것도 단독으로 수어 전체의 정답/오답 확정이 아니다. 개별9참고 결과는 내부 분석용이며 사용자가 원한 최종 피드백 화면이 아니다.

시간 근거
'''
for row in timings:
    report+=f"{row['public_clip']}: 전체 {row['request_seconds']:.2f}초, 입력 관측 {row['input_observer_seconds']:.2f}초, 비교·검사·기타 {row['comparison_guard_serialization_other_seconds']:.2f}초.\n"
report+='참고 비교도 비용이 들지만 이 기록에서는 입력 관측이 대부분이다. 9대조를 빼면 9배 빨라진다는 해석은 틀리다. 사용자 최신 촬영의 속도를 재측정한 결과가 아니다.\n\n'
report+='목표에 맞춘 설계 기준\n'+next_action+'\n\n'
report+='목표 구조: [공개 여러 사람+설명+시간별 정답으로 사전 학습] → [사용자 영상+목표 단어] → [학습된 단계/특징/차이 판단] → [한 번의 통합 피드백].\n'
report+='이 구조는 설계 기준이며 구현 완료나 성능 입증이 아니다. 현재 관측기·특징학습·시간축·누락 보류·공개자료 분리 작업은 재사용한다. 최종 의미 판단 학습을 대신한 것으로 포장하지 않는다.\n'
report+='다른 모든 모델을 새로 검색하거나 9참고 UI만 숨기는 것으로 해결했다고 하지 않는다. 중요한 오류 예시와 허용 변화의 감독 신호, 미사용 사람 평가가 필요하다.\n'
(U/'RESULT_KO.txt').write_text(report,encoding='utf8')
record=dict(status='ARCHITECTURE_AUDITED_TARGET_MODEL_RECONFIRMED_NOT_IMPLEMENTED',updated_at=now,
            summary=summary,next_action=next_action,public_timings=timings,
            source_pins={str(B/n):sha(B/n) for n in ['adapter.py','engine.py','live_engine.py','RESULT_KO.txt','TRAINING_REPORT.json']},
            new_inferences=0,new_training_runs=0,user_inputs_read=0,live_code_changed=False,
            evidence=str(U/'RESULT_KO.txt'),review='Codex source and saved-log audit; no new experiment or independent KSL judgment')
with (U/'REVIEW.json').open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf-8-sig')
for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','참고 대조 데모와 목표 학습 모델의 차이 확인',summary),
                          ('## 9. 다음 단일 작업','목표 단어 조건의 학습 피드백 모델 명세',next_action)]:
    assert marker in text;text=text.replace(marker,marker+'\n\n### 2026-11-02 '+title+'\n'+body,1)
p.write_text(text,encoding='utf8')
p=O/'STATE.json';state=read(p);state['current_stage']=record['status'];state['next_single_action']=next_action;state[U.name]=record
p.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_action+
    '\n\n기존8804는word100_seated_web_framing_20261102/run_server.py로 현재 비교 데모 유지, PID31864(재확인 필요). '
    '8802는이전판PID26000(재확인 필요). 새서버/학습/추론 없음. C/T3F 보존. '
    '이번에는 사용자 최신 촬영을 읽지 않고 코드와 기존 공개시험 시간만 확인했다.\n근거: '+str(U/'RESULT_KO.txt')+'\n',encoding='utf8')
with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:
    f.write('\n\n'+now+' 목표 재확인: 다인 참고 대조와 학습된 판단 모델의 구분\n'+summary+
            '\n경위: 여러 사람의 다양성을 다루기 위해 구현한9참고 대조 도구가 사용자가 원한 학습된 피드백 모델처럼 받아들여질 수 있었다. '
            '사용자는 각 참고를 실행 때마다 대조하는 최종구조를 원하지 않음을 명확히 했다. 새로운 목표가 생긴 것보다는 기존 목표와 구현의 차이를 바로잡는 재확인이다.\n'+next_action+'\n근거: '+str(U/'RESULT_KO.txt')+'\n')
assert read(O/'STATE.json')[U.name]['new_inferences']==0
print(record['status'])
