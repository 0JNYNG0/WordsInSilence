from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
from relation_model import V,read
O=V.parent;R=O.parents[2];U=O/'word100_learned_feature_retention_20261102';P=O/'word100_learned_guide_feedback_20261102'
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
 now=datetime.now(timezone(timedelta(hours=9))).isoformat();verify=read(V/'SAVED_VERIFICATION.json');ui=read(V/'DEMO_UI_VERIFICATION.json');render=read(V/'DEMO_RENDER_VERIFICATION.json');controls=read(U/'CONTROL_VERIFICATION.json')
 assert verify['status'].startswith('PASS') and ui['status']==render['status']=='PASS'
 launch=read(V/'DEMO_LAUNCH.json')
 summary=('기존 손모양 판정부를 보존하고 학습된 시간흐름과 연결해 36개 저장 DEV clip-seed에서 기존 특징과 DTW 경로를 유지했다. 실제 공개영상 속도·대기자르기·동작자르기·화면가림 24개를 새 추출해 검사했다. 정상변형16개에서 확정차이 경고0, 검은구간4개 시간대 그대로 보류. 양손 모음 구간을 지운 이집트가 기존7특징만으로는 비슷함으로 남는 원인을 확인했다. 공개144시점에 얼굴앞 오른손·양손 긴손가락끝 모음·왼팔 부근 오른손 3관계의 물리적 주석을 추가했다. FIT ID01~09 108장면/DEV ID11~13 36장면, 관절만/영상+관절 각3seed 총6회 새 신경망 학습. 관계 균형정확도 평균91.10%→95.84%(+4.74%p), 전체 수어 정확도 아님. 삭제된 양손모음 관계를3seed 모두 확인못함으로 분리. 정상 속도·대기 변형의60관계판정(20관계×3seed)은 유지, 관측부족 보존. 새 격리8806 데모 공개 업로드7.46초/두영상/관절선/구간재생/정상·삭제시험 UI 검증. C·T3F·8804·8805와 사용자자료 보존. 전체목표 active.')
 next_step=('다음 단위는 정지 위치관계에서 실제 이동·순서 판단으로 확대한다. 현재 잘하다의 왼팔부근 관계는 남아 있어 팔을 따라 이동한 구간 삭제를 그 관계만으로 구분하지 못한다. FIT 공개영상에서 오른손과 왼팔의 상대 이동, 양손 교대 진행의 시점별 물리적 근거를 확인하고, 단어명 없이 읽는 공통 움직임 감독을 준비한다. 정상 몸흔들림/속도/교차와 실제 이동·순서 차이를 분리하고 필요한 작은 시간판정부를 학습한다. 기존10특징(7손모양/높이+3관계)·단계경로 보존, DEV새점수로 문턱/seed 선택 금지. 실제 움직임삭제는 언어적오답으로 자동승격하지 않고 명시된 설명항목과 학습근거에 한정. 이미끝난6관계학습·24원영상변형·144새관계판정·72정상DEV판정·공개업로드 재실행금지. 8806 새촬영은추론만, 개인자료 학습·튜닝·선택 금지. 계속목표 미완료, 다음 진행을 위한 승인 재요청 없음.')
 report=f'''이번에 달라진 점

1. 손모양을 잘 맞추던 기존 지식을 보존했다.
   앞선 새 전체시간 모델은 기존 정답을 회차별 4~6개 놓쳤다. 기존 신경망 관측 분기를 유지하고 새 단계/시간 흐름 분기와 연결했다.
   기존7특징 194확인항목의98.24%와 새 학습DTW경로를 그대로 유지한다. 새 성능향상 수치가 아니라 퇴행을 막은 것이다.

2. 실제 공개 영상을 고쳐 시험했다.
   4표현 × 정상/0.8배속/1.25배속/초기대기제거/관찰동작구간제거/주요구간검정화면 =24영상.
   변경한 영상에서 MediaPipe와 SignRep 특징을 처음부터 다시 추출했다. 특징배열만 잘라 앞뒤 RGB문맥에 정보가 남는 시험과 다르다.
   정상적인 속도·대기 변형16개에 확정차이 경고는0개. 일부 항목은 계속 보류된다. 따라서16개 전체를 올바른 수어로 판정했다는 뜻은 아니다.
   검은 구간4개는 실제 빠진 시간대를 모두 남겼다. 관측부족을 맞음으로 바꾸지 않았다.
   양손을 모으는 장면을 삭제한 이집트는 기존 지원 손모양만 비슷하다고 나왔다. 판정부가 그 양손 관계를 배우고 검사하지 않았기 때문이다.

3. 부족했던 신체 관계를 실제로 지도학습했다.
   기존 공개4표현48영상의144시점을 예측 없이 눈으로 확인했다. 학습9개기록ID/108시점, 개발비교3개기록ID/36시점이다.
   오른손이 얼굴 앞에 있음 / 양손 긴 손가락 끝이 한곳에 모임 / 오른손이 왼팔 부근에 있음.
   관절만 쓰는 조건과 SignRep 영상특징도 쓰는 조건을 각각3회 학습했다. 단어명·설명을 관측 판정부에 넣지 않는다.
   관절만: 89.44%,94.40%,89.44% → 평균91.10%.
   영상+관절:96.07%,96.07%,95.38% → 평균95.84%. +4.74%p.
   이것은 새3관계의 균형정확도이며 기존7손모양 점수/TCN단어인식률/전체피드백정확도와 서로 다른 수치다.
   새 학습은 약4.8만개 가중치의 관계판정부다. 이번에 SignRep본체를 다시 미세조정한 것은 아니다. 앞선 마지막투영층 rank8 미세조정 모델은 시간흐름 분기에 그대로 있다.

4. 설명·시간과 연결했을 때도 확인했다.
   FIT에서 직접 확인한 양성관계가 여러 기록ID와 기준영상에 함께 있는4개구간만 연결했다. 단어명으로 관계 정답을 채우지 않았다.
   학습에 넣지 않은12개공개영상의15관계 ×3seed=45관계에서 해당관계를 찾았다. 이는 독립된45영상이 아니다.
   실제 정상/속도/대기변형16영상의20관계 ×3seed=60개도 유지했다.
   양손모음 구간 삭제는3회 모두 '그 관계를 확인하지 못함'으로 바뀌었다. 화면가림은 모두 근거부족을 유지했다.
   관찰구간 삭제는 인위적인 시험이다. 실제 학습자의 오류나 모든 한국수어의 필수규칙 정답으로 취급하지 않는다.

5. 직접 넣어 볼 별도 데모가 있다.
   {launch['url']}
   이집트 선택 → 정상 공개 영상 시험 / 주요 구간을 뺀 공개 시험을 번갈아 확인하면 변화가 보인다.
   새 촬영/영상 파일도 입력 가능. 실제 공개파일 업로드 분석 {ui['upload_seconds']:.2f}초. 사용자 카메라는 켜지 않았다.
   손·팔 검출과 영상특징 추출은1번, 학습된 기준흐름과 DTW1회, 참고원영상 개별대조0회다.
   새 관계판정부 계산은 영상당 대략0.005초. 전체촬영 분석시간과 혼동하면 안 된다. 표시용 저장 공개 예제의 시간은 앞서 측정한 관측추출 경로 시간이다.
   정상/삭제영상8개 재생, 관절선, 구간버튼, 한글 결과, 화면 대비·폭을 브라우저에서 확인했다.

아직 남은 핵심
   '손이 왼팔 부근에 있음'과 '왼팔을 따라 실제로 이동했음'은 다르다. 후자는 아직 확인하지 못한다.
   정확한 손끝 접촉·엄지·손바닥 방향·반복횟수·이동 방향과 중요한 순서의 전체 판정은 미완료다.
   검증자료는 이미 사용했던 개발자료다. 새 사람 최종검증이나 농인강사 검토가 아니다.
   주석과 별도검산은 같은 Codex가 했다. 독립 사람의 정답 검토로 부르지 않는다.
   '반복이 몇 회여야 의미가 맞는가' 같은 교육적 허용범위는 이번 물리적관계학습으로 해결됐다고 주장하지 않는다.
   C·T3-F 및8804·8805를 보존했다. 개인·멘토 촬영 학습/튜닝/후보선택0. 외부전송/추가다운로드/Git0.

다음 작업
{next_step}

근거
   RELATION_TARGETS.json / TRAINING_CONTRACT.json / TRAINING_RESULT.json
   GUIDE_RELATION_SUPPORT.json / RELATION_COMPARISON.json / DEVELOPMENT_RELATIONS.json
   SAVED_VERIFICATION.json / DEMO_UI_VERIFICATION.json / DEMO_RENDER_VERIFICATION.json
   이전 단위 CONTROL_VERIFICATION.json / RETAINED_COMPARISON.json
'''
 (V/'RESULT_KO.txt').write_text(report,encoding='utf8')
 (U/'RESULT_KO.txt').write_text('기존손모양 보존 및 실제영상24대조 완료.\n'+summary+'\n후속 신체관계 학습 결과: '+str(V/'RESULT_KO.txt')+'\n',encoding='utf8')
 current=dict(status='RELATION_NEURAL_DEMO_VERIFIED_MOVEMENT_SEQUENCE_PENDING',at=now,summary=summary,next=next_step,
  goal_status='active',last_goal_turn='progress: six new relation-head training runs, actual edited-video analysis, isolated working demo',
  live_server=launch,active_training=None,active_exec_sessions=[],old_models_preserved=True,user_mentor_training_tuning_selection=False,
  report=str(V/'RESULT_KO.txt'),independent_human_validation=False,verification=str(V/'SAVED_VERIFICATION.json'))
 write(V/'CURRENT_EXECUTION.json',current)
 write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_RETAINED_FEATURES_AND_ACTUAL_VIDEO_CONTROLS',at=now,active_exec_session=None,next_unit=str(V),report=str(U/'RESULT_KO.txt')))
 state_path=O/'STATE.json';state=read(state_path);state['current_stage']=current['status'];state['next_single_task']=next_step;state['updated_date_kst']='2026-11-02'
 state[U.name]=read(U/'CURRENT_EXECUTION.json');state[V.name]=current;write(state_path,state)
 context=R/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text(encoding='utf8')
 for heading,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','기존손모양 보존 + 신체관계 신경망6학습·8806 데모',summary),('## 9. 다음 단일 작업','관계의 존재에서 실제 이동·순서로',next_step)]:
  assert heading in text;text=text.replace(heading,heading+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
 context.write_text(text,encoding='utf8')
 (R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_step+'\n\n현재 단위: '+str(V)+'\n8806 실제 PID '+str(launch['pid'])+'. 8805 기존 후보 보존. 실행중 학습/비교 명령 없음. 구현/별도검산 Codex, 사람독립정답 아님.\n',encoding='utf8')
 plan=R/'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt';text=plan.read_text(encoding='utf8');begin=text.index('최신 실행 현황');end=text.index('1. 사용자가 확정한 목표')
 text=text[:begin]+'최신 실행 현황 — 아래 과거 설계/과거 미수행 문구보다 이 항목을 우선한다.\n'+summary+'\n'+next_step+'\n\n'+text[end:];plan.write_text(text,encoding='utf8')
 with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:f.write('\n\n'+now+' — 손모양 보존과 별도 관계학습으로 지원 범위 확대\n'+summary+'\n목표변경이 아니라 누락된 관계 감독을 보충했다. 기존 강점을 보존하고 새특징/이동/순서의 책임을 구분한다.\n'+next_step+'\n')
 write(V/'FINAL_ARTIFACTS.json',dict(at=now,files={str(p.relative_to(V)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [V/n for n in ['RESULT_KO.txt','run_demo.py','demo_adapter.py','relation_runtime.py','demo.js','demo.html','TRAINING_CONTRACT.json','RELATION_TARGETS.json','SAVED_VERIFICATION.json','DEMO_UI_VERIFICATION.json','DEMO_RENDER_VERIFICATION.json']]}))
 assert read(state_path)[V.name]['status']==read(V/'CURRENT_EXECUTION.json')['status'];assert str(V) in (R/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
 print(json.dumps(dict(status=current['status'],demo=launch,report=str(V/'RESULT_KO.txt'),goal='active'),ensure_ascii=False))
if __name__=='__main__':main()
