from setup import *
import numpy as np,datetime
if __name__=='__main__':
 r=read(U/'RECHECK.json');v=read(U/'VERIFICATION.json');ui=read(U/'BROWSER_VERIFIED.json');integ=read(U/'RUNTIME_INTEGRATION.json')
 protected=[]
 for path,pin in read(U/'FROZEN.json')['protected'].items():
  expected=pin['sha256'] if isinstance(pin,dict) else pin;assert sha(path)==expected;protected.append(path)
 changes=[D/'consumer.py',D/'app.js',D/'style.css',D/'index.html',D/'EXAMPLES.json',U/'positive_evidence.py']
 report=dict(status='PUBLIC_ENGINEERING_COMPARISON_VERIFIED_AND_8814_DEPLOYED',conditions=93,unique_development_clips=12,recorded_development_ids=[11,12,13],words=4,
  effect=dict(closed_only={'before_green':3,'after_green':0,'review_cue':3,'clips':3},reverse_only={'before_green':3,'after_green':0,'review_cue':3,'clips':3},both={'before_green':3,'after_green':0,'review_cue':3,'clips':3},
   normal_full={'clips':12,'before_green_cards':24,'after_green_cards':21,'orange_cards':0},normal_crop={'clips':12,'before_green_cards':20,'after_green_cards':17,'orange_cards':0},
   unexplained_green_removed='The three Egypt gather-hands cards lacked an explicit learned handshape/relationship target; now unknown rather than green.',
   original_remaining='Cropped obedience ID11/12 initial hands remain unknown; cropped ancestor ID13 contradictory model/projection stays uncertain. Phase alignment still imperfect.'),
  latency=dict(refinement_median_ms=v['combined_refinement_median_ms'],direction_median_ms=v['direction_median_ms'],scope='Added coordinate/evidence calculations only. Not total webcam-to-display latency.',runtime_integration=integ['timing']),
  checks=dict(engineering_and_saved_results=v['checks'],browser=ui['count'],runtime=len(integ['checks']),protected_hashes=len(protected)),
  limitations=['Existing exposed DEV; conservative correction added after a normal false signal was identified. Not a fresh generalization result.',
   'Nine perturbations are three public clips x three modifications, not nine independent people/videos.',
   'Closed-hand controls replace coordinates, not RGB. Reversal controls reverse existing main-interval coordinates. Not natural learner-error ground truth.',
   'Neural/projected geometry uses the same detector: consistency is not independent truth. 2D does not prove hidden handshape or physical contact.',
   'Specific along-forearm direction contract connected to existing guide; primitive supports either arm. Other words do not automatically gain all motion/repetition checks.',
   'Exact extent, contact, thumb orientation, all handshapes, omitted-stage correctness and global educational pass remain unvalidated.',
   'Normal green coverage decreases where explicit positive features were never learned; this is uncertainty correction, not a rise in recognition accuracy.'],
  architecture=dict(input='Shared RTMW keypoints from recording',tcn='Unchanged word classifier',feedback='Existing pose temporal convolution + bidirectional GRU, seven learned attribute outputs and phase/difference heads',
   new='Positive attribute support + projected geometry consistency + signed palm progress along opposite forearm + separate uncertain state',dtw='Playback alignment after evidence decisions',rag=False,rgb_model=False,new_training=False),
  private_used=False,camera_opened_by_agent=False,training=False,download=False,weights_changed=False,c_t3f_8812_preserved=True,
  verification_identity='Same Codex implementation and separate engineering checks, not independent human annotation',changed={str(p):sha(p) for p in changes},
  failure_history=['FIT v0 projected forearm length cutoff rejected foreshortening. Fixed before DEV; original retained.',
   'First compare save failed on duplicate timestamp_us keyword in reused prediction. Completed outputs reused; no completed neural cases rerun.',
   'Initial neural contradiction produced a normal crossed-hand warning. RGB review and a conservative projected-chain consistency gate added; old results retained.',
   'Browser test harness top-level await error fixed. Screenshots now wait for actual video decoding. No app camera request.'],
  next='User may retest 8814 without training/tuning on the recording. Next development: expand public reviewed handshape and motion contrast/phase-boundary supervision, especially model/pose disagreement and brief omitted stages; no new model needed solely for this gate fix.')
 write(U/'RESULT.json',report)
 ms=report['latency']['refinement_median_ms']
 text=f'''2026-11-09 손모양·이동 방향 피드백 보완

무엇을 바꿨나
이전에는 차이 신호가 약하면 비슷하다고 표시했다. 이제 손모양이 일치한다는 긍정 근거가 필요하다.
손모양을 판단한 신경망과 손가락의 투영된 굽힘 근거가 충돌하면 오답 안내가 아니라 확인 어려움으로 남긴다.
팔꿈치-손목을 기준으로 움직이는 손바닥의 진행 방향을 확인한다. 도착 위치만 보고 반대 방향 동작까지 통과시키지 않는다.
현재는 RAG 모델이 아니다. RTMW 관절→이미 학습한 시간 합성곱+GRU 및 판정부→수어 설명 연결 구조이다. 이번 추가 학습은 0회다.

공개 비교 결과
잘하다 공개 ID11/12/13 세 영상에 대해 손모양만 변경, 방향만 역순, 둘 다 변경한 9개 좌표 시험.
각 조건에서 첫 주요 동작을 비슷하다고 표시하던 3/3이 0/3으로 줄고, 차이 확인 안내 3/3으로 바뀌었다.
이것은 3개 공개 영상에서 만든 합성 시험이지 실제 사용자 오류 9건을 맞췄다는 뜻이 아니다.
정상 4단어×3명=12영상 전체화면, 같은 영상 상반신 화면, 0.75/1.25배 속도 및 위치·크기 변형에서 새 주황색 안내 0.
전체화면 초록 카드 24→21, 상반신 20→17: 이집트 손끝 모으기 3카드는 대응 단계만 있고 명시적 손모양/접촉 근거가 없어 회색으로 정정했다.
정상 상반신 조상 ID13에서는 짧게 신경망이 손을 편 것으로 잘못 읽었다. 원본 RGB에서 닫힌 손을 확인했고, 좌표와 모순되므로 회색으로 남긴다. 임계값을 조정해 초록색으로 바꾸지 않았다.
정상 교차에서 새 주황색 안내는 없으나, 세부 손모양 관측 자체가 모두 해결된 것은 아니다.

속도·검증
추가 근거 판정 중앙 {ms:.2f}ms. 전체 촬영 종료 후 대기시간이 아니다.
저장 결과/공학 검사 {v['checks']}건, 실제 브라우저 {ui['count']}건, 소비자 코드 연결 {len(integ['checks'])}건, 보호 파일 {len(protected)}해시 확인.
같은 Codex의 별도 검산이며 전문가 정답 검증이 아니다.
총 93조건, 기존 노출 개발 자료이며 새 독립 평가로 주장하지 않는다. 최초 실패와 수정 전 결과는 보존했다.

실제 적용
http://127.0.0.1:8814/ 에 반영. 새로고침 후 촬영 가능.
초록: 문장에 적힌 확인된 특징만 비슷함. 주황: 관측된 차이를 참고와 확인. 회색: 근거 부족/충돌.
손모양은 차이가 확인된 짧은 구간만, 방향은 해당 이동 구간을 주황색으로 표시한다. 사용자의 전체 수행 정답률 %는 만들지 않는다.
TCN, C·T3-F, 8812, 학습 가중치, 녹화/자동종료 방식 보존. 사용자·멘토 학습/튜닝/후보 선택 0. 카메라 자동 개방 0.

남은 일
실제 사용자의 모든 손모양 오류/동작 생략을 정확히 찾아낸다는 보장은 아직 없다.
손끝 모으기, 엄지 방향, 반복 횟수와 누락 단계, 가림 속 손모양은 학습 정답 및 검증 범위를 더 갖춰야 한다.
공개 정상/명확한 차이의 시간별 정답을 늘려 신경망의 판별과 단계 경계를 보완하는 것이 다음 개발 과제다.
RAG는 필요하면 설명을 찾는 데 쓸 수 있지만, 손모양과 방향을 판정하는 능력을 대신 만들어 주지는 않는다.
'''
 (U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
 status='''### 2026-11-09 손모양 긍정 근거·팔 기준 이동 방향 보완 — 8814 적용

word100_positive_direction_20261109. 기존 신경망의7속성 긍정근거/반대속성/투영 손가락 구조 일관성을 확인하고,팔꿈치→손목 기준 이동방향을 별도 비교한다. RAG/새학습 없음. DTW이전판정 유지. 잘하다 ID11/12/13의 수형변경/주요구간역순/복합 각3개에서 첫카드 초록3→0,차이안내3. 좌표합성9조건이며 자연오류정확도아님. 정상4단어12영상 full/crop/속도.75·1.25/위치크기변형 새주황0. full초록24→21/crop20→17은 이집트모으기 명시특징없음→보류 정정. 조상ID13 정상교차의 짧은 수형모델오류는RGB확인 및투영구조충돌→회색,오답아님. 기존노출DEV개발보완으로새독립성능주장금지.

93조건/835저장·공학/26브라우저/8연결/66보호해시확인. 같은Codex별도검산,독립사람정답아님. 추가계산약''' + f'{ms:.2f}' + '''ms(전체대기시간아님). 모델/C·T3F/8812/사용자·멘토 보호. 8814 consumer/UI/새공개예시연결,기존예시JSON보존. 주황실제근거시간·회색불확실분리,부분특징외전체정답/유사도% 없음. 재학습0/다운로드0/개인자료열람0. 최초실패·FIT단축팔문제·정상오경고개발보완기록보존. 근거RESULT_KO.txt/RESULT.json/RECHECK.json/VERIFICATION.json/RUNTIME_INTEGRATION.json.
'''
 next_action='''### 2026-11-09 손모양·방향 보완 적용 이후

8814새로고침 후 사용자재촬영 가능. 개인촬영은학습·튜닝·후보선택에사용금지. 추가개발은공개정답의 손모양·동작경계/누락차이를보강해 동일신경망의 오관측·단계경계문제를줄인다. 손끝모으기/엄지/정확한접촉/반복과생략정오미완,초록은확인된부분특징만. 새 along_forearm 공통부품은양손대칭이나현재설명연결은팔쓸기범위,모든단어움직임검증완료주장금지. 이번93조건/모델학습재실행금지. 기존C·T3F/8812/사용자보호유지. RAG미사용,모델추가보다현재공개감독정답의부족/충돌부터보완.
'''
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8-sig');h='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';s=s.replace(h,h+'\n\n'+status,1);h='## 9. 다음 단일 작업';s=s.replace(h,h+'\n\n'+next_action,1);p.write_text(s,encoding='utf-8')
 p=O/'STATE.json';s=read(p);s['current_stage']='POSITIVE_SHAPE_SIGNED_DIRECTION_VERIFIED_DEMO8814_APPLIED';s['next_single_action']='8814 사용자 확인; 공개 손모양/동작경계·누락 감독정답 보강. 개인학습튜닝금지. 부분특징 안내만, 전체 정오 미검증.'
 s[U.name]=dict(status=report['status'],report=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFICATION.json'),training=False,private_used=False,web_changed=True,conditions=93,independent_generalization_claim=False);write(p,s)
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+status+'\n'+next_action+'\n8814 서버 재시작 후 PID는 본 작업 DEPLOYMENT_FINAL.json/SERVER.json과 실제 상태를 확인한다. 촬영 프로세스 중복실행 금지. 사용자 질문 반복 답변 금지.\n',encoding='utf-8')
 write(U/'DOCUMENT_SYNC.json',{str(p):sha(p) for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']})
 print(json.dumps({'report':str(U/'RESULT_KO.txt'),'refine_ms':ms,'summary':report['effect']},ensure_ascii=False))
