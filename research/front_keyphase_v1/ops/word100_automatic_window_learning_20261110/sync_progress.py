from window_core import *
from datetime import datetime
status='자동 구간의 준비·마무리 혼합 보완 진행. 공개899FIT100단어 기반 유지, 136움직임·75순서쌍 감독으로 균등비교/핵심구간 학습2조건×3seed. 기존 모양모델 고정. 1698사전검사통과. 평가 전이며 성능개선미확인.'
next_action='CURRENT_EXECUTION/EVENTS와 실제 프로세스 확인 후 이어받기. 완료학습 중복금지. 자동창 학습6후보의 방향/순서와414전체경로를 동결조건으로 비교·별도산술검산. 준비/복귀 제외가 실제왕복동작삭제로 이어지지 않는지 확인. 기존C·T3F/웹8814보존,사용자자료접근0,새평가튜닝금지.'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text('utf-8-sig');tag='### 2026-11-10 자동 비교 구간의 방향·순서 학습 진행'
if tag not in s:
 s=s.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+tag+'\n'+status+'\n숫자 설명 정정: 이전227/228의228은 순종44·잘하다92·조상92·이집트0. 4표현 영상묶음이지만 이집트 손끝닫힘 검증수치가 아님.32움직임은순종9·잘하다6·조상11·이집트6.\n근거: '+str(U)+'\n',1)
 s=s.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n'+tag+'\n'+next_action+'\n',1);p.write_text(s,'utf-8')
p=O/'STATE.json';s=read(p);s.update(current_stage=U.name,next_single_action=next_action,last_updated=datetime.now().astimezone().isoformat());s[U.name]=dict(status='TRAINING',summary=status,next=next_action,execution=str(U/'CURRENT_EXECUTION.json'),precheck=str(U/'PRETRAIN_CHECK.json'),default_web_changed=False,private_access=False);write(p,s)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+status+'\n\n'+next_action+'\n\n이번 작업: '+str(U)+'\nCURRENT_EXECUTION/EVENTS를 먼저 확인. 새 준비 완료, 학습시작.\n이전 word100_broad_factorized_learning_20261110 완료6학습/414비교는 재실행하지 않음.\n','utf-8')
