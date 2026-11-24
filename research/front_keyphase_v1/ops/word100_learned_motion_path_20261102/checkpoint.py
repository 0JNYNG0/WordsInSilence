from pathlib import Path
import json
from path_model import Q,N,O,read
from prepare_real_fit import clock,write

def main():
 R=O.parents[2];now=clock()
 summary=('실제 FIT9기록ID·4표현의108속도입력 검증과6회 미세조정을 완료했다. 제한5방향 균형정확도89.07→93.33%, 정상변형 확인47→48/60, 주요팔이동삭제 충분확인2→0/3. 그러나 원래DEV확인40→36/45로 악화돼 웹미적용. '
  '정확한구간을 주는 진단에서도 정상변형52/60·원래36/45여서 경계만의 문제가 아님. 손모양/관계/DTW288비교 보존. '
  '다음 word100_learned_motion_path_20261102에서 자기이전위치 대비 이동입력 추가와 방향/연속성 별도신경망을 비교하는6학습을 시작했다. '
  '동결조건·초기출력일치·패딩/위치이동불변·분리기울기 검증통과. 분리기울기 최초검사는0을None으로잘못요구해실패했고 최초기록보존후검사정정;모델결과/문턱변경아님. 새결과는아직미확인.')
 next_step=('exec84510와 TRAIN_PROGRESS/TRAINING_RESULT를 먼저 확인해 중복학습하지 않는다. 완료후 고정288개비교와 저장산술/보존검산. 기존N/이전결과 재추론금지. '
  '주후보 separated_relative/seed42는 결과전에고정, 새DEV로문턱/seed선택금지. 정상/주요이동근거/관측부족을함께통과할때만 ID14/15의8자료 잠금평가와 별도연구데모로진행. '
  '이동검출≠이동과정충분≠언어정답을분리. C·T3F/8804~8806/사용자·멘토자료보존,개인자료학습·튜닝·선택금지,외부다운로드없음. 전체목표active.')
 s=dict(status='RELATIVE_MOTION_AND_CONTINUITY_FINETUNING',at=now,summary=summary,next=next_step,active_exec_session=84510,
  prior_unit=str(N),report=str(N/'RESULT_KO.txt'),goal_status='active',web_default_changed=False)
 write(Q/'CURRENT_EXECUTION.json',s)
 ns=read(N/'CURRENT_EXECUTION.json');ns.update(status='COMPLETE_LIMITED_GAIN_NOT_READY_FOR_WEB',active_exec_session=None,next=str(Q),report=str(N/'RESULT_KO.txt'));write(N/'CURRENT_EXECUTION.json',ns)
 state=read(O/'STATE.json');state.update(current_stage=s['status'],next_single_action=next_step,next_single_task=next_step,updated_date_kst='2026-11-02');state[Q.name]=s;state[N.name]=ns;write(O/'STATE.json',state)
 p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf8')
 for h,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','상대이동·연속성 분리학습 진행',summary),('## 9. 다음 단일 작업','상대이동 신경망6학습 확인·공동 검증',next_step)]:
  text=text.replace(h,h+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
 p.write_text(text,encoding='utf8')
 (R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_step+'\n\n현재단위: '+str(Q)+'\n구현·별도검산 Codex, 독립사람/KSL전문가정답 아님.\n',encoding='utf8')
 p=R/'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt';text=p.read_text(encoding='utf8');a=text.index('최신 실행 현황');b=text.index('1. 사용자가 확정한 목표');p.write_text(text[:a]+'최신 실행 현황\n'+summary+'\n'+next_step+'\n\n'+text[b:],encoding='utf8')
 with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:f.write('\n\n'+now+' — 실제 속도108입력·6학습 검증 및 자기이동/연속성분리 후속\n'+summary+'\n'+next_step+'\n')
 assert read(O/'STATE.json')[Q.name]['active_exec_session']==84510
 print('Current stage, next task, handoff and goal/history synchronized')

if __name__=='__main__':main()
