from pathlib import Path
import numpy as np
from path_model import Q,N,M,O,read
from prepare_real_fit import write,exclusive,sha,clock

def main():
 R=O.parents[2];next_unit=O/'word100_learned_rgb_motion_20261102';now=clock();v=read(Q/'SAVED_VERIFICATION.json');assert v['status'].startswith('PASS')
 assert read(next_unit/'INPUT_READY.json')['training_done'] is False
 summary=('공개 FIT36원본의108실제속도입력을 검증하고 두단위에서 총12회 움직임 신경망 미세조정을 완료했다. '
  '자기이전위치 대비 이동을 추가하고 방향/연속성 인코더를 분리한 후보는 정상속도·대기 확인47/60→56/60, 주요팔이동 삭제 충분확인2/3→0/3. '
  '제한5방향 균형정확도89.07→93.33%, 기존손모양/관계/DTW 각단위288비교 보존,검은구간15/15보류. '
  '원래DEV확인은40/45→39/45여서 사전전체채택기준미달·8806미교체. 주후보seed42고정이며 15/15·13/15·11/15의seed간차이도숨기지않음. '
  '현재 움직임신경망은RGB를 손모양/관계확률로간접활용하고768차원영상특징을직접읽지않는다는구조를확인했다. '
  '다음 RGB 직접연결 비교계획과 FIT기존144특징파일·정규화만준비했으며 새RGB연결모델학습은아직미실행. 전체목표active.')
 next_step=('word100_learned_rgb_motion_20261102의 PLAN/INPUT_READY를 따른다. 새다운로드/추출없이 기존SignRep768영상특징을 움직임판정부에 직접 연결하는 zero-init잔차와 RGB없이같은추가학습하는 비교조건을 구현한다. '
  '이전Q separated_relative 각seed를초기값으로2조건×3seed,120epoch 최종checkpoint고정. FIT81구간/1044증강창/144저장입력만사용. '
  '초기출력일치·관측마스크·시간매핑·기울기검증후학습,고정288비교와별도산술검산. 원래개발자료도정상으로확인하고중요누락/관측부족도유지해야잠금ID14/15의8영상평가와별도데모로진행. '
  '이미완료한108추출/12학습/576비교 재실행금지. 기존DEV는이미노출된개발자료이며문턱/seed/지원단어예외선택금지. '
  '잠금ID14/15는과거TCN평가이력있지만새후보학습/선택/추론0,18물리적구간은예측전에시각검토. 사용자·멘토자료접근/학습/튜닝/선택금지. C·T3F/8804~8806보존. 전체목표미완료이며다음승인재요청없음.')
 report=f'''이번 작업 — 정상적인 속도 차이와 중요한 이동 누락을 함께 구분하기

1. 무엇을 바꿨나
같은 동작을 느리게·빠르게 한 실제 공개영상108입력을 준비하고 원본/시간/관측마스크를 확인했다. 9개 기록ID·4표현·36원본이다.
기존 학습된 움직임 판정부를 이어받아, 자기 이전 위치와 비교한 손·팔 이동 정보를 추가했다. 방향을 배우는 부분과 과정의 연속성을 배우는 부분을 따로 둔 조건도 비교했다. 두 단위에서 각6회, 총12회 실제 가중치 미세조정이다.
기존 SignRep의 영상특징과 손모양/관계 판정부, 학습된 기준흐름DTW는 그대로다. 이번에 SignRep 본체 전체를 추가 학습한 것은 아니다.

2. 이전 → 현재
정상속도·손내림/대기 차이에서 중요한 이동 확인: 47/60 → 56/60
핵심 팔이동을 삭제해도 충분하다고 남은 판정: 2/3 → 0/3
검은구간의 관측부족 보류: 15/15 → 15/15
원래 개발영상의 이동 확인: 40/45 → 39/45
실제역순과 원래 구간의5방향 균형정확도: 89.07% → 93.33%

60은20물리적이동을3seed로 확인한 수,45는15이동을3seed로 확인한 수다. 독립영상60개/45개가 아니며 전체TCN인식률이나 전체피드백정확도가 아니다. 56/60은20개단위 중 항상통과가56개라는뜻도 아니다. 확정오류경고 대신 확인보류된 사례를 세었다.
후속조건 내부에서는 함께배우는조건46/60, 분리해서배우는조건56/60이었다. 단순 이동입력추가만으로 해결됐다고 주장하지 않는다.

3. 남은 문제
원래 개발영상 기준은 미리40/45이상을 정했는데39/45다. 사후에기준을낮추거나 가장좋은seed를골라통과시키지않았다. 고정seed42는15/15,43은13/15,44는11/15로 안정성 문제가 남는다.
정상빠른영상의4판정도 아직연속성보류다. 정확한시간을알려주는 앞선진단에서도 문제가남았으므로 DTW만의잘못이라고할수없다.
팔이동삭제영상에는 도착자세와 작은잔여변화가 남는다. 이동이 조금보였다는것과 주요이동과정이충분히확인됐다는것은다르다. 편집삭제는 실제학습자의언어오답정답이아니다.
현재 직접학습한 범위는5가지물리적방향이다. 손목회전·엄지모양·정확한접촉·필수반복횟수·모든단어의의미정오까지 완성한것은아니다.

4. 실제 적용과 시간
기존손모양·관계·DTW는두단위각288개비교에서보존했다. 이번움직임판정부 자체시간은 중앙값{v['motion_head_ms_median']:.2f}ms다. 영상관측부터피드백까지전체시간이3ms라는뜻이아니며 전체시간을새로측정한시험도아니다.
웹8806/8804/8805와 C·T3-F는그대로보존했다. 이번후속후보는 아직웹기본에적용하지않았다.

5. 다음 작업
{next_step}

공개자료만사용. 사용자·멘토촬영 접근/학습/튜닝/후보선택0. 외부다운로드/전송0. 같은Codex가별도프로그램으로산술·해시·출력보존을검산했고 독립사람/한국수어전문가정답검토는아니다.
주요근거: CONTRACT/PREFLIGHT/TRAINING_RESULT/COMPARISON/SAVED_VERIFICATION.json. 최초검사실패는PREFLIGHT_FIRST_FAILURE.json에보존했다. gradient가없는경우None뿐아니라0으로반환되는것을검사에서누락했던문제를고쳤고실제학습전재검증했다.
전체목표는진행중이다. 이번부분개선을전체프로젝트완료로처리하지않는다.
'''
 (Q/'RESULT_KO.txt').write_text(report,encoding='utf8')
 status=dict(status='NORMAL_SPEED_GAIN_VERIFIED_ORIGINAL_REGRESSION_REMAINS',at=now,summary=summary,next=next_step,active_exec_sessions=[],active_training=None,
  goal_status='active',report=str(Q/'RESULT_KO.txt'),verification=str(Q/'SAVED_VERIFICATION.json'),web_default_changed=False)
 write(Q/'CURRENT_EXECUTION.json',status)
 current=dict(status='PUBLIC_RGB_MOTION_INPUT_READY_NOT_TRAINED',at=now,summary=summary,next=next_step,active_exec_sessions=[],goal_status='active',prior_unit=str(Q),
  input_ready=str(next_unit/'INPUT_READY.json'),plan=str(next_unit/'PLAN.json'),new_model_trained=False)
 write(next_unit/'CURRENT_EXECUTION.json',current)
 state=read(O/'STATE.json');state.update(current_stage=current['status'],next_single_action=next_step,next_single_task=next_step,updated_date_kst='2026-11-02');state[Q.name]=status;state[next_unit.name]=current;write(O/'STATE.json',state)
 p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf8')
 for heading,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','정상속도56/60 개선·원래DEV39/45 회귀보류',summary),('## 9. 다음 단일 작업','기존 RGB 영상특징의 움직임 직접연결 비교',next_step)]:
  assert heading in text;text=text.replace(heading,heading+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
 p.write_text(text,encoding='utf8')
 (R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_step+'\n\n현재단위: '+str(next_unit)+'\n진행중학습/추출/평가없음. exec14497/22013/84510/90490 종료0확인. 기존8806는마지막기록만보존하며사용시실제상태확인.\n',encoding='utf8')
 p=R/'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt';text=p.read_text(encoding='utf8');a=text.index('최신 실행 현황');b=text.index('1. 사용자가 확정한 목표');p.write_text(text[:a]+'최신 실행 현황\n'+summary+'\n'+next_step+'\n\n'+text[b:],encoding='utf8')
 with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:f.write('\n\n'+now+' — 自身이동과방향/연속성분리로정상속도개선,남은영상직접근거연결\n'+summary+'\n'+next_step+'\n')
 exclusive(Q/'FINAL_ARTIFACTS.json',dict(at=now,status=status['status'],files={p.name:sha(p) for p in Q.iterdir() if p.is_file() and p.name!='FINAL_ARTIFACTS.json'},goal_status='active'))
 assert read(O/'STATE.json')[next_unit.name]['status']==current['status'];assert str(next_unit) in (R/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
 print('Verified reports, status, goals/history and next RGB task synchronized; goal active')

if __name__=='__main__':main()
