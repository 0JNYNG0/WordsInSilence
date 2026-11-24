from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,numpy as np
from motion_model import M,P,U,V,read
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
 O=M.parent;R=O.parents[2];now=datetime.now(timezone(timedelta(hours=9))).isoformat();v=read(M/'SAVED_VERIFICATION.json');c=read(M/'COMPARISON.json');training=read(M/'TRAINING_RESULT.json');assert v['status'].startswith('PASS')
 scores={cond:float(np.mean([r['five_motion_balanced_accuracy'] for r in v['actual_reverse_plus_native_scores'] if r['condition']==cond])) for cond in ['orderless_pool','ordered_gru']}
 latency=float(np.median([r['seconds'] for r in c['rows'] if r['condition']=='ordered_gru']));extraction=[r['seconds'] for r in read(M/'REVERSE_EXTRACTION.json')['rows']]
 summary=(f'公開48영상의108물리적이동/유지구간을 RGB로 검토하고 FIT81구간(ID01~09)/DEV27구간(ID11~13)으로 분리했다. '
  f'순서를버린비교신경망/순서를보는GRU 각3회 총6회 지도학습. 원래구간과 실제역순12영상 재추출의5방향 균형정확도 평균 {scores["orderless_pool"]*100:.2f}%→{scores["ordered_gru"]*100:.2f}%. '
  '이는 동일입력 비교용 새신경망 간 물리적방향 점수이며 C·T3F/기존웹/전체피드백 정확도 개선 수치가 아니다. 기존학습DTW 연결의 정상15이동×3seed에서 확인4→40/45. '
  '정상속도/대기16영상20이동×3seed는47/60만확인,13보류. 잘하다 실제이동삭제를2/3seed가 이동관측으로남겨 전체피드백 적용기준 미달. '
  '정확한시간을별도로제공한진단에서 경계이동으로일부개선되나 속도에따른판단흔들림도남아 두원인이겹침을확인. 관측부족15/15보류, 기존손모양/관계/DTW출력288비교보존. '
  '이번이동신경망은오프라인연구결과이며8806기본판정부에적용하지않음. C·T3F/기존데모/개인자료보존, 전체목표active.')
 next_step=('다음 단위 word100_learned_motion_invariance_20261102에서 실제 FIT 공개영상을 현재 실행경로로 관측해 정상속도/인코딩 차이와 단계경계 편차를 함께 학습한다. '
  '먼저 FIT ID01~09만의4표현 정상/0.8/1.25배속 입력을 준비하고 추출성공·원본시간대응·좌우/손모양보존을확인한다. 단순특징배열속도증강과 실제영상재추출의차이를측정한다. '
  '현재5방향관측의이득은보존하되 연속성0=언어오답으로쓰지않고, 시작부위→이동→도착부위의관측근거와 시간구간연결을공동보완한다. 단어예외코드/정답인것처럼문턱완화금지. '
  '새조건은FIT근거로고정한뒤학습하고 기존DEV는개발회귀로만보고; 별도공개ID사용이력/중복을확인해잠금평가를추가한다. 기존6학습/12역순영상추출/288비교 재실행금지. '
  '정상속도보류와움직임삭제오통과를동시에줄이기전8806기본교체안함. 개인·멘토자료학습/튜닝/선택금지. 전체목표아직미완료이며승인재요청없이계속.')
 report=f'''이번 목표와 실제 작업

목표는 그대로다. 여러 사람의 공개 수어와 동작 설명을 배운 신경망이 새 사용자의 행동을 분석하고, 설명 문구와 사용자 시간대에 연결해 피드백하는 것이다.
기존 목표를 삭제한 뒤 등록한 새 계속 목표는 active다. 이번 작은 시험만으로 전체 목표를 완료 처리하지 않는다.

무엇을 가르쳤나
- 손이 왼팔 부근에 있다는 것만으로 팔을 따라 움직였다고 할 수 없어, 실제 이동 방향을 별도로 지도학습했다.
- 공개4표현48영상에서 직접 확인한108구간: 학습9기록ID81구간, 개발3기록ID27구간.
- 팔을 따라 손목 쪽으로 이동 / 양손 벌어짐 / 오른손의 상대적 상승·하강 / 얼굴에서 몸통 쪽 이동.
- 작은 몸 흔들림이 남은 유지구간을 음성 예로 포함했다. 정확한 접촉이나 필요한 반복 횟수의 언어적 정답은 아니다.
- 출처 영상과 원래 프레임·시간을 기록했고, 확인하지 않은 특징의 학습 손실은 제외했다.
- 순서를 버리는 비교 신경망과 순서를 기억하는 GRU 신경망을 각각3회, 총6회 학습했다. 모델에 목표단어·사람ID·정답시각을 넣지 않는다.
- 기존 SignRep/손모양/관계/학습된DTW는 보존했다. 이번 학습은 새로운 작은 움직임 판정부이며, SignRep 전체 본체를 추가 미세조정한 것이 아니다.

실제로 개선된 부분
- 원래 공개 구간과 실제 영상12개를 역순 편집 후 새로 관측한 구간을 함께 비교했다.
- 5가지 방향의 균형정확도: 순서없는 비교조건 평균 {scores['orderless_pool']*100:.2f}% → 순서있는 조건 평균 {scores['ordered_gru']*100:.2f}%, +{(scores['ordered_gru']-scores['orderless_pool'])*100:.2f}%p.
- 이 수치는 새 비교용 두 신경망 사이의 제한된 물리적 동작 시험이다. 기존 TCN/웹의 인식률 또는 사용자 피드백 정확도가 이만큼 오른 것이 아니다.
- 기존 학습DTW와 연결한 정상12영상의15이동 판정×3회에서 방향을 확인한 수는4/45→40/45였다. 45개의 독립 영상이 아니다.
- 검은화면으로 관측을 없앤4영상의5이동×3회=15판정은 모두 관측부족으로 보류했다.
- 기존 손모양·신체관계 출력과 DTW 경로는288회 비교에서 그대로 유지됐다. 이는48개공개/변형 영상×2조건×3회다.
- 추가 움직임 판정부는 영상당 중앙값 {latency*1000:.2f}ms. 이번 역순영상의 기존 관측/피드백 전체 실행은 {min(extraction):.2f}~{max(extraction):.2f}초이며 모델 로딩 대기는 별도다.

아직 해결하지 못한 부분
- 정상/속도차이/대기제거16영상의20이동×3회=60판정 중47개만 충분히 확인하고13개를 보류했다. 틀렸다고 경고한 것은 아니지만 정상 동작을 충분히 평가하지 못한다.
- 잘하다에서 팔을 따라 가는 중간을 삭제한 실제 영상은3회 중2회가 여전히 이동을 확인했다고 추정했다. 새 연속성 학습만으로 누락 방지가 해결되지 않았다.
- 이집트 삭제영상의 뒤쪽에는 양손이 더 벌어지는 짧은 움직임이 실제로 남는다. 그 이동을 찾았다는 것과 앞의 양손모음 단계까지 수행했다는 것은 다르다. 기존 관계 판정부의 누락 결과를 함께 보존해야 한다.
- 실제 역순 영상에도 일부 같은 방향의 준비/마무리·반복이 존재한다. 역순 영상 전체를 무조건 잘못된 수어로 정의하지 않았다. 대응 시점의 방향 시험과 전체 흐름 판단을 구분했다.
- 올바른 비교 시점을 알려주는 진단도 별도로 했다. 경계를 고치면 일부 판정은 나아졌지만, 같은 정상 속도변형의 불안정성이 남았다. 시간정렬만 고치거나 좌표만 고치는 한쪽 대응으로 끝낼 수 없다.

적용 상태
- 새 신경망은 실제 학습·평가·코드 연결까지 됐지만, 위 회귀 때문에8806의 기본 결과는 바꾸지 않았다.
- 기존8804·8805·8806과 C·T3-F 보존. 사용자·멘토 촬영 접근/학습/튜닝/선택0. 새 외부 모델·데이터 다운로드0.
- 주석과 별도 산술검산은 같은 Codex가 했다. 독립 사람/수어 전문가의 정답 검토가 아니다. DEV ID는 이전에도 사용된 개발 자료다.

다음에 실제로 할 일
{next_step}

근거: MOTION_TARGETS.json, TRAINING_CONTRACT.json, TRAINING_RESULT.json, REVERSE_MANIFEST.json,
COMPARISON.json, SAVED_VERIFICATION.json, EXACT_TIME_DIAGNOSIS.json. 새 HTML은 만들지 않았다.
'''
 (M/'RESULT_KO.txt').write_text(report,encoding='utf8')
 state=dict(status='MOTION_DIRECTION_GAIN_VERIFIED_CONTINUITY_AND_SPEED_UNRESOLVED',at=now,summary=summary,next=next_step,goal_status='active',
  last_goal_turn='progress: new108episode supervision, six NN trainings,12 real reversed-video extractions,288 preserved-path comparisons, exact-time diagnosis',
  active_training=None,active_exec_sessions=[],web_default_changed=False,preserved_demo='http://127.0.0.1:8806/',report=str(M/'RESULT_KO.txt'),verification=str(M/'SAVED_VERIFICATION.json'))
 write(M/'CURRENT_EXECUTION.json',state)
 s=read(O/'STATE.json');s['current_stage']=state['status'];s['next_single_action']=s['next_single_task']=next_step;s['updated_date_kst']='2026-11-02';s[M.name]=state;write(O/'STATE.json',s)
 p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf8')
 for heading,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','동작방향 신경망6학습·실제역순12영상 검증',summary),('## 9. 다음 단일 작업','실제속도 입력변이·구간경계·이동근거 공동보완',next_step)]:
  assert heading in text;text=text.replace(heading,heading+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
 p.write_text(text,encoding='utf8')
 (R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_step+'\n\n현재 단위: '+str(M)+'\n실행 중 학습/추출 없음. 기존8806보존, 마지막기록PID38820은실제확인후사용. 구현·별도검산Codex, 사람독립정답아님. 전체목표active.\n',encoding='utf8')
 p=R/'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt';text=p.read_text(encoding='utf8');a=text.index('최신 실행 현황');b=text.index('1. 사용자가 확정한 목표');text=text[:a]+'최신 실행 현황 — 아래 과거 설계보다 우선한다.\n'+summary+'\n'+next_step+'\n\n'+text[b:];p.write_text(text,encoding='utf8')
 with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:f.write('\n\n'+now+' — 정지관계에서 이동·순서 신경망으로 확장\n'+summary+'\n목표를 바꾼 것이 아니라 동작순서 관측을 실제 지도학습했다. 연속성 학습 실패도 보존하며 개별 후보 실패를 원점복귀로 기록하지 않는다.\n'+next_step+'\n')
 write(M/'FINAL_ARTIFACTS.json',dict(at=now,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in M.iterdir() if p.is_file() and p.name!='FINAL_ARTIFACTS.json'}))
 assert read(O/'STATE.json')['next_single_action']==next_step;assert read(O/'STATE.json')[M.name]['status']==state['status']
 assert str(M) in (R/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
 print(json.dumps(dict(status=state['status'],scores=scores,motion_head_ms=latency*1000,report=str(M/'RESULT_KO.txt'),goal='active'),ensure_ascii=False))
if __name__=='__main__':main()
