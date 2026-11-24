"""Record executable prototype milestone without completing the persistent goal."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,statistics
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 audit=read(U/'SAVED_VERIFICATION.json');assert audit['status'].startswith('PASS')
 now=datetime.now(timezone(timedelta(hours=9))).isoformat();phase=read(U/'STAGE_EVALUATION.json');ui=read(U/'DEMO_UI_VERIFICATION.json')
 table=[]
 for r in phase['candidates']:table.append(f"{r['condition']} / seed{r['seed']}: 구간 연결 {r['dtw']['answered']}/92, 맞춘 연결 {r['dtw']['exact_correct']}/92")
 summary=('공개4표현48영상(FIT ID01~09 36개/DEV ID11~13 12개)으로 새 전체시간 신경망과 실제 SignRep 마지막 투영층 rank8 미세조정 후보를 각3seed, 총6회 학습했다. '
  '개별 참고9회 비교 대신 학습된 단어 기준흐름과 DTW1회로 새 영상을 처리하는 격리 데모8805를 만들고 공개 업로드·재생·관절선·시간버튼을 검증했다. '
  '단계92시점의 맞춘 연결은 기존64에서 후보75~80으로 늘었으나 보류율이 달라 순수 정확도 향상으로 단정하지 않는다. '
  '같은194개 확인 항목의 균형정확도는 기존98.24%, 새 두조건 평균96.30%로 낮아졌다. 부분미세조정 추가 이득은 확인되지 않았다. '
  '병렬화는 동일 공개4영상의 특징·출력·DTW경로를 정확히 보존하며 약7.75~12.67초→5.54~7.14초, 실제브라우저 업로드 분석7.60초. '
  '전체 목표는 아직 active이며 학습된 소규모 연구 데모가 생긴 단계다.')
 next_action=('새 단위 word100_learned_feature_retention_20261102에서 기존 특징 성능 보존과 중요한 단계 누락 검출을 우선한다. '
  '먼저 기존 protected 판정부와 새 모델의 저장된 공개 예측을 같은 정답 중앙시점·사람별로 분해하여, 손모양 오류와 단계/문구 연결 오류를 분리한다. '
  '학습에 쓰지 않는 사용자의 촬영은 열지 않는다. FIT 공개자료에서 확인된 주요 단계와 손모양만 대조 대상으로 고정하고, 정상 속도/몸통 흔들림/짧은 준비동작 삭제를 주요동작 삭제와 구분한다. '
  '필요한 학습은 FIT에서만 기존 특징 지식을 보존하는 감독/증류와 단계 차이 감독을 비교한다. DEV 결과로 반복 문턱·seed를 고르지 않고 고정 조건의 진단으로 보고한다. '
  '완료한6학습·기존128대조·4영상 병렬검증을 다시 실행하지 않는다. 현재8805 후보·8804·C·T3-F 보존. '
  '짧은 중앙7프레임 삭제는 언어적 필수동작 정답이 아니므로 이를 놓쳤다는 이유만으로 모든 삭제를 오류로 만들지 않는다. '
  '현재 데모는 검지/네손가락·상하관계 일부와 물리적 단계만 지원하며 엄지·방향·접촉·반복횟수·전체정오 미완료다.')
 text='''학습한 지식으로 확인하는 피드백 신경망 — 첫 실행 데모

지금 무엇이 생겼나
예전8804는 여러 참고를 각각 비교했다. 새8805는 여러 사람의 공개 자료로 학습한 신경망에 새 영상과 목표 단어를 넣고, 모델이 배운 한 흐름과 DTW로 시간을 맞춘다.
표시되는 참고 한 편은 따라 하기용이다. 실제 평가에서 참고 9개를 다시 읽거나 비교하지 않는다.
촬영/파일 입력 → 전체화면 손·팔 관측과 RGB 특징 → 시간별 신경망 → 학습된 흐름 정렬 → 설명·시간 구간·특징/보류 표시.
기존 C·T3-F와8804는 그대로다. 새 화면은 연구 후보이며 운영 모델 교체가 아니다.

실제로 학습한 것
- 공개 순종·잘하다·조상·이집트4표현, FIT 기록ID01~09 36영상, DEV ID11~13 12영상.
- 144개의 관찰 중앙시점과7개 물리적 특징. 전체 프레임 모두에 사람 정답이 있는 것은 아니다.
- 13개 물리적 단계 설명과11개 출처 문구 연결. ID01의 관찰 구간을 다른 FIT영상으로 DTW 이전한 부분은 낮은 비중의 약한 정답이다.
- 신경망의 시간판정부·설명 특징 연결·단어별 기준 흐름을 지도학습했다. SignRep고정/마지막 fc.1 투영층 rank8 미세조정 조건 각각3번,120epoch 고정.
- 마지막 층의 원래 가중치가 실제로 변했다. 영상 본체 전체를 미세조정한 것은 아니다.
- 한국어 설명을28개 구조화 특징으로 정리해 연결했다. 자유로운 한국어와 수어 의미 전체를 이해하는 언어모델을 만든 것은 아니다.
- DTW는 처음부터 영상을 강제로 같게 바꾸는 정답 장치가 아니다. 원래 프레임과 시간을 남긴 상태에서 학습 대응과 실행 비교에 쓴다.

결과를 과장하지 않고 보면
'''+summary+'\n\n시간 연결 세부값\n기존:92시점 중67개 연결,64개 맞음. 나머지25개는 보류.\n'+'\n'.join(table)+'''
새 후보도 주요 구간 밖으로 연결된 시점은 보류한다. 위92시점은 동일Codex가 예측을 보지 않고 공개 이미지를 확인한 물리적 단계 표본이며, 독립 수어 전문가 정답이 아니다.
이집트의 짧은 중간단계는 확실한 추가 정답 시점이 없어 검증 완료로 세지 않았다.

중요한 버그와 보완
새 모델이 ‘손가락이 접혔다’ 쪽으로 분류해도 확신이 조금 낮으면 차이 경고가 되던 문제가 있었다.
확신 정도를 손가락의 실제 굽힘 정도처럼 해석한 오류다. 같은 손모양 판단이면 오류 경고 대신 확신 부족으로 구분했다.
개발 정상12영상×3seed에서 미세조정 후보의 경고1개가 보류로 바뀌었다. 기존v1 결과를 지우지 않았다.
이 수정 뒤 정상 차이 후보0개라는 사실은 손모양을 모두 정확히 안다는 뜻이 아니다. 학습 근거 부족 구간과 보류가 남는다.
관절+RGB를 동시에 지운 대조에서는 빠진 시간을 별도로 보존하여 정상으로 통과시키지 않았다.
속도변환·삭제·역순 시험은 추출된 특징에 가한 공학적 대조다. 실제 학습자 오류의 교육적 정답으로 부풀리지 않았다.
짧은7프레임 제거4개 중3개에서 검사한 특징이 여전히 비슷했다. 그7프레임이 실제 필수동작인지도 정답이 없으므로, 현재는 짧은 누락을 검증했다고 주장하지 않는다.

속도와 화면
모든 프레임·해상도·정밀도·특징을 그대로 두고 CPU관절 관측과 GPU영상 특징을 병렬 처리했다.
4개 공개영상에서 원래좌표,RGB,모델출력,DTW경로,판정이 정확히 같았다. 속도만 줄였다.
반복 측정의 통계적 평균이나 모든PC 보장치가 아니다. 첫 영상/라이브러리 초기화와 동시 부하 영향이 있다.
HTTP 업로드→새 추론→구간버튼→영상/관절선 재생,4공개사례,문자대비,좁은화면,JS오류 없음을 확인했다. 사용자 카메라는 실행하지 않았다.

남은 핵심 문제
1. 기존에 잘 구분하던 손모양 성능을 새 전체시간 모델에서 회복해야 한다. ‘시간 연결이 늘었으니 전체 향상’이라고 하지 않는다.
2. 순서/짧은 필수동작/반복의 중요성을 판단할 감독 신호가 부족하다. 모호한 정상 중간동작을 무조건 오답으로 만들지 않는다.
3. 엄지·손바닥 방향·피부접촉·정확한 반복횟수와 교육적 허용 기준은 아직 학습·검증하지 못했다.
4. RGB 프레임이 존재한다는 것만으로 해당 손가락이 잘 보인다는 보장은 없다. 모델의 추정과 현재 관측을 구분해야 한다.
5. 실제 새 사용자에 대한 일반화는 아직 검증되지 않았다. 개발 자료는 이전 연구에서도 사용됐으므로 새 독립 최종시험이 아니다.

다음 실행
'''+next_action+'''

보존과 검증
사용자·멘토 촬영 학습/튜닝/선택0, 새 외부 다운로드0. 공개 브라우저 시험도 기존 ID12 공개영상만 사용했다.
실행/구현:Codex. 별도 저장 산술검산:같은Codex의verify_saved.py. 독립 인간/수어 전문가 검증과 다르다.
첫학습전 오류·기존결과·6회학습·검증파일은 보존했다. 추가 수정은 새 결과로 기록한다.
주요근거: INPUT_MANIFEST.json, FROZEN_TRAINING.json, TRAIN_READY.json, TRAINING_REPORT.json, STAGE_EVALUATION.json,
FEEDBACK_EVALUATION.json, FEEDBACK_V2_AUDIT.json, PARALLEL_PUBLIC_VERIFICATION.json, DEMO_UI_VERIFICATION.json, SAVED_VERIFICATION.json.
'''
 (U/'RESULT_KO.txt').write_text(text,encoding='utf8')
 status='LEARNED_NEURAL_DEMO_BUILT_PERSISTENT_GOAL_ACTIVE'
 record=dict(status=status,at=now,summary=summary,next_action=next_action,unit=str(U),goal_status='active',
  training_complete=True,training_runs=6,active_training_sessions=[],demo_url='http://127.0.0.1:8805/',demo_pid=read(U/'DEMO_LAUNCH.json')['pid'],
  implementation='Codex',verification='Separate saved-arithmetic script by same Codex; not independent human truth',
  report=str(U/'RESULT_KO.txt'),verification_report=str(U/'SAVED_VERIFICATION.json'),user_mentor_training=False,
  full_project_or_goal_complete=False,production_model_replaced=False)
 write(U/'CURRENT_EXECUTION.json',record)
 write(U/'MILESTONE.json',record)
 p=O/'STATE.json';d=read(p);d['current_stage']=status;d['next_single_action']=next_action;d['updated_date_kst']=now[:10];d[U.name]=record;write(p,d)
 p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig')
 for h,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','학습된 신경망 피드백 데모8805·총6회 비교 완료, 계속목표 유지',summary),('## 9. 다음 단일 작업','손모양 성능 보존·중요 단계 차이 감독을 다음 검증 단위로 진행',next_action)]:
  assert h in t;t=t.replace(h,h+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
 p.write_text(t,encoding='utf8')
 (R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음 단일 작업: '+next_action+
  '\n\n현재 단위: '+str(U)+'\n현재 진행 학습/평가 프로세스 없음. 완료6학습·전체평가 재실행 금지. 8805서버 PID '+str(record['demo_pid'])+'만 로컬 시험을 위해 유지. '+
  'CURRENT_EXECUTION/RESULT_KO/SAVED_VERIFICATION을 읽고 이어간다. 새 목표는 앱active다. 전체성과미달을 완료로 바꾸지 않는다.\n',encoding='utf8')
 p=R/'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt';t=p.read_text(encoding='utf8');old='개정: 2026-11-02 / 상태: 목표·설계 확정, 새 평가 모델 구현·학습 전'
 t=t.replace(old,'개정: 2026-11-02 / 상태: 아래 설계 이후 첫 신경망 데모 구현·6회 학습 완료, 전체 목표 진행 중',1)
 marker='\n\n1. 사용자가 확정한 목표';assert marker in t
 t=t.replace(marker,'\n\n최신 실행 현황 — 아래 과거 설계의 “학습 전/미수행/SignRep 동결”은 작성 당시 상태이며 현재는 다음이 우선한다.\n'+summary+'\n'+next_action+marker,1);p.write_text(t,encoding='utf8')
 with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:f.write('\n\n'+now+' — 개별참고 비교에서 학습된 신경망·기준흐름 데모로 첫 전환\n'+summary+'\n'+next_action+'\n')
 code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in U.iterdir() if p.is_file() and p.suffix in ('.py','.js','.html')}
 write(U/'DEMO_CODE_MANIFEST.json',dict(files=code,at=now,prior_model_and_demo_edits=0))
 print(status)
if __name__=='__main__':main()
