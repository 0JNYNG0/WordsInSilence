from pathlib import Path
import json,hashlib
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;R=U.parents[3];S=O/'word100_rtmw_shared_input_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 v=read(U/'NATIVE_PARALLEL_VERIFICATION.json');sv=read(S/'VERIFICATION.json');assert v['status']==sv['status']=='PASS'
 freeze=read(U/'FROZEN.json')
 for p,h in freeze['protected'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 at=datetime.now(timezone(timedelta(hours=9))).isoformat();nchecks=len(v['checks']);cold=read(U/'NATIVE_PARALLEL_COMPLETE.json')
 table='\n'.join(f"{x['label']}: {x['frames']}프레임, 마지막 예정 입력→녹화 저장·RTMW 재사용·기존 TCN 완료 {x['last_capture_to_all_ms']/1000:.3f}초" for x in v['runs'])
 text=f'''2026-11-08 로컬 촬영·관절 재사용 검증 결과

판정: 같은 로컬 프로그램 안에서 촬영 입력·녹화·관절 추출을 겹쳐 처리하는 공개 재생 경로는 검증 통과. 실제 카메라/기존8812 통합/RTMW용 새 모델 학습은 아직이다.

무엇을 바꿨나
- 브라우저에서 큰 원시 화면을 계속 전달하던 첫 시도에서 지연·누락을 발견했다. 결과를 적용하지 않고 모든 최초 실패와 시험 기록을 보존했다.
- 브라우저 경로에서는 WebCodecs VP9로 녹화한 같은 패킷을 즉시 분석하고 녹화 파일에 저장했다. 마지막 browser_a423697e는 받은358프레임+취소18프레임의 입력·녹화가 일치했고 PTS 최대0.495ms, 녹화와 분석 픽셀 동일. 원본360프레임 중 브라우저가 받지 못한2프레임이 있어 원본 전체보존이라고 말할 수 없다. 종료후1.10~2.08초로 사전1초목표는 미통과. 기존 화면 미적용.
- 사용자가 Unity와 동일하게 로컬 촬영 프로그램 안에서 RTMW를 직접 실행하는 구조를 지적했고 그 구조를 추가 구현·비교했다. 원본 공개 영상을 실제30fps 시각으로 넣으며 녹화와 관절 추출을 같이 수행한다. 브라우저로 이미지를 넘기는 과정이 없다.
- 기존 C·T3-F 보존을 위해 함께 실행하는 MediaPipe의 독립 몸/손 계산을 병렬로 배치했다. 모델, 좌우할당, ROI, 신뢰도 기준, 입력 수치는 바꾸지 않았다. TCN 가중치는 촬영 전 준비한다. 공개 파일 디코더는 최대2프레임만 미리 준비해 촬영 입력 루프를 막지 않게 했다.

최종 실제 측정: 공개3영상(기록ID02), 총360프레임, 각1회. 새 사람 일반화 시험 아님.
{table}

검산: {nchecks}항목 통과. 원본360프레임 모두 처리/녹화, 누락0, 원본PTS와 관절PTS 및 녹화PTS 정확히 동일. RTMW 좌표·점수는 같은 픽셀의 이전 동결 출력과 동일. 병렬화 전후 MediaPipe 7개배열, C/T3-F 순위·점수도 동일. 재검출0. 촬영 예정시각 지연 최대7.63ms. 녹화 압축에 따른 썸네일 픽셀 평균절대차는 프레임별최대1.23/255 미만이며 관절 정확도 지표는 아니다.

이전 대비: 몸/손 순차 실행에서는 초기 로딩이 끝난 두 영상의 전체 잔여 대기가1.44~1.67초였다. 병렬/사전준비/제한된소스버퍼 결합 후0.35~0.53초. 여러 실행순서 보완이 함께 바뀐 효과이므로 전부 병렬화 하나의 효과라고 하지 않는다. 첫 순종의 이전5.74초에는 TCN 첫 로딩4.22초가 포함되어 공정한 속도배수 계산에서 제외한다.
초기 준비는 별도: RTMW컴파일{cold['engine_compile_seconds']:.2f}초+예열{cold['engine_warm_seconds']:.2f}초, C/T3-F준비{cold['tcns_prepare_seconds']:.2f}초. 매 촬영이 아니라 프로세스 시작 전 준비. 촬영 중 RTMW 좌표 완성 지연p95는0.33~0.49초이므로 무지연이라고 말하지 않는다.

의미와 남은 문제
- 0.35~0.53초는 녹화 저장·관절 묶음·기존 TCN까지다. 새 피드백 신경망/DTW/화면 렌더링/실제 카메라·Unity 전체 시간이 아니다.
- 인식률 향상 시험이 아니다. 순종 영상은 전후 모두 T3-F가 공항시장으로 분류했다. 조상·수어는 각각 해당 단어였다. 이3건을 새로운 일반화 정확도로 보고하지 않으며 특정 단어 예외를 만들지 않았다.
- RTMW 엄지/교차 오류와 가려진 점의 신뢰도 문제는 입력 재사용으로 해결되지 않는다. raw score는 가시성 확률이 아니다.
- 현재 C·T3-F는 MP입력이 필요해 병행했다. 사용자 승인 최종 방향은 RTMW한번→RTMW용TCN후보+관절전용피드백후보 공유다. 이 새 모델들은 아직 학습0. 기존 모델을 RTMW입력에 그대로 연결하지 않는다.
- 공개 재생 검증이지 실제웹캠 시험이 아니다. 기존8812/기존C·T3-F/운영DTW 변경0. 사용자·멘토자료, 카메라, 새다운로드, 외부전송, Git0. 임시 시험 서버 종료.

실행/검증 담당: 같은 Codex의 구현 및 별도 저장자료 검산. 독립 사람/전문가 검토가 아니다.
근거: NATIVE_PARALLEL_COMPLETE.json, NATIVE_PARALLEL_VERIFICATION.json, native_runs/, 각browser_*/FAILURE·COMPLETE·VERIFICATION. 최초 시도와 중간 실패 삭제/소급수정 없음.
다음 입력 준비: ../word100_rtmw_shared_input_20261108/PREPARED.json 및 VERIFICATION.json.
'''
 (U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
 (S/'RESULT_KO.txt').write_text('RTMW 한 번의 관절 추출 결과를 TCN과 세부 피드백 후보가 공유하도록 입력 어댑터를 준비했다.\n\n공개3영상360프레임: 원시133점·raw 점수·시각·프레임번호 보존. 기존 TCN과 같은280차원 입력 형태와 몸·손 기준 상대 좌표를 생성. 25검사 통과, 별도100800개 수치 검산 최대차이0. 관절 재추론0, 학습0.\n\n중요: 차원이 같다고 기존 C·T3-F와 호환되는 학습분포가 되는 것은 아니다. 좌표 계산 가능 마스크는 실제 가시성 정답이 아니다. 얼굴좌표는 손과 얼굴의 부위 관계를 위해 보존하고 표정은 다루지 않는다. 현재 전체시각을 유지하며 RTMW 점 존재만으로 시작/종료나 학습범위를 자르지 않는다.\n\n자료 목록: TCN100단어899학습후보(기록ID01~09)/300개발(ID11~13). 피드백4단어36FIT/12DEV. 기존에 사용한 개발자료이며 새 평가가 아니다. 시간별 중심특징 정답과 DTW로 옮긴 약한 구간 주석을 구분한다. 실제 원본 접근성·중복·주석시각·화면 경계/가림 정책을 고정한 뒤 추출·학습을 진행해야 한다.\n\n기존 모델과 웹화면을 바꾸지 않았다. 사용자·멘토자료와 카메라 사용0. 다음은 공통 RTMW 학습 입력·FIT/DEV·정답 적격성 고정 및 공개 추출 준비. PLAN_KO.txt의 전체4단계에 따라 진행한다.\n',encoding='utf-8')
 write(U/'CURRENT_EXECUTION.json',dict(status='NATIVE_PUBLIC_CAPTURE_REUSE_VERIFIED',at=at,running_processes=[],training=0,production_changed=False,next_unit=str(S),verification=str(U/'NATIVE_PARALLEL_VERIFICATION.json')))
 write(S/'CURRENT_EXECUTION.json',dict(status='COMMON_INPUT_ADAPTER_VERIFIED_DATA_AUDIT_NEXT',at=at,running_processes=[],training=0,source_inventory_only=True,next='Resolve and hash approved public sources; fix crop/missing masks and time-label provenance; freeze train/dev candidate conditions before extraction/training.'))
 next_action='RTMW共通入力を基に' # overwritten below to keep state Korean.
 next_action='RTMW 공통입력 어댑터 검증 완료. 다음은 기존 공개100단어899FIT/300DEV 및 피드백4단어36FIT/12DEV의 원본접근·중복·주석시각과 화면경계/가림 정책을 확인하고 학습조건을 동결해 RTMW 입력을 준비한다. 새 TCN은 미세조정/동일구조 새학습 후보 비교, 피드백은 관절전용 시간합성곱+GRU 별도학습. 원본 C·T3-F/8812 유지, 사용자자료 금지. 현재 실제카메라·새학습0.'
 state=read(O/'STATE.json');state['current_stage']='RTMW_LOCAL_CAPTURE_VERIFIED_SHARED_INPUT_READY_TRAINING_PENDING';state['next_single_action']=next_action;state['updated_at']=at
 state[U.name]=dict(status='NATIVE_PUBLIC_CAPTURE_REUSE_VERIFIED',at=at,report=str(U/'RESULT_KO.txt'),verification=str(U/'NATIVE_PARALLEL_VERIFICATION.json'),frames=360,clips=3,post_capture_seconds=[x['last_capture_to_all_ms']/1000 for x in v['runs']],full_feedback_timing=False,camera=False,production_changed=False,training=0,protected_files=len(freeze['protected']),remaining='Real webcam integration and new RTMW models; thumb uncertainty and baseline 순종 misclassification unresolved.',browser_condition='Recorded input sync passed, original2frame capture gap and one_second timing failed; not adopted')
 state[S.name]=dict(status='COMMON_INPUT_ADAPTER_VERIFIED_DATA_AUDIT_NEXT',at=at,report=str(S/'RESULT_KO.txt'),verification=str(S/'VERIFICATION.json'),frames=360,scalars_checked=100800,training=0,next_single_action=next_action)
 write(O/'STATE.json',state)
 ctx=R/'WordsInSilence_PROJECT_CONTEXT.md';s=ctx.read_text(encoding='utf-8')
 heading='### 2026-11-08 로컬 촬영 재사용 검증과 RTMW 공통입력 전환'
 entry=f'''{heading}

사용자가 브라우저 이미지전달보다 Unity처럼 로컬 촬영·RTMW추출·관절전달 구조로 여기서도 시험할 것을 요청했고, RTMW한번으로 TCN과 세부피드백을 함께 제공하는 방향을 승인했다. word100_rtmw_browser_capture_20261108에서 브라우저 RGBA전송/MediaRecorder 경로의 누락·지연을 보존하고 WebCodecs경로 및 로컬직접경로를 비교했다. 선택한 로컬경로는 공개3영상360프레임 모두 보존, 녹화/관절PTS동일, 추가검출0. 독립몸/손MP계산 병렬화·TCN사전준비·최대2프레임파일버퍼 후 종료→저장/RTMW재사용/C·T3-F 0.348~0.529초. 초기 RTMW준비10.16초+TCN3.88초 별도. 최종피드백·실제카메라·Unity·웹렌더링 시간은 아니다. {nchecks}검산통과; 전후MP좌표/C·T3-F점수동일, RTMW기존출력동일. 순종의 기존T3-F오인식(공항시장)은 그대로여서 정확도향상 주장은 하지 않는다. 원래8812/모델변경0, 서버종료.

word100_rtmw_shared_input_20261108에서 raw133점/점수/PTS/프레임을 보존하며 TCN280입력·몸/손 상대좌표를 함께 만드는 어댑터를 준비했다. 25검사와 별도360프레임100800수치검산최대차이0. finite/in-frame은가시성정답아님, rawscore확률아님, 자동보간/미관측정답채움0. 현재전체시각보존. 새TCN/피드백학습0. 공식wholebody인덱스와기존학습코드확인. TCN899FIT(ID01~09)/300DEV(ID11~13), 피드백36FIT/12DEV의기존공개목록확인; 실제원본/정답시간/화면경계감사는다음. 기존C·T3F는비교기준보존, 관절전용학습시SignRep상시RGB가지제외. 사용자·멘토자료/카메라/다운로드/외부전송/Git0. 구현및별도검산은같은Codex이며독립사람검증아님. 과거AO중단유지.

'''
 marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n';assert marker in s and heading not in s;s=s.replace(marker,marker+entry,1)
 marker9='## 9. 다음 단일 작업\n\n';assert marker9 in s;s=s.replace(marker9,marker9+'### 2026-11-08 RTMW 통일 학습 입력 준비\n\n'+next_action+' 완료된 공개재생시험을 재추론하지 않는다. Native_PARALLEL 검산과 공통어댑터검산은 완료됐고 실제카메라나최종피드백완료로격상하지않는다.\n\n',1);ctx.write_text(s,encoding='utf-8')
 handoff=f'''# WordsInSilence 최신 인수인계

{at} 최신 사용자 승인: RTMW로 관절을 한번 추출해 TCN과 세부피드백을 공유하는 방향. 사용자는 같은 로컬프로그램에서 촬영과관절추출을 동시에 하고 종료후관절만 전달할 것을 명확히 요청했다. 외부서버불필요,127.0.0.1은로컬통로. 기존8812화면/모델은변경하지않았다.

마지막 완료: word100_rtmw_browser_capture_20261108. 브라우저큰화면전달의지연·MediaRecorder누락을그대로기록하고수정. 최종선택은 native_replay.py/native_parallel.py의 로컬촬영모사→RTMW+녹화+기존TCN 동시처리. MP의독립pose/hand계산을 DetectorPair로병렬실행,동결소비코드는그대로. 공개3영상순종/조상/수어ID02,360프레임누락0,PTS정확동일,원시RTMW기존동일,MP/C·T3-F전후동일. 종료후0.471/0.529/0.348초. NATIVE_PARALLEL_VERIFICATION.json {nchecks}항목PASS. 기본값을바꾸지않은새격리부품이다. 실제카메라/웹렌더링/Unity/새피드백시간아님. 초기RTMW10.16+TCN3.88초별도. 촬영중좌표완성p95약0.33~0.49초. 순종기존TCN오인식(공항시장)잔존;이번은정확도향상아님.

이전browser_a423697e는받은358프레임보존/PTS0.495ms이하/분석픽셀동일이나원본360중2프레임미수신,잔여1.10~2.08초로사전1초실패. 브라우저를최종적용했다고말하지말것. 초기실패/계측부하/코덱지원/큐포화모든기록보존. native첫순종5.74초는TCN초기4.22초포함,병렬비교속도배수계산에쓰지않는다. 모든시험프로세스종료,신규카메라0.

현재다음단위: word100_rtmw_shared_input_20261108. PLAN_KO.txt/adapter.py/check_prepare.py/PREPARED.json/verify_saved.py/VERIFICATION.json. raw133/score/time/fid공유→TCN280및몸/손상대좌표. 25검사,360프레임100800수치독립식계산최대0. 새추론/학습0. shape호환이기존가중치성능호환은아님. finite/inframe은실제가시성아님;점수확률아님. 전체시각유지하며RTMW점존재로시작끝자르지않는다. 얼굴은손과부위관계용,표정제외.

{next_action}
후속조건: 기존데이터좌표의source/frame/time주석대응확인, 정상촬영의신체/카메라변화·가림·짧은주요동작을포함한평가. oldDEV는미공개시험아님,REAL16중복격리/과거노출보존. TCN에는목표단어주입금지;피드백에는선택된표현의설명/학습흐름사용. 지도정답과DTW약한주석구분. 기존AO학습자동재개아님. 사용자/멘토자료학습·튜닝·선택금지. W:/sources/과거동결읽기전용. 새다운로드/외부전송/Git금지.
'''
 (R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
 write(U/'CODE_MANIFEST.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in U.iterdir() if p.suffix in ['.py','.js','.html']})
 print('closeout synchronized',nchecks,'native checks;',len(freeze['protected']),'old files preserved')
if __name__=='__main__':main()
