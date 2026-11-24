"""Write reviewed results and synchronize authoritative handoff, no model calls."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,sys,hashlib,html
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent
ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

assert not (U/'COMPLETION.json').exists()
plan=read(U/'PLAN.json');comparison=read(U/'COMPARISON.json');verify=read(U/'INDEPENDENT_VERIFICATION.json')
design=read(U/'DESIGN_AUDIT.json');trace=read(U/'ROOT_TRACE_SUPPORT.json')
assert verify['status']=='PASS' and len(comparison['rows'])==20
assert 'REVIEWED' in design['status'] or 'COMPLETE' in design['status']
model=ROOT/'research/front_keyphase_v1/runs/word100_tcn_v1/whole_context/best.pt'
model_hash=sha(model)
assert model_hash=='e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f'
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
next_action='사용자와 추가 보정의 효과가 미미하고 초기·종료 손목 잔존이 남는 결과를 검토한다. 이 손목 제거 필터는 채택 보류한다. 검토 후 전체 검출+상반신 마스크와 실제 화면 잘림에서 생기는 미관측을 포함할 제한 학습·개발검증 조건을 확정한다. 승인 전 학습·899영상 재추출·운영 TCN/DTW 교체는 시작하지 않는다.'
summary='안정된 초기 팔내림 구간 선택+경계 손목1점의 시간 기반 제외 후보를 공개 TRAIN20영상(3촬영자)2536프레임에 비교했다. 초기 보정20/20성공, 실제RGB 새검출10·동일범위 기존검출10재사용. 손21점 관측은 전체검출1733→1734(증가2/손실1), 실제crop1706→1705(증가3/손실4)로 사실상 변화 미미. 필터 자체는 홍성 종료의 손목3프레임만 제외, 손가락·몸점 추가삭제0. 잘하다 처음5프레임 잔존5→12점, 홍성 마지막5 잔존5→4점으로 완전제외 실패. 냉커피 중간 재진입은 보존했지만 기존sensor누락은 남음. 구현25검사·독립33반례/120조건280채널4260480값/1068보호파일 검증PASS. 추가필터 채택보류; 학습·분류추론·사용자자료·운영변경0.'
status='VERIFIED_STABLE_REST_ROI_LIMITED_EFFECT_NOT_ADOPTED_REVIEW_BEFORE_TRAINING'
report=f'''안정된 준비 자세·경계 손목 처리 비교 결과
갱신: {now}

결론
구현과 독립 검증은 완료했다. 다만 이번 추가 보정만으로 뚜렷한 관측 개선이 나오지 않았으며, 휴식 손을 모두 제외하지 못했다. 추가 손목 제거 필터를 학습 입력에 바로 채택하지 않는 것을 권고한다. 앞서 만든 전체 영상 검출→상반신 입력 분리 구조와는 별도의 판정이다. TCN 학습이나 인식률 시험은 시작하지 않았다.

1. 정확히 바꾼 것
- 기존: 시작5프레임 손목 중앙값으로 하단 경계 결정.
- 후보: 처음1초 안에서 양쪽 어깨·팔꿈치·손목이 관측되고, 팔이 아래로 내려가며 최소0.3초 안정된 구간을 선택. 그 손목 바로 위에 고정 하단을 잡음. 얼굴·머리 위·좌우 폭은 기존처럼 유지.
- 경계 안에 손가락 없이 손목 한 점만 남고, 준비 자세의 위치 근처에서0.3초 이상 정지한 경우에만 현재 프레임 손목을 제외. 손가락이 보이거나 움직이거나 관측이 끊기면 시간을 다시 셈. 과거 프레임까지 지우지 않음.
- 어깨·팔꿈치·몸 포즈 손목을 손21점과 혼동하지 않았으며, 이 필터는 몸 관절을 지우지 않음. 손을 새로 만들거나 빈 구간을 보간하지 않음.
- 영상 전체 프레임과 원래 시각은 보존. 단어별 예외·실사용자 자료·결과에 맞춘 임계값 재조정 없음.

2. 비교한 자료와 방법
이미 문제를 검토했던 공개 TRAIN20영상, 촬영자3명,2536프레임. 독립 평가셋은 아님.
기존 손목경계2조건과 후보 경계·필터 전후4조건, 총120조건. 새 경계가 달라진10영상만 실제RGB를 잘라 MediaPipe 재추출(약50초), 동일한10영상은 고정된 기존 검출 재사용. 전체원본 검출도 저장 관측을 재사용.
초기 보정은20/20성공. 잘하다는 첫0~4프레임 대신3~12프레임을 선택하여 하단877→892px. 나머지는 경계가 그대로이거나1~2px 위로 이동. '안정'은 어깨너비5% 이하 이동을 허용하는 수치 조건이며 완전 정지는 아니다(잘하다 최대4.43%). 이것이 검출 또는 인식 성능 향상을 뜻하지는 않음.

3. 핵심 결과
아래 숫자는 '한 손의21점이 모두 사용 가능했던 프레임 수'를 양손·영상에 걸쳐 합한 값이다. 정답 수나 인식률이 아니다.
                         기존     후보     새로 보존 / 사라짐
전체 검출 후 영역 마스크  1733     1734       +2 / -1
실제 잘린RGB에서 검출     1706     1705       +3 / -4

필터 자체: 홍성군주민지원센터의 종료 근처 왼손 손목 f93~95, 총3프레임만 제외. 손가락·몸점 추가삭제0. 실제 잘린RGB에서는 제거 대상0.
전체검출 조건의 첫5프레임 잔존 손관절점:5→12. 모두 잘하다에서 손을 내려놓는 초기 구간이다. 안정된 뒤의 더 낮은 경계를 쓰면서 초기 조각이 오히려 더 남았다.
전체검출 조건의 마지막5프레임 잔존:5→4. 홍성 손목4프레임은 아직 남는다. 마지막0.5초 잔존은17→11점으로 줄었지만 완전제외는 아니다.
실제RGB잘림 조건: 처음5프레임3점 잔존, 마지막5프레임0점으로 이전과 같다.
'점' 합계는 관절점×프레임 단위이며, 점12개가 한 프레임의 손가락12개라는 뜻은 아니다.

4. 왜 완전히 해결되지 않았나
홍성 f93~95는 충분히 오래 정지해 제외되었지만, 이후 움직임·미관측이 발생해 정지 시간이 초기화됐다. 마지막 f102~105는 새 관측 후0~0.1초이므로 고정된 보호 규칙에 따라 남는다. 짧게 내려갔다 다시 올라오는 실제 동작을 성급히 지우지 않기 위한 결과다.
냉커피의 중간 손 하강·재진입에서 새 필터가 손을 추가 삭제하지 않았다. 그러나 실제crop의 기존 원검출 누락(예:51,53프레임)은 이 필터로 복구되지 않는다. 수영·전멸의 기존 한손 원검출 부재/중복검출 문제도 해결한 것이 아니다.
1~2px 경계 이동에도 일부 관절점이 포함/제외되고 실제 검출값이 달라질 수 있다. 전체 검출 조건의 손21점 손실1건은 상처79프레임의 검지끝이 경계 밖이 된 경우다. 실제crop 손21점 손실4건은 수어35·잘하다47·별35·상처38프레임으로, 모두 이전 관절에 새 경계만 적용하면21점이 남는다. 따라서 이4건은 필터 삭제나 단순 마스크가 아니라 새crop의 검출·배정 출력 변화다. 총계의 작은 순증을 성능 향상으로 해석하지 않았다.
이 필터와 구별되지 않는 매우 낮고 정적인 손목만의 동작은 여전히 오인 제외 가능성이 있다. 모든 수어에 안전하다는 결론은 내리지 않는다.

5. 검증과 한계
구현자 합성검사25개 통과. 별도 에이전트의 독립 반례33개,120조건280채널4,260,480값 재계산, 실제crop20조건 좌우배정 재생 일치. 영역밖135,826점 교란 무영향, 보호파일1,068개 해시 일치. 초기 보정과 변경·삭제 장면은 별도 시각 검토했다.
전체원본의 휴식 손목을 사용한 오프라인 경계 보정과 실제 웹캠 입력은 다르다. 웹캠 화면 밖 손은 볼 수 없으며, 이번에는 새 사용자/카메라 시험을 하지 않았다.
최초 실행 오류는 이전 완료기록의 상대경로를 절대경로처럼 읽은 준비 단계 문제였고, FIRST_ERRORS.log와 RECOVERY.json에 보존했다. 경로 해석만 고쳐 전후 자료 해시 검증 후 이어갔으며 후보 규칙은 변경하지 않았다.

6. 다음 결정
{next_action}
계속 경계선을 조금씩 조정해 이20영상에 맞추는 방향은 권고하지 않는다. 본래 목표인 '화면 아래에 반대손이 없는 사용자 입력'에 견디는지는 고정된 학습/개발검증 비교로 확인해야 한다. 사용자 테스트 기록을 학습·튜닝·후보선택에 넣지 않는 원칙은 유지한다.

구현: stable_roi_impl. 실행·비교·최종판정: root. 독립 계산: stable_roi_verify. 시각·설계 감사: stable_roi_design.
근거: SCOPE.json, PLAN.json, COMPARISON.json, INDEPENDENT_VERIFICATION.json, DESIGN_AUDIT.json, ROOT_TRACE_SUPPORT.json.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
images=sorted((U/'design_images').glob('*.png'))
imagehtml=''.join(f'<figure><img src="design_images/{html.escape(p.name)}" loading="lazy"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in images)
(U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>준비 자세·손목 경계 비교</title><style>body{margin:0;background:#edf1f5;color:#16293d;font:17px/1.8 "Malgun Gothic",sans-serif}main{max-width:1100px;margin:30px auto;padding:40px;background:white;border-radius:16px}pre{white-space:pre-wrap;font:inherit}h1{font-size:30px}img{width:100%;height:auto}figure{margin:30px 0}figcaption{font-size:14px;color:#44566b}</style><main><h1>준비 자세·손목 경계 보완 결과</h1><pre>'+html.escape(report)+'</pre><h1>원본 프레임과 검출 근거</h1>'+imagehtml+'</main></html>',encoding='utf-8')
entry=dict(status=status,updated_kst=now,unit='ops/'+U.name,summary=summary,next_single_action=next_action,
    clips=20,unique_video_frames=2536,conditions=120,new_capture=10,reused_capture=10,
    independent_verification='PASS',implementation_tests=25,independent_synthetic_checks=33,
    full_all21_before=1733,full_all21_after=1734,sensor_all21_before=1706,sensor_all21_after=1705,
    suppressed_hand_wrist_frames=3,training=0,classifier_inference=0,production_changes=0,user_data=0,
    current_model_sha256=model_hash,process_status='capture exit0 (PID47988/exec81267); compare exit0; verification exit0; complete')
write(U/'ROOT_ACCEPTANCE.json',dict(engineering='VERIFIED',adoption='NOT_ADOPTED_LIMITED_BENEFIT_AND_RESIDUALS',
    source_and_model_preserved=True,summary=summary,reviewed=[str(U/n) for n in ['SCOPE.json','PLAN.json','COMPARISON.json','INDEPENDENT_VERIFICATION.json','DESIGN_AUDIT.json','ROOT_TRACE_SUPPORT.json']],
    user_review_required_before_training=True,model_sha256=model_hash))
write(U/'EXIT.json',dict(prepare_recovery=0,capture=0,compare=0,implementation_tests=0,independent_verification=0,training_started=False))
state_path=ROOT/'research/front_keyphase_v1/ops/STATE.json';old=read(state_path)
old['updated_date_kst']=now;old['current_stage']='WORD100_STABLE_REST_ROI_LIMITED_EFFECT_REVIEW'
new={};inserted=False
for k,v in old.items():
    new[k]=v
    if k=='current_stage':new[U.name]=entry;inserted=True
assert inserted
if 'next_single_action' in new:new['next_single_action']=next_action
state_path.write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
context_path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';context=context_path.read_text('utf-8-sig')
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
assert h8 in context and h9 in context
context=context.replace(h8,h8+'\n\n### 2026-10-17 안정된 준비 자세·경계 손목 보정 비교 — 추가필터 채택 보류\n'+summary+' 상세 `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`. 후보규칙 결과후 수정 없음.\n',1)
context=context.replace(h9,h9+'\n\n### 2026-10-17 다음 — 추가 보정 효과·잔존 검토 후 학습 조건 확정\n'+next_action+'\n',1)
context_path.write_text(context,encoding='utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {now}. 사용자 “그럼 너의 계획대로 진행해봐”에 따라 안정된 초기 팔내림 보정·경계 손목 조각 필터만 구현/비교. **학습은 결과 검토 후**라는 제한 유지.

## 마지막 완료
{summary}

## 실제 변경과 해석
- 새 연구단위 `{U.name}`의 `stable_rest.py` 및 검증/실행 코드만 작성. 기존검출·ROI어댑터·운영TCN/DTW/웹은 변경 없음.
- 처음1초 중 최소0.3초 안정 구간, 손목중앙값−어깨너비2%의 고정하단. 잘하다만3~12프레임, 나머지0~9. 후자는기존하단과동일 또는1~2px위.
- ROI안 손목1점만 남고 준비위치근처 정지0.3초일 때만 현재점 제외. 손가락이 보이면보호. 삭제3점은홍성f93~95. f96움직임/f99~101미관측후 재시작하므로 f102~105잔존. 몸pose손목과손21점분리.
- 처음/끝5프레임 full손점0인영상은각19/20, 모두동시에18/20. 초기잘하다 잔존5→12, 종료홍성5→4. 실제crop처음3/끝0점 이전과동일.
- 냉커피 재진입을 새필터가삭제하지않음. 실제crop원검출누락51/53과 수영/전멸 기존문제 해결은아님.
- 이번은이미조사한공개TRAIN20의공학비교. 관측수는정답률아님. 오프라인초기원본손목보정은실제웹캠화면밖손정보와다름. 시각근거 `design_images/`.
- 원본·이전동결결과1068핀불변. 최초prepare상대경로오류는FIRST_ERRORS.log/RECOVERY.json보존, 규칙변경없이복구.

## 다음 한 작업
{next_action}

## 담당·근거·프로세스
구현stable_roi_impl, 실행/판정root, 독립stable_roi_verify, 시각stable_roi_design. RESULT_KO.txt/html, COMPARISON, INDEPENDENT_VERIFICATION, DESIGN_AUDIT, ROOT_TRACE_SUPPORT, ROOT_ACCEPTANCE, COMPLETION 참조.
새capture PID47988/exec81267 종료0(10새검출/10재사용,50.09초), compare/독립검증종료0. 새학습프로세스없음. 기존완료검출재실행금지.

## 운영 상태 유지
운영체크포인트 `research/front_keyphase_v1/runs/word100_tcn_v1/whole_context/best.pt`, SHA {model_hash}. 학습9명899/개발3명300,96.67%는개발검증. 사용자12회Top1 6/12·Top5 8/12는실사용기록이며학습·튜닝·후보선택금지.
TCN은몸9점+양손42점280채널,어깨·팔꿈치·손목포함. 기존진단웹 http://127.0.0.1:8767/ 은이번연구결과로갱신하지않음. sources/원본/과거동결산출물읽기전용, Git변경없음.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
# Pin completed experiment artifacts only; mutable root state remains outside.
pins={str(p.relative_to(U)):sha(p) for p in U.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
write(U/'COMPLETION.json',dict(**entry,artifact_pins=pins))
assert read(state_path)[U.name]['summary']==read(U/'COMPLETION.json')['summary']
assert summary in context_path.read_text('utf-8') and summary in (ROOT/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
print(json.dumps(dict(status=status,artifact_pins=len(pins),report=str(U/'RESULT_KO.html')),ensure_ascii=False))
