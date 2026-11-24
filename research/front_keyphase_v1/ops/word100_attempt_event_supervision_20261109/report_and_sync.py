"""Close the fixed experiment; do not change deployed weights or rerun inference."""
from pathlib import Path
import json,hashlib,shutil,datetime
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 verify=read(U/'FINAL_CHECK.json');browser=read(U/'BROWSER_CHECK.json');media=read(U/'MEDIA_CHECK.json');assert verify['status']=='PASS_SAVED_ARTIFACT_AUDIT' and browser['status']==media['status']=='PASS'
 base=read(U/'scores/final_normal_only_42.json');new=read(U/'scores/final_attempt_supervised_42.json');legacy=read(U/'LEGACY_COMPARE.json');a=new['by_kind'];b=base['by_kind']
 protocol=dict(internal_split_limitation='FIT07-09 are withheld from new inner-head updates only. Both frozen shared100 backbone and initial deployed heads already saw FIT01-09. This is not independent unseen-person validation. Inner metrics were not used to select any configuration.',historical_DEV='IDs11-13 are held out of all current training, but repeatedly observed in earlier project work. This is regression evidence, not a fresh final test.',omission_limit='Whole later event removed, but an abrupt edit joins the prior action to rest. A network can use edit artifacts; real learner omissions remain unverified.',adoption='NOT_DEPLOYED: normal withholding increased and alternate handshape control lost the required stage.',reviewer='Same Codex original RGB review and separate saved arithmetic. No expert semantic ground truth.')
 write(U/'PROTOCOL_AUDIT.json',protocol)
 nextplan=dict(objective='단계 구분이 손모양 오류와 함께 무너지는 문제를 해결하되 누락/미관측 분리 이득과 정상 판단을 보존',next_single_action='공개 FIT에서 움직임은 같은데 손모양·손바닥 방향이 달라지는 짝의 단계/전환 경계를 보강한다. 특정 DEV 실패를 정답으로 베끼지 않고 FIT만으로 새 비교조건을 동결한다.',evidence=[dict(problem='기존 다른 손모양 대조3개에서 필수 동작 구간0/3',observed='REAL11 대표에서는 쓸기 단계 peak0.078, 다음 선택 단계 peak0.988. 동작 존재 확률은 높으나 단계 배정이 틀림. 단순 손 미검출 문제가 아님.'),dict(problem='정상 한 장면 추가 보류',observed='잘하다 REAL11 쓸기 구간49-72→47-66, 왼손닫힘의 짧은 시작 전환47-48이 포함돼 보류. 임계값 완화로 감추지 않음.')],steps=[dict(step=1,task='공개 FIT에서 유지 모양·모양 변화·실제 팔 이동·양손 관계·전환을 분리 주석. 현재 손가락 변형 한 가지 외에 손바닥 방향이 달라도 단계가 유지되는 사례 지원 확인.',gate='반복 횟수/접촉/엄지 세부처럼 미확정 특징을 임의 정답으로 만들지 않기'),dict(step=2,task='손모양 정오와 독립적인 움직임 후보 경로 + 기존 전체특징 경로를 고정 조건으로 비교. 모든 후보/미배정 구간의 근거 보존; 후단은 단계확신·모양확신·관측여부를 별도로 사용.',gate='학습조건 선택이 필요하면 초기 모델까지 사람 제외가 된 FIT 분리 필요. 지금 inner결과는 그 용도로 사용 금지'),dict(step=3,task='정상/반대모양/역방향/복합/손교차/짧은변화/완전누락/관측가림을 함께 확인. 현재 DEV는 회귀 전용; 새 일반화 평가는 출처/중복/노출 감사부터.',gate='정상 미확인 증가 없이 실제 오류 구간이 비교에 남고, 미관측을 정상/누락으로 단정하지 않아야 함'),dict(step=4,task='통과한 기능만 현재 연구 웹에 연결하고 실제 촬영 종료→표시 시간을 측정. 이번 오프라인 화면의 편집 대응표를 DTW 성능으로 주장하지 않기.',gate='C·T3F/기존TCN 유지, 사용자 자료 학습·튜닝·후보선택 금지')],not_started=True,no_new_detector=True,no_word_specific_exception=True)
 write(U/'NEXT_PLAN.json',nextplan)
 steps=[dict(step=1,status='DONE_WITH_LIMITS',work='552입력 조건의 단계/모양/존재/관측여부 분리. 공개 원본48개, 누락12편의 전후 RGB 검토. 연속 전프레임 전문가 정답은 아님.'),dict(step=2,status='DONE',work='정상 중심 대 분리 감독 2조건. 내부진단2회 + 최종3seed×2조건=8고정학습. 공통 Conv+GRU 관측은 고정; 단계/차이/사건존재 판정부만 학습. 방향은 기존 별도학습 MotionNet을 동결 재사용.'),dict(step=3,status='DONE_PARTIAL_GAIN_FULL_GATE_FAILED',work='각 최종모델 공개DEV144조건 + 대표모델 과거93조건 추가검증. 누락 개선, 다른 손모양 분포/정상 확인률 회귀.'),dict(step=4,status='OFFLINE_VIEW_DONE_DEPLOYMENT_WITHHELD',work='정상12+누락6+관절가림6 실제 저장 결과, 원본/편집 영상 나란히, 단계색/판정색 분리. 웹 기본 모델 교체 안 함.')]
 result=dict(status='COMPLETE_PARTIAL_OMISSION_GAIN_NOT_DEPLOYED',unit=U.name,steps=steps,data=dict(original_fit=36,fit_record_ids=list(range(1,10)),original_dev=12,dev_record_ids=[11,12,13],words=['WORD0739','WORD1133','WORD1567','WORD2523'],conditions=552,primary_dev_cases=144,legacy_cases=93,independent_expert_ground_truth=False),comparison_definition='normal_only_42 vs attempt_supervised_42: matched new heads, same initialization/data/steps/thresholds; not old C/TCN or all previously deployed feedback. Deployed stage71/84 equals new71/84.',primary=dict(baseline=base,candidate=new),legacy=legacy,protocol=protocol,adoption_gates=verify['deployment_gates'],deployed=False,web_code_changed=False,private_access=False,training_or_tuning_on_private=False,new_download=False,shape_backbone_changed=False,direction_head_retrained=False,verification=dict(saved_checks=verify['checks'],browser_checks=len(browser['checks']),media_checks=media['checks']),visual=str(U/'RESULT_KO.html'),report=str(U/'RESULT_KO.txt'),next=str(U/'NEXT_PLAN.json'))
 write(U/'RESULT.json',result)
 report='''동作 구간 찾기와 모양·방향·누락 판단 분리 — 2026-11-09

이번 1~4단계의 학습·공개 비교·저장 결과 화면을 마쳤습니다.
마지막 동작이 빠진 경우와 관절만 안 보인 경우를 더 잘 구분했습니다.
다만 다른 종류의 손모양 오류에서 비교 구간을 놓쳤고, 정상 한 장면을 더 보류했습니다.
따라서 새 후보 전체를 기존 웹 모델로 교체하지 않았습니다. 피드백 기능 완성이라고 말할 단계는 아닙니다.

무엇을 바꿨나
예전에는 손모양이 다르면 “이 동작이 있는가”부터 흔들려 비교에서 빠질 수 있었습니다.
이번에는 ① 어떤 동작을 시도한 구간인가 ② 그 구간의 모양·방향은 어떤가 ③ 필요한 동작이 빠졌는가를 따로 다뤘습니다.
공개 영상의 손가락 모양을 바꾼 공학 대조에도 동작 단계 이름은 남겼고, 잘못된 모양을 정상 손모양으로 가르치지는 않았습니다.
마지막 주요 동작 전체를 뺀 입력과, 영상/시간은 그대로인데 관절만 가린 입력을 별도로 만들었습니다.
최고점 구간 하나 외의 다른 후보 근거도 기록합니다. DTW가 빠진 동작을 만들어 채우지는 않습니다.
검토 가능한 설명·주석을 조합하는 공통 처리이며 단어 이름별 성공 예외나 DEV 사후 문턱 조정은 없습니다.
현재 사건 연결은 지원4표현 범위입니다. 모든 단어에서 검증됐다는 뜻은 아닙니다.

무엇을 학습했나
기존 100단어 공통 학습에서 얻은 Conv+GRU 관측 모델은 고정했습니다.
그 뒤의 작은 동작구간/차이 판정부를 추가 학습하고, 필요한 사건이 존재하는지 보는 출력부를 더했습니다.
방향은 기존에 따로 지도학습한 MotionNet과 팔 기준 이동 계산을 재사용했습니다. 이번에 방향 모델 가중치를 다시 학습한 것은 아닙니다.
모양 차이 판정부는 직접 확인한 실제 특징과 기대 특징의 짝으로 학습했습니다. 합성 손모양을 검증된 실제 사람 정답으로 둔갑시키지 않았습니다.
정상 중심 비교와 새 분리 감독 비교는 같은 초기값·학습횟수·입력 분리·고정 문턱을 사용했습니다.
내부 진단2회와 최종2조건×3seed, 총8회로 고정했고 마지막 가중치를 사용했습니다. 가장 잘 나온 seed를 고르지 않았습니다.
단, 내부07~09는 새 학습 업데이트에서만 제외됐고 초기 모델은 과거 그 ID를 본 적이 있습니다. 독립 새 사람 검증으로 해석하면 안 됩니다.

자료와 숫자 읽는 법
학습: 공개4단어×기록ID01~09 =36원본. 비교: 같은4단어×기록ID11~13 =12원본.
정상·속도·관절가림·모양변경·역방향·복합·사건삭제 등을 만들어 전체552입력 조건을 준비했습니다. 원본 영상552개가 아닙니다.
각 최종 모델의 비교는144조건, 대표 모델은 예전에 고정한93조건도 추가 회귀검사했습니다. 기존 DEV를 새 독립 평가라고 부르지 않습니다.
아래의 “비교→새 학습”은 이번 정상 중심 비교모델과 분리 감독 후보 사이의 변화입니다. TCN C의 인식률과 무관합니다.

확인한 결과
- 정상 영상에서 동작 단계를 맞춘 검토 시점: 67/84→71/84. 84는12영상에 있는 검토된 순간 수이지84개 단어가 아닙니다. 기존 웹 단계출력도71/84여서 기존 웹보다 좋아졌다는 증거는 아닙니다.
- 마지막 주요 동작 전체를 뺀 시험: 누락 후보3/6→6/6. 빠졌는데 필요한 항목을 모두 비슷하다고 표시한 경우2/6→0/6.
  순종·이집트 각3개 공개DEV 영상의 편집 시험입니다. 새 학습의 세seed 모두6/6 후보를 냈습니다. 실제 사람 실수 검출률100%라는 뜻은 아닙니다.
  새 존재 판정부 단독으로 낮은 존재 확률을 낸 것은 대표모델3/6입니다. 기존 단계/관측/종료 근거와 연결한 최종 결과가6/6입니다.
- 마지막 구간의 관절만 가린 시험: 누락으로 몰아간 경우0/6 유지. 필요한 항목을 전부 비슷하다고 넘긴 경우2/6→0/6, 모두 미확인으로 남겼습니다.
- 모든 손 관절을 가린12조건: 전부 비슷함0, 누락후보0으로 유지했습니다.
- 학습 때 쓴 형태와 같은 손가락 변경 시험: 오른손 변경 시 단계63/84→72/84, 필요한 단계가 모두 남은 영상10/12→12/12.
  하지만 이미 주먹인 구간도 포함한 공학 대조이므로 “오류12개를 모두 찾았다”는 뜻은 아닙니다.
- 기존의 넓은 주요구간 역방향 시험3개: 차이 경고2/3→3/3. 두 손모양·역방향 복합3개도 차이 경고3/3 유지.
  반면 새 짧은 핵심 구간만 뒤집은 대조에서는 정상 방향 부분이 남아 있어 방향 판정을 완성했다고 말할 수 없습니다.

남은 문제와 적용 판단
1. 정상12개에 차이 경고는0개로 유지했지만, 미확인 영상이4→5개로 늘었습니다.
   잘하다 ID11의 쓸기 구간이49~72에서47~66으로 바뀌어 시작 부분의 왼손 모양 전환이 함께 들어왔습니다. 눈에 보이는 짧은 전환을 어떻게 구간별로 다룰지 남았습니다.
2. 손바닥 배치까지 달라지는 기존 손모양 변경3개에서는 차이 경고2/3→0/3, 모두 미확인이 됐습니다. 필수 쓸기 구간도3/3→0/3으로 사라졌습니다.
   대표 저장출력은 쓸기보다 그 뒤의 선택 단계로 배정됐습니다. 손이 안 보여서만 생기는 문제는 아니며, 모양 변화에 단계 구분이 아직 끌려갑니다.
3. 상반신 자르기·속도 차이에서 오경고를 늘리지는 않았지만, 다른 고정93조건에서는 미확인이 늘었습니다. 정상 속도 차이까지 완전히 보존했다고 주장하지 않습니다.
   새 시험의 느린 속도 단계70→72/84, 빠른 속도54→53/64, 짧은2프레임 가림73→72/84입니다. 빠른 조건은 일부 검토 프레임이 생략돼 분모64이며 짧은 변화 보존의 완전한 증거가 아닙니다.
4. 기존 정상 교차 영상도 포함했지만 별도의 교차 오류율이나 손가락 변화·반복 횟수 정확도를 확보하지 못했습니다. 관절 검출기와 관측 가중치를 그대로 보존했으며 교차 성능 향상을 주장하지 않습니다.
5. 전체 사건 삭제는 앞 동작과 대기 자세가 편집으로 바로 이어집니다. 편집 흔적을 배웠을 가능성을 배제하지 못하므로 실제 자연스러운 누락에는 추가 확인이 필요합니다.
6. 현재 관측은 손모양·손높이의7특징 범위입니다. 엄지·정확한 손끝 접촉·모든 손모양·의미상 허용범위·반복 횟수·짧은 내부 동작 생략 전체를 해결한 것은 아닙니다.
7. 별도로 학습한 방향 모델이 높은 확신으로 기존 이동 근거와 충돌하는 사례도 남았습니다. 그 결과로 기존 근거를 덮어쓰지 않았습니다.

처리시간
저장된 관측 특징을 받아 새 판정부 두 번(기대 특징0/1)을 계산한 CPU 시간은 중앙값2.65ms,95백분위4.94ms였습니다.
비교 판정부 중앙값은2.42ms였습니다. 관절 추출·전체 Conv+GRU·DTW·영상전송·화면표시를 뺀 시간입니다.
따라서 사용자의 촬영 종료 후 전체 대기시간이 줄었다고 보고하지 않습니다. 이번에는 실제 카메라를 열지 않았습니다.

결과 화면
RESULT_KO.html에서 정상12개, 실제 마지막 사건을 삭제한6개, 관절 입력만 가린6개를 고를 수 있습니다.
원본 공개 영상과 시험 영상을 나란히 보고, 누락 후보는 “이 동작을 이어서 했는지 확인”과 참고 구간 재생으로 보여줍니다.
없는 사용자 동작을 만들어 보여주지 않고, 회색은 미확인으로 남깁니다. 편집의 원본 대응표로 재생하며 DTW 성능 시험 화면은 아닙니다.
화면의 손뼈대는 실제 저장 관절입니다. 색은 검사한 특징만 뜻하며 전체 수어 정답률은 아닙니다.

검증·보존
저장값/자료분리/해시/산술 검산10144건, 공개 영상·관절 투영/디코딩2045건, 실제 Chrome화면25항목 통과.
같은 Codex의 별도 검산·원본 시각검토이며 독립 전문가의 교육적 정오 확인과 다릅니다.
기존 C·T3F, 현재RTMW TCN, 웹8814 코드/모델을 교체하지 않았습니다. 사용자·멘토자료 접근/학습/튜닝/후보선택0, 다운로드/외부전송/Git0.
초기 legacy import 오류는 최초 기록 보존 후 경로를 명시해 복구했습니다. 브라우저 제한환경 시작 실패와 일시 재생오류도 보존했습니다. 재생중단 Promise 처리를 보완한 최종25검사를 통과했습니다.
학습과 모델 평가는 완료 결과를 재실행하지 않았습니다. 화면 검사는 실제 UI 수정/실패 이후에만 다시 했습니다.

다음 핵심 작업
새 모델을 또 바꾸기보다, 공개 FIT에서 “손모양이 달라도 같은 동작 단계”와 “앞 단계에서 다음 단계로 넘어가는 전환”의 정답을 보강해야 합니다.
팔·양손 이동으로 동작 후보를 찾는 경로와 손모양을 검사하는 경로를 따로 비교하고, 비교에서 빠진 구간도 남깁니다.
이때 정상 동작 미확인이 늘지 않는지, 서로 다른 손모양 대조에서도 오류 동작 구간이 사라지지 않는지를 함께 통과해야 합니다.
새 사람 일반화를 주장하려면 초기 모델의 학습 이력까지 분리된 평가가 필요합니다. 이미 본 DEV를 다시 맞춰서 개선으로 만들지 않습니다.
세부 다음 계획은 NEXT_PLAN.json. 이번 범위는 완료했지만 위 보완과 웹 모델 적용은 아직 진행하지 않았습니다.
'''.replace('동作','동작')
 (U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
 write(U/'VISUAL_REVIEW.json',dict(status='INSPECTED',reviewer='same Codex',screenshots=['VIEW_OMISSION.png','VIEW_MASK.png'],tail_sheet='TAIL_REVIEW.jpg',findings='12 public omissions: last reviewed predecessor remains; later action anchors removed; final tail hands lowered. Screen text readable; actual video/skeleton; uncertain distinct from wrong. Not independent expert review.'))
 write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_PARTIAL_GAIN_NOT_DEPLOYED',training=None,evaluation=None,temporary_browser_server='closed',web_default_changed=False))
 status='공개48원본(36FIT/12역사적DEV)에서 사건 위치·모양차이·사건존재를 분리한8고정학습과144조건/과거93조건 비교 완료. 정상 중심 비교→새 감독 대표42: 단계67→71/84(기존웹71과같음), 전체사건삭제 누락후보3→6/6 및거짓비슷함2→0, 관절만가림 거짓비슷함2→0/누락오판0유지. 세seed누락6/6. 정상오경고0유지이나미확인4→5; 다른손바닥형상대조3에서필수단계3→0/3·차이경고2→0/3으로회귀. 따라서새모델웹미적용. 손특징Conv+GRU/방향MotionNet고정, 단계·차이·존재heads만학습; 방향완성/전체속도개선아님. 순수head2.65ms. 저장검산10144/미디어2045/Chrome25통과. 같은Codex검산,전문가정답아님. inner07~09는초기모델이이미본ID라독립검증아님;기존DEV회귀전용. 공개편집누락은편집흔적가능성있고자연누락미검증. 기존C·T3F/현재TCN/웹8814보존,개인자료접근0. 모든학습평가종료.'
 nexttext=nextplan['next_single_action']+' '+nextplan['steps'][1]['task']+' 정상미확인·오경고 증가/오류구간소실이없어야웹적용검토. 기존8학습·평가완료재실행금지,DEV사후튜닝금지. 내부진단을새사람성능으로해석금지. 자세한NEXT_PLAN은 '+U.name+'/NEXT_PLAN.json. 새모델다운로드/개인자료/카메라/운영교체로확대하지않는다.'
 statepath=O/'STATE.json'
 if not (U/'before/STATE_before_finalize.json').exists():shutil.copy2(statepath,U/'before/STATE_before_finalize.json')
 state=read(statepath)
 entry=dict(status=result['status'],report=str(U/'RESULT_KO.txt'),result=str(U/'RESULT.json'),visual=str(U/'RESULT_KO.html'),verification=str(U/'FINAL_CHECK.json'),protocol_audit=str(U/'PROTOCOL_AUDIT.json'),execution=str(U/'CURRENT_EXECUTION.json'),summary=status,next=nexttext,new_model_deployed=False,web_code_changed=False)
 state[U.name]=entry;state={U.name:entry,**state};state['current_stage']=U.name;state['next_single_action']=nexttext;state['updated_date_kst']='2026-11-09';write(statepath,state)
 ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=ctx.read_text(encoding='utf-8');title='### 2026-11-09 시도 동작·모양·누락 분리 감독 — 누락 부분 개선, 전체 적용 보류\n\n'
 for marker,text in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',status+' 근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`.'),('## 9. 다음 단일 작업',nexttext)]:
  assert marker in s
  if not s.split(marker,1)[1].lstrip().startswith(title.strip()):s=s.replace(marker+'\n',marker+'\n\n'+title+text+'\n',1)
 ctx.write_text(s,encoding='utf-8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n현재 목표: 어떤 손모양인지와 어떤 동작을 언제·어느 방향으로·빠짐없이 했는지 분리 학습하고 연결한다.\n\n'+status+'\n\n다음 한 작업: '+nexttext+'\n\n완료4단계: 자료/정답분리,8고정학습,공개144+93조건비교,24사례시각화. 검증조건 미달로 기본 웹8814 모델은 유지. 새 결과 화면은 파일 RESULT_KO.html이며 새 웹캠서버가 아니다.\n\n방향 모델은 이번 재학습0. 84는12DEV영상의 검토시점; 누락6은순종/이집트각3편을편집한대조. 552는48원본의조건수. 손가림을정답/누락으로강제채택하지않음.\n\n구현·실행·검산: Codex 로컬; 독립사람정답 아님. 처음실패와범위한계는 RECOVERY_LOG/PROTOCOL_AUDIT. 모든학습/평가종료, 임시브라우저서버종료. 기존 웹서버의현재생존여부는이번확인안함. 카메라미접근/새다운로드없음.\n\n근거: '+str(U/'RESULT_KO.txt')+'\n실행: '+str(U/'CURRENT_EXECUTION.json')+'\n검산: '+str(U/'FINAL_CHECK.json')+'\n사용자·멘토자료 학습/튜닝/선택/접근 금지. 기존 C·T3F/현재RTMW TCN과 sources/원본·기존동결결과는 보존.\n',encoding='utf-8')
 # Artifact/document cross checks only. Never call training/evaluation here.
 links=[Path(result[k]) for k in ['visual','report','next']];assert all(p.exists() for p in links)
 assert read(statepath)['current_stage']==U.name and status in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf-8') and status in ctx.read_text(encoding='utf-8')
 for p,h in read(U/'FROZEN.json')['protected'].items():
  if isinstance(h,str) and len(h)==64:assert sha(p)==h,p
 write(U/'CLOSURE_CHECK.json',dict(status='PASS',root_context=True,handoff=True,state=True,report=True,protected_hashes=True,model_deployed=False,completed_execution=True))
 print(json.dumps(dict(status=result['status'],report=str(U/'RESULT_KO.html'),verified=verify['checks'])))
if __name__=='__main__':main()
