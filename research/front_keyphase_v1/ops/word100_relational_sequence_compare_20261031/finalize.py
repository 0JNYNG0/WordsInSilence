from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,urllib.request
U=Path(__file__).resolve().parent;ROOT=U.parents[3];OPS=U.parent;OLD=OPS/'word100_core_stage_evidence_20261031'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone(timedelta(hours=9))).isoformat();review_names=['SAVED_RELATIONAL_VERIFICATION.json','CHANGE_CONTROL_RANGE_VERIFICATION.json','CODE_REVIEW_RESOLUTION.json','INVARIANCE_CHECK.json']
for n in review_names:assert read(U/'review'/n)['status'].startswith('PASS'),n
ui=read(U/'UI_VERIFICATION.json');assert ui['status']=='PASS' and ui['html_sha256']==sha(U/'index.html') and ui['data_sha256']==sha(U/'REPORT_DATA.json')
frozen=read(U/'FROZEN_V2.json');assert sha(U/'FROZEN.json')==frozen['parent_sha256'];assert all(sha(p)==h for p,h in frozen['pins'].items())
protected=read(OPS/'word100_semantic_episode_pilot_20261031/FROZEN.json')['protected'];assert all(sha(p)==h for p,h in protected.items())
old=read(OLD/'COMPLETION.json');assert all(sha(OLD/n)==h for n,h in old['pins'].items())
write(U/'review/PROTECTED_CHECK.json',dict(status='PASS',protected_files=len(protected),previous_unit_files=len(old['pins']),mismatches=[]))
health=json.load(urllib.request.urlopen('http://127.0.0.1:8792/health',timeout=3));assert health['read_only'] and not health['inference']
result=read(U/'RESULT_SUMMARY.json');report='''목표와 이번 네 단계 작업 결과

목표는 여러 사람의 참고와 사용자의 손 모양·팔/몸 관계·앞뒤 순서를 비교해, 중요한 차이와 관측 부족을 구분할 근거를 만드는 것이다.
이번 완료는 공통 특징/참고 범위/고정 비교와 통제/표시·검산의 네 단계이며, 전체 교육 피드백 AI 완성은 아니다.
화면: http://127.0.0.1:8792/ (이전 http://127.0.0.1:8791/ 유지)

보라색 구간 설명
순종 개발 P036의89번 원본프레임=2.966667초에 붙인 전환 미확정 표시다.
약2.833333~3.2초(85~96f)의 손 모양·방향 전환 경계를 확정하지 못했다. 전환이 실제1프레임이라는 뜻도, 관절 미검출이라는 뜻도 아니다.
사전 설명에 담기지 않은 앞선 동작은 이 작은 표시가 아니라 얼굴 앞에 펼친 손을 가져가 유지하는 전체 선행 동작이다.

1. 공통 비교 항목과 지원 확인
공개4표현16영상,참고12/개발4를 유지. 좌우 손가락14투영각도, 실제손위치, 손의화면축, 팔꿈치·전완관계, 양손배치, 반대전완을따라손이어디있는지, 이동방향을13수치그룹으로 분리했다.
설명11항목을 연결한12개 상위공통개념과 수치13그룹은 같은 숫자의 모델이 아니다. 수치13은 좌우 이동방향을 별도 포함하고 순서개념은 별도진단으로 처리한다.
fit핵심후보440프레임 중 오른손각도427,팔관계440에 수치지원. 손내림으로ROI밖인왼손을 오류로 만들지 않는다.
원래 설명과 특정시점 연결은 표현별fit1명에집중돼있어,다른2명에게설명을정답으로복사하지않았다. 주먹/접촉/손바닥법선/깊이/정확한얼굴부위는 해당수치만으로확정하지않는다.

2. 사람별 참고 범위
참고3명을 각각유지하고 한사람을빼서나머지2명과비교한24쌍을 조건별로사용했다.
개별reference/core/group에 대해 다른fit2명 모두관측지원되는 경우에만 두차이의최댓값을작은경험범위로저장했다. DEV로범위나가중치를조정하지않았다.
이 범위는 한국수어에서허용되는정답폭이나새사람전체의정상범위가아니다. 유한한개발자료에서관측된차이일뿐이다.

3. 고정 비교와 통제
A기존저장R중심결과·경로재사용, B같은경로에양손/팔근거추가, C새13그룹균형DTW+같은근거. 새학습0,새검출0;수치비교/DTW계산만실행했다.
72개비교(조건별fit24/dev12). DEV의507개reference구간×특징에서 B범위안259/초과88/참고부족146/현재관측부족14, C범위안238/초과104/참고부족149/관측부족16.
C가더좋다는근거없어운영채택하지않는다. 범위초과는관측된차이후보이지틀린수어라는뜻이아니다.
순서제약없는별도창검색은원본query시간에서후보를찾는다. 단조DTW에서정렬된순서를다시읽어순서정답이라고주장하지않는다. 반복된비슷한자세의회차를혼동해원본동일비교P063에서도역순후보1개발생,교정판정미채택.
126개수치통제:원본/느린재표본에서R모양차이0(각16). R모양5f변경16모두수치차이,원래범위적용가능fit12중11초과. L관측있는12개모두수치차이지만fit관측9중3초과만발생. 손전체누락16모두근거부족.
5프레임삭제16에서수치차이는있으나fit12초과0. 해당시점이언어적으로필수라는답안이없으므로중요행동누락검출성능으로해석못한다.
단계블록교환은가능한14개에서13개역순후보(개발잘하다·조상은core1개여서swap미구성),원본역순오검출도있어교육피드백불가. 빠른재표본은원본시점이빠지는수치시험이며실제빠른촬영/재검출시험아니다. fit12범위초과는0.
기존시간경로유지+근거분리의진단방식이현재다음개발기준이다. 운영이나기존C/T3F모델을바꾸지않았다.

4. 표시·검산·시간
24개DEV참고/조건화면,13그룹그래프,양쪽영상재생,밝고어두운테마,좁은화면검증PASS. 실제앱화면자동화는이번에도사용하지않았으며Chrome로컬검사이다.
기존관절입력이준비된뒤3참고새비교는CPU0.234~0.409초/영상(1회). 관절검출·TCN·화면표시를합친실제게임전체시간은아니다.
별도코드검산:72쌍/126배열138505확인;변화량/범위165453확인;기대분모9568확인불일치0. 확대·이동/누락칸교란제어검사PASS. 이검사는실제체형·원근·촬영환경다양성을입증하지않는다.
검토중identity의PTS보존과재개해시검사보완,공유관측으로참고평균이달라지던change필드의새fixed-reference근거를별도작성했다. 최초코드/FROZEN/결과는보존했고가중치/기준/기존점수는변경하지않았다.
기존C/T3F/운영/32보호파일과이전보고서핀모두보존. 사용자·멘토자료0,다운로드0,외부전송0,카메라0,Git0.

다음 우선순위
반복되는자세를한장면의가까움으로고르지말고진입→자세→이탈의연결과각회차를함께유지하는방법을고정비교한다. 원본identity의역순오판부터원인분리해야한다.
왼손의짧은차이가범위에묻히는원인은구간전체평균·2D투영·다른사람변이로분리한다. 이번DEV점수에맞춰허용폭을좁히지않는다.
실제로구별에필요한완결동작이누락된것과속도·유지시간차이를구분할근거가있어야학습정답/교정정오를주장한다. 작은공개범위의기술개발은계속하되불확실성을정답으로채우지않는다.

담당:root구현/실행/화면/최종통합, rgb_labels_a지원·불변성검사, guide_stage_map출처/공통개념, rgb_labels_b별도코드·저장결과검산.
같은Codex의별도검산과표본검토는독립사람한국수어정답이아니다.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf8')
(U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="color-scheme" content="only light"><title>관계·순서 비교 결과</title><style>body{background:#fff;color:#17283b;font:18px/1.8 system-ui;max-width:1000px;margin:35px}a{color:#125f93}pre{white-space:pre-wrap;color:#17283b}</style><h1>손 모양·팔 관계·순서 비교</h1><p><a href="http://127.0.0.1:8792/">영상과 그래프로 결과 열기 →</a></p><pre>'+report.replace('&','&amp;').replace('<','&lt;')+'</pre></html>',encoding='utf8')
completion=dict(status='COMPLETE_FOUR_STAGE_RELATIONAL_DIAGNOSTIC_NOT_EDUCATIONAL_JUDGE',at=now,active=None,result_summary=result,new_neural_training=0,new_detection=0,comparison_pairs=72,control_cases=126,production_changed=False,user_mentor_used=False,report_url='http://127.0.0.1:8792/',read_only_report_server=health,previous_report_preserved='http://127.0.0.1:8791/',protected_unchanged=True,review_counts=[138505,165453,9568],pins={n:sha(U/n) for n in ['FROZEN.json','FROZEN_V2.json','relational.py','run_compare.py','REFERENCE_RANGES.json','DEVELOPMENT.json','CONTROL_SUMMARY.json','CONTROL_RANGE_DIAGNOSTIC.json','CHANGE_EVIDENCE.json','REPORT_DATA.json','index.html','RESULT_KO.txt']})
write(U/'COMPLETION.json',completion);write(U/'CURRENT_EXECUTION.json',dict(status=completion['status'],active=None,new_neural_training=0,report_server=health,exec_session=6746))
statepath=OPS/'STATE.json';state=read(statepath);state['updated_date_kst']=now;state['current_stage']='RELATIONAL_SEQUENCE_DIAGNOSTIC_COMPLETE';state[U.name]=completion;write(statepath,state)
eight='''### 2026-10-31 손 모양·팔 관계·순서 비교 4단계 완료 — 새 정렬/순서 교정 미채택

- `word100_relational_sequence_compare_20261031` 공개4표현16영상(fit12/dev4),13수치그룹/12상위개념·11설명연결. source직접fitanchor는표현별1명이라다인정답전파X. 새신경망학습/검출0,72비교+126수치통제실행.
- B기존저장경로+분리근거 vs C13그룹새경로,각조건fit24LOPO쌍만범위. DEV507구간×그룹 B안259/초과88/참고부족146/관측부족14; C안238/초과104/참고부족149/관측부족16. 새정렬우월근거없고운영미채택. 순서무제약창검색은identityP063에서도역순후보1로교정사용X.
- 통제R5f모양변경16수치차이·fit12중11범위초과; L관측12수치차이·fit관측9중3초과. R전체미관측16부족. 5f삭제fit12범위초과0(언어필수정답아님). 블록교환14중13역순후보이나정상오검출존재. 원본/느린재표본R차이0,빠른재표본은drop포함/fit범위초과0. 실제오류·실제속도정확도아님.
- 별도저장검산138505/새change·범위165453/기대분모9568불일치0,fit2변환/누락교란불변성PASS. 원FROZEN/코드/결과보존후V2는controls실행전identityPTS·재개검사복구/최대5창명확화. pair.changes는미사용,CHANGE_EVIDENCE는고정ref평균+지원제한별도.
- CPU저장입력3참고비교0.234~0.409초(전체게임아님). 새로컬읽기전용8792/exec6746는열람용유지,이전8791유지. Chrome24대응/13그룹/양쪽재생/테마PASS. C/T3F/운영/개인·멘토/32보호파일·이전결과유지. 정확순서/중요누락/교육정오AI완료X.

'''
nine='''### 2026-10-31 다음 단일 작업 — 반복 회차·짧은 특징의 진입/유지/이탈 구분

완료단위 `word100_relational_sequence_compare_20261031`의원본결과를재실행하지않고B기존경로+분리근거를개발기준으로유지한다. 다음은identityP063에서도역순후보가난평균자세검색의반복회차혼동을진입→자세→이탈의연결로구분하는고정검사다. 왼손짧은차이가fit범위에묻히는것은구간평균/투영/다인변이로원인분리하고DEV기준튜닝X. 5프레임삭제를중요동작누락정답으로승격X. source직접anchor한사람분포를다인학습정답으로전파X. 기존C/T3F/운영/개인보존,새다운로드/모델순회X. 보고서8792와이전8791은읽기전용열람용이므로health확인후중복실행금지. 과거완료비교재채점하지않는다.

'''
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=context.read_text(encoding='utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업';assert h8 in txt and h9 in txt
if eight.splitlines()[0] not in txt:txt=txt.replace(h8,h8+'\n\n'+eight,1)
if nine.splitlines()[0] not in txt:txt=txt.replace(h9,h9+'\n\n'+nine,1)
context.write_text(txt,encoding='utf8')
handoff='# WordsInSilence 최신 인수인계\n\n'+now+'\n\n목표: 한국수어 설명과 여러 사람의 참고 영상을 함께 이용해 손·모양·방향·몸관계·움직임·순서를 파악하고 중요한 차이와 관측 부족을 구분한다. 사용자 승인 네 단계 비교를 완료했으나 전체 교육 피드백 AI는 아직 미완성이다.\n\n'+eight+nine+'\n현재 active=null(계산종료). 열람서버8792 exec6746/PID'+str(health['pid'])+' 유지,이전8791 exec9048/PID52400보존. 실제health를먼저확인. Python word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. 로컬HTTP/Chrome은require_escalated필요. GPU모델실행없음,이번수치비교CPU.\n\n현재근거: COMPLETION/RESULT_KO/REPORT_DATA, FROZEN+FROZEN_V2, PAIR_PROVENANCE,72pairs/126controls, REFERENCE_RANGES/DEVELOPMENT/CONTROL_RANGE_DIAGNOSTIC/CHANGE_EVIDENCE, review검산4개,UI_VERIFICATION. 원본파일과최초동결보존,개인/멘토0,운영교체0. 구현root; 지원·불변성rgb_labels_a; 설명guide_stage_map; 산술rgb_labels_b. 같은Codex검산은독립사람한국수어정답이아님.\n'
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8');print('COMPLETE synced',health['pid'])
