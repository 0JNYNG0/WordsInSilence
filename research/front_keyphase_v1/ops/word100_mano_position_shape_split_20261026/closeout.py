from pathlib import Path
from datetime import datetime,timezone,timedelta
from html.parser import HTMLParser
import ast,json,hashlib
U=Path(__file__).resolve().parent;ROOT=U.parents[3];assert ROOT.name=='WordsInSilence'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  self.links += [v for k,v in attrs if k in ['href','src']]
l=Links();l.feed((U/'RESULT_KO.html').read_text('utf-8'))
for p in l.links:assert (U/p).is_file(),p
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'),str(p))
for p in U.glob('*.json'):read(p)
c=read(U/'COMPLETION.json');assert c['verification']['status']=='PASS'
f=read(U/'FROZEN.json')
for p in f['prior_pins']:assert sha(p['path'])==p['sha256']
for path,h in f['protected'].items():assert sha(path)==h
new=c['next_single_action'];now=datetime.now(timezone(timedelta(hours=9))).isoformat()
entry='''### 2026-10-26 HaMeR 위치·모양 근거 분리 — 추가 근거16건·새 좌표 복원0
- ops/word100_mano_position_shape_split_20261026: 공개TRAIN7영상4명5표현812프레임/저장HaMeR243요청(원래손없음57칸)+기존독재자접촉/추출겹침2장면. 원본전체640영상의손영역ShiTomasi/LK추적,현재팔연결·동결팔윤곽정보와정밀손끝정보분리. 기존좌우충돌관측은기준점제외,앞뒤.5초최근깨끗한관측만. 가려진손가락/정확한손바닥중심확정금지.
- 기존정밀근거63건보존:새영역도지지44+기존만19. 추가영역좌우보조16건,16원본시각검토에서보이는팔/손과제안측일치(동일Codex·독립GT아님). 15건은원래21점이있던장면,실제누락1건대지REAL06f36왼손도이전G/GT에이미HaMeR추정이있었다. 따라서새손복원0/좌표교체0/인식률향상미측정.
- 기존후원77~86의10좌우충돌미해결. f82전74영역12점중8추적(이전손끝2/5),후88역추적11중5로실패. 현재팔윤곽은서로다른뿌리의두후보모두그럴듯하게지지할수있어단독채택금지. 독재자접촉의새영역은오른손후보를왼영역과연결할모호함:현재팔불일치로보류,기존정밀양손근거유지. 추출오른손지지/왼손보류유지. 새영역은보조sidecar이며기존정밀정보대체없음.
- 원본관측14149점/정상교차20프레임840점/정상대조94프레임및기존배열·마스크모두보존. 이는보존계약이지새일반화정확도증거아님. 별도scalar3198경로3595단계1814판단7127영역거리검산PASS,계약7PASS,보호32/이전1963파일해시불변. 구현/시각/별도수치검산동일Codex·공개진단자료재사용. 7그림검토/HTML정적QA,브라우저렌더미검증.
- 실행32960종료0(약79.4초오프라인진단,저장추론재사용/미래영상사용이므로실시간FPS아님). 새학습/신경망추론/사용자·멘토자료접근/운영C·T3-F·웹·DTW변경0. RESULT/SEPARATED_EVIDENCE/VERIFICATION/VISUAL_REVIEW/COMPLETION/RESULT_KO.html참조. 진행중프로세스없음.

'''
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';handoff=ROOT/'NEW_CHAT_HANDOFF.md';state=U.parent/'STATE.json';backup=U/'before_closeout';backup.mkdir()
for p in [context,handoff,state]:(backup/p.name).write_bytes(p.read_bytes())
s=context.read_text('utf-8');assert '### 2026-10-26 HaMeR 위치·모양 근거 분리' not in s
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업';assert h8 in s and h9 in s
s=s.replace(h8+'\n',h8+'\n\n'+entry,1).replace(h9+'\n',h9+'\n\n### 2026-10-26 위치·모양 분리 이후 — 실제 손목 연결부 평가 기준\n'+new+'\n\n',1);context.write_text(s,encoding='utf-8')
st=read(state);st.update(updated_date_kst=now,current_stage=c['status'],next_single_action=new,latest_visual_review=str(U/'RESULT_KO.html'));st[U.name]={**c,'updated_at':now};write(state,st)
handoff.write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}
현재 목표: 손 영역/팔 근거와 정확한 손목·손가락 근거를 분리해 안전하게 복원후보를 검토한다. C·T3-F/운영보존, 사용자·멘토자료 접근·학습·튜닝·선택금지.
마지막 완료: {U}
상태: {c['status']}
RESULT_KO.html/txt, RESULT.json, SEPARATED_EVIDENCE.json, VERIFICATION.json, VISUAL_REVIEW.json, COMPLETION.json.

공개TRAIN7영상4명5표현812프레임/저장HaMeR243요청(원래누락57손칸)+기존300대조에서고정된독재자접촉/추출겹침2장면. 새300전체추론이나100단어전수아님. 새학습/신경망추론없음.
원본전체640너비이미지의손영역ShiTomasi최대32특징점/LK정역FB1.5px·오차20·6점/60%·변위잔차.10어깨,영역거리.12어깨. 기존image_tracking_pair상수재사용. 가장가까운앞뒤.5초깨끗한관측만. 시작점상대손영역제외. 이전identity_recheckV3의교차좌우중복관측격리앵커재사용. 현재팔.25/.10·이전동결팔윤곽근거와대조. 수치문턱검색/단어예외없음.
기존정밀손목+손끝4/5근거63건그대로보존. 새영역도지원44,기존만19,새영역만16. 새영역을기존근거의대체재로쓰지않음. 새16건원본·겹침그림시각검토제안측일치지만정밀손목/손가락GT아님. 15건원래full21점있음. 실제누락대지REAL06f36왼손1건도이전G/GT에이미HaMeR추정이채택되어있음. 이번은그위치근거추가. 새좌표복구0,인식률향상주장금지.
후원REAL01의77~86의10충돌여전히보류. f82기존왼손칸뿌리는왼팔,HaMeR오른손뿌리는오른팔에가까움;각각팔윤곽통과할수있어단독판정불가. 전74영역12점중8추적(지난정밀tips2/5),후88역추적11중5·실패. 단순영역추적도회전/겹침한계. 손이없다는판정이아님.
독재자REAL01f59새영역오른손query가왼영역지지함:현재오른팔과불일치해새제안보류. 원래정밀양손근거는유지. 추출REAL02f53은오른손지지/왼손미확정유지. 대지f57교차오른손영역근거추가. 기존기준배열/마스크/identityeligibility전부동일,기존관측14149점·교차20프레임840점·정상94프레임보존. 보존계약검증이지새후보모든배정정확도증거아님.
별도scalar/단조체인polygon검산3198경로3595단계1814판단7127영역거리PASS. 계약7PASS. 보호32/과거1963파일불변. 팔윤곽은과거독립수치검산코드재사용. 구현·시각·별도검산동일Codex,독립전문가GT아님. 공개진단재사용. 그림7장시각검토,HTML정적/링크검사. 브라우저렌더미확인(이전file정책우회없음).
실행32960종료0,79.4초는저장추론재사용한오프라인진단으로실시간FPS가아님. 이후프레임도사용. 진행프로세스없음. 사용자·멘토자료/카메라/다운로드/외부전송/Git/운영변경0.

다음 단일 작업: {new}

이전단위는읽기전용: word100_mano_identity_recheck_20261026 최종RESULT_V3. V1오염된앵커의왼손잘못재확인5→V3 0이지만실좌표교체는처음부터0. 단어모델오답5개교정뜻아님. duplicate_guard최종V2:추가중복7차단,기존잘못된뿌리미해결. 정확도감사 missing_hand_accuracy최종REVIEWED_SCORES만(손끝17범위/63불확실),초기22점은사용금지.
원본W:/sources/과거동결결과읽기전용. 원본후보를소거하거나추정관절을관측정답으로승격하지않는다.
''',encoding='utf-8')
assert new in handoff.read_text('utf-8') and new in context.read_text('utf-8');assert read(state)[U.name]['status']==c['status']
write(U/'CLOSEOUT_VERIFICATION.json',dict(status='PASS',at=now,links_checked=len(l.links),python_ast='PASS',json='PASS',context_state_handoff_synchronized=True,browser_render_checked=False))
print('State/context/handoff synchronized; static links',len(l.links))
