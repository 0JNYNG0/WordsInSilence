from pathlib import Path
from datetime import datetime,timezone,timedelta
from html.parser import HTMLParser
import ast,json
U=Path(__file__).resolve().parent;ROOT=U.parents[3];assert ROOT.name=='WordsInSilence'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['href','src']:self.links.append(v)
l=Links();l.feed((U/'RESULT_KO.html').read_text('utf-8'))
for p in l.links:assert (U/p).is_file(),p
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'),str(p))
for p in U.glob('*.json'):read(p)
c=read(U/'COMPLETION.json');assert c['verification']['status']=='PASS';now=datetime.now(timezone(timedelta(hours=9))).isoformat()
next_action='현재 프레임의 팔→손목→손바닥 연결과 손 영역 대응으로 위치·좌우 신원을 확인하는 근거를, 손가락 모양의 정밀 추적 근거와 분리하는 공개 연구 후보를 검토한다. 손끝 추적이 부족해도 남은 위치 근거는 보존하되 영역 추적만으로 가려진 손가락을 확정하지 않는다. 기존 10충돌·정상교차·독재자 접촉·추출 겹침·손회전을 공통 대조로 고정하고 같은 영상의 앞뒤 정보는 신원오염 없는 기준점에서만 활용한다. 미해결이면 원본과 후보를 보존하고 보류한다. C·T3-F/운영 유지, 사용자·멘토 촬영 학습·튜닝·선택 금지, 추가학습·운영 DTW 연결 없음.'
entry='''### 2026-10-26 HaMeR 팔·앞뒤 영상 신원 재확인 — 안전근거 보완·좌표교체0
- ops/word100_mano_identity_recheck_20261026: 공개7영상4명5표현812프레임, 저장HaMeR243요청/57빈손칸. 원본전체RGB에서앞뒤0.5초가까운팔지원관측을정·역추적. 새신경망추론/학습0,미래정보사용하는촬영후진단.
- V1에서잘못된기존좌우관측을기준점으로써후원79~83의5프레임을왼손으로재확인하는오류발견. V3은알려진서로다른좌우칸충돌을기준점에서제외해5→0. V1도실제좌표교체는0이므로좌표오류5개교정이라는뜻아님. V2의동일좌우재확인까지보류하던범위오류도V3에서수정,전체이력보존·임계값완화없음.
- 손목근거실패인손끝근접가설을정상별개손의경쟁자로쓰지않도록수정. 독재자정상접촉진단[-1,1]→[0,1],관절그대로. 추출겹침은[-1,1]로왼손불확실/오른손지지,원본교정없음. 지난300영상중이2원본장면만추가추적,300전체재실행아님.
- 실제충돌10프레임=지난신규중복추가보류7+기존보류3(f78/85/86). 최종좌표교체0. f82오른손이전74에서손끝2/5유효,이후88역추적1/5·손목근거없음. 빈점을메우거나문턱을낮추지않고좌우미확정유지. 충돌양쪽20손칸비교가능상태별도보류(20개의보이는손삭제아님),원본과G배열전체보존.
- 기존14149점/정상교차20프레임840점/대조94프레임결과와비교가능상태유지. 별도scalar637추적경로1949단계/257판단검산PASS,계약7PASS. 보호32/과거609파일불변. 같은Codex구현·시각·별도검산,독립전문가GT/미사용자료평가아님.
- 5개설명이미지시각검토+지난단위시각근거재사용. HTML정적/링크/문법확인,브라우저렌더미확인. C·T3-F·운영웹DTW·사용자자료변경0. 실행91306/50038/3165종료0,진행프로세스없음. RESULT_V3/VERIFICATION/COMPLETION/RESULT_KO.html참조.

'''
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';handoff=ROOT/'NEW_CHAT_HANDOFF.md';state=U.parent/'STATE.json'
before=U/'before_closeout';before.mkdir()
for p in [context,handoff,state]:(before/p.name).write_bytes(p.read_bytes())
s=context.read_text('utf-8');assert '### 2026-10-26 HaMeR 팔·앞뒤 영상 신원 재확인' not in s
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
s=s.replace(h8+'\n',h8+'\n\n'+entry,1).replace(h9+'\n',h9+'\n\n### 2026-10-26 신원 재확인 이후 — 위치와 정밀 모양의 선택 근거 분리\n'+next_action+'\n\n',1)
context.write_text(s,encoding='utf-8')
st=read(state);st.update(updated_date_kst=now,current_stage=c['status'],next_single_action=next_action,latest_visual_review=str(U/'RESULT_KO.html'));st[U.name]={**c,'unit':str(U),'updated_at':now,'next_single_action':next_action};write(state,st)
handoff.write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}
현재 목표: 기존 손과HaMeR같은손충돌에서 위치/좌우근거와정밀손가락근거를분리. C·T3-F·운영유지,사용자·멘토촬영학습/튜닝/선택금지.
마지막 완료: {U}
RESULT_KO.html/txt,RESULT_V3.json,VERIFICATION.json,COMPLETION.json,VISUAL_REVIEW.json.

최종V3연구보관/운영미적용/새좌표채택0. 실제좌표개선·인식률향상주장금지.
공개7영상4명5표현812프레임,기존HaMeR243요청/57빈손칸재사용. 새신경망추론/학습0. 원본전체RGB640너비로앞뒤0.5초가장가까운팔지원관측을현재까지LK정역추적. 이후프레임쓰므로오프라인/촬영후검증,실시간성능아님. .25/.10팔,.12일치,.5경쟁비율,FB1.5px/오차20,4/5손끝고정. 시간창.5초는공학적진단가설,성공에맞춘튜닝없음.
V1은잘못배정된이웃기존관측을기준점으로삼아후원79~83의5프레임왼손신원을내부재확인. V3은교차좌우같은손충돌관측기준점제외로5→0. V1도실제좌표교체는0이므로5개좌표수정으로오해금지.
V2모든일치검출보류는정상동일좌우재확인까지제외하는범위오류. V3은existing_side!=requested_side에한해양쪽기준점제외. V1/V2/V3산출물·FROZEN모두보존,기준완화없음.
V3또한손목근거실패상대가설이손끝근접만으로정상별개손을막지않도록함. 독재자근접주먹진단V1[-1,1]→V3[0,1],추출[-1,1]유지. 두Broad사례는지난300좌표대조중2원본장면추가검사이며300새추적/HaMeR재실행아님. Broad원본수정없음,가려진왼손모양확정금지.
중복신원충돌10프레임(후원77~86)=지난추가보류7+그전부터보류3(78/85/86),새오류3증가아님. 후보교체0. f82오른손전74에서추적손끝2/5,후88에서1/5및손목실패. 해당손이시각적으로없다는뜻아님;단순영상점추적의한계. 신원확정못하므로후보강제채택안함. R/G원본좌표·마스크전체동일,identity_eligible만10프레임양쪽20손칸보류. rawG/후보보존. 사용자피드백DTW아직연결안함.
보존14149점/지정교차20프레임840점/정상대조94프레임. 별도scalar637경로1949단계·257판단검산PASS,계약7PASS. 보호32/과거609파일불변. 구현·시각·검산동일Codex,독립전문가GT/미사용자료평가아님. 시각5그림확인+지난13장면근거재사용. HTML정적링크/문법확인,브라우저렌더미확인(기존file정책우회없음).
실행91306(V1)/50038(V2)/3165(V3)종료0. 검산/보고종료0. 진행프로세스없음. 새오류발견을지우거나이전결과덮어쓰기없음. 카메라/다운로드/외부전송/Git/운영교체/사용자촬영접근0.

다음 한 작업: {next_action}

이전 word100_mano_duplicate_guard_20261025 G/GT: H33→G26,HT31→GT24의중복추가7보류. 같은실제손두번세는것만막음,기존잘못된왼손칸/손목/숨은주먹미해결. 해당최종V2보존.
그전정밀감사 word100_mano_missing_hand_accuracy_20261025는REVIEWED_SCORES만최종(손끝17범위/63불확실). 초기22점은E11번호불확실로정정되어성능근거금지. 대략RGB범위는3DGT아님.
원본W:/sources/과거동결결과읽기전용. 다음에도근거없는매끄러운복원/손가락정답주장금지.
''',encoding='utf-8')
assert read(state)[U.name]['report']==str(U/'RESULT_KO.html')
assert next_action in handoff.read_text('utf-8') and next_action in context.read_text('utf-8')
write(U/'CLOSEOUT_VERIFICATION.json',{'status':'PASS','at':now,'links_checked':len(l.links),'python_ast':'PASS','json':'PASS','context_state_handoff_synchronized':True,'browser_render_checked':False})
print('Closeout complete',len(l.links),'links verified')
