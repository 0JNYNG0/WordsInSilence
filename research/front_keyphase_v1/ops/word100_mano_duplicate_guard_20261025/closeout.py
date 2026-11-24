from pathlib import Path
from datetime import datetime,timezone,timedelta
from html.parser import HTMLParser
import json,ast,hashlib
U=Path(__file__).resolve().parent; ROOT=U.parents[3];assert ROOT.name=='WordsInSilence'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['src','href']:self.links.append(v)
parser=Links();parser.feed((U/'RESULT_KO.html').read_text('utf-8'))
for ref in parser.links:assert (U/ref).is_file(),ref
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'),str(p))
for p in U.glob('*.json'):read(p)
v=read(U/'VERIFICATION.json');assert v['status']=='PASS';r=read(U/'RESULT_V2.json');c=read(U/'COMPLETION.json')
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
next_action='중복 추가 방지 결과를 바탕으로 기존 손과 새 HaMeR 후보가 충돌할 때 현재 영상의 팔 연결과 앞뒤 관측으로 어느 실제 손인지 다시 확인하는 공통 후보를 검토한다. 기존 좌우 칸·손목 오류를 그대로 정답으로 두지 않되, 정상 교차/E01·E16/독재자 접촉과 추출 불확실 사례를 함께 보존 검증한다. 원본 좌표와 두 가설은 보존하고 신뢰가 부족한 위치·손가락 근거를 분리한다. 단어별 예외·사용자자료 튜닝 금지, C·T3-F·운영 유지, 추가학습·운영 DTW 연결은 아직 하지 않는다.'
entry='''### 2026-10-25 HaMeR 동일 손 중복 추가 방지 — 공개 재생 검증 완료·운영 미적용
- ops/word100_mano_duplicate_guard_20261025: 전체 영상 좌표에서 기존손과HaMeR의 손가락 마디·끝을 비교. 다른팔라벨이라는 이유의 중복검사 우회를 막는 별도G/GT후보. 충돌시새추정추가만보류,기존/원본후보보존,양쪽identity_conflict기록. 기존값을정답으로확정하지않음.
- 공개7영상4명5표현812프레임/빈손칸57에서H33→G26,HT31→GT24. 후원REAL01 f77,79~84의같은펼친손중복추가7프레임보류. 기존왼손칸오류·손목치우침·숨은주먹복원은미해결. 정확복원수/TCN인식률향상아님.
- 공개100단어·TRAIN3명씩300영상/36869프레임/완전양손34466쌍의저장MP좌표대조. HaMeR300영상재추론아님. V1정상두주먹접촉(독재자)의심→V2손끝엄지·검지등상호최소거리일대일대응추가로유지. .08어깨·4/5조건유지,단어예외없음. V1/V2모두보존. 남은추출f53은두손교차+기존좌표중복/혼합가능성불확실. 동일자료개발검사,독립GT오류율아님.
- 기존14149점/지정교차20프레임840점/정상대조94프레임유지;E01/E16정상겹침보완유지,E02/E06기존보류유지. 별도scalar417후보쌍+34466대조검산PASS,V1자체11/V2자체9PASS. 기존보호32/과거결과564파일불변. 같은Codex구현·시각·별도검산이며전문가독립GT아님.
- 13장면(변경7/대조4/확대자료2)+확대자료전후패널시각검토. HTML/링크/문법정적확인,브라우저렌더미확인. 최초검산role이름오기만수정·기록보존. 새추론/학습/사용자자료접근/운영교체0. 실행완료/진행프로세스없음. 근거 RESULT_V2/VERIFICATION/VISUAL_REVIEW/COMPLETION/RESULT_KO.html.

'''
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';handoff=ROOT/'NEW_CHAT_HANDOFF.md';state=U.parent/'STATE.json'
before=U/'before_closeout';before.mkdir(exist_ok=False)
for p in [context,handoff,state]:(before/p.name).write_bytes(p.read_bytes())
s=context.read_text('utf-8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
assert '### 2026-10-25 HaMeR 동일 손 중복 추가 방지' not in s
s=s.replace(h8+'\n',h8+'\n\n'+entry,1)
s=s.replace(h9+'\n',h9+'\n\n### 2026-10-25 중복 추가 방지 이후 — 남은 신원·손목 충돌 분리\n'+next_action+'\n\n',1)
context.write_text(s,encoding='utf-8')
st=read(state);key=U.name
st['updated_date_kst']=now;st['current_stage']='MANO_DUPLICATE_GUARD_VERIFIED_EXISTING_IDENTITY_UNRESOLVED_NOT_DEPLOYED';st['latest_visual_review']=str(U/'RESULT_KO.html');st['next_single_action']=next_action
st[key]={**c,'updated_at':now,'unit':str(U),'verification_summary':v,'C_T3F_change':False,'running_processes':[],'next_single_action':next_action}
dump(state,st)
handoff.write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}
현재 목표: 공개 수어에서 기존 손+HaMeR 통합의 동일손 중복을 방지하고, 남은 좌우 신원·손목 오류를 분리 검증. C·T3-F/운영 유지, 사용자·멘토 촬영 학습·튜닝·선택 금지.
마지막 완료: {U}
RESULT_KO.html/txt, RESULT_V2.json, VERIFICATION.json, VISUAL_REVIEW.json, COMPLETION.json.

후보V2완료·연구보관·운영미적용. 기존7영상4명5표현812프레임/57빈손칸의저장HaMeR243요청재사용, 새추론/학습0. G/GT는H/HT에공통손가락중복검사추가. 단어/사람/프레임입력없음.
H33→G26,HT31→GT24. 후원REAL01 f77,79,80,81,82,83,84의같은펼친손중복추가7프레임을보류. 기존잘못된왼손칸/손목위치는남음. 청록기존뼈대는정답아님;원본후보보존,양쪽identity_conflict표시. 손복원완료/인식률개선이라고보고금지.
공통조건: .08×어깨너비 이내 손가락마디중앙거리+손끝거리,5중4손가락;V2각손끝이상대동일손가락과엄격한상호최소거리일대일대응추가. 동률/손끝부족은새중복확정안함,기존gate는유지. 전면가림/뭉개짐이면놓칠가능성있음.
V1가넓은대조의독재자두주먹접촉을과도보류의심→원본전후확인후V2일대일대응추가로정상유지. FROZEN_V2설명/V1원본결과보존,임계값조정/단어예외없음.
확대대조100단어TRAIN3명300영상/36869프레임/34466완전양손쌍:V1의심2→V2의심1(추출REAL02f53). 저장MP좌표쌍대조이며HaMeR300재추론/전수RGBGT아님. 추출실제양손교차이나좌표가앞손에겹침,숨은손가락신원미확정. 같은자료개발검사라독립최종성능아님.
기존14149점/교차20프레임840점/정상대조94프레임보존, E01/E16정상겹침보완유지,E02/E06기존보류/과도보류가능성그대로,E15손목오차그대로. 보호32/과거564파일해시동일. V1/V2주비교배열전부같음확인하여V1비교이미지재사용.
구현·시각검토Codex,별도scalar검산도같은Codex(417후보쌍+34466대조쌍),전문가독립GT아님. V1자체11/V2자체9PASS. 13장면+확대자료2전후패널시각확인. HTML정적·이미지·링크확인;브라우저렌더미검증(이전file자동화차단우회없음).
최초검산36646은manifest정상role이름오기로실패. 검산기만existing_observation_control로수정후종료0. VERIFICATION_FIRST_ERROR.txt보존. 입력/후보/동결결과변경없음. 준비55354/재생68756/시각85869/V2재생34568모두종료0,진행프로세스없음.

다음 한 작업: {next_action}

이전손목/손끝정밀감사 word100_mano_missing_hand_accuracy_20261025는REVIEWED_SCORES.json만최종(17손끝범위/63불확실). 초기22점은E11손가락번호불확실로보류하여성능근거금지. 대략RGB범위는3DGT아님. 과거FreiHAND대체32정지사진결과와이번실제누락시험혼동금지.
카메라/다운로드/외부전송/Git/운영교체없음. 원본W:/sources/과거동결파일읽기전용. 사용자자료접근0.
''',encoding='utf-8')
assert read(state)[key]['report']==str(U/'RESULT_KO.html')
assert next_action in handoff.read_text('utf-8') and next_action in context.read_text('utf-8')
dump(U/'CLOSEOUT_VERIFICATION.json',{'status':'PASS','at':now,'local_links_checked':len(parser.links),'python_ast':'PASS','top_level_json':'PASS','context_state_handoff_synchronized':True,'browser_render_checked':False,'source_image_review_count':13})
print('Closeout/static verification PASS',len(parser.links),'local links')
