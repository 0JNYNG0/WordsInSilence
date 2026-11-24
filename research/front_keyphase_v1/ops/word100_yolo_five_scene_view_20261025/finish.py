from pathlib import Path
import json,datetime,shutil,re,subprocess
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
page=(U/'RESULT_KO.html').read_text('utf8');script=re.search(r'<script>(.*?)</script>',page,re.S).group(1)
(U/'viewer.js').write_text(script,encoding='utf8');subprocess.run(['node','--check',str(U/'viewer.js')],check=True)
assert read(U/'VALIDATION.json')['status']=='PASS'
(U/'before').mkdir()
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
at=datetime.datetime.now().astimezone().isoformat();key=U.name
state=read(O/'STATE.json');state[key]={'status':'VISUAL_REVIEW_READY','at_kst':at,'report':str(U/'RESULT_KO.html'),'scenes':5,'original_video_copies_hash_identical':5,'new_inference':0,'training':0,'runtime_changes':0,'user_recordings':0,'purpose':'Show prior results, no scientific-result changes'}
state['updated_date_kst']=at;state['latest_visual_review']=str(U/'RESULT_KO.html');write(O/'STATE.json',state)
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');assert key not in t
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
t=t.replace(h8,h8+f'\n\n### 2026-10-25 보류5장면 사용자 직접 확인 화면 준비\n- `research/front_keyphase_v1/ops/{key}/RESULT_KO.html`: 원본전체영상5개, 문제시점이동·앞뒤0.5배속·프레임이동·동일시점4조건비교·확대후보전체·전체화면/확대전환. 기존결과의표시만추가, 새추론/학습/운영교체/사용자촬영접근없음. 원본복사해시5개일치·링크·JS문법확인. 보류5개미해결판정유지.\n',1)
t=t.replace(h9,h9+'\n\n### 2026-10-25 보류5장면 시각 확인\n\n사용자가 준비된5장면의 원본과 검출결과를 직접 확인할 수 있도록 제공한다. 연구상의 다음작업과 제한은 아래 선택적YOLO검증이후 계획을 그대로 유지한다. 시각자료를 새성능개선이나정밀정답으로해석하지않는다.\n',1);p.write_text(t,encoding='utf8')
p=ROOT/'NEW_CHAT_HANDOFF.md';t=p.read_text('utf-8-sig');t=t.replace('# WordsInSilence 최신 인수인계','# WordsInSilence 최신 인수인계\n\n최신 표시 작업 '+at+'\n사용자 요청: 보류5장면 직접 확인. '+str(U/'RESULT_KO.html')+' 준비. 원본영상·4조건 정지화면·모든 확대후보·보류이유, 느린재생과문제시점이동 제공. 새추론/학습/결과변경없음. 연구상태/다음계획은 아래 유지.\n',1);p.write_text(t,encoding='utf8')
v=read(U/'VALIDATION.json');v['javascript_syntax']='PASS';v['documents_synced']=True;write(U/'VALIDATION.json',v)
print('READY',U/'RESULT_KO.html')
