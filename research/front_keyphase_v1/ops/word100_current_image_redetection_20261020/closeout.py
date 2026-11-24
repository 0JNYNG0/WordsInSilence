"""Synchronize current-state documents after evidence is complete."""
import json,hashlib,re,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
f=read(U/'FROZEN.json');a=read(U/'FINAL_AUDIT.json')
assert a['status'].startswith('COMPLETE_')
for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
    assert sha(p)==sha(U/'before'/p.name),'Concurrent root document change: '+str(p)
page=(U/'RESULT_KO.html').read_text('utf8')
links=re.findall(r'(?:src|href)="([^"]+)"',page)
local=[x for x in links if not x.startswith(('http:','https:','#'))]
for link in local:assert (U/link).is_file(),link
for i in range(1,32):assert (U/'evidence'/f'added_{i:02d}.jpg').is_file()
script=re.search(r'<script>(.*?)</script>',page,re.S).group(1)
js=U/'report_script_check.js';js.write_text(script,encoding='utf8')
q=subprocess.run(['C:/Program Files/nodejs/node.exe','--check',str(js)],capture_output=True,text=True)
assert q.returncode==0,q.stderr
for p in U.glob('*.json'):read(p)
next_action=('현재 이미지 재검출 비교 결과를 사용자와 검토한다. 보이는 손 추가 후보는 전체294/국소303손시점'
             '(본동작76/77), 그러나 기존 오배정과 같은 실제 손 중복1건, 형상 왜곡/화면 밖 손가락이 남아 미채택. '
             '다음 제안은 현재 팔·손 근거로 기존/신규 후보의 동일 실제 손 충돌을 분리하고 관절별 유효성을 확인하는 '
             '공통 후보 비교다. 정상 근접·교차를 단순 거리/상자 겹침으로 제거하지 않는다. '
             'C·T3-F/운영DTW/웹 유지, 사용자·멘토 촬영 학습/튜닝/선택 금지. 추가 학습은 결과 확인 후.')
title='### 2026-10-20 현재 이미지 손 재검출 비교 완료 — 추가 관측 가능성·중복/형상 반례, 미채택'
status=f"""{title}
- 단위 `research/front_keyphase_v1/ops/{U.name}`. `RESULT_KO.html`/`RESULT_KO.txt`/`FINAL_AUDIT.json`. Codex 로컬 구현·실행·별도 저장 산술검산·시각 검토 완료. 다른 사람/에이전트 독립 검토 아님.
- 공개 TRAIN36영상/23단어/9촬영자/4641프레임, 같은 sensor 픽셀/기존 C 관측. 기존 로컬 MediaPipe Hand Landmarker IMAGE 모드로 전체 현재 이미지 및 현재 팔 근처 국소영역을 별도 비교. 새 모델 다운로드/분류재평가/학습 없음.
- 초기 설명 정정: 같은36원본 전체(masked의 raw) 첫 프레임은 모두 손 후보2개. sensor 제한 화면은35개0손/1개1손. 알려진12좌우 오류는 sensor, 그중첫프레임1/이후11. 원본정지자세손을모두처음부터못찾았다는해석은부정확. 후보2개가좌우·정밀관절정답이라는뜻도아님. C는masked/sensor혼합학습.
- 동일픽셀365프레임/3영상/2단어를 VIDEO1배/0.75배/0.5배/1배재시험, timestamp만 변경: 손수/라벨/좌표차0. 원추출은이미프레임순차처리. 실제빠른촬영의흔들림/촬영FPS/실시간지연문제와재생속도변경을구분. 모든빠른동작개선여부시험아님.
- 추가후보 전체294/국소303손시점, 주석본동작76/77. 공통289/전체만5/국소만14,고유308손시점31시트전부검토. 정상42/교차54기존값그대로(덮어쓰지않는설계);알려진좌우오류12그대로. 숫자는관절정확도/수어정답률아님.
- 중요한반례: target307 공항09 f41, 기존실제오른손이왼손으로오배정된채새오른손추가→같은손중복. 중심거리guard가왜곡차이로통과. 결과그대로보존,임계값재탐색/조용한제외없음.
- 추출09 f81손가락선붕괴,대지05 f83/84·06 f66손목/손바닥연장형상반례. 전체26/국소32손시점은손바닥안이나손가락일부sensor밖. 손존재/좌우/중심/각손가락유효성을분리해야함. 고유308중같은손중복1이있어새관측을자동입력채택하지않음.
- 계약11PASS/저장판단4820검산/동결핀·32보호파일유지. VERIFY의pending_visual은검산당시상태,최종VISUAL_REVIEW/FINAL_AUDIT로검토완료. 알려진정상보존은관측고정설계이며보조합성출력무오류보장아님.
- 전체이미지검출자체중앙22.28ms,p95 24.31ms;국소1091호출평균11.91ms. Pose/읽기/TCN제외,실시간전체FPS아님. 매프레임둘다운영적용하지않음.
- C·T3-F/운영DTW/웹/사용자촬영불변. 사용자자료접근0,학습0,분류재평가0,다운로드0,채택0. 새후보코드만별도연구폴더. 실행종료/잠금없음/최초오류없음.
- 보고서이미지·상대링크·JS구문검사. 과거file URL정책차단우회없음,브라우저자동미리보기미확인. 다음은결과검토후중복통합/관절별근거보완계획,학습은아직금지.

"""
cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';cs=cp.read_text('utf-8-sig')
for heading,new in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',status),('## 9. 다음 단일 작업',title+'\n'+next_action+'\n\n')]:
    assert cs.count(heading)==1
    cs=cs.replace(heading+'\n',heading+'\n\n'+new,1)
cp.write_text(cs,encoding='utf8')
state=read(O/'STATE.json');state['updated_date_kst']='2026-10-20'
state['current_stage']='WORD100_CURRENT_IMAGE_REDETECTION_REVIEW'
state['next_single_action']=next_action
state[U.name]=dict(a,unit=U.name,final_audit='FINAL_AUDIT.json',visual_review='VISUAL_REVIEW.json',
                   next_single_action=next_action,execution_complete=True,running=False)
write(O/'STATE.json',state)
handoff=f"""# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 보이는 손의 검출 누락을 줄이면서 실제 손·좌우·정밀 손가락 관측을 혼동하지 않기.
마지막 완료: 현재 이미지 재검출 공개 TRAIN36 비교와 초기검출/배속 감사. 추가 후보는 얻었지만 중복·형상 반례로 미채택.
다음 한 작업: 결과를 사용자와 검토 후 기존/신규 손 충돌 및 관절별 유효성 보완 계획. 추가 학습은 아직 하지 않는다.

- 단위 `{U.as_posix()}`. `RESULT_KO.html`, `RESULT_KO.txt`, `FINAL_AUDIT.json`.
- 공개 TRAIN36/23단어/9명/4641프레임. 사용자·멘토/DEV/보류14~18 미접근. 기존 같은 sensor 픽셀과 C 저장값. 공식 로컬 Hand Landmarker 가중치 IMAGE 재검출, 새 다운로드0.
- 원본전체 최초프레임은36모두손후보2. sensor제한화면은35개0손/1개1손. 이전초기설명은경로를혼동하지말것. 알려진sensor오류12중첫프레임1/이후11;모두원본첫프레임오류가아님. C는full검출후ROI(masked)와sensor혼합,전체검출방식폐기아님. 후보2개가정밀좌우정답이라는보장아님.
- 배속:3영상/2단어/365동일픽셀프레임,VIDEO timestamp만1/.75/.5/1변경,수/라벨/XY차0. 보간·중복프레임없음. 실제촬영속도·흔들림/FPS와재생속도별개.
- 전체IMAGE294/국소303추가후보손시점(본동작76/77). 289공통+5전체만+14국소만=308고유. 전31시트시각검토+겹침8/형상3확대. 단어정답률·정밀관절성공수아님.
- 기존정상42/교차54보존은기존값무수정설계. 기존좌우오류12유지. target307 공항09f41 기존오른손→왼손오배정과새오른손이동일실제손중복. 거리guard불충분. 결과숨기거나계수조정하지않음.
- 형상반례:target227추출09f81손가락선붕괴;141/142대지05f83/84,159대지06f66손목/손바닥길어짐. 전체26/국소32후보손가락일부화면밖(중복가능). 자동입력추가/각도확정금지.
- 계약11PASS,별도저장판단4820검산,32보호파일/동결핀유지. same Codex 구현·별도검산·시각검토,독립사람/에이전트아님. `VERIFY.json` pending_visual은당시기록,`VISUAL_REVIEW.json`과`FINAL_AUDIT.json`이최종.
- 전체IMAGE중앙22.28ms/p95 24.31ms,국소1091호출평균11.91ms(검출만). sensor밖픽셀/미래픽셀사용0. 운영단일경로로확정한것아님.
- C·T3-F6가중치포함32보호파일,운영DTW/웹그대로. 학습/분류재평가/사용자접근/다운로드/카메라/Git0. 실행완료/잠금없음/최초오류없음. 완료검출재실행하지말것.
- 새단위보고서이미지·상대링크·JS구문확인;file URL정책차단우회없음,브라우저미리보기미확인. 로컬링크제공.

{next_action}
"""
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
assert read(O/'STATE.json')[U.name]['training']==0
write(U/'DELIVERY_CHECK.json',dict(status='PASS',at=at,local_links=len(local),gallery_sheets=31,
     javascript_syntax=True,root_documents_synchronized=True,protected_files=32,
     frozen_sources_preserved=True,browser_preview=False,
     browser_reason='Prior file URL policy block; no alternate-route workaround',
     visual_assets='31 contact sheets, initial source/sensor image, 11 detailed duplicate/shape checks inspected',
     current_model_and_web_changes=0))
print('CLOSEOUT PASS',len(local),'local links;31 sheets;32 protected')
