"""Synchronize the exact resumed unit; only root writes project state."""
from pathlib import Path
import sys,json,datetime
U=Path(__file__).resolve().parent;H=U.parent;G=H.parent;O=G.parent;root=O.parents[2]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def write(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8');q.replace(p)
at=datetime.datetime.now().astimezone().isoformat()
mode=sys.argv[1]; session=int(sys.argv[2]) if len(sys.argv)>2 else None
status={'prepare':'ACTIVE_BODY_PROGRESS_INPUT_REVIEW','train':'ACTIVE_BODY_PROGRESS_TRAINING','evaluate':'ACTIVE_BODY_PROGRESS_EVALUATION'}[mode]
summary='추가8 공개 FIT 영상의 dense/overview 실제 RGB를 확인했다. 기존16의 저장 RGB검토와 합쳐24영상(6명)의 몸 움직임/유지 학습을 준비했다. 원주석 training_allowed=false는 보존하고 새 body-only admission을 기록했다. 애매한 끝 경계는 유지 음성에서 제외하며 정상 유지 자체를 오답으로 가르치지 않는다.'
nxt='body_progress_witness_learning의 명시 주석승인·입력/보호해시 검증 후 고정3seed 학습. 기존 L의 두 고정 구간 선택은 바꾸지 않고 새 진행근거만 원래 PTS에 연결해 정상/속도/역순/관측누락 비교 및 별도 검산한다. 실패원인에 맞춰 그래프를 이어가며 G2 통과 전 웹/최종8은 보류한다.'
e=read(G/'CURRENT_EXECUTION.json');e.update(at=at,status=status,active_unit=str(U),current_command=mode,session_id=session,jobs=[],training_running=mode=='train',process_running=session is not None,next=nxt,last_completed=str(U/'SOURCE_ADMISSION.json'));write(G/'CURRENT_EXECUTION.json',e)
st=read(O/'STATE.json');st.update(updated_at=at,last_updated=at,next_single_action=nxt,next_single_task=nxt)
st['word100_web_completion_graph_20261111'].update(at=at,status=status,active_unit=str(U),training_running=mode=='train',progress=summary,remaining=nxt,latest_evidence=str(U/'SOURCE_ADMISSION.json'));write(O/'STATE.json',st)
s=read(H/'SUBGRAPH.json');s.update(at=at,status=status)
for n in s['nodes']:
    if n['id']=='C5':n.update(status=status,task=nxt,evidence=[str(U/'SOURCE_ADMISSION.json')])
write(H/'SUBGRAPH.json',s)
p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 추가8 주석 확인 · 움직임과 자세 유지 분리 학습 재개'
if title not in t:
    for head,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:
        assert head in t;t=t.replace(head,head+'\n\n'+title+'\n'+body+'\n근거: `'+str(U.relative_to(root)).replace('\\','/')+'/SOURCE_ADMISSION.json`.\n',1)
    p.write_text(t,'utf-8')
(root/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 2026-11-13 움직임/유지 학습 재개

{summary}

{nxt}

현재 {status}, exec session {session}. CURRENT_EXECUTION과 실제 완료파일/프로세스를 먼저 확인해 중복 실행하지 않는다. body_progress_witness_learning 폴더를 이하 BP로 부른다(이전 다른 M실험과 혼동 금지). PY=ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8.

기존 C5 402는 old progress가 양성인 hold표본수이며 모두 강음성이라는 뜻이 아니다. CORRECTION_BOUNDARY_HOLD.json 보존. 추가8은 REAL03/07, 기존16은 REAL02/04/05/06. 원본 false를 수정하지 않고 SOURCE_ADMISSION+ADDITIONAL_RGB_REVIEW를 새범위로 연결. 프로브08/09는 학습에서 제외. Codex RGB검토와 별도코드검산은 독립 사람/KSL정답이 아니다.

BP는 부모 boundary_extended의150몸context->96->5 진행출력을 복사한 별도 판정부. 원래822양성 보존, 경계음성102 제외, 유지648관측 적격·72경계보류(뷰×정역×family 반복수). 의미적 완료/오답 정답은 아니다. 기존100 공통학습/손모양/방향/TCN C·T3F/웹 불변.

L 완료결과는 FIT3seed460/460, DEV 정상/느림9/9·빠름단독8/9/반반7/9. 순서편집 구간18/18·역순9/9, 정상의미불변. 약33ms전체범위중첩은실패로보존. 새 BP는 extent/관측움직임순서/완료를 구별하며 새 허용문턱으로실패를지우지않는다. C5/progress_order.py 모든양성섬유지,2개이상원본PTS,기대순서금지,completion None.

G0/G1PASS,G2REWORK,G3이후보류. 완료실험재학습/재추론금지. 사용자/멘토/최종8·카메라·다운로드·외부전송·Git·운영교체금지. protected477 전후검산. 첫실패보존. 대기/사용량중단시다음단계와실제프로세스기록.
''','utf-8')
print(status,session)
