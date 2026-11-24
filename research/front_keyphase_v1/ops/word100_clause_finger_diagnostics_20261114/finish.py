from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,ast,hashlib
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=read(U/'VERIFIED.json');assert v['status']=='PASS_LIMITED_RESEARCH'
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8-sig'))
at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
next_action='손가락별 판정부가 새로 틀린 장면의 관측·정답·시간 문맥을 분리 감사하고, 검증자료를 학습으로 옮기지 않으면서 부족한 손가락 조합의 공개 FIT 정답을 보강한다. 새 촬영 웹 교체 전 별도 손가락 오류 검증과 현재 통과한 방향 보완의 실제 입력 연결을 확인한다. G6 사용자 촬영은 보류 유지.'
summary='설명 7행동·14문구를 관측 구간과 연결하는 계약을 정리하고 공개 손 이미지50개를 확인했다. 손가락별 시간합성곱+GRU 학습과 기존 움직임 가지 미세조정(고정3seed, 두차례 원인보완)을 완료했다. 팔 기준축 변화가 방향을 뒤집는 계산 문제를 공통 상대좌표 방식으로 수정했다. 20공개영상122조건에서 기존 정상·속도64조건 판정 보존, 복합 모양+역방향 검출2/3→3/3(합성대조). 손가락 개발895항목은807→875~879일치지만 기존 정답을 새로 틀리는1~2항목/seed가 남아 개별 손가락 교정은 미연결. 엄지와 약지·새끼손가락 개별구별 범위도 부족. 방향 보완은 격리 비교 통과, 웹은 그대로. C·T3-F·공통100 및 기존 보호파일 유지, 사용자자료 미접근. 전체 자연어 이해/모든 수어 정오 완성 아님.'
report=dict(status='FOUR_STEP_RESEARCH_COMPLETE_LIMITED',at=at,implementation='Codex',verification='Same Codex, separate saved-output audit; no independent KSL expert',steps={
 'A':dict(status='COMPLETE',result='50 public RGB hand observations reviewed; finger unknowns preserved;7 action instructions/14 phrase fragments linked; no arbitrary Korean parser'),
 'B':dict(status='COMPLETE',result='Known and automatic intervals both missed one reverse control; isolated shape/context interference and varying forearm reference-axis artifact'),
 'C':dict(status='COMPLETE',result='12 fixed small training runs: new finger heads and movement-branch fine-tuning; common100 frozen; both failed and repaired versions kept'),
 'D':dict(status='COMPLETE_LIMITED',result='122 archived public cases,64 normal/speed/complete controls; existing supported/shape/order/candidates preserved; synthetic joint error2/3 to3/3')},
 decisions=v['decisions'],metrics=read(U/'INTEGRATED_RESULTS.json')['summary'],finger_metrics=v['finger_metrics'],coverage=v['finger_coverage'],
 limits=['No real learner error accuracy claim: direction errors include time-reversed/edited coordinate controls.',
 'All development and archived G4 sources already exposed; this is regression, not a new blind test.',
 'Finger895 means individual visible-finger-at-time cells across226 hand anchors from33 public DEV clips/12 words, not895 videos or whole-word accuracy.',
 'Baseline finger output was conservatively converted from existing coarse attributes at.85, not a direct comparison to every deployed web cue.',
 'FIT has no examples where ring and little have different labeled extension states. Ordinary DEV has no thumb labels; no independent thumb correction.',
 'Generic relation/finger code is shared; actual evidence is limited to the tested actions. Four feedback words/7actions are not full100-word detailed coverage.',
 'Clause requirements are manually structured. Neural models observe motion/shape and select intervals; templates compose the explanation. Sentences were not learned by an LLM.',
 'Isolated direction inference supplement median0.005s,p95 0.010s excludes RTMW,base inference,DTW,capture anddisplay; not total latency.',
 'Web not changed; G6 physical capture remains deferred. Partial-completion, repetition, contact, detailed hand orientation and full sign correctness are not solved by this task.'],
 protected=dict(paths=v['protected_paths'],hashes_checked=v['total_hashes'],C_T3F=True,common100=True,user_training=False,user_tuning=False,user_access=False),next_action=next_action)
write(U/'RESULTS_REVIEW.json',report)
write(U/'FAILURES_AND_REPAIRS.json',dict(items=[
 dict(stage='Input preparation',issue='NumPy scalar JSON serialization, hand-side/schema names and shape-availability plumbing required correction before training.',repair='Prepared inputs and source roles were checked before fixed training; no evaluation labels were edited to fit predictions.'),
 dict(stage='Finger v1',issue='Unrelated whole-body context changed identical hand predictions; known-cell regressions remained.',repair='FIT context permutation audit, then hand-local temporal branch; v1 checkpoints/results retained.'),
 dict(stage='Motion v1',issue='A normal slow example acquired an opposite-direction warning; combined control still missed.',repair='Found moving reference-axis artifact when the forearm is foreshortened; common observed median axis plus same training schedule. No wordID branch or threshold adjustment.'),
 dict(stage='Motion v2',issue='One old supported normal became uncertain with the new branch alone.',repair='Supplementary evidence preserves old support if new evidence is weak; strong disagreement is gray. All3 new readings must agree to recover previously undecided evidence.'),
 dict(stage='Presentation audit',issue='Resolved verdict could retain the old probability vector; generic old draft generator could emit a correction despite card conflict.',repair='Separate isolated output view names baseline probabilities, exports resolving witnesses, and suppresses correction for conflict/uncertain action links. Web unchanged.'),
 dict(stage='Finger v2 remaining',issue='3–4 wrong known finger cells;1–2 were previously correct. Ring/little independent patterns absent.',repair='Shadow only, not discarded or deployed. Record precise data gap and next source review; no unobserved individual fingers invented.')],scope='Local development history, not evidence of final production accuracy'))
text=f'''설명과 행동 구간 연결 · 손가락별 판단 보완
갱신: {at}

1. 이번 작업의 결론
승인된 정답 점검 → 원인 분리 → 추가 학습 → 기존 방식 비교를 끝냈습니다.
방향 판단은 격리 시험에서 개선됐습니다. 손가락별 판단도 더 많이 맞혔지만,
새로 틀리는 장면이 남아 실제 웹의 개별 손가락 교정으로 연결하지 않았습니다.

2. 문장과 영상을 어떻게 연결하나요?
예: “오른손을 펴고 / 왼팔을 따라 / 손목 쪽으로 쓸어 보세요.”
앞부분은 손가락 상태, 가운데는 두 신체 부위의 관계, 뒷부분은 이동 방향입니다.
세 문구는 같은 쓸기 행동에서 함께 확인할 조건입니다. 문구3개=행동3개가 아닙니다.
현재는 사람이 설명을 구조화하고, 모델이 찾은 실제 행동 구간의 관측 결과와 연결합니다.
4단어의7행동·14문구 연결을 확인했습니다. 임의의 한국어 설명을 스스로 읽어
모든 구간을 알아내는 모델이 완성된 것은 아닙니다.

3. 실제로 달라진 점
손모양을 닫고 이동 방향도 뒤집은 공개자료 대조3개에서 두 차이를 함께 찾은 결과:
이전2개 → 보완3개. 잘못된 손모양이어도 원래 찾은 행동 구간을 버리지 않았습니다.
원인 하나는 비교 기준인 팔 방향이 프레임마다 바뀌면서 정상 이동도 반대로 계산되던 점입니다.
같은 행동 구간에서 관측한 팔 기준을 일관되게 사용하도록 고쳤습니다.
그 입력으로 움직임 신경망을 미세조정했습니다. 단어별 if문으로 오류를 지정한 것이 아닙니다.
단, 새 근거가 약하면 기존 판단을 유지하고 충돌하면 보류하는 공통 규칙은 함께 사용합니다.

20개 공개 원본에서 만든122조건을 비교했습니다. 정상·속도·완료 대조64조건에서
추가 경고0, 기존 정상 방향 판정 누락0, 기존 오류 근거 소실0입니다.
122개는122명의 새 촬영이 아니며, 이미 사용한 공개자료의 재검증입니다.
합성 오류 시험이 실제 사용자 오류 전체의 정확도를 뜻하지는 않습니다.

4. 손가락 학습 결과
공개 FIT80영상/15단어의420손-시점 정답으로 작은 시간합성곱+GRU를 학습했습니다.
DEV33영상/12단어의226손-시점 중 관측 가능한895손가락 항목을 확인했습니다.
기존 속성 출력을 같은 보수적 기준으로 환산하면807항목 일치,
새 후보3회는875 / 879 / 878항목 일치입니다. 이 수치는 수어 전체 정답률이 아닙니다.
새 후보는3~4항목을 틀렸고, 그중1~2항목은 기존에 맞히던 내용이었습니다.
약지와 새끼손가락의 상태가 서로 다른 학습 정답은0개였습니다.
그래서 둘을 독립적으로 잘 구분한다고 말할 수 없습니다. 일반 단어의 엄지도 미검증입니다.
“검지를 펴 보세요”처럼 말할 출력 연결은 만들었지만, 새 개별 손가락 문장은 아직 꺼두었습니다.

5. 무엇을 학습했고 무엇은 유지했나요?
공통100단어 관측 신경망과 단어 인식 C·T3-F는 유지했습니다.
새 손가락 시간판정부를 지도학습하고, 기존 움직임 가지의 가중치를 미세조정했습니다.
손가락·움직임 각각3개 고정 초기값, 원인 보완 전후로 총12회 소규모 학습을 했습니다.
학습에 사용하지 않은 수행자를 개발 비교에 썼습니다. 이 자료는 과거 검토 이력이 있어
새로운 독립 시험이라고 부르지 않습니다. 사용자 촬영은 읽거나 학습·튜닝에 쓰지 않았습니다.

6. 현재 적용 상태와 남은 일
현재 웹은 그대로입니다. 방향 보완은 별도 공개 재생 검증을 통과한 상태입니다.
손가락 판정은 아직 연구용이며, 공개 자료의 남은 오판 원인과 부족한 손가락 조합을
보강한 다음 실제 교정 안내에 연결해야 합니다. 반복 횟수·접촉·전체 수어 정오도 별도 과제입니다.
G6 실제 웹캠 확인은 사용자가 나중에 촬영하기로 한 상태를 유지합니다.

다음 작업: {next_action}

구현·검산: Codex. 별도의 저장 결과 재계산과 불변성 검사를 했지만 독립 수어 전문가 검토는 아닙니다.
근거: RESULTS_REVIEW.json, VERIFIED.json, INTEGRATED_RESULTS.json,
FINGER_RESULTS_V2.json, FINGER_PATTERN_COVERAGE.json, CLAUSE_LINKS.json.
'''
(U/'RESULT_KO.txt').write_text(text,'utf-8')

# A small, readable progress graphic; no decorative score or completion percentage.
im=Image.new('RGB',(1660,960),'#f5f7fa');d=ImageDraw.Draw(im)
font='C:/Windows/Fonts/malgun.ttf';bold='C:/Windows/Fonts/malgunbd.ttf'
f=lambda n,b=False:ImageFont.truetype(bold if b else font,n)
navy='#17334b';gray='#617181'
d.text((66,42),'설명과 행동을 연결하는 피드백 학습',font=f(46,True),fill=navy)
d.text((66,113),'이번 4단계 작업 완료 · 방향 보완 검증 통과 / 개별 손가락 교정은 추가 보강',font=f(25),fill=gray)
boxes=[('A  정답 확인','공개 손 이미지 재확인','보이는 손가락만 정답으로'),('B  원인 구분','행동 구간과 판정을 따로 점검','팔 기준축 변화 문제 확인'),('C  추가 학습','손가락 상태와 움직임 분리','시간합성곱 + GRU'),('D  공개 비교','정상 판정 유지','복합 차이 검출 개선')]
for i,(h,a,b)in enumerate(boxes):
    x=66+i*398;d.rounded_rectangle((x,205,x+354,394),radius=18,fill='white',outline='#c9d2db',width=2)
    d.text((x+21,228),h,font=f(29,True),fill=navy);d.text((x+21,287),a,font=f(22),fill=navy);d.text((x+21,328),b,font=f(22),fill=gray)
    if i<3:d.line((x+360,298,x+385,298),fill=navy,width=3);d.polygon([(x+385,298),(x+376,291),(x+376,305)],fill=navy)
d.rounded_rectangle((66,448,1594,657),radius=18,fill='#e8eef4')
d.text((92,468),'“오른손을 펴고 / 왼팔을 따라 / 손목 쪽으로 쓸어 보세요.”',font=f(33,True),fill=navy)
d.text((92,535),'손가락 상태                   두 신체 부위의 관계                  이동 방향',font=f(27),fill=navy)
d.text((92,597),'같은 행동 구간에서 함께 확인할 조건입니다. 문구마다 별도 행동으로 나누지 않습니다.',font=f(23),fill=gray)
d.text((66,706),'남은 핵심',font=f(32,True),fill=navy)
d.text((66,763),'손가락별 새 오판 줄이기 → 부족한 손가락 조합 검증 → 실제 교정 안내 연결',font=f(29),fill=navy)
d.text((66,823),'현재 웹 · C·T3-F · 공통100 유지   |   사용자 촬영 학습·튜닝 없음',font=f(24),fill=gray)
d.text((66,881),'방향 개선은 공개자료의 제한된 비교 결과이며, 전체 수어 피드백 완성을 뜻하지 않습니다.',font=f(23),fill=gray)
im.save(U/'PROGRESS_GRAPH.png')

title='### 2026-11-15 설명 구간·손가락별 판단 4단계 완료 / 방향 개선·손가락 추가 보강'
doc=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
    doc=doc.replace(heading+'\n\n',heading+'\n\n'+title+'\n\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULTS_REVIEW.json`, `VERIFIED.json`, `RESULT_KO.txt`.\n\n',1)
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(doc,'utf-8')
state=read(O/'STATE.json');state['current_followup']=U.name;state['next_single_action']=next_action
state[U.name]=dict(status=report['status'],at=at,steps=report['steps'],report=str(U/'RESULTS_REVIEW.json'),verified=str(U/'VERIFIED.json'),graph=str(U/'PROGRESS_GRAPH.png'),training_running=False,web_changed=False,private_access=False,next_action=next_action)
write(O/'STATE.json',state)
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 설명 구간·손가락별 판단 4단계 완료 — {at}

{summary}

다음 한 작업: {next_action}

현재 결과 폴더: `{U}`. RESULT_KO.txt / RESULTS_REVIEW.json / VERIFIED.json / PROGRESS_GRAPH.png 참고.
작성·학습·실행 Codex, 별도 저장 결과 검산도 같은 Codex. 독립 전문가 검토 아님.
기존 보호481경로 포함 총676개 해시 확인. 모든 학습·검산 프로세스 종료, 카메라 미접근.
실제 웹 서버가 현재 살아 있는지는 이번 작업에서 확인하지 않았다. 과거 PID를 재사용하지 않는다.

방향 후보: direction_adapter.py + stable_geometry.py + motion_core.py + motion_models_v2.
입력은 정규화 관절/시간/자동 선택구간과 검토한 설명조건. wordID나 시험변형종류로 판정하지 않는다.
presentation_evidence.py는 충돌 보류와 갱신 판정의 확률 출처를 분리하는 격리 출력 연결부다. 웹 교체 없음.
새 finger_models_v2는 shadow 전용. 주어진895항목의807→875~879는 전체 수어 정확도 아님.
개별 손가락 DEV 오답3~4개, 기존 정답을 오답으로 만든1~2개가 있어 출시에 쓰지 않는다.
약지/새끼손가락 상태가 다른 FIT/DEV 정답0개; 일반 DEV 엄지 정답0개. 손가락 이름만 나눠 출력한다고 독립 구별 학습이 입증되는 것은 아니다.

G0~G5 과거 제한 PASS, G6 실제 사용자 확인은 보류 유지. 이번 연구가 G6/DONE을 완료시키지 않는다.
LLM/3D 추가작업 제외. 사용자·멘토 촬영 접근/학습/튜닝/후보선택 금지. C·T3-F·공통100 그대로.
실패·수정 이력 FAILURES_AND_REPAIRS.json, v1/v2 모두 보존. 기존 결과/원본/sources/ 동결 파일 수정 금지.
''','utf-8')
write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETED_LIMITED',at=at,pid=None,training_running=False,next_action=next_action))
for p in [R/'NEW_CHAT_HANDOFF.md',R/'WordsInSilence_PROJECT_CONTEXT.md']:assert '2026-11-15'in p.read_text('utf-8')
assert read(O/'STATE.json')[U.name]['status']==report['status']
write(U/'DOCUMENT_CHECK.json',dict(passed=True,linked_files_exist=all((U/n).exists()for n in['RESULT_KO.txt','RESULTS_REVIEW.json','VERIFIED.json','PROGRESS_GRAPH.png']),at=at,files={str(p):sha(p)for p in[U/'RESULT_KO.txt',U/'RESULTS_REVIEW.json',U/'PROGRESS_GRAPH.png',R/'NEW_CHAT_HANDOFF.md',R/'WordsInSilence_PROJECT_CONTEXT.md',O/'STATE.json']}))
print(json.dumps(dict(status=report['status'],graph=str(U/'PROGRESS_GRAPH.png'),result=str(U/'RESULT_KO.txt')),ensure_ascii=False))
