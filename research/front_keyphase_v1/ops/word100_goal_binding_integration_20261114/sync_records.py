from pathlib import Path
from datetime import datetime
import json,hashlib,shutil
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
review=read(U/'RESULTS_REVIEW.json');now=datetime.now().astimezone().isoformat()
assert review['release']['whole_project_complete']is False
for p,h in read(G/'PROTECTIONS.json').items():assert sha(Path(p))==h,p
for n,h in review['evidence'].items():assert sha(U/n)==h,n
backup=U/'before_final_records';backup.mkdir(exist_ok=True)
for p in [G/'TASK_GRAPH.json',O/'STATE.json',R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',U/'CURRENT_EXECUTION.json',U/'RESULT_KO.txt']:
 if not(backup/p.name).exists():shutil.copy2(p,backup/p.name)
shutil.copyfile(U/'RESULT_SIMPLE_KO.txt',U/'RESULT_KO.txt')
summary='자동 구간·도달 근거 연결과 관측 부족/구간 혼동 보완, 공개 정상·오류 비교 완료. 손목 쪽/양손 분리 관측8→12/15·8→14/15, 정상속도81쌍 근거 유지. 손가림 방향지지17·반대7→0. 끝 삭제/가림36조건은31미확인·4연결충돌·1순간자세로 구분. 실제 사용자 누락 정확도나 전체 완료 아님. 웹/C·T3-F/공통100 유지.'
next_ko='공개 FIT에서 손 닫힘·손과 가슴/얼굴 관계·팔뚝 방향 변화를 동시에 가능하게 분리한 직접 정답을 준비한다. 정상 변이를 단순 아래 이동량/전완 각도 기준으로 오답 처리하지 않는다. 이번 구간보존·손관측·소유충돌 보완을 유지하며 실제 요구 상태를 확인하는 별도 판정부를 검증한다. G2/G3 통과 전 봉인8/웹 전체 승격 금지. 완료된 비교·6보정학습 반복 금지.'
s=read(O/'STATE.json');s['current_followup']=U.name;s['next_single_action']=next_ko
s[U.name]=dict(status=review['status'],updated_at=now,summary=summary,next=next_ko,unit=str(U),
 training_running=False,evaluation_running=False,whole_project_complete=False,web_changed=False,
 report=str(U/'RESULT_KO.txt'),graph=str(U/'PROGRESS_GRAPH.png'),verification=str(U/'RESULTS_REVIEW.json'),
 source_scenarios=82,three_seed_cases=246,web_lineage_cases=93,final8_unopened=True)
save(O/'STATE.json',s)
g=read(G/'TASK_GRAPH.json');node=next(n for n in g['nodes']if n['id']=='G2')
attempt=dict(at=now,status='REWORK',checks=dict(normal_preserved=True,false_support_reduced=None,
 not_abstention_only=True,three_seeds=True),
 evidence=[dict(path=str(U/'RESULTS_REVIEW.json'),sha256=sha(U/'RESULTS_REVIEW.json'))],
 note='Completed all82x3 normal/error source comparisons. Added goal/visibility/ownership safeguards; normal support retained. Source-order observations and safe withholding are not a validated whole-action completion/error target. No whole-G2 promotion from changed interpretation.',scoped_results=review['comparison'])
if not any(a.get('evidence',[{}])[0].get('path')==str(U/'RESULTS_REVIEW.json')for a in node.get('attempts',[])if a.get('evidence')):node.setdefault('attempts',[]).append(attempt)
node['status']='REWORK';node['latest_followup']=str(U/'RESULT_KO.txt')
g3=next(n for n in g['nodes']if n['id']=='G3');g3['preflight_comparison']=dict(status='COMPLETED_NOT_RELEASE_PASS',path=str(U/'JOINT_SAVED_COMPARISON.json'),remaining=['terminal/action identity validation','unsupported guide requirements','actual learner error scope'])
save(G/'TASK_GRAPH.json',g)
save(U/'CURRENT_EXECUTION.json',dict(status='COMPARISONS_FINISHED_G2_G3_REWORK',updated_at=now,
 training_running=False,evaluation_running=False,process_ids=[],web_changed=False,whole_graph_complete=False,
 next=next_ko,no_claim_of_background_execution=True))
p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-14 도달·관측·구간 충돌 보완 / 정상·오류 고정 비교 완료'
if title not in t:
 for header,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary+' 별도 검산 통과, 477보호파일 유지. 신규 4가중치 물리변화 보정6학습은 목표 타당성 문제로 미채택. G2 REWORK/G3 전체 미통과, 봉인평가·웹교체 미실행.'),('## 9. 다음 단일 작업',next_ko)]:
  marker=header+'\n\n';assert marker in t;t=t.replace(marker,marker+title+'\n\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `RESULTS_REVIEW.json`, `PROGRESS_GRAPH.png`.\n\n',1)
 p.write_text(t,'utf-8')
handoff=f'''# 최신 인수인계 — 2026-11-14 정상·오류 비교 완료 / 전체 그래프 미완료

{summary}

현재 단위: {U}
현재 실행 중 학습/평가 없음. CURRENT_EXECUTION의 재개점부터 진행한다. 서버/카메라 변경 없음.
기존 graph G2 REWORK, G3는 사전 비교 완료이지만 전체 합격 아님. final8 계속 봉인.

완료 결과 재실행 금지
- FULL_GRAPH_COMPARISON:82조건×3seed=246. 기존234 role출력 재사용, 새 dynamic4×3=12 고정role호출.
- JOINT_SAVED_COMPARISON: full_bag_rate_rank의 전체구간 손모양/방향을 그대로 연결. 좁은 role core로 오류 구간을 지우지 않음.
- CONTEXT/COVERAGE: 웹93, 정상60. 손목 쪽8→12/15, 양손분리8→14/15. 각3공개영상×5조건, 전체정확도 아님.
- VISIBILITY: 손가림 방향지지17/반대7→0, 정상/오류 기존근거 회귀0. 원측정값 유지.
- OWNERSHIP: 서로 다른 행동 구간이 상대 근거를 함께 가져오는4조건 분리. 첫검산에서6건으로 가정한 오류 보존, V2 실제모델별확인 통과.
- ASSOCIATION_SCOPE: 마지막삭제/가림36조건에서31미확인/4충돌/1순간자세, 완료/교육적오답 추론 없음.
- relational_change:4파라미터 물리변화 보정2조건×3seed 총6학습 완료, 미채택. 정상 기록08은 손중심 높이변화가 거의 없어도 손닫힘/팔관계 변화가 있음. TARGET_REVIEW를 읽고 단순 아래이동을 의무 목표로 쓰지 말 것. 노출probe 정답 사후 변경/재채점 없음.

다음 한 작업
{next_ko}

새 모델 다운로드, 사용자/멘토 자료 접근·학습·튜닝·선택, Git, 외부전송 없음.
공개FIT02~07 / 이미노출probe08~09 / 이미노출DEV11~13 유지. 기록ID를 외형으로 동일인 판단하지 않음.
기존 C·T3-F/공통100/운영웹과 원본·이전동결 보존. 별도검산은 같은Codex이며 독립KSL전문가정답과 구분.
전체 원인/결과/남은 범위는 RESULT_KO.txt와 RESULTS_REVIEW.json. 현재 그림 PROGRESS_GRAPH.png.
'''
(R/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8')
assert read(O/'STATE.json')['current_followup']==U.name
assert read(U/'CURRENT_EXECUTION.json')['process_ids']==[]
assert next(n for n in read(G/'TASK_GRAPH.json')['nodes']if n['id']=='G2')['status']=='REWORK'
assert title in (R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8')
assert str(U)in (R/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
for n in ['RESULT_KO.txt','RESULTS_REVIEW.json','PROGRESS_GRAPH.png','GRAPH.json']:assert(U/n).is_file()
save(U/'RECORDS_SYNCED.json',dict(at=now,passed=True,
 files={str(p):sha(p)for p in [O/'STATE.json',G/'TASK_GRAPH.json',R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',U/'CURRENT_EXECUTION.json',U/'RESULT_KO.txt',U/'PROGRESS_GRAPH.png']},
 whole_graph_complete=False,verification_not_independent_human=True))
print('Records synchronized; no model/web promotion and no running job claimed.')
