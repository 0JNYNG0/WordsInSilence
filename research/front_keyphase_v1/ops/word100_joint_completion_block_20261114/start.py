from pathlib import Path
import json,datetime,shutil,importlib.util
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
assert not(U/'PLAN.json').exists()
write(U/'PLAN.json',dict(at=stamp,authorization='User requests completion of all G3, then graph reminder/report',
 goal='Same-action integration of shape, body relation, direction, chronology and visible goal; no whole-sign correctness from an isolated feature',
 stages=['Audit fixed joint requirements and FIT source observations',
 'Repair body-relation/forearm observation only where FIT evidence supports it; keep original shape branches',
 'Join same native candidate evidence; compare normal/short/partial/missing/reversed/combined-error controls',
 'Separate saved-output verification and original G3 gate decision; synchronize graph/records'],
 gates=['component_preserved','combined_errors','missing_not_pass','original_time_evidence'],
 additional_admission=['no false completion from goal posture only or missing evidence',
 'retain normal short state transition; no improvement by abstention alone',
 'never use required order to choose observations','unknown and unsupported requirements explicit'],
 data='Existing public FIT records02-07, exposed probe08/09, exposed DEV11-13; no new test claimed',
 protected=['C/T3F','common100 model','operating web','original endpoint model','sealed final8','user and mentor data'],
 no_web_change=True,no_download=True,no_private_access=True,implementer='Codex',verification='Separate local saved-output and source audit by same Codex; not human KSL review'))
(U/'before_start').mkdir()
for p in [O/'STATE.json',G/'TASK_GRAPH.json',G/'CURRENT_EXECUTION.json',R/'NEW_CHAT_HANDOFF.md',R/'WordsInSilence_PROJECT_CONTEXT.md']:
 shutil.copy2(p,U/'before_start'/p.name)
spec=importlib.util.spec_from_file_location('graph_api',G/'graph.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.record('G3','RUNNING',{},[U/'PLAN.json'],'Continue the whole G3 box; research evidence assembly and completion qualification before any final8 or web rollout.')
write(U/'CURRENT_EXECUTION.json',dict(status='SOURCE_AND_CONTRACT_AUDIT',training_running=False,at=stamp))
s=read(O/'STATE.json');s['current_followup']=U.name;s['next_single_action']='G3 전체: 신체 관계·팔 자세 관측 보완과 동일 행동 구간의 모양/방향/순서/끝 상태 통합→정상·부분·가림·복합오류 검증. 칸 완료 전 세부실험별 종료하지 않음.'
s[U.name]=dict(status='RUNNING',unit=str(U),plan=str(U/'PLAN.json'),at=stamp,graph_box='G3',training_running=False,web_changed=False)
write(O/'STATE.json',s)
(R/'NEW_CHAT_HANDOFF.md').write_text(f'# G3 전체 진행 중 — {stamp}\n현재 단위 {U}\nPLAN/CURRENT_EXECUTION 확인. G2는 이전 action_evidence_block 단위의 개발 기준 통과. G3는 아직 미완료. 기존 완료9학습/평가 재실행금지.\n현재: 정상 신체영역/팔자세 실패를 원본과 비교한 뒤 FIT 근거가 있을 때 공통 판정부 보완. 같은 행동 근거 통합/정상·오류 대조까지 끝내고 보고.\nC·T3F/공통100/웹 유지, 사용자·멘토자료 접근·학습·튜닝·선택 금지. Final8 미개봉. 카메라·다운로드·외부전송·Git 없음.\n',encoding='utf-8')
print('G3 started; no existing model execution repeated.')
