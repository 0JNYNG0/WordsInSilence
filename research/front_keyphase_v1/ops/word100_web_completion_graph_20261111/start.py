from graph import *
import shutil
O=U.parent;ROOT=O.parents[2]
initialize()
if not (U/'AUTHORIZATION.json').exists():
 write(U/'AUTHORIZATION.json',dict(request='사용자: 작업 그래프를 만들고 성공 시 다음 단계, 실패 시 원인검토/수정/검증 반복으로 웹 목표 끝까지 자율진행. ClaudeCode 분할업무 허용.',scope='web-only four supported words, shared100 retained',private_data_allowed=False,default_model_replacement='only evidence-qualified component, preserve C/T3F',camera='user starts actual webcam test',author='Codex',reviewers=['Codex delegated code/data reviewers'],claude_attempt='CLI2.1.282 generic text-only review; no response after bounded wait, interrupted; no completion or quota claim'))
 write(U/'INITIAL_TOOL_ERROR.json',dict(error='graph.py initial relative path duplicated under subdirectory',fix='run python graph.py in correct directory',model_or_data_changes=False))
 protections=read(O/'word100_guide_requirement_learning_20261111/PROTECTIONS.json')
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protections.items())
 write(U/'PROTECTIONS.json',protections)
 write(U/'GATES.json',dict(before_new_scores=True,stage2=['normal/speed observed order support must not decrease in any seed','combined reversal false support must decrease with at least one positive difference identification, not just withholding','normal cross/shape/direction baseline preserved','no all-hand-missing positive supports','no expected order used for correspondence'],stage4=['new public source exposure audit before final scoring','final public score cannot select/tune checkpoint','unsupported capabilities explicit'],unresolved='If no unexposed evidence exists, report final independent scope incomplete; do not rename repeated DEV fresh.'))
 record('G0','PASS',dict(public_only=True,protected_hashes=True,predeclared_gates=True),[U/'AUTHORIZATION.json',U/'PROTECTIONS.json',U/'GATES.json'],'Public cached data only; graph contracts fixed before new candidates.')
 b=U/'before';b.mkdir(exist_ok=True)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']:
  shutil.copy2(p,b/p.name)
 msg='웹4단어 피드백 완성을 작업그래프로 실행 중. G0범위고정→G1설명/정답검증→G2자동연결→G3통합판정→G4분리평가→G5시각화→G6촬영/속도. 실패는보존후원인별루프. Unity제외,공통100/C·T3F보존,개인학습/튜닝없음.'
 ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=ctx.read_text(encoding='utf-8-sig')
 for section,title,body in [('8.','2026-11-11 웹 완료 작업그래프 자율 실행 시작',msg),('9.','2026-11-11 현재 단일 실행 — 설명/실제행동 연결부터',str(U/'TASK_GRAPH.json')+'에 따라 단계통과 후 진행. 현재직접주석/설명표현형감사 및 상류순서강제연결원인수정. 기존완료결과재실행금지.')]:
  lines=t.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if l.startswith('## '+section));lines.insert(i+1,'\n### '+title+'\n'+body+'\n\n');t=''.join(lines)
 ctx.write_text(t,encoding='utf-8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계\n\n'+msg+'\n\n현재 실행그래프: '+str(U/'TASK_GRAPH.json')+'\n마지막완료근거: word100_guide_requirement_learning_20261111. 지정구간순서개선과자동연결실패구분. 현재새웹적용없음.\n다음: G1검증→기존학습역할출력을실제시각으로연결하는범용수정/검증. 실행기록: '+str(U/'CURRENT_EXECUTION.json')+'\n',encoding='utf-8')
 state=read(O/'STATE.json');state['current_stage']=U.name;state['last_updated']=stamp();state['next_single_action']='작업그래프 G1 설명/직접주석 감사→G2 실제행동연결 검증; 이후그래프순서로웹완료까지 진행';state[U.name]=dict(status='ACTIVE_GRAPH_EXECUTION',graph=str(U/'TASK_GRAPH.json'),execution=str(U/'CURRENT_EXECUTION.json'),web_changed=False,training_running=False,private_access=False);write(O/'STATE.json',state)
 with (ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf-8') as f:f.write('\n2026-11-11 작업그래프 자율완료 승인\n'+msg+'\n이전미통과후중단반복을줄이기위해 단계통과/실패근거와재작업분기를영속화. 서류완료를웹완료로간주하지않음.\n')
write(U/'CURRENT_EXECUTION.json',dict(status='RUNNING_SOURCE_AUDIT_AND_BINDER_FIX',pid=None,claude_running=False,camera=False,training_running=False))
