from prepare import *
import ast,datetime

def main():
 plan=read(U/'PLAN.json');checks=[]
 for path,h in plan['protected'].items():
  actual=sha(path);assert actual==h,(path,h,actual)
  checks.append(dict(path=path,unchanged=True))
 previous=read(O/'word100_game_acceptance_20261115/PROTECTED.json');changed=[]
 for path,h in previous.items():
  now=sha(path)
  if now!=h:changed.append(dict(path=path,before=h,now=now))
 allowed={str(O/'word100_live_g6_20261114/engine.py'),str(O/'web_tutorial_live_integration_20261113/contract.py'),str(O/'web_tutorial_live_integration_20261113/site/live.js')}
 assert all(r['path'] in allowed for r in changed),changed
 compiled=[]
 for path in list(U.glob('*.py'))+[O/'web_tutorial_live_integration_20261113/feedback_acceptance.py',O/'web_tutorial_live_integration_20261113/contract.py']:
  ast.parse(path.read_text('utf8'),filename=str(path));compiled.append(str(path))
 V=U/'source_corrected';j=read(V/'CONFIRMATION_RESULTS.json');aa=j['anchors'];assert j['summary']['correct']==sum(a['state']==a['pred']for a in aa);assert all(r['diagnostic_pass']for r in j['rows'])
 t=read(V/'TRAINED.json');run_count=len(t.get('runs',[]))
 # Data ownership and performer split, including unchanged source hashes.
 rows=read(U/'INPUTS.json')['rows'];f={r['person']for r in rows if r['role']=='fit'};d={r['person']for r in rows if r['role']=='development'}
 assert f==set(range(1,10))and d=={11,12,13}and not f&d
 for r in rows:
  assert r['clip_id'].startswith('NIA_SL_WORD1290_REAL')
  for key,file in [('source','video'),('raw','raw'),('channels','channels')]:assert sha(r[file])==r[key+'_sha256']
 actual_q=read(U/'QUEST_VERIFIED.json');assert all(r['passed']for r in actual_q['contract_cases'])
 local=read(U/'local_shape/ADMISSION.json');assert local['admitted']is False
 sd=read(U/'local_shape/CONSENSUS_DEVELOPMENT.json');sz=np.load(U/'local_shape/CONSENSUS_DEVELOPMENT.npz');mask=sz['dev'];y=sz['target']
 for key,array in [('existing','baseline'),('supplemented','states')]:
  q=sz[array];assert sd[key]['correct']==int((mask&(q==y)).sum());assert sd[key]['wrong']==int((mask&(q>=0)&(q!=y)).sum())
 assert sd['old_correct_to_wrong']==0 and sd['old_correct_to_unknown']==0
 sf=read(U/'local_shape/fresh/RESULTS.json');assert sf['summary']['predicted']['wrong']==0
 assert all(r['opposite_unchanged']is True for r in sf['rows']if r['kind']in['left_fist','right_fist'])
 assert all(h['unknown']==h['frames']for r in sf['rows']if r['kind']=='missing'for h in r['hands'].values())
 live=read(U/'BASIC_RUNTIME_VERIFIED.json');assert [len(r['events'])for r in live['rows']]==[2,1]
 for r in live['rows']:assert read(r['result'])['whole_word_correct']is None
 for path in [O/'STATE.json',U/'TASK_GRAPH.json',U/'local_shape/CONSENSUS_ADMISSION.json']:read(path)
 assert read(O/'STATE.json')['current_stage']=='basic_demo_feedback_20261115'
 assert '기초단어 반복·손모양 공개학습 비교' in (R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf8')
 assert '정상완료생성기가없어현재정상도보류' in (R/'NEW_CHAT_HANDOFF.md').read_text('utf8')
 result=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),passed=True,
  protected_current_pins=checks,older_protected_count=len(previous),older_unchanged=len(previous)-len(changed),intentional_differences=changed,
  python_ast_files=len(compiled),public_input_videos=len(rows),fit_people=sorted(f),development_people=sorted(d),
  source_corrected_confirmation=j['summary'],quest_contract_cases=len(actual_q['contract_cases']),hand_consensus_development=sd,hand_confirmation=sf['summary'],later_relation_counts=[2,1],
  actors=dict(implementation='Codex',verification='Codex separate audit script; not independent agent or KSL expert'),
  execution=dict(no_camera=True,no_server_restart=True,no_private_training=True,no_live_admission_of_new_weights=True),
  limitations=['Whole new vocabulary lesson completion not qualified; actual positive completion producer absent.',
   'Current quest source now holds all outputs without qualified completion, including old normal outputs. This is an unfinished integration, not measured normal-acceptance gain.',
   'Initial149 relation anchors included12 erroneous labels; corrected copy preserves original history.',
   'local_shape CONTROLS missing rows are score-only; MISSING_CHECK is the real unavailable-coordinate test.',
   'Shape controls synthetic, speed controls timestamp changes retaining frames, confirmation people not globally new.'])
 write(U/'VERIFIED.json',result);print('VERIFIED',len(previous),'protected;',len(changed),'documented differences;',j['summary'])
if __name__=='__main__':main()
