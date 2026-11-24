from pathlib import Path
import json,hashlib,datetime,shutil,importlib.util
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
assert not(U/'FROZEN.json').exists()
(U/'before').mkdir(exist_ok=True)
for p in [R/'NEW_CHAT_HANDOFF.md',R/'WordsInSilence_PROJECT_CONTEXT.md',O/'STATE.json',G/'TASK_GRAPH.json']:
 shutil.copy2(p,U/'before'/p.name)
sources=[]
for w in ('0739','1133','1567','2523'):
 for person in (17,18):
  cid=f'NIA_SL_WORD{w}_REAL{person}_F';meta=O/'word100_t3f_heldout_20261019/data/records'/f'{cid}.json';r=read(meta);p=Path(r['source_file'])
  assert p.exists()and sha(p)==r['source_sha256']
  sources.append(dict(clip_id=cid,word='WORD'+w,record_id=person,path=str(p),sha256=sha(p),metadata=str(meta),metadata_sha256=sha(meta),prior_use='TCN heldout evaluation 2026-10-19; not globally unseen'))
ready=read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']
assert not {s['clip_id']for s in sources}&{r['clip_id']for r in ready}
assert not {s['sha256']for s in sources}&{r.get('source_sha')for r in ready}
scope=[O/n for n in ['word100_goal_state_learning_20261114','word100_terminal_relation_targets_20261114','word100_action_evidence_block_20261114','word100_joint_completion_block_20261114','word100_action_completion_binding_20261114','word100_goal_binding_integration_20261114']]
audit=[];pins={};hits=[]
for directory in scope:
 for p in directory.rglob('*.json'):
  if 'before' in p.parts or p.stat().st_size>45000000:continue
  text=p.read_text('utf-8-sig')
  found=[s['clip_id']for s in sources if s['clip_id']in text or s['sha256']in text]
  if found:hits.append(dict(path=str(p),ids=found))
  audit.append(str(p))
 for p in directory.rglob('*.pt'):pins[str(p)]=sha(p)
assert not hits,hits
for p in [O/'word100_shared_motion_learning_20261109/READY.json',O/'word100_joint_completion_block_20261114/VERIFIED.json',G/'GATES.json']:pins[str(p)]=sha(p)
write(U/'EXPOSURE_AUDIT.json',dict(sources=sources,active_feedback_metadata_checked=audit,feedback_matches=hits,shared100_records=len(ready),shared100_exact_source_overlap=0,
 scope='New to the current feedback development and its common100 training lineage, previously used for TCN evaluation. Recording IDs are not proven biometric identities. Historical reference to Final8 did not identify a definitive manifest; these8 are explicitly designated here before feedback scores.',
 exclusion='10,14,15 already feedback-exposed;16 identity quarantine;01-09 FIT/probe,11-13 DEV',private_access=False))
write(U/'FROZEN.json',dict(at=datetime.datetime.now().astimezone().isoformat(),objective='G4 frozen public generalization evaluation of admitted G2/G3 evidence, with unchanged old baseline',
 sources=sources,model_pins=pins,seeds=[42,43,44],
 protocol=['Review public RGB and freeze sparse state/ordered action witnesses before feedback predictions','All8 sources retained; ambiguity stays unknown','Extract all native frames once; preserve timestamps and missing masks','Normal and0.75/1.25 timestamp-speed controls; source-truncated, reversed and missing-observation controls are diagnostic edits, not real learner errors','Evaluate current state/region path and automatic role association separately; never hand-select windows and claim automatic success','No new training or threshold selection from these sources'],
 gates=dict(protection='all protected hashes unchanged',normal='no new confidently wrong correction on reviewed normal witnesses; report both coverage and errors',comparison='paired baseline/new scores with unknowns, all3seeds and sources',negative='no positive complete-flow claim on truncated/reversed/all-missing controls',
 automatic='report true autonomous localization coverage separately from directly specified witness intervals',unsupported='thumb/contact/repetition/whole-sign correctness remain unsupported',freshness='this8 is consumed after first evaluation; repairs cannot turn repeated scores into fresh evidence'),
 repair_policy='Implementation/provenance/wording defects may be repaired and separately audited. Model changes must use FIT only; this final batch then becomes exposed diagnostic evidence, requiring another independent batch before a new generalization claim.',
 no_web_replacement=True,no_private=True,no_C_T3F_change=True))
spec=importlib.util.spec_from_file_location('g4_graph',G/'graph.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.record('G4','RUNNING',{},[str(U/'FROZEN.json'),str(U/'EXPOSURE_AUDIT.json')],'Eight public17/18 sources fixed; prior TCN exposure disclosed. No scores yet.')
write(U/'CURRENT_EXECUTION.json',dict(status='AUDITED_FROZEN_INPUT_PREPARATION',next='Extract native RTMW and review RGB before feedback inference',operator='Codex',verification='separate saved-output verification by same Codex, not human expert'))
print('G4 frozen',len(sources),'sources;',len(audit),'feedback metadata checked')
