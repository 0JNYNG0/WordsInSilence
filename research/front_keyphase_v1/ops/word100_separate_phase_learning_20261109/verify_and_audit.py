"""Separate saved-artifact verification. No training, neural forward or re-scoring."""
from pathlib import Path
import json, hashlib, collections, math
import numpy as np
import torch

P=Path(__file__).resolve().parent; O=P.parent
B=O/'word100_shared_motion_learning_20261109'
D=O/'word100_action_feedback_demo_20261108'
J=O/'word100_rtmw_joint_learning_20261108'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
checks=[]
def check(ok,label):
 checks.append(label)
 if not ok: raise AssertionError(label)
def stats(records):
 def one(rr): return dict(correct=sum(r['target']==r['pred'] for r in rr),total=len(rr))
 perclip={k:one([r for r in records if r['clip_id']==k]) for k in sorted({r['clip_id'] for r in records})}
 perword={k:one([r for r in records if k in r['clip_id']]) for k in sorted({r['clip_id'].split('_')[2] for r in records})}
 perclass={str(k):one([r for r in records if r['target']==k]) for k in sorted({r['target'] for r in records})}
 return dict(**one(records),perclip=perclip,perword=perword,perclass=perclass,
             macro_recall=float(np.mean([v['correct']/v['total'] for v in perclass.values()])))

def main():
 cfg=read(P/'FROZEN.json'); prepared=read(P/'PREPARED.json'); ev=read(P/'EVAL_INPUTS.json'); results=read(P/'EVALUATION_COMPLETE.json')
 for path,h in {**cfg['source_pins'],**cfg['code_pins']}.items(): check(sha(path)==h,'frozen '+path)
 for path,h in read(J/'FROZEN.json')['pins'].items():
  if isinstance(h,str) and len(h)==64: check(sha(path)==h,'protected '+path)
 cp=J/'tcn_runs/finetune_C_20261013';check(sha(cp/'model.pt')==read(cp/'COMPLETE.json')['checkpoint_sha256'],'current TCN unchanged')
 check(sha(P/'evaluate.py')==read(P/'EVAL_FROZEN.json')['code_sha256'],'evaluation code unchanged')
 check(sha(P/'EVAL_INPUTS.json')==read(P/'EVAL_FROZEN.json')['controls_sha256'],'evaluation inputs manifest unchanged')
 check(sha(P/'prepare_controls.py')==ev['code_sha256'],'control preparation unchanged')
 fit=[r for r in prepared['rows'] if r['role']=='fit'];dev=[r for r in prepared['rows'] if r['role']=='development']
 check(len(fit)==36 and len(dev)==12,'36 FIT / 12 DEV')
 check(not ({r['person'] for r in fit}&{r['person'] for r in dev}),'record IDs disjoint')
 check(not ({r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev}),'source SHA disjoint')
 rows={};counts=np.zeros(14);label_counts=collections.Counter()
 for r in prepared['rows']:
  check(sha(r['path'])==r['sha256'],'prepared hash '+r['clip_id']);z=np.load(r['path']);rows[r['clip_id']]=z
  check(np.isfinite(z['h_full']).all() and np.isfinite(z['g_full']).all(),'finite '+r['clip_id'])
  if r['role']=='development': check((z['stage_y']==-1).all() and (z['stage_w']==0).all(),'DEV excluded '+r['clip_id'])
  else:
   y=z['stage_y'];w=z['stage_w'];counts+=np.bincount(y[y>=0],weights=w[y>=0],minlength=14)
   label_counts.update(background=int(((y==0)&(w>0)).sum()),direct_stage=int(((y>0)&(w>=1)).sum()),weak_stage=int(((y>0)&(w>0)&(w<1)).sum()),ignored=int((w==0).sum()))
 check(np.allclose(counts,read(P/'PRETRAIN_CHECK.json')['weighted_phase_counts']),'weighted phase label arithmetic')
 controls={}
 for r in ev['rows']:
  check(sha(r['path'])==r['sha256'],'control hash '+r['clip_id']+r['kind']);z=np.load(r['path']);controls[(r['clip_id'],r['kind'])]=z
  check(len(z['frame_id'])==len(z['raw_x'])==len(z['h'])==len(z['p'])==len(z['g']),'control dimensions')
  if r['kind']=='full_normal':
   check(np.array_equal(z['frame_id'],rows[r['clip_id']]['frame_id']),'stage anchor frame identity')
   check(np.array_equal(z['timestamp_us'],rows[r['clip_id']]['timestamp_us']),'stage anchor timestamp identity')
  if r['kind']=='all_hand_missing':check((z['raw_x'][:,237:279]==0).all(),'hand availability completely masked')
 check(len(controls)==129,'129 distinct controls')
 log=[json.loads(s) for s in (P/'EVENTS.jsonl').read_text(encoding='utf-8').splitlines()]
 check(sum(r['kind']=='run_complete' for r in log)==6,'six training completions')
 check(sum(r['kind']=='evaluated' for r in log)==6,'six evaluations')
 baseline=read(P/'BASELINE_STAGE_RECORDS.json');audit={'baseline':stats(baseline),'models':{},'fit_label_counts':dict(label_counts)}
 check(audit['baseline']['correct']==results['baseline_stage']['correct'] and len(baseline)==84,'baseline 84 arithmetic')
 initial=torch.load(D/'runs/42/model.pt',weights_only=True,map_location='cpu')
 for name,result in results['models'].items():
  run=P/'runs'/name;done=read(run/'COMPLETE.json');state=torch.load(run/'model.pt',weights_only=True,map_location='cpu')
  check(sha(run/'model.pt')==done['checkpoint_sha256'],'head hash '+name)
  check(done['epochs']==100 and len(done['history'])==100,'100 epochs '+name)
  check(all(math.isfinite(e['loss']) for e in done['history']),'finite loss '+name)
  delta=sum(float(((v[:,:71] if k=='temporal.0.weight' else v)-initial[k]).float().square().sum()) for k,v in state.items() if k!='guide_words')
  check(delta>0,'actual head weights learned '+name)
  rr=read(P/'scores'/f'{name}_records.json');ss=stats(rr)
  check(ss['correct']==result['stage']['correct'] and ss['total']==84,'stage arithmetic '+name)
  check([(r['clip_id'],r['frame'],r['target']) for r in rr]==[(r['clip_id'],r['frame'],r['target']) for r in baseline],'same stage targets '+name)
  cs=read(P/'scores'/f'{name}_cases.json');check(len(cs)==129,'129 decisions '+name)
  for c in cs:
   check(sha(c['decision'])==c['sha256'],'decision hash '+name)
   d=read(c['decision']);check(c['display']==[x.get('display_status') for x in d['cards']],'display trace '+name)
  for kind,summary in result['auto'].items():
   subset=[c for c in cs if c['kind']==kind]
   computed=dict(clips=len(subset),difference_clips=sum('difference' in c['display'] for c in subset),uncertain_clips=sum('uncertain' in c['display'] for c in subset),missing_candidate_clips=sum('missing_candidate' in c['statuses'] for c in subset),all_required_similar=sum(all(s=='similar' for s,req in zip(c['display'],c['required']) if req) for c in subset),all_required_phase_present=sum(all(s for s,req in zip(c['phase_presence'],c['required']) if req) for c in subset))
   check(computed==summary,'automatic aggregation '+name+kind)
  check(result['auto']['all_hand_missing']['all_required_similar']==0,'no success on missing hands '+name)
  gains=[r for r,b in zip(rr,baseline) if r['pred']==r['target'] and b['pred']!=b['target']]
  losses=[r for r,b in zip(rr,baseline) if r['pred']!=r['target'] and b['pred']==b['target']]
  audit['models'][name]=dict(stage=ss,gains=gains,losses=losses,head_weight_delta_l2=math.sqrt(delta))
 # Trace the false normal from the saved candidate outputs without changing rules.
 cid='NIA_SL_WORD1133_REAL12_F';kind='right_closed_hand_main';z=controls[(cid,kind)]
 target=read(P/'decisions/matched_relations_42'/f'{cid}_{kind}.json')['cards'][0]
 other=read(P/'decisions/matched_latent_42'/f'{cid}_{kind}.json')['cards'][0]
 interval_checks=[]
 for span in target['phase_evidence'][0]['available_runs']:
  a,b=span;idx=np.arange(a,b+1);interval_checks.append(dict(span=span,mean_R_open=float(z['p'][idx,1].mean()),mean_R_closed=float(z['p'][idx,3].mean()),selected=span==target['query_indices']))
 audit['failure_trace']=dict(clip_id=cid,kind=kind,scope='Synthetic public joint intervention; not real user error accuracy',same_observation_features=True,latent_head=dict(status=other['display_status'],span=other['query_indices'],difference_spans=other['difference_spans']),relations_head=dict(status=target['display_status'],span=target['query_indices'],intervals=interval_checks),mechanism='evidence.assess selects only the candidate run with the highest single-frame phase peak; later candidate runs are not checked for shape in this card. Shape weights and attributes are identical between the two conditions.')
 audit['reverse_phase_loss']=[]
 for r in ev['rows']:
  if r['kind']!='main_reversed':continue
  z=controls[(r['clip_id'],r['kind'])];d=read(P/'decisions/matched_relations_42'/f"{r['clip_id']}_{r['kind']}.json");hand=z['raw_x'][:,237:279].reshape(-1,2,21).sum(2)>=5
  audit['reverse_phase_loss'].append(dict(clip_id=r['clip_id'],visible_both_frames=int(hand.all(1).sum()),required_card_span=d['cards'][0]['query_indices'],phase_peak=d['cards'][0]['phase_evidence'][0]['peak'],display=d['cards'][0]['display_status'],motion=d['cards'][0]['motion_evidence']['state']))
 audit['limitations']=['Historical development data, not fresh independent test.', '84 timestamps are prior Codex RGB-reviewed anchors, not expert-validated whole-event boundaries.', 'A missing keypoint is not an omitted event. Legacy time cuts do not necessarily remove a complete required event.', 'Counterfactual explanation from saved outputs is not a newly validated correction.', 'Geometric movement checker and neural direction model are separate; neural contradictions are not deployed.']
 write(P/'FAILURE_AUDIT.json',audit)
 write(P/'FINAL_CHECK.json',dict(status='PASS',checks=len(checks),items=checks,reviewer='Same Codex, separate saved-artifact arithmetic/hash verification; not independent human/expert verification',no_neural_forward=True,no_training_repeat=True,web_model_changed=False,private_data_access=False,model_adoption='REJECTED: primary stage/shape/reverse regression'))
 print(json.dumps(dict(checks=len(checks),baseline=audit['baseline']['correct'],labels=label_counts,trace=audit['failure_trace'],reverse=audit['reverse_phase_loss']),ensure_ascii=False))

if __name__=='__main__': main()
