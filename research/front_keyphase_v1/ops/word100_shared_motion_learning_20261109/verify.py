"""Separate saved-output arithmetic and provenance checks. No training or inference."""
from core import *
from collections import Counter,defaultdict

def main():
 cfg=read(U/'FROZEN.json');ev=read(U/'EVALUATION_COMPLETE.json');checks=[]
 def ck(name,ok):
  assert bool(ok),name
  checks.append(name)
 ck('fixed training settings unmodified',sha(U/'FROZEN.json')==read(U/'PRETRAIN_CHECK.json')['frozen_sha256'])
 for p,h in {**cfg['source_pins'],**cfg['code_pins']}.items():
  if Path(p)==D/'app.js':ck('pre-edit UI matches frozen bytes',sha(U/'before/app.js')==h)
  else:ck('pin '+p,sha(p)==h)
 for p,h in read(J/'FROZEN.json')['pins'].items():
  if Path(p).suffix=='.pt':ck('protected historical model '+p,sha(p)==h)
 for p in [J/'tcn_runs/finetune_C_20261013',V/'feedback_runs/pose_transfer_42',D/'runs/42']:
  c=read(p/'COMPLETE.json');expected=c.get('checkpoint_sha256',c.get('model_sha256'));ck('protected runtime checkpoint '+str(p),expected is not None and sha(p/'model.pt')==expected)
 rows=load_rows();fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development']
 ck('split counts',len(fit)==899 and len(dev)==300)
 ck('record IDs separated',{r['person'] for r in fit}==set(range(1,10)) and {r['person'] for r in dev}=={11,12,13})
 ck('source bytes separated',not ({r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev}))
 for r in rows:
  ck('prepared input '+r['clip_id'],sha(r['path'])==r['sha256'] and sha(r['channel'])==r['channel_sha256'])
  ck('finite input '+r['clip_id'],np.isfinite(r['x_full']).all() and np.isfinite(r['x_upper']).all())
 extra=read(U/'EXTRA_EVAL_LABELS.json');ck('extra labels frozen',sha(U/'EXTRA_EVAL_LABELS.json')==ev['extra_labels_sha256'])
 ck('extra labels not in FIT',not ({r['clip_id'] for r in fit}&{r['clip_id'] for r in extra['rows']}))
 ck('extra annotation denominator',extra['known_cells']+extra['null_cells']==216 and extra['known_cells']==133)
 for e in extra['rows']:ck('review image '+e['image'],sha(e['image'])==e['image_sha256'])
 breakdown={}
 for name,out in ev['models'].items():
  comp=read(U/'runs'/name/'COMPLETE.json');ck('model '+name,comp['checkpoint_sha256']==sha(U/'runs'/name/'model.pt'))
  ck('epochs '+name,comp['history'][-1]['epoch']==24 and comp['steps']==912 and not comp['development_training'])
  ck('all shared training data exposed '+name,comp['aux_supervised_clips']==(899 if name.startswith('shared100') else 0))
  ck('neural weights actually learned '+name,comp['deltas']['temporal']>0 and comp['deltas']['context']>0)
  rec=read(U/'scores'/f'{name}_records.json');word=read(U/'scores'/f'{name}_words.json')
  ck('word score '+name,len(word)==300 and sum(r['pred']==r['target'] for r in word)==out['aux_word']['correct'])
  for v,tasks in out['by_variant'].items():
   for t,sc in tasks.items():
    rr=[r for r in rec if r['variant']==v and r['task']==t];acc=[r for r in rr if r['eligible'] and r['confident']]
    expected=[len(rr),sum(r['pred']==r['target'] for r in rr),len(acc),sum(r['pred']==r['target'] for r in acc),len(rr)-len(acc)]
    ck('arithmetic '+name+'/'+v+'/'+t,expected==[sc[k] for k in ['n','raw_correct','accepted','accepted_correct','withheld']])
    for f,fs in sc['by_feature'].items():
     fr=[r for r in rr if r['feature']==int(f)];ck('feature count '+name+'/'+v+'/'+t+'/'+f,len(fr)==fs['n'] and sum(r['pred']==r['target'] for r in fr)==fs['correct'])
  long=[r for r in rec if r['variant']=='missing_long' and r['task'] in ['reviewed_shape','original_shape_labels','normal_order']]
  ck('absent hands/order are withheld '+name,all(not r['eligible'] for r in long))
  d={}
  for task in ['reviewed_shape','expanded_shape','semantic_stage']:
   rr=[r for r in rec if r['variant']=='normal' and r['task']==task];sub={}
   for group in sorted({(r.get('feature',-1),r['target']) for r in rr}):
    rs=[r for r in rr if (r.get('feature',-1),r['target'])==group];sub[str(group)]=dict(n=len(rs),correct=sum(r['pred']==r['target'] for r in rs),withheld=sum(not(r['eligible'] and r['confident']) for r in rs))
   d[task]=sub
  breakdown[name]=d
 ck('one execution per frozen run',len(read(U/'TRAINING_COMPLETE.json')['runs'])==6)
 events=[json.loads(l) for l in (U/'EVENTS.jsonl').read_text().splitlines()]
 ck('one evaluation per model',Counter(r['model'] for r in events if r['kind']=='evaluated')==Counter(ev['models'].keys()))
 for f in ['REPLAY_LOGIC_CHECK.json','REPLAY_BROWSER_CHECK.json']:ck(f,read(U/f)['status']=='PASS')
 tail=read(U/'TAIL_ALIGNMENT_CHECK.json');ck('tail diagnostics no backwards mapping',len(tail['rows'])==24 and all(r['backward_steps']==0 for r in tail['rows']))
 # Additional label bookkeeping alone is not an accuracy test.
 lpos=[]
 for r in fit:
  for i in np.flatnonzero((r['y'][:,2]>.5)&(r['ym'][:,2]>.5)):
   lpos.append(dict(clip_id=r['clip_id'],word=r['word_id'],person=r['person'],available=bool(available(r['x_full'])[i,2])))
 write(U/'DIAGNOSTICS.json',dict(by_model=breakdown,left_open_fit=dict(label_cells=len(lpos),eligible_cells=sum(r['available'] for r in lpos),words=sorted({r['word'] for r in lpos}),record_ids=sorted({r['person'] for r in lpos})),identity_limit='Partition uses dataset record IDs and source SHA; no biometric identity verification across all words.',stage_limit='Semantic-stage raw/confident scores are outputs at reviewed moments, not complete-action correctness. Long-hand-missing stage predictions must not establish completion.'))
 log=read(U/'REPLAY_CODE_CHANGE.json');log['final_verified_sha256']=sha(D/'app.js');log['initial_intermediate_sha256']=log.pop('new_sha256');log['final_readback_fix']='Removed accidental recursive helper introduced by text replacement before browser validation.';write(U/'REPLAY_CODE_CHANGE.json',log)
 write(U/'FINAL_CHECK.json',dict(status='PASS',checks=len(checks),items=checks,reviewer='same Codex separate arithmetic/provenance script; not an independent human or external agent review',neural_reinference=False,private_used=False,web_model_deployed=False))
 print(json.dumps(dict(status='PASS',checks=len(checks),left_open_fit=read(U/'DIAGNOSTICS.json')['left_open_fit'])))
if __name__=='__main__':main()
