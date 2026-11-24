"""Separate saved-output arithmetic and preservation audit. No model inference."""
from pathlib import Path
import json,hashlib,collections,statistics,math,time,traceback
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;A=O/'word100_automatic_window_learning_20261110'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
checks=[]
def ck(name,value):
 checks.append(dict(name=name,passed=bool(value)))
 assert value,name
def compact(d):
 statuses=[c['display_status'] for c in d['cards']]
 return dict(display=statuses,warning='difference' in statuses,similar_no_warning=all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True)) and 'difference' not in statuses)
def counts(rr,side):
 n=len(rr);w=sum(r[side]['warning'] for r in rr);s=sum(r[side]['similar_no_warning'] for r in rr)
 return dict(total=n,warning=w,similar_without_warning=s,uncertain=n-w-s)
def main():
 cfg=read(U/'FROZEN.json');ecfg=read(U/'EVAL_FROZEN.json');audit=read(U/'LABEL_AUDIT.json')
 for group in [cfg['protections'],cfg['pins'],cfg['code_pins'],ecfg['pins']]:
  for p,h in group.items():ck('hash:'+p,sha(p)==h)
 ck('pretrain_checks',read(U/'PRETRAIN_CHECK.json')['passed'])
 ck('word_namespace_audit',read(U/'WORD_MAPPING_AUDIT.json')['passed'])
 manifest=read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'];fit=[r for r in manifest if r['role']=='fit'];dev=[r for r in manifest if r['role']=='development'];fitset={r['clip_id'] for r in fit}
 ck('public100fit899dev300',len(fit)==899 and len(dev)==300 and len({r['class_index'] for r in fit})==100)
 ck('recordIDsdisjoint',set(r['person'] for r in fit)==set(range(1,10)) and set(r['person'] for r in dev)=={11,12,13})
 counts_cells=collections.Counter();direct=0
 for row in fit:
  with np.load(U/'cache'/f"{row['clip_id']}.npz",allow_pickle=False) as z:
   y=z['phase_y'];w=z['phase_w'];n=len(y)
   ck('cachefinite:'+row['clip_id'],np.isfinite(y).all() and np.isfinite(w).all() and z['g_full'].shape==(n,66) and z['q_full'].shape==(n,10,10))
   counts_cells.update({str(k):int(((y.argmax(-1)==k)&(w>0)).sum()) for k in range(5)});direct+=int(z['direct'].sum())
 ck('phase_label_denominators',dict(counts_cells)==audit['phase_cells'] and direct==audit['reviewed_endpoint_cells'])
 training=[]
 for seed in cfg['seeds']:
  d=read(U/'runs'/str(seed)/'COMPLETE.json');ck(f'fitseen{seed}',set(d['seen'])==fitset and len(d['history'])==10 and all(e['all100_seen']==899 for e in d['history']))
  ck(f'learned_and_fixedbase{seed}',d['encoder_delta']>0 and d['shape_direction_checkpoint']==cfg['pins'][str(O/'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt')])
  ck(f'modelhash{seed}',sha(U/'runs'/str(seed)/'model.pt')==d['sha256']);training.append(dict(seed=seed,seconds=d['seconds'],encoder_delta=d['encoder_delta'],epochs=10,fit_clips=899))
 evaluation=read(U/'EVALUATION_COMPLETE.json');components={}
 for seed in cfg['seeds']:
  scores=read(U/'scores'/f'{seed}.json');ck(f'componentdenominator{seed}',len(scores)==32*7*2)
  components[str(seed)]={}
  for variant in ecfg['conditions']:
   components[str(seed)][variant]={}
   for linked in [False,True]:
    m=[r for r in scores if r['variant']==variant and r['linked']==linked and r['task']=='movement_interval'];h=[r for r in scores if r['variant']==variant and r['linked']==linked and r['task']=='hold'];ma=[r for r in m if r['matched']]
    ck(f'componentcounts{seed}_{variant}_{linked}',len(m)==21 and len(h)==11 and len({r['clip_id'] for r in m+h})==12)
    out=dict(moving_intervals=len(m),matched=len(ma),direction_correct=sum(r['pred']==r['target'] for r in m),eligible=sum(r['eligible'] for r in m),mean_overlap=statistics.mean(r['overlap'] for r in m),mean_boundary_error_matched=statistics.mean(r['boundary_error_frames'] for r in ma) if ma else None,hold_intervals=len(h),hold_correct=sum(r['pred']==r['target'] for r in h))
    expected=evaluation['models'][str(seed)]['component'][variant][str(linked)]
    ck(f'arithmetic{seed}_{variant}_{linked}',all(out[k]==v if not isinstance(v,float) else math.isclose(out[k],v,abs_tol=1e-10) for k,v in expected.items()))
    components[str(seed)][variant][str(linked)]=out
  wr=read(U/'scores'/f'{seed}_words.json');ck(f'wordaux{seed}',len(wr)==300 and sum(r['target']==r['pred'] for r in wr)==evaluation['models'][str(seed)]['aux_word_correct'])
 baserows=read(A/'ORDER_ONLY_COMPLETE.json')['rows'];base={r['clip_id']+'_'+r['kind']:r for r in baserows}
 for row in baserows:ck('baseline_outputhash:'+row['clip_id']+'_'+row['kind'],sha(row['result'])==row['sha256'])
 allrows=read(U/'FULLPATH_COMPLETE.json')['rows'];part=read(U/'PARTIAL_COMPLETE.json')['rows'];ck('automatic234plus78',len(allrows)==234 and len(part)==78)
 for r in allrows+part:
  key=r['clip_id']+'_'+r['kind'];b=read(base[key]['result']);a=read(r['result']);ck('outputhash:'+r['result'],sha(r['result'])==r['sha256']);ck('summary:'+r['result'],compact(b)==r['before'] and compact(a)==r['after'])
  ck('preserve:'+r['result'],a['tcn_top3']==b['tcn_top3'] and a['path']==b['path'] and a['frame_id']==b['frame_id'] and [c.get('shape_evidence') for c in a['cards']]==[c.get('shape_evidence') for c in b['cards']] and [c.get('motion_evidence') for c in a['cards']]==[c.get('motion_evidence') for c in b['cards']])
  n=len(a['frame_id']);ee=a['phase_events']+a['phase_partials'];ck('span_bounds:'+r['result'],all(0<=e['a']<=e['b']<n and len(e['phases'])==e['b']-e['a']+1 for e in ee))
  ck('complete_not_forced:'+r['result'],all(e['complete'] and all(k in e['phases'] for k in [1,2,3]) and all(x<=y for x,y in zip(e['phases'],e['phases'][1:])) for e in a['phase_events']))
  ck('no_educational_overclaim:'+r['result'],not any(c.get('educational_wrong',False) for c in a['cards']))
  ck('no_positive_promotion:'+r['result'],not r['after']['similar_no_warning'] or r['before']['similar_no_warning'])
  if 'seed' in r:ck('no_frame_discard:'+r['result'],r['frames']==r['model_frames']==n)
 summary={}
 for label,rr in [('phase42',[r for r in allrows if r['seed']==42]),('phase43',[r for r in allrows if r['seed']==43]),('phase44',[r for r in allrows if r['seed']==44]),('partial42',part)]:
  ck('same78inputs:'+label,{r['clip_id']+'_'+r['kind'] for r in rr}==set(base))
  summary[label]={}
  for kind in dict.fromkeys(r['kind'] for r in rr):
   rows=[r for r in rr if r['kind']==kind];summary[label][kind]={side:counts(rows,side) for side in ['before','after']}
  cross=[r for r in rr if r['kind']=='full_normal' and 'WORD1567' in r['clip_id']];summary[label]['crossing_normal']={side:counts(cross,side) for side in ['before','after']};ck('crossing_preserved:'+label,len(cross)==3 and all(r['before']==r['after'] for r in cross))
 changed=[dict(clip_id=r['clip_id'],kind=r['kind'],added=r['added'],before=r['before'],after=r['after']) for r in part if r['before']!=r['after'] or r['added']]
 timings=[r['phase_seconds'] for r in allrows if r['seed']==42]
 result=dict(training=training,label_audit=audit,components=components,automatic=summary,posthoc_partial_changes=changed,primary_seed=42,no_seed_selection=True,existing_tcn_not_retrained=True,original_shape_direction_weights_and_outputs_preserved=True,phase_aux_seconds=dict(median=statistics.median(timings),max=max(timings),scope='phase branch plus order supplement only; excludes pose extraction,frozen encoder,TCN,DTW,web. Partial direction followup latency not measured.'),automatic_unique_conditions=78,original_dev_videos=12,independent_human_or_expert_review=False,historical_development_only=True,default_web_changed=False)
 write(U/'RESULT.json',result);write(U/'VERIFICATION.json',dict(passed=True,count=len(checks),checks=checks,reviewer='Separate saved-output verifier by same Codex; no independent human labels, no rerun of completed inference.',result_sha256=sha(U/'RESULT.json')))
 print(json.dumps(dict(passed=True,checks=len(checks),changed_cases=changed,crossing=summary['partial42']['crossing_normal'],aux_word={k:v['aux_word_correct'] for k,v in evaluation['models'].items()}),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'VERIFY_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
