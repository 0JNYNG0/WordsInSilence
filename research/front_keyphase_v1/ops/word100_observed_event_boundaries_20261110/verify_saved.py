from pathlib import Path
import json,hashlib,collections,statistics,math,traceback,time
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;PREV=O/'word100_event_phase_learning_20261110'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
def compact(d):
 statuses=[c['display_status'] for c in d['cards']];return dict(display=statuses,warning='difference' in statuses,similar_no_warning='difference' not in statuses and all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True)))
def count(rows,side):
 n=len(rows);warn=sum(r[side]['warning'] for r in rows);similar=sum(r[side]['similar_no_warning'] for r in rows);return dict(n=n,warning=warn,similar_without_warning=similar,uncertain=n-warn-similar)
def main():
 cfg=read(U/'FROZEN.json')
 for pins in [cfg['pins'],cfg['protections'],read(U/'CODE_PINS.json'),read(U/'EVAL_PINS.json')]:
  for p,h in pins.items():ck('pin:'+p,sha(p)==h)
 ck('prechecks',read(U/'PRETRAIN_CHECK.json')['passed'] and read(U/'BOUNDARY_CONTRACT_CHECK.json')['passed'])
 manifest=read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'];fit=[r for r in manifest if r['role']=='fit'];dev=[r for r in manifest if r['role']=='development'];fitids={r['clip_id'] for r in fit}
 ck('publicsplit',len(fit)==899 and len(dev)==300 and len({r['class_index'] for r in fit})==100 and not {r['person'] for r in fit}&{r['person'] for r in dev})
 ck('sourcehashdisjoint',not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev})
 ann=read(U/'OBSERVED_BOUNDARIES.json')['rows'];trainids={r['clip_id'] for r in ann if r['role']=='fit'};ck('annotationsplit',trainids<=fitids and not trainids&{r['clip_id'] for r in ann if r['role']=='development'})
 ck('labeldenominators',len(trainids)==18 and sum(len(r['events']) for r in ann if r['role']=='fit')==29 and sum(len(r['holds']) for r in ann if r['role']=='fit')==17 and sum(len(r['events']) for r in ann if r['role']=='development')==7)
 for r in fit:
  with np.load(U/'cache'/f"{r['clip_id']}.npz") as z:ck('trainingtargets:'+r['clip_id'],np.isfinite(z['boundary_y']).all() and np.isfinite(z['boundary_w']).all() and ((z['boundary_w'].sum()>0)==(r['clip_id'] in trainids)))
 training=[]
 for seed in cfg['seeds']:
  d=read(U/'runs'/str(seed)/'COMPLETE.json');ck('learning:'+str(seed),len(d['history'])==12 and all(e['all100_seen']==899 for e in d['history']) and set(d['seen'])==fitids and d['encoder_delta']>0 and sha(U/'runs'/str(seed)/'model.pt')==d['sha256']);training.append({k:d[k] for k in ['seed','seconds','encoder_delta','sha256']})
 rows=read(U/'FULLPATH_COMPLETE.json')['rows'];ck('234outputs',len(rows)==234);baseline={r['clip_id']+'_'+r['kind']:r for r in read(PREV/'PARTIAL_COMPLETE.json')['rows']};conditions={};changes=[];order_causes=[]
 for seed in cfg['seeds']:
  rr=[r for r in rows if r['seed']==seed];ck('same78:'+str(seed),len(rr)==78 and {r['clip_id']+'_'+r['kind'] for r in rr}==set(baseline));conditions[str(seed)]={}
  for r in rr:
   k=r['clip_id']+'_'+r['kind'];br=baseline[k];b=read(br['result']);a=read(r['result']);ck('hash:'+r['result'],sha(r['result'])==r['sha256'] and sha(br['result'])==br['sha256']);ck('summary:'+r['result'],compact(b)==r['before'] and compact(a)==r['after'])
   ck('protected:'+r['result'],a['path']==b['path'] and a['tcn_top3']==b['tcn_top3'] and a['frame_id']==b['frame_id'] and [c.get('motion_evidence') for c in a['cards']]==[c.get('motion_evidence') for c in b['cards']] and [c.get('shape_evidence') for c in a['cards']]==[c.get('shape_evidence') for c in b['cards']])
   ck('framecoverage:'+r['result'],r['frames']==r['model_frames']==len(a['frame_id']));ck('no_false_completion_promotion:'+r['result'],not r['after']['similar_no_warning'] or r['before']['similar_no_warning'])
   for e in a['phase_events']:
    ck('both_boundaries_observed:'+r['result'],0<=e['a']<=e['b']<r['frames'] and e['complete'] and e['onset_range'][0]<=e['a']<=e['onset_range'][1] and e['offset_range'][0]<=e['b']<=e['offset_range'][1] and e['active_fraction']>=.6 and e['observed']>=.8)
   if r['before']!=r['after']:changes.append(dict(seed=seed,key=k,before=r['before'],after=r['after']))
   if seed==42 and r['kind']=='reversed_core_auto_order':order_causes.append(dict(clip_id=r['clip_id'],display=r['after']['display'],checks=a['phase_order_checks']))
  for kind in dict.fromkeys(r['kind'] for r in rr):
   ss=[r for r in rr if r['kind']==kind];conditions[str(seed)][kind]={s:count(ss,s) for s in ['before','after']}
  cross=[r for r in rr if r['kind']=='full_normal' and 'WORD1567' in r['clip_id']];conditions[str(seed)]['crossing']={s:count(cross,s) for s in ['before','after']}
 ev=read(U/'EVALUATION_COMPLETE.json');components={}
 for seed in cfg['seeds']:
  ss=read(U/'scores'/f'{seed}.json');ck('coarse_denominator:'+str(seed),len(ss)==32*7*2);components[str(seed)]={}
  for variant in cfg['conditions']:
   components[str(seed)][variant]={}
   for linked in [False,True]:
    mm=[r for r in ss if r['variant']==variant and r['linked']==linked and r['task']=='movement_interval'];hh=[r for r in ss if r['variant']==variant and r['linked']==linked and r['task']=='hold'];d=dict(moving=len(mm),matched=sum(r['matched'] for r in mm),direction_correct=sum(r['pred']==r['target'] for r in mm),hold=len(hh),hold_correct=sum(r['pred']==r['target'] for r in hh));ck('saved_arithmetic',d==ev['models'][str(seed)]['component'][variant][str(linked)] and len(mm)==21 and len(hh)==11);components[str(seed)][variant][str(linked)]=d
  bounds=read(U/'scores'/f'{seed}_boundaries.json');ck('fine_boundaries_denominator',len(bounds)==7 and len({r['clip_id'] for r in bounds})==4)
  ck('fine_boundary_arithmetic',dict(total=7,matched=sum(r['matched'] for r in bounds),both_boundaries_in_range=sum(r['both_in_reviewed_ranges'] for r in bounds))==ev['models'][str(seed)]['reviewed_boundaries'])
  for r in bounds:
   if r['matched']:
    a,b=r['predicted_span'];dist=[max(r['onset'][0]-a,0,a-r['onset'][1]),max(r['offset'][0]-b,0,b-r['offset'][1])];ck('reviewed_range_distance',r['boundary_distance_to_reviewed_ranges']==dist and r['both_in_reviewed_ranges']==(dist==[0,0]))
 oldfine=read(U/'BASELINE_NEW_BOUNDARY_SCORE.json');oldcoarse=read(PREV/'EVALUATION_COMPLETE.json')['models'];times=[r['seconds'] for r in rows if r['seed']==42]
 result=dict(primary_seed=42,training=training,labels=read(U/'LABEL_AUDIT.json'),old_primary_coarse=oldcoarse['42']['component'],new_components=components,old_fine=dict(total=7,matched=sum(r['matched'] for r in oldfine),both_boundaries_in_range=sum(r['both_in_reviewed_ranges'] for r in oldfine)),new_fine={s:v['reviewed_boundaries'] for s,v in ev['models'].items()},automatic=conditions,changed_clip_displays=changes,order_linkage_audit=order_causes,auxiliary_latency=dict(median=statistics.median(times),max=max(times),scope='boundary branch,order supplement,localized frozen direction; excludes detector,TCN,DTW,frozen encoder andUI; not comparable total web latency'),decision='BETTER_BOUNDARY_RESEARCH_NO_DEFAULT_ADOPTION',limitations=['Repeated historicalDEV; newfinegold4videos from1recordID, not independent multi-person denseboundaryproof.','Model and labels/decoder both changed; isolated contribution of each not established.','Newhold diagnostics8/11primary, oldhold preserved not replaced.','Guide-stage association and order failures remain despite better boundary coverage.'],same_codex_verification=True,private_access=False,default_change=False)
 write(U/'RESULT.json',result);write(U/'VERIFICATION.json',dict(passed=True,count=len(checks),checks=checks,result_sha256=sha(U/'RESULT.json'),method='Separate saved-output arithmetic by sameCodex, no re-inference or independentexpertclaim.'));print(json.dumps(dict(passed=True,count=len(checks),changed_displays=len(changes),fine=result['new_fine']),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'VERIFY_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
