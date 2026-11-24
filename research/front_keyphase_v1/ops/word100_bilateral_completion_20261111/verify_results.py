from bilateral_core import *
from collections import Counter

def main():
 checks=[]
 def check(name,v):
  assert v,name
  checks.append(name)
 for p,h in read(U/'PROTECTIONS.json').items():check('preserved '+p,sha(p)==h)
 for f in ['FROZEN.json','EVAL_PINS.json','CROSS_FROZEN.json']:
  for p,h in read(U/f)['pins'].items():check('pin '+p,sha(p)==h)
 for r in read(U/'TRAINING_COMPLETE.json')['runs']:check('trained model '+str(r['seed']),sha(r['model'])==r['sha256'])
 labels=read(U/'LABELS.json')['rows'];fit=[r for r in labels if r['role']=='fit'];dev=[r for r in labels if r['role']!='fit']
 check('fit dev source disjoint',not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev});check('record ID disjoint',not {r['person'] for r in fit}&{r['person'] for r in dev})
 direct=read(U/'DIRECT_SCORES.json');cross=read(U/'CROSS_SCORES.json');rows=read(U/'FULLPATH_COMPLETE.json')['rows'];summary={};ds={};cs={}
 for seed in [42,43,44]:
  for side in ['R','L']:
   rr=[r for r in direct if r['seed']==seed and r['event']['moving']==side]
   ds[f'{seed}/{side}']=dict(n=len(rr),whole=sum(r['target']==2 for r in rr),partial=sum(r['target']==1 for r in rr),quality_before=sum(r['before']==r['target'] for r in rr),quality_after=sum(r['after']['quality']==r['target'] for r in rr),whole_before=sum(r['target']==2 and r['before_complete'] for r in rr),whole_after=sum(r['target']==2 and r['after']['physical_event_complete'] for r in rr),false_complete_before=sum(r['target']!=2 and r['before_complete'] for r in rr),false_complete_after=sum(r['target']!=2 and r['after']['physical_event_complete'] for r in rr),partial_false_complete_before=sum(r['target']==1 and r['before_complete'] for r in rr),partial_false_complete_after=sum(r['target']==1 and r['after']['physical_event_complete'] for r in rr),direction_n=sum(r['direction_target']>=0 for r in rr),direction_correct=sum(r['direction_target']>=0 and r['after']['direction']==r['direction_target'] for r in rr))
  rr=[r for r in cross if r['seed']==seed and r['known']];cs[str(seed)]=dict(known=len(rr),unknown=sum(r['seed']==seed and not r['known'] for r in cross),quality_before=sum(r['before']==r['target'] for r in rr),quality_after=sum(r['after']['quality']==r['target'] for r in rr),whole_n=sum(r['target']==2 for r in rr),whole_before=sum(r['target']==2 and r['before_complete'] for r in rr),whole_after=sum(r['target']==2 and r['after']['physical_event_complete'] for r in rr),false_complete_before=sum(r['target']!=2 and r['before_complete'] for r in rr),false_complete_after=sum(r['target']!=2 and r['after']['physical_event_complete'] for r in rr))
 for r in rows:
  check('output sha '+r['result'],sha(r['result'])==r['sha256']);out=read(r['result']);oldout=read(out['previous']);check('previous output sha',sha(out['previous'])==out['previous_sha256']);check('stored DTW and TCN unchanged',out['protected']==oldout['protected'])
  for a,b in zip(out['cards'],oldout['cards']):
   check('original handshape/direction intact',a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence'))
   if b['display_status']=='difference':check('previous warning not erased',a['display_status']=='difference')
  key=f"{r['seed']}/{r['mode']}/{r['kind']}";s=summary.setdefault(key,dict(n=0,warnings_before=0,warnings_after=0,similar_before=0,similar_after=0,new_warnings=0,lost_warnings=0,uncertain_cards_before=0,uncertain_cards_after=0))
  s['n']+=1
  for label in ['before','after']:
   s['warnings_'+label]+=r[label]['warning'];s['similar_'+label]+=r[label]['similar_no_warning'];s['uncertain_cards_'+label]+=sum(c=='uncertain' for c in r[label]['display'])
  s['new_warnings']+=r['after']['warning'] and not r['before']['warning'];s['lost_warnings']+=r['before']['warning'] and not r['after']['warning']
 check('78 x 3 seeds x 3 variants',len(rows)==702)
 gates={}
 for mode in ['head_only','binding_only','combined']:
  ss=[(k,s) for k,s in summary.items() if '/'+mode+'/' in k]
  gates[mode]=dict(no_new_normal_warning=all(s['new_warnings']==0 for k,s in ss if k.split('/')[-1] in ['full_normal','speed075','speed125']),normal_similar_preserved=all(s['similar_after']>=s['similar_before'] for k,s in ss if k.split('/')[-1] in ['full_normal','speed075','speed125']),previous_errors_retained=all(s['lost_warnings']==0 for k,s in ss),no_missing_observation_success=all(s['similar_after']==0 for k,s in ss if k.split('/')[-1] in ['all_hand_missing','mask_final_event']),no_reverse_false_similar=all(s['similar_after']==0 for k,s in ss if k.split('/')[-1]=='reversed_core_auto_order'),direct_whole_not_worse=all(s['whole_after']>=s['whole_before'] for s in ds.values()),no_partial_false_complete=all(s['partial_false_complete_after']==0 for s in ds.values()),cross_whole_not_worse=all(s['whole_after']>=s['whole_before'] for s in cs.values()))
  gates[mode]['pass']=all(gates[mode].values())
 timing=[r['additional_seconds'] for r in rows if r['mode']=='combined']
 final=dict(status='RESEARCH_COMPLETE_MODEL_NOT_ADOPTED' if not any(g['pass'] for g in gates.values()) else 'REVIEW_REQUIRED',direct=ds,cross=cs,feedback=summary,gates=gates,additional_time=dict(median_seconds=float(np.median(timing)),p95_seconds=float(np.quantile(timing,.95)),scope='New completion + direction candidate analysis only. No camera/RTMW/TCN/DTW/video rendering time.'),training_videos=len({r['clip_id'] for r in fit}),training_events=sum(len(r['events']) for r in fit),review_added_left_videos=12,new_left_training_videos=6,new_left_development_videos=6,common100_backbone_preserved=True,private_data=0,independent_human=False,natural_learner_error_accuracy=None)
 write(U/'FINAL_RESULT.json',final);write(U/'VERIFICATION.json',dict(status='PASS',checks=len(checks),check_names=checks,verifier='Separate saved-output arithmetic and integrity script, same Codex; not independent human',training_or_inference_repeated=False))
 event('verified',checks=len(checks),gates=gates,direct=ds,cross=cs)
if __name__=='__main__':main()
