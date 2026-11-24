"""Post-hoc evidence coverage repair, cached predictions only. Initial result is immutable.

Two implementation issues: only first repeated event reached shape review, and
new completion uncertainty could hide a previously observed shape/direction difference.
No probability threshold, checkpoint, or word-specific exception is changed.
"""
from core import *
from association import consumer,prior,project
metrics=load('_completion_metrics',E/'evaluate.py');compact=metrics.compact;summary=metrics.summary
import copy
def repair(d,old_decision,attrs,raw,ts,fid,p):
 out=copy.deepcopy(d)
 for c in out['cards']:
  if c.get('previous_status')!='supported':continue
  spans=[ca['span'] for e in c['phase_evidence'] for ca in e.get('candidates',[])];cores=[co for e in c['phase_evidence'] for ca in e.get('candidates',[]) for co in ca['cores']]
  if not spans:continue
  c['initial_selected_span']=c['query_indices'];c['reviewed_occurrence_spans']=spans;c['core_intervals']=cores
  # Preserve all visible occurrences; no interpolation of coordinates or unseen events.
  c['query_indices']=[min(s[0] for s in spans),max(s[1] for s in spans)];c['status']='supported'
  c.pop('uncertainty_reason',None)
 out=consumer.apply_coverage(consumer.refine(out,attrs,raw,ts,fid,project(p)));out=prior.follow.apply_guard(out,attrs,raw,ts)
 for c,oc in zip(out['cards'],old_decision['cards']):
  c['completion_separate_from_difference']=True
  if oc['display_status']!='difference' or c['display_status']=='difference':continue
  span=oc.get('query_indices')
  incompatible=any(a['route']=='deployed' and a['reidentified_as_other_core'] and a['span']==span for a in c['candidate_archive'])
  if incompatible:continue
  # The new output may doubt event completion. It cannot erase the old observed difference.
  c['retained_difference_evidence']=dict(span=span,shape_evidence=oc.get('shape_evidence'),motion_evidence=oc.get('motion_evidence'),difference_spans=oc.get('difference_spans'),message=oc['message'])
  c['prior_completion_status']=c['status'];c.update(display_status='difference',status='difference_candidate',message=oc['message'],candidate_message=oc['message'],educational_wrong=False)
 out['followup_version']='all_occurrences_and_preserve_observed_differences_v1';return out
def main():
 assert not (E/'FOLLOWUP.json').exists();write(E/'FOLLOWUP_FROZEN.json',dict(kind='post-hoc code contract repair after first fixed comparison',changes=['all supported repeated occurrences reach shape review','completion uncertainty cannot erase prior observed shape/direction contradiction'],no_retraining=True,no_new_inference=True,no_threshold_change=True,no_word_branch=True,first_comparison_sha256=sha(E/'COMPARISON.json'),code_sha256=sha(E/'followup.py'),independent_evaluation=False))
 mainrows={(r['clip_id'],r['kind']):r for r in read(U/'INPUTS.json')['rows'] if r['role']=='development'};legacy={(r['clip_id'],r['kind']):r for r in read(P/'EVAL_INPUTS.json')['rows']};result={}
 for seed in [42,43,44]:
  name=f'final_{seed}';rows=[];changes=[]
  for r in read(E/'scores'/f'{name}_cases.json'):
   key=r['key'];src=(mainrows if r['inventory']=='main144' else legacy)[(r['clip_id'],r['kind'])];z=np.load(src['path']);raw=z['raw'] if r['inventory']=='main144' else z['raw_x'];d=read(r['decision']);base=read(N/'decisions/web'/f'{key}.json');web=np.load(N/'predictions/web'/f'{key}.npz');p=np.load(E/'predictions'/name/f'{key}.npz')['prob'];new=repair(d,base,web['p'],raw,z['timestamp_us'],z['frame_id'],p)
   dest=E/'followup_decisions'/name/f'{key}.json';write(dest,new);row=dict(r,initial=r['after'],after=compact(new),decision=str(dest),sha256=sha(dest));rows.append(row)
   if row['initial']!=row['after']:changes.append(dict(key=key,before=row['initial'],after=row['after']))
  write(E/'followup_scores'/f'{name}_cases.json',rows);s=summary(rows);result[name]=dict(by_kind=s,changes=changes);event('coverage_repaired',run=name,changes=len(changes),normal=s['main144/full_normal'],omission=s['main144/omit_final_event'])
 write(E/'FOLLOWUP.json',dict(models=result,web_deployed=False,posthoc=True,independent_evaluation=False))
if __name__=='__main__':main()
