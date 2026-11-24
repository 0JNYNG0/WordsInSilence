"""Error and input-information audit. No inference, fitting or threshold search."""
from boundary_core import *
from collections import Counter,defaultdict

def main():
 direct=read(U/'DIRECT_SCORES.json');cfg=read(U/'FROZEN.json');cached={r['key']:r for r in read(R/'EVALUATION_INPUTS.json')['rows']};errors=[];cache={}
 for r in direct:
  if r['mode']!='endpoints':continue
  b,p=r['previous'],r['after'];target=b['target'];complete=p['complete_joint']
  if (target==2)==complete:continue
  clip=b['clip_id'];e=b['event']
  if clip not in cache:cache[clip]=dict(np.load(cached[clip+'_full_normal']['path']))
  z=cache[clip];vals,valid=prior.wrist_series(z['x'],prior.motion.KINDS[e['axis']],e['moving'],prev.opposite(e['moving']))
  errors.append(dict(clip_id=clip,seed=b['seed'],event=e,target=target,kind=b['kind'],complete=complete,quality_probability=p['quality_probability'],boundary_probability=p['boundary_probability'],observed_coverage=p['coverage'],body_relative_start=vals[e['a']].tolist(),body_relative_end=vals[e['b']].tolist(),meaning='Start/end are normalized body relationships, not screen pixels. Current new completion branch removes the start value from every sample.'))
 mixed=[]
 for r in read(U/'FULLPATH_COMPLETE.json')['rows']:
  if r['condition']!='endpoints_joint' or r['kind'] not in ['full_normal','speed075','speed125']:continue
  out=read(r['result']);oldout=read(out['previous'])
  for c,b in zip(out['cards'],oldout['cards']):
   if b['display_status']=='similar' and c['display_status']=='uncertain':mixed.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['id'],state=c.get('completion_evidence',{}).get('state'),events=c.get('completion_evidence',{}).get('observed')))
 # Exact invariance of the input transform to a constant relationship offset.
 # Synthetic arithmetic check only, not a claim of a physically equivalent sign.
 rng=np.random.default_rng(42);v=rng.normal(size=(31,4));delta=np.array([.5,-.2,.3,.1]);mask=np.ones(31)
 err=float(np.max(np.abs((v-v[7])*mask[:,None]-((v+delta)-(v+delta)[7])*mask[:,None])))
 assert err<1e-12
 report=dict(errors=errors,normal_extra_uncertain=mixed,input_audit=dict(body_relative_offset_invariance_max_error=err,current_input='4 movement channels minus candidate-start values + coverage/time/axis; no absolute body-relative starting relation, no guide identity, no shared100 hidden features',old_direction_branch='Its separate geometry includes the start relationship. Existing direction was preserved, not replaced.',interpretation='Verified information loss. Its causal effect on errors has NOT been isolated in a trained comparison. Endpoint residual alone did not solve it.'),fit_axis_support=dict(Counter(str((r['axis'],r['moving'])) for r in read(U/'LABELS.json')['rows'] if r['role']=='fit')),limitations=['Background labels denote outside the reviewed target action; they must not automatically be described as no physical movement.','Quality supervision uses clipped intervals of normal videos; this is not equivalent to a learner who stops early and then holds.','New left record14 annotations are 3 movements and 22 overlapping derived spans in 2 previously exposed clips.','No new errors were used to retune labels, thresholds or select a preferred seed.'])
 write(U/'ERROR_DIAGNOSIS.json',report)
 print(json.dumps(dict(errors=len(errors),normal_extra_uncertain=len(mixed),states=dict(Counter(r['state'] for r in mixed)),invariance_error=err),ensure_ascii=False))

if __name__=='__main__':main()
