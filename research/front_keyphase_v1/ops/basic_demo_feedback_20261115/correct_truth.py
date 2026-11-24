"""Append-only source correction; preserve original labels/training/results."""
from prepare import *
import copy
def main():
 v=U/'source_corrected';v.mkdir(exist_ok=True)
 for d in ['runs','results','cache']: (v/d).mkdir(exist_ok=True)
 original=read(U/'SOURCE_TRUTH.json');updated=copy.deepcopy(original)
 corrections={(1,74):1,(2,79):1,(3,77):1,(4,114):1,(5,65):1,(6,63):1,(8,81):1,(9,81):1,(12,67):1,(13,73):2,(13,76):2,(13,85):2}
 log=[]
 for a in updated['rows']:
  key=(a['person'],a['frame'])
  if key in corrections:
   b=corrections[key];log.append(dict(person=key[0],frame=key[1],before=a['state'],after=b,reason='Larger native RGB crop shows maintained separated/near state, not a new approach',evidence=str(U/'close_review')))
   a['state']=b;a['state_name']=['outside_interaction','vertically_separated','near_in_projection'][b]
 updated['repetition']='Initial three-cycle interpretation withdrawn: larger RGB review shows two approach cycles followed by separated/near hold. Do not count a hold as an extra cycle.'
 updated['audit']='All non-background anchors reviewed again in larger sequentially decoded RGB crops, not corrected by model probabilities. Exactly matching random-seek/PyAV frame indices confirmed at person12 52/67/70; no timestamp mismatch found.'
 updated['prior_truth']=str(U/'SOURCE_TRUTH.json')
 write(v/'SOURCE_TRUTH.json',updated);write(v/'CORRECTIONS.json',dict(rows=log,initial_runs_valid_for_release=False,reason='Own source annotation errors; preserve original failed run, correct source truth before retry',expert_review=False))
 for n in ['INPUTS.json','PLAN.json']:shutil.copy2(U/n,v/n)
 frozen=read(U/'TRAIN_FROZEN.json');frozen.update(revision='source labels corrected only; same split/features/architecture/hyperparameters',source_truth_sha=sha(v/'SOURCE_TRUTH.json'),previous=str(U/'TRAIN_FROZEN.json'))
 write(v/'TRAIN_FROZEN.json',frozen)
 for r in read(U/'INPUTS.json')['rows']:
  z=dict(np.load(U/'cache'/f'{r["clip_id"]}.npz'));y=np.full(len(z['y']),-1,np.int64)
  for a in updated['rows']:
   if a['clip_id']==r['clip_id']:y[a['frame']]=a['state']
  z['y']=y;np.savez_compressed(v/'cache'/f'{r["clip_id"]}.npz',**z)
 print('SOURCE CORRECTIONS',len(log),'original files retained')
if __name__=='__main__':main()
