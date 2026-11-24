"""Fixed public-normal and representation-level controls, never private videos."""
from pathlib import Path
from collections import Counter
import numpy as np
from prepare import U,WORDS,read,save
from engine import LearnedGuideEngine

def load_arrays(r):
 a=dict(np.load(r['input']));p=np.load(U/'projection_inputs'/f"{r['clip_id']}.npz")
 a.update(rgb=p['rgb'],prelinear=p['prelinear'],image_available=np.ones(len(a['frame_id']),bool));return a
def transform(a,source_indices):
 n=len(a['frame_id']);ix=np.asarray(source_indices);lo=np.floor(ix).astype(int);hi=np.ceil(ix).astype(int);f=(ix-lo)[:,None]
 out={}
 for key in ['x','rgb','prelinear']:out[key]=(a[key][lo]*(1-f)+a[key][hi]*f).astype(np.float32)
 out['mask']=a['mask'][lo]&a['mask'][hi];out['image_available']=a['image_available'][lo]&a['image_available'][hi]
 out['frame_id']=np.arange(len(ix));out['timestamp_us']=np.arange(len(ix))*33333
 return out
def brief(result):
 return dict(statuses=dict(Counter(s['status'] for s in result['stages'])),
  hard_difference_candidates=sum(s['status']=='FEATURE_DIFFERENCE_CANDIDATE' for s in result['stages']),
  evidence_gap_count=len(result['evidence_gaps']),
  all_stages_similar=bool(result['stages']) and all(s['status']=='SUPPORTED_FEATURES_SIMILAR' for s in result['stages']) and not result['evidence_gaps'],
  alignment_seconds=result['alignment_seconds'],head_seconds=result['head_seconds'])
def main():
 assert not (U/'FEEDBACK_EVALUATION.json').exists()
 manifest=read(U/'INPUT_MANIFEST.json');dev=[r for r in manifest['rows'] if r['role']=='development'];rows=[]
 for condition in ['frozen_projection','projection_finetune']:
  for seed in [42,43,44]:
   engine=LearnedGuideEngine(condition,seed)
   for r in dev:
    a=load_arrays(r);p=dict(np.load(U/'runs'/f'{condition}_{seed}'/'predictions'/f"{r['clip_id']}.npz"))
    result=engine.compare(p,a,r['word_id']);name=f"feedback_cases/{condition}_{seed}_{r['clip_id']}_normal.json";save(name,result)
    rows.append(dict(condition=condition,seed=seed,clip_id=r['clip_id'],kind='public_normal',path=str(U/name),**brief(result)))
   if seed!=42:continue
   # Fixed first DEV record per expression; no score-based sample selection.
   for r in [r for r in dev if r['person']==11]:
    a=load_arrays(r);n=len(a['frame_id']);mid=int(round(np.mean(r['intervals'][0])*30));cut=np.arange(max(1,mid-3),min(n-1,mid+4))
    cases=[]
    for factor in [.8,1.25]:cases.append((f'speed_{factor}',transform(a,np.linspace(0,n-1,int(round(n*factor)))),r['word_id']))
    pose_only={k:v.copy() for k,v in a.items()};pose_only['mask'][cut]=False;pose_only['x'][cut]=0;cases.append(('pose_gap_rgb_present',pose_only,r['word_id']))
    both={k:v.copy() for k,v in pose_only.items()};both['rgb'][cut]=0;both['prelinear'][cut]=0;both['image_available'][cut]=False;cases.append(('joint_and_rgb_gap',both,r['word_id']))
    kept=np.array([i for i in range(n) if i not in set(cut)]);cases.append(('short_feature_segment_deleted',transform(a,kept),r['word_id']))
    indices=np.arange(n);lo=max(0,int(r['intervals'][0][0]*30));hi=min(n,int(r['intervals'][0][1]*30)+1);indices[lo:hi]=indices[lo:hi][::-1]
    cases.append(('main_features_reversed',transform(a,indices),r['word_id']))
    cases.append(('different_target_expression',a,WORDS[(WORDS.index(r['word_id'])+1)%4]))
    for kind,edited,target in cases:
     pred=engine.infer(edited,target);result=engine.compare(pred,edited,target)
     name=f"feedback_cases/{condition}_{seed}_{r['clip_id']}_{kind}.json";save(name,result)
     if kind=='joint_and_rgb_gap':assert result['evidence_gaps'] and not brief(result)['all_stages_similar']
     rows.append(dict(condition=condition,seed=seed,clip_id=r['clip_id'],target=target,kind=kind,path=str(U/name),
       control_scope='feature-space engineering intervention, not actual edited-video validation or real learner-error gold',**brief(result)))
   print(condition,seed,'done',flush=True)
 save('FEEDBACK_EVALUATION.json',dict(status='PUBLIC_NORMAL_AND_ENGINEERING_CONTROLS_COMPLETE',rows=rows,
  new_model_inferences=sum(r['kind']!='public_normal' for r in rows),
  normal_saved_predictions_reused=True,original_videos_or_models_modified=False,
  model_choice_or_threshold_tuning_on_these_results=False,
  limitations=['Public normal data are development, not independent final test.','Feature-space speed/deletion/reversal is a pipeline control, not a new real performance video.','No real learner-error educational truth.','Only seed42 controls; all seeds public-normal and stage diagnostic.']))
 for c in ['frozen_projection','projection_finetune']:
  rr=[r for r in rows if r['condition']==c and r['kind']=='public_normal'];print(c,'normal difference candidates',sum(r['hard_difference_candidates'] for r in rr),'over',len(rr),'clips')
if __name__=='__main__':main()
