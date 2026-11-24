"""New stage-anchor diagnostic from saved outputs; no re-inference/retraining."""
import sys,json
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
sys.dont_write_bytecode=True
from prepare import U,O,OLD,WORDS,read,save,sha
from alignment import dtw,representative,stage_at

def score(rows):
 n=len(rows);valid=[r for r in rows if r['prediction'] is not None]
 return dict(anchors=n,answered=len(valid),coverage=len(valid)/n,
   exact_correct=sum(r['prediction']==r['truth'] for r in rows),
   same_configuration_correct=sum(r['prediction'] in r['allowed'] for r in rows),
   per_clip={cid:dict(anchors=sum(r['clip_id']==cid for r in rows),exact_correct=sum(r['clip_id']==cid and r['prediction']==r['truth'] for r in rows),
     same_configuration_correct=sum(r['clip_id']==cid and r['prediction'] in r['allowed'] for r in rows)) for cid in sorted({r['clip_id'] for r in rows})})
def main():
 assert not (U/'STAGE_EVALUATION.json').exists()
 annotations=read(U/'BLIND_STAGE_ANCHORS.json');anchors=annotations['rows'];support=read(U/'GUIDE_SUPPORT.json');guides=support['guides'];ids=[g['canonical_stage_id'] for g in guides]
 byclip=defaultdict(list)
 for a in anchors:byclip[a['clip_id']].append(a)
 results=[]
 for condition in read(U/'FROZEN_TRAINING.json')['conditions']:
  for seed in [42,43,44]:
   directory=U/'runs'/f'{condition}_{seed}';record=read(directory/'COMPLETE.json');assert sha(directory/'model.pt')==record['checkpoint_sha256']
   state=torch.load(directory/'model.pt',map_location='cpu',weights_only=True);prototypes=state['prototype'].numpy();lengths=state['prototype_lengths'].numpy()
   direct=[];aligned=[];paths={}
   for cid,ann in byclip.items():
    word=ann[0]['word_id'];wi=WORDS.index(word);p=np.load(directory/'predictions'/f'{cid}.npz')
    lookup={int(fid):i for i,fid in enumerate(p['frame_id'])};alignment=dtw(p['latent'],prototypes[wi,:lengths[wi]])
    paths[cid]=dict(path=alignment['path'].tolist(),axes=['query_native_index','learned_canonical_index'],mean_cost=alignment['mean_cost'])
    for a in ann:
     ix=lookup[a['frame_id']];base=dict(clip_id=cid,frame_id=a['frame_id'],truth=a['stage_id'],allowed=a['allowed_same_physical_configuration'])
     direct.append(dict(base,prediction=ids[int(p['stage_probability'][ix].argmax())]))
     j=representative(alignment['path'],ix);aligned.append(dict(base,prediction=stage_at(j,guides,word),canonical_index=j))
   results.append(dict(condition=condition,seed=seed,direct=score(direct),dtw=score(aligned),direct_rows=direct,dtw_rows=aligned,paths=paths))
 old=read(OLD/'FEEDBACK_EVALUATION.json');oldcases={c['query']:c for c in old['case_index'] if c['kind']=='same_expression'}
 baseline=[]
 for cid,ann in byclip.items():
  case=read(oldcases[cid]['path']);selected=case['modes']['all9']['selected_whole_references'];pairs={r['reference']:read(r['path']) for r in case['comparisons']}
  mapped={ref:read(OLD/'stage_mappings'/f'{ref}.json') for ref in selected}
  for a in ann:
   labels=[]
   for ref in selected:
    pair=pairs[ref];p=np.asarray(pair['path']);j=representative(p,a['frame_id'])
    k=representative(mapped[ref]['path'],j) if j is not None else None
    labels.append(stage_at(k,guides,a['word_id']) if k is not None else None)
   pred=labels[0] if labels and len(set(labels))==1 else None
   baseline.append(dict(clip_id=cid,frame_id=a['frame_id'],truth=a['stage_id'],allowed=a['allowed_same_physical_configuration'],prediction=pred,selected_predictions=labels))
 out=dict(status='SAVED_PREDICTION_PHYSICAL_STAGE_DIAGNOSTIC',annotation_sha256=sha(U/'BLIND_STAGE_ANCHORS.json'),
  baseline=score(baseline),baseline_rows=baseline,candidates=results,
  inference=0,new_training=0,limitations=annotations['limitations'],
  meaning='Sampled physical configuration/order check, not full-video boundary accuracy or independent educational correctness. Repeated configuration equivalence and exact occurrences are separate.')
 save('STAGE_EVALUATION.json',out)
 print('baseline',out['baseline'])
 for r in results:print(r['condition'],r['seed'],'direct',r['direct']['exact_correct'],'dtw',r['dtw']['exact_correct'],'configuration',r['dtw']['same_configuration_correct'],'/',r['dtw']['anchors'])
if __name__=='__main__':main()
