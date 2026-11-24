"""Additional actual-image-bounds diagnostic, not used to select weights."""
from common import *
import numpy as np,torch
from feedback_model import PoseGuideNet
from normalization import normalize
from train_feedback import metrics
sys.path.insert(0,str(O/'word100_learned_guide_feedback_20261102'))
from alignment import dtw,representative,stage_at
def main():
 assert not (U/'FEEDBACK_FULL_VIEW.json').exists()
 write(U/'FULL_VIEW_SCOPE.json',dict(reason='Serving uses actual camera image bounds, not a reference-derived wrist cutoff. Check fixed heads against same reviewed labels with full-source coordinate view.',
  weight_changes=False,threshold_changes=False,subset='all 12 existing DEV clips, all six fixed heads',primary_comparison_changed=False))
 torch.set_num_threads(4);stats=read(U/'FEEDBACK_NORMALIZATION.json');f=read(U/'FROZEN.json')['feedback'];old=O/'word100_learned_guide_feedback_20261102'
 man=read(old/'INPUT_MANIFEST.json');dev=[r for r in man['rows'] if r['role']=='development'];anchors=read(old/'BLIND_STAGE_ANCHORS.json')['rows'];guides=read(old/'GUIDE_SUPPORT.json')['guides'];out=[]
 for condition in f['conditions']:
  for seed in f['seeds']:
   model=PoseGuideNet(stats['guide_vectors'],stats['guide_words'],stats['prototype_lengths']);d=U/'feedback_runs'/f'{condition}_{seed}';model.load_state_dict(torch.load(d/'model.pt',map_location='cpu',weights_only=True));model.eval();ys=[];ps=[];ms=[];stages=[];saved=[]
   for r in dev:
    cid=r['clip_id'];rec=read(U/'records'/f'{cid}.json');z=np.load(rec['channel_path']);x=normalize(z['full'],stats);wi=f['words'].index(r['word_id'])
    with torch.inference_mode():p=model(torch.from_numpy(x[None]),torch.tensor([len(x)]),torch.tensor([wi]))
    prob=p['attributes'][0].sigmoid().numpy();stage=p['stage'][0].softmax(-1).numpy();latent=p['latent'][0].numpy()
    path=U/'full_view_predictions'/f'{condition}_{seed}'/f'{cid}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,probability=prob,stage_probability=stage,latent=latent,frame_id=z['frame_id'],timestamp_us=z['timestamp_us'])
    saved.append(dict(clip_id=cid,path=str(path),sha256=sha(path)))
    for l in r['labels']:
     ix=int(np.flatnonzero(z['frame_id']==l['center_frame'])[0]);ys.append(l['y']);ps.append(prob[ix]);ms.append(l['target_mask'])
    alignment=dtw(latent,model.prototype[wi,:stats['prototype_lengths'][wi]].detach().numpy())
    for a in [a for a in anchors if a['clip_id']==cid]:
     ix=int(np.flatnonzero(z['frame_id']==a['frame_id'])[0]);j=representative(alignment['path'],ix)
     stages.append(dict(clip_id=cid,frame_id=a['frame_id'],truth=a['stage_id'],direct=stats['guide_ids'][int(stage[ix].argmax())],dtw=stage_at(j,guides,r['word_id'])))
   out.append(dict(condition=condition,seed=seed,attributes=metrics(np.asarray(ys),np.asarray(ps),np.asarray(ms),man['targets']),stage_rows=stages,
       stage_total=len(stages),stage_direct_correct=sum(r['direct']==r['truth'] for r in stages),stage_dtw_correct=sum(r['dtw']==r['truth'] for r in stages),predictions=saved))
 write(U/'FEEDBACK_FULL_VIEW.json',dict(status='DIAGNOSTIC_COMPLETE',rows=out,not_real_webcam=True,used_for_training_or_selection=False))
 event('feedback_full_view_complete',summary=[{k:r[k] for k in ['condition','seed','stage_direct_correct','stage_dtw_correct']} for r in out])
if __name__=='__main__':main()
