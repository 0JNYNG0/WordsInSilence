"""Fixed two-condition, three-seed supervised public development experiment."""
from pathlib import Path
import json,time,sys,hashlib,traceback
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
sys.dont_write_bytecode=True
from prepare import U,O,OLD,WORDS,read,sha,save
from model import GuideFeedbackNet,collate

def assemble():
 manifest=read(U/'INPUT_MANIFEST.json');support=read(U/'GUIDE_SUPPORT.json')
 projected=read(U/'PROJECTION_PREPARED.json');assert projected['status']=='PASS_PRETRAINED_PROJECTION_INPUTS'
 pmap={r['clip_id']:r for r in projected['rows']}
 guides=support['guides'];guide_ids=[g['canonical_stage_id'] for g in guides]
 # Structured observation descriptors, not hard linguistic correctness rules.
 # All are saved canonical visual spans; source phrase links remain provenance.
 lexicon=['오른손','왼손','양손','펼','닫','주먹','엄지','얼굴','코','입','가슴','어깨','전완','팔꿈치','손목','위','아래','낮','높','모','벌','유지','잠시','옮','바뀌','세우','손가락','삼각']
 vectors=np.array([[float(token in g['description']) for token in lexicon] for g in guides],np.float32)
 guide_words=np.array([WORDS.index(g['word_id']) for g in guides])
 lengths=[next(r['frames'] for r in manifest['rows'] if r['word_id']==w and r['person']==1) for w in WORDS]
 fit=[r for r in manifest['rows'] if r['role']=='fit']
 pose_all=[];rgb_all=[]
 for r in fit:
  a=np.load(r['input']);p=np.load(pmap[r['clip_id']]['path']);pose_all.append(np.where(a['mask'],a['x'],np.nan));rgb_all.append(p['rgb'])
 x=np.concatenate(pose_all);z=np.concatenate(rgb_all)
 pm=np.nanmean(x,axis=0);ps=np.nanstd(x,axis=0);pm=np.nan_to_num(pm);ps=np.where(np.isfinite(ps)&(ps>1e-4),ps,1)
 zm=z.mean(0);zs=np.maximum(z.std(0),1e-4)
 rows=[];conflicts=0
 for r in manifest['rows']:
  a=np.load(r['input']);p=pmap[r['clip_id']];assert sha(Path(p['path']))==p['sha256'];p=np.load(p['path'])
  n=len(a['x']);y=np.zeros((n,7),np.float32);ym=np.zeros_like(y);sy=np.full(n,-1,np.int64);sw=np.zeros(n,np.float32)
  for l in r['labels']:
   ix=int(np.flatnonzero(a['frame_id']==l['center_frame'])[0]);y[ix]=l['y'];ym[ix]=l['target_mask']
  overlaps=np.zeros(n,np.int32)
  for s in r['stages']:
   lo,hi=s['frames'];ix=np.flatnonzero((a['frame_id']>=lo)&(a['frame_id']<=hi));overlaps[ix]+=1
   sy[ix]=guide_ids.index(s['canonical_stage_id']);sw[ix]=1. if r['person']==1 else .2
  conflicts+=int((overlaps>1).sum());sy[overlaps>1]=-1;sw[overlaps>1]=0
  ci=np.full(n,-1,np.int64)
  if r['role']=='fit':
   mapping=read(OLD/'stage_mappings'/f"{r['clip_id']}.json")
   buckets=defaultdict(list)
   # Read the explicit stored convention, never infer orientation from names.
   assert mapping['path_axes']==['other_reference_native_index','canonical_native_index'],mapping['path_axes']
   assert len(mapping['reference_frame_id'])==n
   for ri,ca in mapping['path']:buckets[int(ri)].append(int(ca))
   for i,vs in buckets.items():ci[i]=int(round(np.mean(vs)))
  norm=(a['x']-pm)/ps;norm=np.where(a['mask'],norm,0).astype(np.float32)
  rows.append(dict(clip_id=r['clip_id'],word=WORDS.index(r['word_id']),person=r['person'],role=r['role'],
   pose=np.concatenate([norm,a['mask']],1).astype(np.float32),pose_y=norm,pose_m=a['mask'].astype(np.float32),
   rgb=p['rgb'],prelinear=p['prelinear'],y=y,ym=ym,stage_y=sy,stage_w=sw,canonical_index=ci,
   frame_id=a['frame_id'],timestamp_us=a['timestamp_us']))
 stats=dict(pose_mean=pm.tolist(),pose_std=ps.tolist(),rgb_mean=zm.tolist(),rgb_std=zs.tolist(),
  normalized_fit_people=list(range(1,10)),overlapping_weak_frames_excluded=conflicts,
  guide_lexicon=lexicon,guide_vectors=vectors.tolist(),guide_words=guide_words.tolist(),prototype_lengths=lengths)
 return rows,stats

def metrics(y,p,m):
 per=[]
 for j in range(7):
  known=m[:,j]>0;gt=y[known,j]>.5;pr=p[known,j]>=.5
  pos=gt.sum();neg=(~gt).sum();rec=[float((pr[gt]).mean()) if pos else None,float((~pr[~gt]).mean()) if neg else None]
  per.append(dict(target=read(U/'INPUT_MANIFEST.json')['targets'][j],known=int(known.sum()),positive=int(pos),negative=int(neg),
    accuracy=float((pr==gt).mean()) if len(gt) else None,balanced_accuracy=float(np.mean(rec)) if pos and neg else None,
    positive_recall=rec[0],negative_recall=rec[1]))
 return dict(macro_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in per if r['balanced_accuracy'] is not None])),targets=per)

def run(condition,seed,rows,stats,frozen):
 dest=U/'runs'/f'{condition}_{seed}';assert not dest.exists(),'Do not rerun completed or partial experiment';dest.mkdir(parents=True)
 torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4)
 projection=np.load(U/'PRETRAINED_PROJECTION.npy')
 model=GuideFeedbackNet(projection,np.array(stats['guide_vectors']),np.array(stats['guide_words']),stats['prototype_lengths'],condition=='projection_finetune')
 model.rgb_mean.copy_(torch.tensor(stats['rgb_mean']));model.rgb_std.copy_(torch.tensor(stats['rgb_std']))
 initial_weight=model.tuned_weight().detach().clone();assert torch.equal(initial_weight,torch.from_numpy(projection))
 params=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('update_')]
 groups=[dict(params=params,lr=.001)]
 if condition=='projection_finetune':groups.append(dict(params=[model.update_a,model.update_b],lr=.0002))
 opt=torch.optim.AdamW(groups,weight_decay=.01)
 fit=[r for r in rows if r['role']=='fit'];development=[r for r in rows if r['role']=='development']
 all_y=np.concatenate([r['y'] for r in fit]);all_m=np.concatenate([r['ym'] for r in fit]);pos=(all_y*all_m).sum(0);neg=((1-all_y)*all_m).sum(0)
 pw=torch.tensor(np.clip(neg/np.maximum(pos,1),.25,4),dtype=torch.float32)
 log=[];t=time.perf_counter()
 for epoch in range(frozen['epochs']):
  model.train();order=np.random.permutation(len(fit));losses=[]
  for start in range(0,len(order),6):
   b=collate([fit[i] for i in order[start:start+6]])
   o=model(b['pose'],b['rgb'],b['prelinear'],b['lengths'],b['word'])
   attr=(F.binary_cross_entropy_with_logits(o['attributes'],b['y'],reduction='none',pos_weight=pw)*b['ym']).sum()/b['ym'].sum().clamp_min(1)
   ce=F.cross_entropy(o['stage'].transpose(1,2),b['stage_y'],ignore_index=-1,reduction='none');stage=(ce*b['stage_w']).sum()/b['stage_w'].sum().clamp_min(1)
   pm=b['pose_m']*o['valid'][:,:,None];recon=(F.smooth_l1_loss(o['pose'],b['pose_y'],reduction='none')*pm).sum()/pm.sum().clamp_min(1)
   valid=(b['canonical_index']>=0)&o['valid'];ci=b['canonical_index'].clamp_min(0)
   proto=model.prototype[b['word'][:,None],ci];align=((o['latent']-proto).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
   loss=attr+.35*stage+.05*recon+.1*align
   assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();losses.append(float(loss.detach()))
  if epoch%10==0 or epoch==frozen['epochs']-1:
   rec=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.perf_counter()-t);log.append(rec);print(condition,seed,rec,flush=True)
 model.eval();ys=[];ps=[];ms=[];prediction_rows=[];latency=[]
 with torch.inference_mode():
  for r in rows:
   b=collate([r]);t0=time.perf_counter();out=model(b['pose'],b['rgb'],b['prelinear'],b['lengths'],b['word']);latency.append(time.perf_counter()-t0)
   prob=out['attributes'][0].sigmoid().numpy();stage=out['stage'][0].softmax(-1).numpy();latent=out['latent'][0].numpy()
   path=dest/'predictions'/f"{r['clip_id']}.npz";path.parent.mkdir(exist_ok=True)
   np.savez_compressed(path,probability=prob,stage_probability=stage,latent=latent,frame_id=r['frame_id'],timestamp_us=r['timestamp_us'])
   prediction_rows.append(dict(clip_id=r['clip_id'],role=r['role'],path=str(path),sha256=sha(path)))
   if r['role']=='development':ys.append(r['y']);ps.append(prob);ms.append(r['ym'])
 weight=model.tuned_weight().detach();delta=float(torch.linalg.norm(weight-initial_weight));base=float(torch.linalg.norm(initial_weight))
 assert (delta>0)==(condition=='projection_finetune')
 ckpt=dest/'model.pt';torch.save(model.state_dict(),ckpt)
 # Merge real adapted pretrained weight for unambiguous fine-tuning provenance.
 np.save(dest/'signrep_fc1_tuned.npy',weight.numpy(),allow_pickle=False)
 report=dict(status='TRAINED_FINAL_EPOCH_PUBLIC_DEVELOPMENT',condition=condition,seed=seed,epochs=frozen['epochs'],
  new_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
  pretrained_layer='SignRep fc.1.weight',low_rank=8,pretrained_weight_delta_l2=delta,relative_weight_delta=delta/base,
  full_backbone_finetuned=False,training_seconds=time.perf_counter()-t,head_median_seconds=float(np.median(latency)),
  whole_video_latency_measured=False,development=metrics(np.concatenate(ys),np.concatenate(ps),np.concatenate(ms)),
  stage_development_accuracy=None,stage_reason='No independent full-stage held-out labels yet; DTW-projected labels are not test truth.',
  checkpoint_sha256=sha(ckpt),predictions=prediction_rows,training_log=log,
  frozen_sha256=sha(U/'FROZEN_TRAINING.json'),new_educational_error_labels=0)
 with (dest/'COMPLETE.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
 print('COMPLETE',condition,seed,report['development']['macro_balanced_accuracy'],flush=True)
 return report

def main():
 frozen=read(U/'FROZEN_TRAINING.json');assert frozen['seeds']==[42,43,44] and frozen['conditions']==['frozen_projection','projection_finetune']
 assert not (U/'TRAINING_REPORT.json').exists()
 rows,stats=assemble()
 if not (U/'NORMALIZATION.json').exists():save('NORMALIZATION.json',stats)
 else:assert read(U/'NORMALIZATION.json')==stats
 reports=[]
 for condition in frozen['conditions']:
  for seed in frozen['seeds']:
   dest=U/'runs'/f'{condition}_{seed}'
   if (dest/'COMPLETE.json').exists():
    rec=read(dest/'COMPLETE.json');assert rec['frozen_sha256']==sha(U/'FROZEN_TRAINING.json');assert sha(dest/'model.pt')==rec['checkpoint_sha256'];reports.append(rec)
   else:reports.append(run(condition,seed,rows,stats,frozen))
 save('TRAINING_REPORT.json',dict(status='SIX_RUNS_COMPLETE_NOT_DEPLOYMENT_ACCEPTANCE',runs=reports,
   held_out_same_words=True,development_is_previously_used=True,independent_educational_validation=False,
   no_user_mentor_data=True,next='Verify arithmetic, guide alignment and feedback controls before adoption.'))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'TRAIN_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
