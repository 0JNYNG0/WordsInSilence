"""Fixed stored-prediction diagnostics and incremental-head latency only."""
import time,numpy as np,torch
from experiment import U,read,write,sha,assemble,projection,metrics
from head import Head
def main():
 info=read(U/'DATASET.json');plan=read(U/'TRAIN_FROZEN.json');targets=info['selected_targets'];ids=info['selected_indices']
 y=np.array([r['y'] for r in info['rows']])[:,ids];m=np.array([r['target_mask'] for r in info['rows']],bool)[:,ids]
 details=[];contradictions=[];subgroups=[]
 for condition in plan['conditions']:
  probabilities=[]
  for seed in plan['seeds']:
   p=np.load(U/'runs'/f'{condition}_{seed}'/'probabilities.npy',allow_pickle=False);probabilities.append(p)
   for group_key in ['person','word_id']:
    for value in sorted({r[group_key] for r in info['rows']}):
     for role in ['fit','development','new_feature_expression_evaluation']:
      ix=[i for i,r in enumerate(info['rows']) if r[group_key]==value and r['role']==role]
      if ix:subgroups.append(dict(condition=condition,seed=seed,role=role,group_key=group_key,group_value=value,windows=len(ix),**metrics(p[ix],y[ix],m[ix],targets)))
   for role in ['fit','development','new_feature_expression_evaluation']:
    ix=[i for i,r in enumerate(info['rows']) if r['role']==role];px=p[ix];mx=m[ix]
    groups=[]
    for h in 'RL':
     shape=[j for j,t in enumerate(targets) if t.startswith(h+'_') and t.split('_',1)[1] in ['index_only','open_four','closed_four']]
     motion=[j for j,t in enumerate(targets) if t.startswith(h+'_motion_')]
     for kind,jj in [('shape',shape),('motion',motion)]:
      if len(jj)>1:groups.append(dict(side=h,kind=kind,targets=[targets[j] for j in jj],multi_positive_windows=int(((px[:,jj]>=.5).sum(1)>1).sum()),windows=len(ix)))
    jj=[j for j,t in enumerate(targets) if t in ['R_higher','L_higher']]
    if len(jj)==2:groups.append(dict(side='both',kind='bilateral_height',targets=[targets[j] for j in jj],multi_positive_windows=int(((px[:,jj]>=.5).sum(1)>1).sum()),windows=len(ix)))
    contradictions.append(dict(condition=condition,seed=seed,role=role,exclusive_groups=groups,unknown_label_cells=int((~mx).sum()),unknown_label_cells_predicted_positive=int(((~mx)&(px>=.5)).sum()),unknown_not_scored_as_true=True))
  probs=np.stack(probabilities)
  for i,r in enumerate(info['rows']):
   details.append(dict(condition=condition,sample_id=r['sample_id'],role=r['role'],targets={t:dict(known=bool(m[i,j]),truth=int(y[i,j]) if m[i,j] else None,probabilities=probs[:,i,j].tolist(),all_three_wrong=bool(m[i,j] and np.all((probs[:,i,j]>=.5)!=y[i,j]))) for j,t in enumerate(targets)}))
 # Runtime repeats run only the chosen head checkpoint on identical prepared input; no new scores or epoch choice.
 torch.set_num_threads(4);(x,z,_,_),_=assemble(info,read(U/'NORMALIZATION.json'));latencies=[]
 for condition in plan['conditions']:
  head=Head(len(targets));path=U/'runs'/f'{condition}_42'/'head.pt';head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));head.eval()
  with torch.inference_mode():
   sample_x=x[:3];sample_z=z[:3] if condition=='pose_plus_signrep' else sample_x@torch.from_numpy(projection())
   for _ in range(10):head(sample_x,sample_z)
   tt=[]
   for _ in range(100):
    start=time.perf_counter();head(sample_x,sample_z);tt.append((time.perf_counter()-start)*1000)
  latencies.append(dict(condition=condition,seed=42,unit='milliseconds per3 prepared13-frame windows',median=float(np.median(tt)),p95=float(np.percentile(tt,95)),raw_milliseconds=tt,samples=100,threads=4,checkpoint_sha256=sha(path)))
 prep=read(U/'INPUTS.json')['rows'];new=[r for r in prep if not r['reused_input']]
 write(U/'DIAGNOSTICS.json',dict(details=details,subgroups=subgroups,contradictions=contradictions,latencies=latencies,encoder_preparation=dict(new_clips=len(new),total_rgb_seconds=sum(r['rgb_seconds'] for r in new),per_clip_seconds=[r['rgb_seconds'] for r in new]),limitations=['Head latency excludes video decode, MediaPipe, SignRep and feedback construction. It is not recording-end-to-feedback latency.','Unknown RGB labels are not negative evidence. No confidence-based abstention has been validated.','Prediction supplied exact annotation windows; full-video automatic stage segmentation is not evaluated.','Projected2D height and motion labels do not prove contact,3D direction,mandatory repetition or linguistic correctness.','Frozen SignRep features use16-frame source windows with interpolation and can see outside the13-frame head window; noncausal, not13-RGB-frame-only.']))
 print(json.dumps(latencies))
if __name__=='__main__':main()
