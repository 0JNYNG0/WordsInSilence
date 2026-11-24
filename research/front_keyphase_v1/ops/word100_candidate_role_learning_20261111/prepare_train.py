from role_core import *
from collections import Counter
import os,traceback

def main():
 assert not (U/'TRAIN_INPUTS_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cfg=read(U/'FROZEN.json');assert sha(U/'ROLE_LABELS.json')==cfg['pins'][str(U/'ROLE_LABELS.json')]
 labels={r['clip_id']:r for r in read(U/'ROLE_LABELS.json')['rows']};anns=read(A/'FIT_LABELS.json')['rows'];rr=[r for r in old.rows() if r['role']=='fit' and (r['clip_id'] in labels or any(a['clip_id']==r['clip_id'] for a in anns))]
 net=frozen_model();sys.modules['joint_core']=jc;assignmod=load('_role_old_assignment',J/'proposal_adapter.py');base=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));base.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval();base.requires_grad_(False)
 index=[];pairs_manifest=[];summary=Counter();write(U/'CURRENT_EXECUTION.json',dict(status='PREPARING_FEATURES',pid=os.getpid()))
 for r in rr:
  n=len(r['frame_id']);lab=labels.get(r['clip_id']);yy=np.asarray(lab['y']) if lab else np.full(n,-1);ww=np.asarray(lab['w'],np.float32) if lab else np.zeros(n,np.float32);aa=[a for a in anns if a['clip_id']==r['clip_id']];cache=np.load(B/'cache'/f"{r['clip_id']}.npz")
  for view in ['full','upper']:
   for reverse in ([False,True] if lab else [False]):
    x=prior.physical(r['x_'+view]);ts=r['timestamp_us'];y=yy.copy();w=ww.copy();an=copy.deepcopy(aa);idx=np.arange(n);positive=np.flatnonzero((y>0)&(y<8)&(w>0))
    if reverse:
     lo,hi=int(positive.min()),int(positive.max());idx[lo:hi+1]=idx[lo:hi+1][::-1];x=x[idx].copy();y=y[idx];w=w[idx]
     fh,_=lc.be.ev.frozen_forward(base,x,r['word'])
     for a in an:
      for name in ['holds','background']:
       converted=[]
       for u,v in a[name]:
        if lo<=u<=v<=hi:converted.append([lo+hi-v,lo+hi-u])
        elif v<lo or u>hi:converted.append([u,v])
       a[name]=converted
      ev=[]
      for e in a['events']:
       u,v=e['onset'][0],e['offset'][1]
       if lo<=u<=v<=hi:ev.append(dict(e,onset=[lo+hi-z for z in e['offset'][::-1]],offset=[lo+hi-z for z in e['onset'][::-1]],direction=2-e['direction']))
       elif v<lo or u>hi:ev.append(e)
      a['events']=sorted(ev,key=lambda e:e['onset'][0])
    else:fh=cache['frozen_h_'+view]
    ob=features(net,x,ts,fh);q=ob['q'][0].numpy()
    # Family targets require the relation actually observed, not imputed joints.
    for i,c in enumerate(y):
     ks=[0] if 1<=c<=3 else [4,5] if c==4 else [6,7] if 5<=c<=7 else []
     if ks and not all(q[i,k,-2:].max()>.5 for k in ks):w[i]=0
    row=dict(r,annotations=an);ss,sy,ps,py=jc.example_specs([row]);sources=['reviewed_episode_or_fragment']*len(ss)
    cand=jc.candidate_spans(ob['boundary'],q,[])
    for e in cand:
     a,b,k=e['a'],e['b'],e['axis']*2+(e['moving']=='L');target,eid=assignmod.assign(a,b,k,an)
     if target>=0:ss.append((0,a,b,k));sy.append(target);sources.append('actual_candidate')
    sf=segment_features(net,ob,ss);valid=np.asarray([(q[a:b+1,k,-2:].max(-1)>.5).mean()>=.8 for _,a,b,k in ss],dtype=bool);pp=[];pty=[]
    for (a,b,gap),target in zip(ps,py):
     if valid[a] and valid[b]:pp.append(np.r_[sf[a],sf[b],gap]);pty.append(target)
    # Paired action-role probes use frame source labels only, never score-based labels.
    role_specs=[];role_targets=[]
    for e in cand:
     a,b,k=e['a'],e['b'],e['axis']*2+(e['moving']=='L');known=(w[a:b+1]==1);ys=y[a:b+1][known]
     if len(ys)>=2 and len(set(ys))==1:
      role_specs.append([a,b,k]);role_targets.append(int(ys[0]))
    key=f"{r['clip_id']}_{view}_{'core_reverse' if reverse else 'normal'}";p=U/'train_cache'/f'{key}.npz';p.parent.mkdir(exist_ok=True)
    np.savez_compressed(p,z=ob['z'],y=y,w=w,quality=sf[valid],quality_y=np.asarray(sy,dtype=np.int64)[valid],pair=np.asarray(pp,np.float32).reshape(-1,609),pair_y=np.asarray(pty,np.int64),old_role=ob['old_role'],candidate_spans=np.asarray(role_specs,np.int64).reshape(-1,3),candidate_y=np.asarray(role_targets,np.int64))
    index.append(dict(clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],source_sha256=r['source_sha256'],view=view,reverse=reverse,path=str(p),sha256=sha(p),role_frames=int((w==1).sum()),quality=len(sf[valid]),pairs=len(pp),candidates=len(cand),role_candidate_labels=len(role_specs)));summary.update(dict(sequences=1,role_frames=int((w==1).sum()),quality=len(sf[valid]),pairs=len(pp)));pairs_manifest.extend(dict(clip=r['clip_id'],view=view,reverse=reverse,label=int(y)) for y in pty)
  event('training_inputs_clip',clip=r['clip_id'])
 write(U/'TRAIN_INPUTS_COMPLETE.json',dict(rows=index,counts=dict(summary),unique_public_videos=len(rr),unique_words=len({r['word_id'] for r in rr}),private_data=0,raw_frames_not_independent_samples=True));write(U/'PAIR_TARGET_AUDIT.json',dict(counts=dict(Counter(r['label'] for r in pairs_manifest)),labels=['same_observed_event_fragments','separate_repeat_or_movement_same_axis','different_axis_event'],limitations='Pair labels originate reviewed episode identity; not proof of semantic repeat count or all left-hand motion.'))
 write(U/'CURRENT_EXECUTION.json',dict(status='TRAIN_INPUTS_READY',pid=None));event('training_inputs_complete',**dict(summary))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
