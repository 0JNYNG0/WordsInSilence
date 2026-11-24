from model import *
assert not(U/'TRAIN_CONFIG.json').exists()
base=np.load(N/'TRAIN_INPUTS.npz');meta=read(N/'TRAIN_INPUTS.json')['shape_rows'];sx=list(base['shape_x']);sy=list(base['shape_y']);sm=list(base['shape_mask']);newflag=[False]*len(meta);lookup={(r['clip_id'],r['side'],r['frame']):i for i,r in enumerate(meta)};cache={}
def source(r):
 key=r['channel_path'],r['hand']
 if key not in cache:
  x,f=load_channels(r['channel_path']);s,a=features(x,r['hand']);cache[key]=(s,a,f)
 return cache[key]
new=[r for r in read(U/'LABELS.json')['rows']if r['role']=='fit']
for r in new:
 assert r['person']<=9
 s,a,f=source(r);j=int(np.flatnonzero(f==r['frame'])[0]);y=np.array(r['shape']);mask=(y>=0)&(np.array(r['quality'])==1)&a[j];key=r['clip_id'],r['hand'],r['frame']
 if key in lookup:
  i=lookup[key];sy[i]=y;sm[i]=mask;meta[i]=r|dict(side=r['hand'],origin='dense_source_audit_replaces_overlap');continue
 lookup[key]=len(meta);sx.append(windows(s,[j])[0]);sy.append(y);sm.append(mask);meta.append(r|dict(side=r['hand'],origin='new_dense_fit'));newflag.append(True)
px=[];py=[];pm=[];pmeta=[]
for r in read(Q/'FIT_LABELS_V2.json')['rows']+new:
 if not any(v>=0 for v in r['phase']):continue
 assert r['person']<=9 and r['role']=='fit';s,a,f=source(r);j=int(np.flatnonzero(f==r['frame'])[0]);y=np.array(r['phase']);mask=(y>=0)&(np.array(r['quality'])==1)&a[j]
 px.append(windows(s,[j])[0]);py.append(y);pm.append(mask);pmeta.append(r)
sx=np.array(sx,np.float32);sy=np.array(sy);sm=np.array(sm);px=np.array(px,np.float32);py=np.array(py);pm=np.array(pm)
assert np.isfinite(sx).all()and np.isfinite(px).all();assert len({(r['clip_id'],r['side'],r['frame'])for r in meta})==len(meta)
np.savez_compressed(U/'TRAIN_INPUTS.npz',state_x=sx,state_y=sy,state_mask=sm,is_new=np.array(newflag),phase_x=px,phase_y=py,phase_mask=pm)
write(U/'TRAIN_INPUTS.json',dict(state_rows=meta,phase_rows=pmeta,summary=dict(state_anchors=len(meta),state_known=int(sm.sum()),words=len({r['word']for r in meta}),videos=len({r['clip_id']for r in meta}),people=sorted({r['person']for r in meta}),new_anchors=sum(newflag),phase_anchors=len(pmeta),phase_known=int(pm.sum()),phase_change=int(((py==1)&pm).sum()))))
pins={str(p):sha(p)for p in [U/'PLAN.json',U/'LABELS.json',U/'model.py',U/'TRAIN_INPUTS.npz',Q/'FIT_LABELS_V2.json',N/'TRAIN_INPUTS.npz']+[P/'shared_finger_chain'/f'{seed}.pt'for seed in [42,43,44]]}
cfg=dict(seeds=[42,43,44],state_steps=450,state_lr=.0002,phase_steps=400,phase_lr=.0005,weight_decay=.0001,state_threshold=.9,phase_threshold=.85,new_weight=.35,batch='all_old + weighted_all_dense',initialization='State branch copies admitted SharedFinger weights but sees ONLY one actual central frame (no replicated neighbors); fine-tune. Separate phase head Conv+BiGRU is newly supervised; no shared gradient.',phase_loss='inverse-frequency balanced from FIT labels only',variants=['center_only','change_override','change_fill'],release_priority=['change_override','change_fill'],selection='No best seed, no early stopping, no threshold changes after diagnostics; all three seeds must agree. No model output before GT froze.',fusion='override: confident current-state + change consensus replaces base. fill: same but only previously unknown and no old conflict. Others retain base.',pins=pins)
write(U/'TRAIN_CONFIG.json',cfg)
write(U/'PLAN_CLARIFICATION.json',dict(before_any_candidate_inference=True,architecture='Use one-frame execution of copied pretrained Conv+GRU rather than random center MLP, preserving learned geometry parameters while physically excluding neighbor influence. Change head separate.',source_limit='Dense FIT yields one clear short open plateau; no new confirmed wrong-chain FIT examples. Existing quality masks retained. No quality repair is released; new held-out little/ thumb confusion1 separately reported.',labels='Ambiguous partial bend/extension centers are -1, not guessed from the word or neighbors. Change may be1 even when center-state is uncertain.'))
print(read(U/'TRAIN_INPUTS.json')['summary'])
