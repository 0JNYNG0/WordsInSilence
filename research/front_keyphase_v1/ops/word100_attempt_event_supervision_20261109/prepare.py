from unit import *
import shutil
def main():
 assert not (U/'INPUTS.json').exists();torch.set_num_threads(4)
 protected=read(P/'FROZEN.json')['source_pins'];protected.update(read(J/'FROZEN.json')['pins'])
 write(U/'AUTHORIZATION.json',dict(request='2026-11-09 user: execute proposed1-4; separate attempted action localization and shape/direction/omission supervision; public-only; preserve C/T3-F',private_access=False,download=False,deployment='Only separately validated visual evidence; no default model replacement'))
 write(U/'DESIGN.json',dict(conditions=['normal_only','attempt_supervised'],primary='attempt_supervised_42',split='Internal FIT01-06 / FIT07-09 check, then36FIT final /12 historical DEV once. No DEV tuning.',epochs=100,lr=.001,batch=6,seeds=[42,43,44],shape_backbone='Frozen shared100_42',architecture='Existing137-input temporal head + separate pooled event-presence head. Shared Conv+GRU observation weights frozen.',direction='Existing independently supervised MotionNet retained and checked on new automatically found intervals, never allowed to override contradictory geometric evidence.',interventions='Public coordinate engineering controls, not independent actual learner errors. Shape interventions preserve palm and wrist. Source-index mapping preserves event identities under time reversal.',gates=['No increased normal false correction','No missing observations counted normal','Opposite shape/direction attempts retained for comparison','No same-word exceptions or DEV thresholds','Event missing separate from sensor missing'],protected=protected))
 for f in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md']:
  target=U/'before'/f;target.parent.mkdir(exist_ok=True);shutil.copy2(O.parents[2]/f,target)
 original={r['clip_id']:r for r in prior.common.load_rows() if r['word']>=0};old={r['clip_id']:r for r in prior.load_rows()};ann={r['clip_id']:r for r in read(S/'ANNOTATIONS.json')['rows']}
 donor=original['NIA_SL_WORD0739_REAL01_F']['raw_x'][70,:102].reshape(51,2)[30:51].copy()
 net=prior.common.SharedNet();net.load_state_dict(torch.load(prior.CHECKPOINT,weights_only=True,map_location='cpu'));net.eval();out=[];pending=[]
 for cid,r in original.items():
  src=old[cid];raw=r['raw_x'];n=len(raw);sy=src['stage_y'].copy();sw=src['stage_w'].copy()
  # DEV targets stored separately for evaluation only; never used by training.
  if r['role']=='development':sy=np.where(r['stage_gold']>=0,r['stage_gold']+1,-1);sw=(sy>=0).astype(float)
  main=np.flatnonzero(sy>0)
  if r['role']=='development':main=np.flatnonzero(r['stage_gold']>=0)
  a,b=int(main[0]),int(main[-1]);variants=['full_normal','upper_normal','close_R','close_L','close_both','reverse_main','close_reverse','speed075','speed125','gap2','all_hand_missing']
  # Only two multi-action descriptions have a reviewed last event separable
  # from its predecessor. Repetition count and optional holds are not omissions.
  cut=None;removed_groups=[]
  if cid in ann and r['word_id'] in ['WORD0739','WORD2523']:
   prev=1 if r['word_id']=='WORD0739' else 10;omitted=[2] if prev==1 else [11,12]
   cut=max(ann[cid]['stages'][str(prev)]);removed_groups=sorted({GROUPS[k] for k in omitted});variants+=['omit_final_event','mask_final_event']
  for kind in variants:
   x=raw.copy();ids=np.arange(n);ts=r['timestamp_us'].copy();altered=[]
   if kind=='upper_normal':
    z=np.load(r['channel']);x=z['feedback'] if 'feedback' in z.files else z['upper_mask']
   if kind in ['close_R','close_L','close_both','close_reverse']:
    altered=[1] if kind=='close_R' else [0] if kind=='close_L' else [0,1]
    for side in altered:x=close_hand(x,side,a,b,donor)
   if kind in ['reverse_main','close_reverse']:ids[a:b+1]=ids[a:b+1][::-1];x=x[ids]
   if kind in ['speed075','speed125']:
    ids=np.rint(np.linspace(0,n-1,round(n/(.75 if kind=='speed075' else 1.25)))).astype(int);x=x[ids]
   if kind=='omit_final_event':
    # Retain the last reviewed pre-event frame, remove every following action
    # including the final event, then append existing rest tail. Not half-main.
    tail=np.flatnonzero((src['stage_y']==0)&(np.arange(n)>b)) if r['role']=='fit' else np.arange(max(b+8,n-6),n)
    if len(tail)<3:tail=np.arange(max(b+1,n-5),n)
    ids=np.r_[np.arange(cut+1),tail[-6:]];x=x[ids]
   if kind in ['gap2','all_hand_missing','mask_final_event']:
    lo,hi=(int((a+b)/2),int((a+b)/2)+2) if kind=='gap2' else (cut+1,n) if kind=='mask_final_event' else (0,n)
    x=prior.common.mask_hand(prior.common.mask_hand(x,lo,hi,0),lo,hi,1)
   ts=np.arange(len(x),dtype=np.int64)*int(np.median(np.diff(r['timestamp_us'])));x[:,-1]=np.r_[0,np.diff(ts)/1e6]
   yy=r['y'][ids].copy();ym=r['ym'][ids].copy();target=sy[ids].copy();weight=sw[ids].copy()
   # Changed fingers are not relabelled as original shape. The pose teacher stays
   # frozen; only the known normal actual-vs-expected labels train difference.
   if altered:
    for side in altered:ym[:,[2,4] if side==0 else [0,1,3]]=0
   avail=prior.common.available(x);ym*=avail
   present=np.zeros(10,np.float32);pmask=(GROUP_WORD==r['word']).astype(np.float32);present[pmask>0]=1
   if kind=='omit_final_event':present[removed_groups]=0
   if kind in ['all_hand_missing','mask_final_event']:pmask[:]=0;weight[:]=0
   if kind=='gap2':weight[lo:hi]=0
   # All visibility absence is excluded from location/shape supervision; not rest.
   shoulder=(x[:,187:189]>.5).all(1);weight[~shoulder]=0
   with torch.inference_mode():pr=net(torch.from_numpy(prior.common.normalize(x)[None]),torch.tensor([len(x)]),torch.tensor([r['word']]))
   data=dict(raw=x,source_indices=ids,timestamp_us=ts,frame_id=r['frame_id'][ids],h=pr['latent'][0].numpy(),p=pr['attributes'][0].sigmoid().numpy(),g=prior.geometry(x),y=yy,ym=ym,sy=target,sw=weight.astype(np.float32),present=present,pmask=pmask,stage_gold=r['stage_gold'][ids])
   path=U/'inputs'/f'{cid}_{kind}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**data)
   out.append(dict(clip_id=cid,word=r['word'],word_id=r['word_id'],person=r['person'],role=r['role'],kind=kind,path=str(path),sha256=sha(path),source_sha256=r['source_sha256'],main=[a,b],cut=cut,removed_groups=removed_groups if kind=='omit_final_event' else [],synthetic=kind not in ['full_normal','upper_normal']))
 # Inner training subjects alone set preprocessing; no validation statistics.
 gs=np.concatenate([np.load(r['path'])['g'] for r in out if r['person']<=6 and r['kind']=='full_normal']);mu=gs.mean(0);sd=gs.std(0).clip(.05);mu[22:33]=0;sd[22:33]=1;mu[55:]=0;sd[55:]=1
 write(U/'NORMALIZATION.json',dict(mean=mu.tolist(),std=sd.tolist(),fit_record_ids=list(range(1,7))))
 for r in out:
  z=np.load(r['path']);d={k:z[k].copy() for k in z.files};g=d['g'];v=g[:,22:33]>.5;vv=g[:,55:]>.5;g=(g-mu)/sd;g[:,:22]*=np.repeat(v,2,1);g[:,33:55]*=np.repeat(vv,2,1);d['g']=g.astype(np.float32);np.savez_compressed(r['path'],**d);r['sha256']=sha(r['path'])
 write(U/'INPUTS.json',dict(rows=out,original_fit=36,original_development=12,private_access=False,synthetic_not_real_error=True,reviewed_omission='Pre-event anchor to last action removed; append original tail; events same Codex RGB reviewed, not expert semantic correctness'))
 write(U/'FROZEN.json',dict(**read(U/'DESIGN.json'),input_sha256=sha(U/'INPUTS.json'),code_pins={str(p):sha(p) for p in [U/'unit.py',U/'prepare.py']},base_sha256=sha(prior.CHECKPOINT)))
 write(U/'CURRENT_EXECUTION.json',dict(status='PREPARED',pid=None));event('prepared',cases=len(out),fit=sum(r['role']=='fit' for r in out),dev=sum(r['role']=='development' for r in out))
if __name__=='__main__':main()
