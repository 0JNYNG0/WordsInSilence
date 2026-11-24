"""Direct candidate ranking with source-extent bags; no educational completion target."""
from ranking_core import *
import time,traceback
def main():
 K.mkdir(exist_ok=True);assert not(K/'INPUTS_COMPLETE.json').exists();torch.set_num_threads(4);old=read(B/'INPUTS_COMPLETE.json');added=read(D/'SOURCE_EPISODES_ADDED.json');assert len(added['rows'])==8 and {r['person']for r in added['rows']}=={3,7};seq=list(old['sequences']);src={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role']=='fit'};cr={r['clip_id']:r for r in read(D/'CROP_INPUTS_COMPLETE.json')['rows']};co={r['clip_id']:r for r in added['contrasts']};rc=load('_rank_role',O/'word100_candidate_role_learning_20261111/role_core.py');net=rc.frozen_model();sys.path.insert(0,str(O/'word100_observation_guard_20261108'));guard=load('_rank_guard',O/'word100_observation_guard_20261108/guard.py');ad=load('_rank_ad',O/'word100_rtmw_shared_input_20261108/adapter.py');pins={};obs=[]
 cfg=dict(at=stamp(),primary='rank_extended',controls=['rank_original','boundary_extended'],references='Saved source_boundary_candidate_v3 models/results only; no old reruns',hypotheses=['Direct within-source candidate ranking better matches selecting an action extent than multiplying independent classifications.','Additional existing FIT performers may improve onset/end generalization.'],seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,rank_architecture='body-only temporal-third means+variation600 ->96 GELU ->5 scores',rank_loss='Per-source/family listwise logsumexp(all sampled candidates)-logsumexp(all source-band-compatible candidates)',rank_batch='2 bags per family per step; all positive candidates plus 56 sampled negative candidates; seeded sampling, final epoch only',rank_sampling='Up to512 negatives per bag from reviewed source disagreement, seed19; true onset/end bands and all observed progression retained; unlabelled empty windows excluded',data_original='16 FIT clips persons02/04/05/06 +8 probe08/09',data_extended='Add8 FIT clips persons03/07; same8 probe; all already upstream common100-exposed',views=['full','upper','actual_crop'],reverse='whole-source reverse for action attempt only; not correct-sign label',body_only=True,rank_runtime='raw rank score, admitted by frozen same-seed body_window presence>=.5 and body visibility. No expected order or new threshold.',boundary_extended='same architecture/loss/steps as B direct_boundary, only more FIT sources',selection=dict(conditions=['baseline','body_window','direct_boundary','rank_original','rank_extended','boundary_extended']),semantic_completion=None,no_final8=True,no_private=True,no_web=True,pins={str(p):sha(p)for p in[Path(__file__),W/'ranking_core.py',W/'source_boundary_core.py',B/'FROZEN.json',B/'INPUTS_COMPLETE.json',B/'VERIFICATION.json',D/'SOURCE_EPISODES_ADDED.json',D/'CROP_INPUTS_COMPLETE.json']});write(K/'FROZEN.json',cfg)
 for s in added['rows']:
  cid=s['clip_id'];sr=src[cid];assert sha(sr['path'])==sr['sha256'] and sha(sr['channel'])==sr['channel_sha256'];ch=np.load(sr['channel']);z=np.load(sr['path']);ts=z['timestamp_us'];groups,bg=source_witnesses(s,co[cid]);r=cr[cid];assert sha(r['path'])==r['sha256'];cz=np.load(r['path']);assert np.array_equal(ts,cz['timestamp_us']);clean=guard.quarantine(cz['xy_px'],cz['score'],ts,640,480);cx=ad.convert(clean['xy_px'],cz['score'],ts,cz['frame_id'],640,480)['tcn280'];views={'full':ch['full'],'upper':ch['feedback'] if 'feedback'in ch.files else ch['upper_mask'],'actual_crop':cx}
  pins[sr['path']]=sr['sha256'];pins[sr['channel']]=sr['channel_sha256'];pins[r['path']]=r['sha256'];obs.append(dict(clip_id=cid,guarded_hands=int(clean['hand_quarantined'].sum()),frames=len(ts),all_timestamps_preserved=True))
  for view,x in views.items():
   q=rc.old.relation_features(x);qn=(q-net.qm.numpy())/net.qs.numpy();qn=np.concatenate([qn[...,:4]*q[...,8,None],qn[...,4:8]*q[...,9,None],q[...,8:]],-1);feat=body_context(qn,ts);rev=body_context(qn[::-1],ts[-1]-ts[::-1]);y=np.zeros((len(ts),5),np.float32);mask=np.zeros_like(y);bands=np.zeros((len(ts),5,2),bool);valid=np.zeros_like(bands)
   for g in s['groups']:
    f=g['family'];j=F.index(f);good=q[:,AX[f]*2,8]>.5;motion=set(groups[f]['motion']);other=set(bg)|{i for k,v in groups.items()if k!=f for i in v['motion']+v['held']}
    for i in motion:
     if good[i]:y[i,j]=1;mask[i,j]=1
    for i in other:
     if good[i]:mask[i,j]=1
    for h,key in enumerate(['onset','offset']):
     aa,bb=g[key];bands[aa:bb+1,j,h]=good[aa:bb+1]
     if bands[:,j,h].any():valid[:,j,h]=good
   for flip in [False,True]:
    a=dict(x=rev if flip else feat,y=y[::-1].copy()if flip else y,mask=mask[::-1].copy()if flip else mask,bands=bands[::-1,:,::-1].copy()if flip else bands,valid=valid[::-1,:,::-1].copy()if flip else valid);p=K/'sequences'/f'{cid}_{view}_{int(flip)}.npz';p.parent.mkdir(exist_ok=True);np.savez_compressed(p,**a);seq.append(dict(clip_id=cid,person=s['person'],view=view,reverse=flip,split='train',path=str(p),sha256=sha(p),source_sha256=s['source_sha256']))
 bags=[];rng=np.random.default_rng(19)
 for i,r in enumerate(seq):
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);n=len(z['x']);grid=np.array([(a,b)for a in range(n-3)for b in range(a+3,n)],np.int32);a,b=grid.T;nn=b-a+1
  for j,f in enumerate(F):
   va=z['valid'][:,j].any(1);bs=z['bands'][:,j,0];be=z['bands'][:,j,1];pos=(z['mask'][:,j]>0)&(z['y'][:,j]>0);neg=(z['mask'][:,j]>0)&(z['y'][:,j]==0)
   if not bs.any()or not be.any()or not pos.any():continue
   counts=lambda v:np.r_[0,v.cumsum()][b+1]-np.r_[0,v.cumsum()][a]
   eligible=va[a]&va[b]&(counts(va)>=.8*nn);positive=eligible&bs[a]&be[b]&(counts(pos)==pos.sum())&(counts(neg)==0)
   negative=eligible&~positive&((counts(neg)>0)|((counts(pos)>0)&(~bs[a]|~be[b])))
   pi=np.flatnonzero(positive);ni=np.flatnonzero(negative)
   if not len(pi)or not len(ni):bags.append(dict(clip_id=r['clip_id'],view=r['view'],reverse=r['reverse'],family=f,skipped=True,positive=len(pi),negative=len(ni)));continue
   if len(ni)>512:ni=np.sort(rng.choice(ni,512,replace=False))
   ix=np.r_[pi,ni];sp=grid[ix];v=pool_windows(z['x'],sp);yy=np.r_[np.ones(len(pi),bool),np.zeros(len(ni),bool)];dest=K/'bags'/f'{i}_{f}.npz';dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,x=v,y=yy,spans=sp);bags.append(dict(clip_id=r['clip_id'],person=r['person'],view=r['view'],reverse=r['reverse'],family=f,split=r['split'],extended_only=r['person']in[3,7],path=str(dest),sha256=sha(dest),positive=len(pi),negative=len(ni),skipped=False,source_sha256=r['source_sha256']));assert np.isfinite(v).all()
  print('rank bags',i+1,len(seq),flush=True)
 assert len(seq)==192;write(K/'INPUTS_COMPLETE.json',dict(at=stamp(),sequences=seq,bags=bags,pins=pins,added_observation_audit=obs,source_clips=32,fit_original=16,fit_extended=24,probe_clips=8,no_new_expert_gold=True,labels_scope='Observed source extent, not mandatory sign duration or repetition count',first_label_source=str(D/'SOURCE_EPISODES_ADDED.json')));print('Ranking and expanded-boundary inputs ready')
if __name__=='__main__':
 try:main()
 except BaseException:K.mkdir(exist_ok=True);(K/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
