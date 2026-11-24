from common import *
import collections,torch
sys.path[:0]=[str(P),str(O/'word100_guide_requirement_learning_20261111')]
import requirement_core as rc
from finger_core import local_sequence,windows as old_windows,baseline_fingers
from finger_model import features,windows
assert not(U/'TRAIN_FROZEN.json').exists()
torch.set_num_threads(2);net=rc.frozen_observer()
old=read(P/'FINGER_INPUTS.json')['rows'];oldarrays=np.load(P/'finger_arrays.npz')
review=read(U/'SOURCE_REVIEW_INPUTS.json');review['reviewer']='Codex original RGB crop visual recheck of42frames before candidate training; not independent KSL expert';review['accepted']=True;review['changed_labels']=False;write(U/'SOURCE_REVIEW.json',review)
audit=read(U/'ERROR_AUDIT_INPUTS.json');write(U/'ERROR_CAUSE_REVIEW.json',dict(rows=[dict(clip_id=r['clip_id'],frame=r['frame'],side=r['side'],label_maintained=True,source_visible_extended_index=True,other_long_fingers_visible_flexed=True,evidence=r['image'],finding='Source RGB agrees with existing finger truth. Wrong predictions occur during changing orientation/partial overlap; exact learned cause not proven. Preserve DEV labels outside training.')for r in audit['rows']],model_revision='Compare expanded whole-hand temporal model with a shared finger-chain temporal encoder; additional labels same for both candidates.',HDD='W not mounted; required source copies/annotations already on C; no download'))
extra=review['rows'];new=[];keys={(r['clip_id'],r['frame'],r['side'])for r in old}
for r in extra:
    key=r['clip_id'],r['frame'],r['side'];assert key not in keys
    # E->extended, B/F->visibly flexed. Preserve B/F provenance, never call B a fist.
    y=[{0:1,1:0,2:0,-1:-1}[v]for v in r['y']];y[0]=-1
    new.append(dict(clip_id=r['clip_id'],word=r['word'],person=r['person'],role=r['role'],frame=r['frame'],side=r['side'],targets=y,origins=['rechecked_public_digit_or_ordinary_finger_label'],three_state_original=r['y'],auxiliary=True,new=True))
allrows=[dict(r,new=False)for r in old]+new
added={r['clip_id']:r for r in read(E/'INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']};cache={};pins={}
def get(cid):
    if cid in cache:return cache[cid]
    if cid in added:
        rec=added[cid];path=rec['channels'];expected=rec['channels_sha256'];z=np.load(path);x=z['tcn280']
    else:
        rec=read(J/'records'/f'{cid}.json');path=rec['channel_path'];expected=rec['channel_sha256'];z=np.load(path);x=z['feedback']if'feedback'in z.files else z['full']
    assert sha(path)==expected;pins[path]=expected
    with torch.inference_mode():o=net.shared(torch.tensor(rc.prior.shared.normalize(x)[None]),torch.tensor([len(x)]),torch.tensor([0]));latent=o['latent'][0].numpy();pr=o['attributes'][0].sigmoid().numpy()
    cache[cid]=(x,z['frame_id'],latent,pr);return cache[cid]
seq=[];seqold=[];valid=[];base=[]
for i,r in enumerate(allrows):
    x,fid,h,pr=get(r['clip_id']);t=int(np.flatnonzero(fid==r['frame'])[0]);f,k=features(x,r['side']);seq.append(windows(f,[t])[0]);valid.append(k[t]);fl,kl=local_sequence(x,r['side']);seqold.append(old_windows(np.c_[fl,h],[t])[0]);base.append(baseline_fingers(pr,r['side'])[t,1:])
    if i<len(old):assert np.array_equal(oldarrays['targets'][i],r['targets'])and np.allclose(seqold[-1],oldarrays['x'][i],atol=1e-5)
fit=np.array([r['role']=='fit'for r in allrows]);assert not({r['person']for r in allrows if r['role']=='fit'}&{r['person']for r in allrows if r['role']=='development'})
seq=np.array(seq);seqold=np.array(seqold);mu=seq[fit].reshape(-1,14).mean(0);sd=seq[fit].reshape(-1,14).std(0).clip(.05);mo=seqold[fit].reshape(-1,127).mean(0);so=seqold[fit].reshape(-1,127).std(0).clip(.05)
np.savez_compressed(U/'INPUTS.npz',x=seq,x_old=seqold,mean=mu,std=sd,old_mean=mo,old_std=so,availability=np.array(valid),y=np.array([r['targets'][1:]for r in allrows]),baseline=np.array(base))
write(U/'INPUTS.json',dict(rows=allrows,pins=pins,unknown_kept=True))
coverage={}
for role in ['fit','development']:
    rr=[r for r in allrows if r['role']==role];coverage[role]=dict(anchors=len(rr),clips=len({r['clip_id']for r in rr}),words=len({r['word']for r in rr}),new_anchors=sum(r['new']for r in rr),distinct_ring_little=[dict(clip_id=r['clip_id'],frame=r['frame'],targets=r['targets'])for r in rr if r['targets'][3]>=0 and r['targets'][4]>=0 and r['targets'][3]!=r['targets'][4]])
write(U/'COVERAGE.json',coverage)
write(U/'TRAIN_FROZEN.json',dict(seeds=[42,43,44],steps=600,lr=.001,weight_decay=.001,confidence=.85,selection='Fixed final step/all seeds. No DEV early stopping. Ensemble requires all3confident compatible states; no threshold tuning.',candidates=['expanded_whole_hand','shared_finger_chain'],labels='Same original+42rechecked public examples; onlyFIT roles in optimizer',normalization='FIT only',baseline='Archived per-finger v2 + existing coarse observer, unchanged',integration='New finger evidence supplementary only; existing shape/candidate/order/direction unchanged; no correction from uncertain observations',pins={str(U/n):sha(U/n)for n in['INPUTS.npz','INPUTS.json','SOURCE_REVIEW.json','finger_model.py']}))
print(json.dumps({k:{n:v for n,v in c.items()if n!='distinct_ring_little'}|{'ring_little_different':len(c['distinct_ring_little'])}for k,c in coverage.items()}))
