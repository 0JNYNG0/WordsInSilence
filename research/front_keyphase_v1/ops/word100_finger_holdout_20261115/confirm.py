from holdout_common import *
import torch
from guard_runtime import Guard,gate
torch.set_num_threads(2);cfg=read(U/'CANDIDATE_FROZEN.json')
for p,h in cfg['pins'].items():assert sha(p)==h
assert not(U/'CONFIRMATION.json').exists();g=Guard();cache={};reports=[]
for r in read(U/'confirmation_reserved_old/RESULTS.json')['rows']:
 cid=r['clip_id']
 if cid not in cache:
  s=next(v for v in read(U/'EXTRACTED.json')['rows']if v['clip_id']==cid);cache[cid]=g.infer(np.load(s['channel_path'])['tcn280'],r['hand'])[0]
 old=np.array(r['pred']);new=cache[cid][r['frame']];reports.append(dict(**r,guard=new.tolist(),veto_only=gate(old,new,'veto_only')[0].tolist(),require_agreement=gate(old,new,'require_agreement')[0].tolist()))
def metric(rr,key):
 y=np.array([r['y']for r in rr]);old=np.array([r['pred']for r in rr]);p=np.array([r[key]for r in rr]);k=y>=0;return dict(known=int(k.sum()),correct=int(((p==y)&k).sum()),wrong=int(((p>=0)&(p!=y)&k).sum()),unknown=int(((p<0)&k).sum()),old_correct_to_wrong=int(((old==y)&k&(p>=0)&(p!=y)).sum()),old_correct_lost=int(((old==y)&k&(p<0)).sum()))
summary={}
for key in ['pred','guard','veto_only','require_agreement']:
 summary[key]={group:metric(rr,key)for group,rr in [('all',reports),('left',[r for r in reports if r['hand']=='L']),('front',[r for r in reports if r['view']=='front_pointing_index']),('different_ring_little',[r for r in reports if r['y'][2]>=0 and r['y'][3]>=0 and r['y'][2]!=r['y'][3]])]}
write(U/'CONFIRMATION.json',dict(summary=summary,rows=reports,selected_before_release=cfg['selected'],new_models_admitted=False,no_post_confirmation_tuning=True,limitations=['Single reserved signer18, sparse RGB anchors, no independent KSL expert','Prior TCN heldout exposure disclosed; new to examined fine-finger candidate manifests','Standalone models and rejected alternatives are shadow reporting only; cannot be selected using18']))
print(json.dumps(summary))
