"""Explicit requested vocabulary subset, no retraining or score adjustment."""
from prepare import *
import torch,numpy as np
def main():
 old=O/'demo_handshape_learning_20261113';source=read(old/'TCN_RESULTS.json');ck=torch.load(old/'appended_classifier.pt',map_location='cpu',weights_only=False)
 assert ck['new_classes']==['WORD2633','WORD2635','WORD2637','WORD0099'];assert sha(ck['base_checkpoint'])==ck['base_sha256']
 assert not(U/'APPLE_SUBSET.json').exists()
 write(U/'APPLE_SCOPE_FROZEN.json',dict(authorization='User previously removed digits from game, requested basic apple/gratitude/greeting. This is a scope-derived subset, not tuning scores to one failing word.',retained_old_indices=list(range(100)),retained_extra='WORD0099',excluded=['WORD2633','WORD2635','WORD2637'],source_sha=sha(old/'appended_classifier.pt'),limits='Retrospective stored-logit comparison only, no new independent test and no production replacement'))
 rr=[]
 for r in source['predictions']:
  if r['new'] and r['truth']!=103:continue
  logits=np.asarray(r['logits'])[[*range(100),103]];rank=np.argsort(-logits,kind='stable');truth=100 if r['truth']==103 else r['truth']
  rr.append(dict(clip_id=r['clip_id'],apple=r['new'],truth=truth,top1=int(rank[0]),top3=rank[:3].tolist(),baseline_top1=r['baseline_top1'],baseline_top3=r['baseline_top3']))
 summary={}
 for label,flag in [('existing100',False),('apple',True)]:
  rows=[r for r in rr if r['apple']==flag];summary[label]=dict(n=len(rows),top1=sum(r['truth']==r['top1']for r in rows),top3=sum(r['truth']in r['top3']for r in rows),old_top1=sum(r['truth']==r['baseline_top1']for r in rows),old_top3=sum(r['truth']in r['baseline_top3']for r in rows))
 torch.save(dict(base_checkpoint=ck['base_checkpoint'],base_sha256=ck['base_sha256'],new_classes=['WORD0099'],state_dict={k:v[3:4].clone()for k,v in ck['state_dict'].items()},schema='frozen100_plus_apple1',source_checkpoint=str(old/'appended_classifier.pt'),source_sha=sha(old/'appended_classifier.pt')),U/'apple_classifier.pt')
 write(U/'APPLE_SUBSET.json',dict(summary=summary,rows=rr,model_sha=sha(U/'apple_classifier.pt'),retrained=False,web_admitted=False,reason='Stored public dev only. Requires independent scope validation and feedback truth before game deployment. Old model has no apple output; do not call old0 to new3 an accuracy gain.'))
 print('APPLE SUBSET',summary)
if __name__=='__main__':main()
