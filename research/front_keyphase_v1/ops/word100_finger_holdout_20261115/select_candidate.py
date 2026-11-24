from holdout_common import *
import torch
from guard_runtime import Guard,gate
torch.set_num_threads(2);g=Guard();olddata=np.load(P/'INPUTS.npz');oldrows=read(P/'INPUTS.json')['rows'];oldpred=np.load(P/'SUPPLEMENT_PREDICTIONS.npz')['pred'];ids=np.array([i for i,r in enumerate(oldrows)if r['role']=='development']);new,pr=g.infer_windows(olddata['x_old'],olddata['availability'])
def metric(y,old,new):
 k=y>=0;c=(old==y)&k;return dict(known=int(k.sum()),correct=int(((new==y)&k).sum()),wrong=int(((new>=0)&(new!=y)&k).sum()),unknown=int(((new<0)&k).sum()),old_correct=int(c.sum()),old_correct_lost=int((c&(new<0)).sum()),old_correct_to_wrong=int((c&(new>=0)&(new!=y)).sum()))
history={};old_y=olddata['y'].copy();old_y[~olddata['availability']]=-1
for mode in ['unchanged','veto_only','require_agreement','standalone_guard']:
 q=oldpred if mode=='unchanged'else new if mode=='standalone_guard'else gate(oldpred,new,mode)[0];history[mode]=metric(old_y[ids],oldpred[ids],q[ids])
reports=[];cache={}
for r in read(U/'diagnostic_old/RESULTS.json')['rows']:
 cid=r['clip_id']
 if cid not in cache:
  s=next(v for v in read(U/'EXTRACTED.json')['rows']if v['clip_id']==cid);x=np.load(s['channel_path'])['tcn280'];cache[cid]=g.infer(x,r['hand'])[0]
 reports.append(dict(**r,guard=cache[cid][r['frame']].tolist()))
summary={}
for mode in ['unchanged','veto_only','require_agreement','standalone_guard']:
 summary[mode]={}
 for group,rr in [('all',reports),('left',[r for r in reports if r['hand']=='L']),('front',[r for r in reports if r['view']=='front_pointing_index']),('different_ring_little',[r for r in reports if r['y'][2]>=0 and r['y'][3]>=0 and r['y'][2]!=r['y'][3]])]:
  y=np.array([r['y']for r in rr]);old=np.array([r['pred']for r in rr]);guard=np.array([r['guard']for r in rr]);q=old if mode=='unchanged'else guard if mode=='standalone_guard'else gate(old,guard,mode)[0];summary[mode][group]=metric(y,old,q)
eligible=[]
for mode in ['veto_only','require_agreement']:
 ok=summary[mode]['all']['wrong']==0
 for m in [history[mode],summary[mode]['left'],summary[mode]['front']]:ok &= m['old_correct_to_wrong']==0 and m['old_correct_lost']<=.1*m['old_correct']
 if ok:eligible.append(mode)
selected=min(eligible,key=lambda k:history[k]['old_correct_lost']+summary[k]['all']['old_correct_lost'])if eligible else 'unchanged'
write(U/'SELECTION.json',dict(historical_development=history,diagnostic=summary,selected=selected,eligible=eligible,rows=reports,confirmation_predictions_seen=False))
write(U/'CANDIDATE_FROZEN.json',dict(selected=selected,allow_web_candidate=selected!='unchanged',reserved_role='confirmation_reserved',pins={str(p):sha(p)for p in [U/'guard_runtime.py',U/'SELECTION.json',U/'TRAIN_FROZEN.json']+list((U/'guard_models').glob('*.pt'))},rules='No further tuning after release of18. Only reject deployment on failure.'))
print(json.dumps(dict(history=history,diagnostic=summary,selected=selected)))
