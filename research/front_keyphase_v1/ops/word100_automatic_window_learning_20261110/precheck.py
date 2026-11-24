from window_core import *
from collections import Counter
torch.set_num_threads(4);torch.set_num_interop_threads(1)
checks=[]
def ck(name,value):
 checks.append(dict(name=name,passed=bool(value)));assert value,name
cfg=read(U/'FROZEN.json');rr=load_rows();fit=[r for r in rr if r['role']=='fit'];dev=[r for r in rr if r['role']=='development']
ck('899FIT100words',len(fit)==899 and len(set(r['class_index'] for r in fit))==100)
ck('300DEV_disjoint_record_ids',len(dev)==300 and not set(r['person'] for r in fit)&set(r['person'] for r in dev))
for p,h in {**cfg['pins'],**cfg['code_pins'],**cfg['protections']}.items():ck('pinned:'+p,sha(p)==h)
net=WindowNet('learned_core',read(U/'NORMALIZATION.json')).eval();hist=Counter()
for r in fit:
 ck('cache:'+r['clip_id'],r['g_full'].shape==(len(r['x_full']),66) and r['frozen_h_full'].shape==(len(r['x_full']),64))
 if not r['pairs']:continue
 for rev in [False,True]:
  x,ids=transformed(r,reverse=rev);ck('no_frame_loss:'+r['clip_id']+str(rev),sorted(ids.tolist())==list(range(len(x))))
  altered=x.copy();pp=altered[:,:102].reshape(-1,51,2);pp[:,10:30]+=10;pp[:,31:51]-=10;altered[:,102:186]+=15
  ck('geometry_finger_independent',np.array_equal(geometry(x,r['timestamp_us']),geometry(altered,r['timestamp_us'])))
  for p in r['pairs']:
   a,b,_=map_interval(ids,p['a'],p['b']);a=max(0,a-12);b=min(len(x)-1,b+12);y,core,intervals=window_truth(r,ids,a,b,p['axis'],p.get('moving','R'));hist[y]+=1
   q=relations(x,p['axis'],p.get('moving','R'),p.get('anchor','L'));q2=relations(altered,p['axis'],p.get('moving','R'),p.get('anchor','L'));ck('relation_finger_independent',np.array_equal(q,q2));ck('truth_has_covered_reviewed_core',len(intervals)>0 and core.sum()>=3)
ck('opposite_strokes_not_called_hold',hist[3]>0)
r=next(r for r in fit if r['pairs']);g,fh,l=batch_g([r]);h,v=net.encode(g,l);p=r['pairs'][0];q=torch.tensor(relations(prior.physical(r['x_full']),p['axis'],p.get('moving','R'),p.get('anchor','L')))
with torch.inference_mode():
 pred,core,att,valid=net.movement(h[0],q,p['a'],p['b'],p['axis']);ck('movement_output_four_classes',pred.shape==(4,) and torch.isfinite(pred).all());ck('attention_preserves_all_observed',bool((att[valid]>0).all()) and abs(float(att.sum())-1)<1e-5);ck('100word_auxiliary',net.classify_words(h,fh,l,v).shape==(1,100))
ck('no_private_rows',all('REAL' in r['clip_id'] for r in rr))
write(U/'PRETRAIN_CHECK.json',dict(passed=True,checks=checks,counts=len(checks),augmented_label_counts=dict(hist),scope='Code/contracts and cached public data audit by same Codex, not independent human labels. Before new model training or DEV evaluation.'))
print(json.dumps(dict(passed=True,checks=len(checks),label_counts=dict(hist))))
