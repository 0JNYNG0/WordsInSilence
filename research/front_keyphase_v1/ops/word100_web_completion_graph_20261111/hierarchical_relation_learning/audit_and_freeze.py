from core import *
from collections import Counter,defaultdict
import itertools

def main():
 assert not (W/'FROZEN.json').exists()
 meta=read(W/'INPUTS_COMPLETE.json');data=np.load(meta['arrays']);target=read(L/'MULTILEVEL_RELATION_TARGETS.json')['rows'];rows=meta['rows']
 assert sha(meta['arrays'])==meta['arrays_sha256']
 for p,h in meta['pins'].items():assert sha(p)==h
 cache={}
 for r in meta['feature_rows']:
  assert sha(r['path'])==r['sha256'] and sha(r['raw'])==r['raw_sha256']
  cache[r['clip_id'],r['view']]=(np.load(r['path'])['z'],np.load(r['raw'])['frame_id'])
 counts=Counter();people=defaultdict(set);sources=defaultdict(set)
 for i,r in enumerate(rows):
  t=target[r['target_index']];assert r['source_sha256']==t['source_sha256'] and r['source_frames']==t['frames']
  z,fi=cache[r['clip_id'],r['view']];a,b=r['indices'];assert fi[a]==t['frames'][0] and fi[b]==t['frames'][1]
  va=z[a,228:456];vb=z[b,228:456];seq=z[a:b+1,228:456];mean=np.sum(seq.astype(np.float64),axis=0)/len(seq);sd=np.sqrt(np.sum((seq-mean)**2,axis=0)/len(seq))
  expected=np.concatenate([(va+vb)*.5,np.abs(vb-va),mean,sd]);assert np.allclose(expected,data['x'][i],atol=1e-5)
  assert np.array_equal(pair_features(z,b,a),data['x'][i])
  assert data['y'][i,0]==t['parent_same']
  assert data['mask'][i,1]==(t['parent_same']==1 and t['atomic_same'] is not None)
  if data['mask'][i,1]:assert data['y'][i,1]==t['atomic_same']
  assert data['train'][i]==(r['split']=='train')
  q=z[:,292:392].reshape(-1,10,10);obs=[all(q[ix,k,8]>.5 for k in slots) for ix,slots in zip(r['indices'],r['required_relation_slots'])]
  assert data['eligible'][i]==all(obs)
  if t['atomic_same'] is None:assert not data['mask'][i,1];counts['unknown_masked']+=1
  people[r['split']].add(r['person']);sources[r['split']].add(r['source_sha256'])
 assert not people['train']&people['probe'] and not sources['train']&sources['probe']
 assert counts['unknown_masked']==72
 # Shortcut diagnostics fit ONLY train; no head candidate/threshold selected with probe.
 shortcuts={};tr=data['train'];y=data['y'];mask=data['mask'];eligible=data['eligible'];full=np.array([r['view']=='full' for r in rows]);tr=tr&full&eligible;pr=~data['train']&full
 for name in ['majority','word','gap','length']:
  p=np.zeros((len(rows),3),np.float32);detail=[]
  for j in range(2):
   use=tr&(mask[:,j]>0);yy=y[:,j];prior=float(np.mean(yy[use]));prob=np.full(len(rows),prior)
   rule={'prior':prior}
   if name=='word':
    words=sorted({r['word_id'] for r in rows});mapping={w:float(yy[use&np.array([r['word_id']==w for r in rows])].mean()) for w in words if np.any(use&np.array([r['word_id']==w for r in rows]))};prob=np.array([mapping.get(r['word_id'],prior) for r in rows]);rule['train_word_probabilities']=mapping
   elif name in ['gap','length']:
    v=np.array([r['dt_us'] if name=='gap' else r['source_frames_total'] for r in rows]);vals=np.unique(v[use]);ths=np.r_[vals[0]-1,(vals[:-1]+vals[1:])/2,vals[-1]+1];best=None
    for th,sgn in itertools.product(ths,[-1,1]):
     pred=(sgn*(v-th)>=0);ba=np.mean([np.mean(pred[use&(yy==k)]==k) for k in [0,1]])
     if best is None or ba>best[0]:best=(ba,float(th),sgn)
    prob=(best[2]*(v-best[1])>=0).astype(float);rule.update(train_balanced_accuracy=best[0],threshold=best[1],sign=best[2])
   p[:,0 if j==0 else 2]=prob;detail.append(rule)
  p[:,1]=p[:,0]*p[:,2];shortcuts[name]=dict(rules=detail,full_probe=metrics(y[pr],mask[pr],p[pr],eligible[pr]))
 # Distinct occurrence contract from FIT source groups; no DEV witness/order used.
 witness=defaultdict(list)
 for r in read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows']:
  if r['person'] not in people['train']:continue
  for a,b in itertools.combinations(r['groups'],2):
   if a['family']!=b['family'] and (a['offset'][1]<b['onset'][0] or b['offset'][1]<a['onset'][0]):witness[tuple(sorted([a['family'],b['family']]))].append(r['clip_id'])
 relations=[dict(families=list(k),source_clips=sorted(set(v)),relation='distinct_parent_occurrences',order_constraint=False) for k,v in witness.items() if len(set(v))>=3]
 audit=dict(at=stamp(),status='PASS_MAPPING_AND_MASKS_LIMITED_SUPERVISION',checked_views=len(rows),source_pairs=len(target),source_clips=len(cache)//2,people={k:sorted(v) for k,v in people.items()},source_counts={k:len(v) for k,v in sources.items()},unknown_atomic_masks=72,shortcuts=shortcuts,relation_contract=relations,training_allowed=True,independent_human=False,limitations=['Probe people excluded from new head only; upstream saw them.','Same video pairs and full/upper views correlated.','This is sampled relation truth, not exact boundary/completion truth.','Word/time baseline audit is diagnostic; no hyperparameter tuning on probe.'])
 write(W/'PRETRAIN_VERIFICATION.json',audit)
 paths=[W/'core.py',W/'prepare.py',Path(__file__),W/'INPUTS_COMPLETE.json',W/'pairs.npz',W/'PRETRAIN_VERIFICATION.json',L/'MULTILEVEL_RELATION_TARGETS.json',U/'episode_stage_contrast/BINDING_V2_SCORES.json',U/'episode_stage_contrast/EVALUATION_COMPLETE.json']
 write(W/'FROZEN.json',dict(at=stamp(),seeds=[42,43,44],steps=1200,lr=.0003,weight_decay=.01,gradient_clip=2,std_floor=.05,hidden=48,batch='all eligible train views',loss='Parent BCE and conditional-atomic BCE; each class/source equal weight, each head equal weight; unknown and different-parent atomic masked.',selection='Last step only. No probe checkpoint selection.',features='Symmetric endpoint mean/absolute difference and interval mean/std of frozen current228 features; no explicit word, frame, gap, source length.',upstream_trainable=False,linkage=dict(conditions=['baseline','internal','joint','combined'],internal='Mean log p_same_parent over distinct first/middle/last observed-motion representative pairs; coefficient1.',joint='Mean log(1-p_same_parent) across representative points of source-verified distinct-family candidates; coefficient1. Same point gives p_same=1; clip eps1e-6. No expected-order constraint, no hard overlap prohibition.',representatives='First/middle/last sorted unique body-observed points with frozen stage p_motion>=.5; need2 for internal, otherwise zero learned bonus. All original candidates and null availability retained.',candidate_score='Original sum of compatible motion/hold log odds times observed seconds + fixed optional relation log terms. No threshold tuning; no endpoint modification.',relation_contract=relations,completion=None),gates='All3 seeds preserve normal/speed witnesses and reduce reordered wrong-action mixing with positive evidence, not abstention alone. No web adoption based on pair classification.',pins={str(p):sha(p) for p in paths},final8_read=False,private_used=False,web_changed=False))
 print('A verified',len(rows),'views;',len(target),'source pairs; shortcut parent BA:',{k:v['full_probe']['parent']['balanced_accuracy'] for k,v in shortcuts.items()},flush=True)

if __name__=='__main__':main()
