from graph import *
from binding import *
from prepare_linker import load
from prepare_block_linker import permutation
from source_decoder import phase_pairs
from collections import Counter
import time,traceback
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111'
def main():
 assert not (U/'BLOCK_EVAL_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_block_eval_req',G/'requirement_core.py');role=load('_block_eval_role',R/'role_core.py');cfg=read(U/'BLOCK_FROZEN.json');gold={r['clip_id']:r for r in read(O/'word100_event_guide_linkage_20261110/LINK_LABELS.json')['rows']};rows={r['clip_id']:r for r in rc.public_rows() if r['role']=='development'}
 src=read(R/'EVALUATION_INPUTS.json')['rows'];baseline={(r['seed'],r['clip_id'],r['kind']):r for r in read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows']};family=rc.FAMILIES
 write(U/'BLOCK_EVAL_FROZEN.json',dict(seeds=cfg['seeds'],old_cases=78,new_cases='whole observed action blocks reversed for public DEV clips with >=2 distinct reviewed contexts',new_case_truth='source action identity order only, abrupt synthetic joins; not natural learner error or semantic error gold',softmax_threshold=.65,old_model_full_evaluation_not_reexecuted=True,post_score_tuning=False,train_file=rc.sha(U/'BLOCK_INPUT_COMPLETE.json'),inference_code=rc.sha(U/'binding.py')))
 cases=[];base=rc.frozen_observer();frozen=role.frozen_model()
 for s in src:
  assert rc.sha(s['path'])==s['sha256'];c={k:v for k,v in np.load(s['path']).items()};cases.append(dict(meta=s,c=c))
  if s['kind']!='full_normal' or s['clip_id'] not in gold:continue
  cid=s['clip_id'];r=rows[cid];y=np.full(len(c['x']),-1);w=np.zeros(len(y));mapping={0:0,1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:6,9:7}
  for a in gold[cid]['anchors']:y[a['frame']]=mapping.get(a['label'],-1);w[a['frame']]=float(y[a['frame']]>=0)
  perm=permutation(y,w,'reverse_blocks')
  if perm is None:continue
  ii,seams,pm=perm;xx=c['x'][ii].copy();fh,_=role.lc.be.ev.frozen_forward(base,xx,r['word']);ob=role.features(frozen,xx,c['ts'],fh)
  cc=dict(x=xx,ts=c['ts'],z=ob['z'],fh=fh);p=U/'block_eval_inputs'/f'{cid}_block_permuted.npz';p.parent.mkdir(exist_ok=True);np.savez_compressed(p,**cc,source_indices=ii)
  meta=dict(clip_id=cid,kind='block_permuted',path=str(p),sha256=rc.sha(p),permutation=pm,synthetic_seams=seams.tolist());cases.append(dict(meta=meta,c=cc))
 write(U/'BLOCK_EVAL_INPUTS.json',[r['meta'] for r in cases]);results=[];anchors=[]
 for seed in cfg['seeds']:
  net=role.RoleAdapter(seed);net.load_state_dict(torch.load(U/'block_runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval()
  for rec in cases:
   s,c=rec['meta'],rec['c'];cid,kind=s['clip_id'],s['kind'];word=cid.split('_')[2];t=time.perf_counter()
   with torch.inference_mode():prob=net(torch.tensor(c['z'])).softmax(-1).numpy()
   sec=time.perf_counter()-t;av=rc.prior.shared.available(c['x']);fs=sorted({v for f in family[word] if f['required'] for v in f['families']});occ=[]
   for f in fs:
    valid=av[:,1] if f in [1,2,3] else av[:,1]&av[:,2];occ+=bind_occurrences(prob,valid,c['ts'],[f],.65)
   pairs=phase_pairs(occ,family[word]);ref=baseline.get((seed,cid,kind));old=read(ref['result']) if ref else None
   if kind=='full_normal' and cid in gold:
    for a in gold[cid]['anchors']:
     target=mapping.get(a['label'],-1)
     if target<0:continue
     i=a['frame'];after=int(prob[i].argmax());before=int(np.asarray(old['role_probabilities'])[i].argmax());anchors.append(dict(seed=seed,clip_id=cid,index=i,target=target,before=before if before<8 else 0,after=after if after<8 else 0))
   if kind=='block_permuted':
    oldnet=role.RoleAdapter(seed);oldnet.load_state_dict(torch.load(R/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));oldnet.eval()
    with torch.inference_mode():bp=oldnet(torch.tensor(c['z'])).softmax(-1).numpy()
    bo=[]
    for f in fs:
     valid=av[:,1] if f in [1,2,3] else av[:,1]&av[:,2];bo+=bind_occurrences(bp,valid,c['ts'],[f],.65)
    before_pairs=phase_pairs(bo,family[word])
   else:before_pairs=None
   results.append(dict(seed=seed,clip_id=cid,kind=kind,occurrences=occ,pairs=pairs,before_pairs=before_pairs,network_seconds=sec))
 write(U/'BLOCK_EVAL_RESULTS.json',results);write(U/'BLOCK_ANCHOR_SCORES.json',anchors)
 summary={str(s):{k:dict(Counter(p['state'] for r in results if r['seed']==s and r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in results})} for s in cfg['seeds']}
 accuracy={str(s):dict(n=sum(r['seed']==s for r in anchors),before=sum(r['seed']==s and r['before']==r['target'] for r in anchors),after=sum(r['seed']==s and r['after']==r['target'] for r in anchors)) for s in cfg['seeds']}
 write(U/'BLOCK_EVAL_COMPLETE.json',dict(summary=summary,anchor_accuracy=accuracy,cases=len(cases),web_changed=False));print(json.dumps(dict(summary=summary,anchor_accuracy=accuracy),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'BLOCK_EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
