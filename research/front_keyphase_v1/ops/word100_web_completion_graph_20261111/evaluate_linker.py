from graph import *
from binding import *
from prepare_linker import load
import time,traceback
from collections import Counter
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111'
def main():
 assert (U/'LINKER_TRAIN_COMPLETE.json').exists() and not (U/'LINKER_EVAL_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_linker_eval_requirement',G/'requirement_core.py');cfg=read(U/'LINKER_FROZEN.json')
 source=read(R/'EVALUATION_INPUTS.json')['rows'];saved={(r['seed'],r['clip_id'],r['kind']):r for r in read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows']}
 auto={(r['seed'],r['clip_id'],r['kind']):r for r in read(G/'ORDER_AUTOMATIC_DIAGNOSTICS.json')};gold={r['clip_id']:r for r in read(O/'word100_event_guide_linkage_20261110/LINK_LABELS.json')['rows']}
 families=rc.FAMILIES;write(U/'LINKER_EVAL_FROZEN.json',dict(conditions=len(source),seeds=cfg['seeds'],scope='Previously exposed public DEV; whole-core reversal changes directions and may leave duplicate stages, not pure order-error truth',methods=['old_card_midpoint','old_learned_roles_independent','new_body_context_independent'],code_sha256=rc.sha(U/'binding.py'),label_sha256=rc.sha(U/'LINKER_MANIFEST.json'),unknown_threshold=.65,absence='missing association is not omission',no_score_based_tuning=True))
 outs=[];frames=[];cache={};t0=time.perf_counter()
 for seed in cfg['seeds']:
  n=BodyContextLinker(np.asarray(cfg['mean'],np.float32),np.asarray(cfg['std'],np.float32));n.load_state_dict(torch.load(U/'linker_runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));n.eval()
  for s in source:
   cid,kind=s['clip_id'],s['kind'];word=cid.split('_')[2];ref=saved[seed,cid,kind];assert rc.sha(ref['result'])==ref['sha256'];old=read(ref['result'])
   if s['key'] not in cache:
    assert rc.sha(s['path'])==s['sha256'];cache[s['key']]={k:v for k,v in np.load(s['path']).items()}
   c=cache[s['key']];ts=c['ts'];av=rc.prior.shared.available(c['x']);xx=body_context(c['z'],ts);t=time.perf_counter()
   with torch.inference_mode():pp=n(torch.tensor(xx)).softmax(-1).numpy()
   secs=time.perf_counter()-t;role=np.asarray(old['role_probabilities']);required=[f for f in families[word] if f['required']];fs=sorted({f for spec in required for f in spec['families']});sets={}
   for method,p in [('old_learned_roles_independent',role),('new_body_context_independent',pp)]:
    occ=[]
    for f in fs:
     valid=av[:,1] if f in [1,2,3] else av[:,1]&av[:,2]
     occ+=bind_occurrences(p,valid,ts,[f],cfg['threshold'])
    occ=sorted(occ,key=lambda e:(e['a'],e['b'],e['family']));pairs=[]
    for a,b in zip(required,required[1:]):
     fa,fb=a['families'][0],b['families'][0]
     if fa==fb:continue
     pairs.append(dict(cards=[a['id'],b['id']],families=[fa,fb],state=order_relation(occ,fa,fb)))
    sets[method]=dict(occurrences=occ,pairs=pairs,missing=[f for f in fs if not any(e['family']==f for e in occ)])
   if kind=='full_normal' and cid in gold:
    for a in gold[cid]['anchors']:
     label=a['label'];target={0:0,1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:6,9:7}.get(label,-1)
     if target<0:continue
     i=a['frame'];before=int(role[i].argmax());after=int(pp[i].argmax());frames.append(dict(seed=seed,clip_id=cid,index=i,target=target,before=before if before<8 else 0,after=after if after<8 else 0))
   outs.append(dict(seed=seed,clip_id=cid,kind=kind,methods=sets,baseline_pairs=auto.get((seed,cid,kind),{}).get('pairs',[]),network_seconds=secs,tcndtw_unchanged=True,old_result=ref['result']))
 summaries={}
 for seed in cfg['seeds']:
  summaries[str(seed)]={}
  for kind in sorted({r['kind'] for r in outs}):
   rows=[r for r in outs if r['seed']==seed and r['kind']==kind];sums={}
   for method in ['old_card_midpoint','old_learned_roles_independent','new_body_context_independent']:
    pairs=[p for r in rows for p in (r['baseline_pairs'] if method=='old_card_midpoint' else r['methods'][method]['pairs'])];sums[method]=dict(Counter(p['state'] for p in pairs))
   summaries[str(seed)][kind]=sums
 write(U/'LINKER_FRAME_SCORES.json',frames);write(U/'LINKER_RESULTS.json',outs);write(U/'LINKER_EVAL_COMPLETE.json',dict(summary=summaries,anchor_accuracy={str(s):dict(n=sum(r['seed']==s for r in frames),before=sum(r['seed']==s and r['before']==r['target'] for r in frames),after=sum(r['seed']==s and r['after']==r['target'] for r in frames)) for s in cfg['seeds']},conditions=len(source),seconds=time.perf_counter()-t0,private_used=False,web_changed=False))
 print(json.dumps(read(U/'LINKER_EVAL_COMPLETE.json'),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
