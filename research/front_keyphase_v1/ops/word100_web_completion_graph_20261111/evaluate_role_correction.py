from graph import *
from prepare_linker import load
from interval_pipeline import select_cards,order_observations
from collections import Counter
import sys,numpy as np,torch
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111';D=O/'word100_action_feedback_demo_20261108';W=U/'occurrence_supervision'


def main():
 assert not (W/'EVALUATION_COMPLETE.json').exists();assert (W/'TRAINING_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_corrected_role_requirements',G/'requirement_core.py');sys.modules['requirement_core']=rc
 role=load('_corrected_role_model',R/'role_core.py');om=load('_corrected_order_model',G/'order_binding.py')
 cfg=read(W/'FROZEN.json');phasecfg=read(D/'FROZEN.json')['phase'];sources=read(R/'EVALUATION_INPUTS.json')['rows']
 labels={r['clip_id']:r for r in read(O/'word100_event_guide_linkage_20261110/LINK_LABELS.json')['rows']}
 write(W/'EVALUATION_FROZEN.json',dict(conditions=cfg['conditions'],seeds=cfg['seeds'],primary_seed=42,
  source_manifest_sha256=role.sha(R/'EVALUATION_INPUTS.json'),
  pins={str(p):role.sha(p) for p in [Path(__file__),U/'interval_pipeline.py',U/'occurrence_binding.py',W/'FROZEN.json',W/'TRAINING_COMPLETE.json']},
  old_models_unchanged=True,previously_exposed_DEV=True,no_final_test_access=True,
  semantics='Source-correspondence and component metrics only. Partial reversal is not 9 proved semantic errors.'))
 cached={};out=[];frame_scores=[];memo={}
 for s in sources:
  assert role.sha(s['path'])==s['sha256'];z={k:v for k,v in np.load(s['path']).items()};b=read(s['baseline']);cached[s['clip_id'],s['kind']]=(z,b)
 for seed in cfg['seeds']:
  order=om.OrderedBinding();order.load_state_dict(torch.load(G/'order_runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));order.eval()
  for condition in cfg['conditions']:
   net=role.RoleAdapter(seed);net.load_state_dict(torch.load(W/'runs'/condition/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval()
   for s in sources:
    cid,kind=s['clip_id'],s['kind'];word=cid.split('_')[2];z,b=cached[cid,kind]
    with torch.inference_mode():logits=net(torch.tensor(z['z']));logits[:,10]=-1e4;rp=logits.softmax(-1).numpy()
    p=W/'predictions'/condition/str(seed)/f'{cid}_{kind}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,role=rp)
    cards=select_cards(rc.CARDS[word],rc.FAMILIES[word],np.asarray(b['phase_probability']),rp,np.asarray(b['observed_hand_mask']),z['ts'],phasecfg)
    pairs=[]
    for a,bb in order_observations(cards):
     aa,bs=a['span'],bb['span'];prob=None;state='association_unknown'
     if aa is not None and bs is not None:
      lo,hi=sorted([int(round(sum(aa)/2)),int(round(sum(bs)/2))]);ga=next(k['groups'][0][0] for k in rc.CARDS[word] if k['id']==a['id']);gb=next(k['groups'][0][0] for k in rc.CARDS[word] if k['id']==bb['id']);key=(cid,kind,seed,lo,hi,ga,gb)
      if key not in memo:
       with torch.inference_mode():memo[key]=float(order(torch.tensor(rc.observation(z['fh'],lo,hi)[None]),torch.tensor(rc.requirement('order',guides=[ga,gb])[None])).sigmoid()[0])
      prob=memo[key];state='supported' if aa[1]<bs[0] and prob>=.65 else 'reversed' if bs[1]<aa[0] and prob<=.35 else 'ambiguous'
     pairs.append(dict(cards=[a['id'],bb['id']],spans=[aa,bs],state=state,order_probability=prob,semantic_completion=None))
    out.append(dict(condition=condition,seed=seed,clip_id=cid,kind=kind,cards=cards,pairs=pairs,role_prediction=str(p),role_sha256=role.sha(p)))
    if kind=='full_normal' and cid in labels:
     for a in labels[cid]['anchors']:
      target=role.jc.ep.MAP.get(a['label'],-1)
      if target>=0:frame_scores.append(dict(condition=condition,seed=seed,clip_id=cid,frame=a['frame'],target=target,prediction=int(rp[a['frame']].argmax())))
 write(W/'RESULTS.json',out);write(W/'FRAME_ROLE_SCORES.json',frame_scores)
 summary={c:{str(seed):{k:dict(Counter(p['state'] for r in out if r['condition']==c and r['seed']==seed and r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in out})} for seed in cfg['seeds']} for c in cfg['conditions']}
 anchors={c:{str(seed):dict(correct=sum(v['prediction']==v['target'] for v in frame_scores if v['condition']==c and v['seed']==seed),total=sum(v['condition']==c and v['seed']==seed for v in frame_scores)) for seed in cfg['seeds']} for c in cfg['conditions']}
 write(W/'EVALUATION_COMPLETE.json',dict(summary=summary,anchors=anchors,web_changed=False,final_test_unopened=True));print('anchors',anchors);print(summary)

if __name__=='__main__':main()
