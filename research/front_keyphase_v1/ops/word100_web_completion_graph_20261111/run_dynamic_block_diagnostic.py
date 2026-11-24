"""New, source-audited block permutations; no previous completed replay rerun."""
from graph import *
from prepare_linker import load
from interval_pipeline import select_cards,order_observations
import numpy as np,torch,sys,time
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111';D=O/'word100_action_feedback_demo_20261108';W=U/'dynamic_blocks'


def main():
 assert not (W/'COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 targets=read(U/'reviews/sequence_target/TARGETS.json');selected=[r for r in targets['rows'] if r.get('candidate')]
 assert len(selected)==4
 sources={s['clip_id']:s for s in read(R/'EVALUATION_INPUTS.json')['rows'] if s['kind']=='full_normal'}
 cfg=read(D/'FROZEN.json')['phase'];rc=load('_dynamic_requirements',G/'requirement_core.py');sys.modules['requirement_core']=rc
 role=load('_dynamic_roles',R/'role_core.py');om=load('_dynamic_order',G/'order_binding.py')
 sys.path.insert(0,str(D));from consumer import FeedbackConsumer
 consumer=FeedbackConsumer();base=rc.frozen_observer();context=role.frozen_model()
 W.mkdir(exist_ok=True)
 write(W/'FROZEN.json',dict(target_sha256=role.sha(U/'reviews/sequence_target/TARGETS.json'),
  conditions=['source_role','continued_old_direct','reviewed_roles'],seeds=[42,43,44],primary_seed=42,
  limited_dynamic_pairs=3,egypt_source_only=True,
  data='Four new permutations of previously exposed DEV public inputs; not fresh final evaluation.',
  scoring='Report assignment, observed descriptor order, and splice involvement separately; never whole-sign correctness.',
  seam_policy='No across-discontinuous-source motion accepted as performed movement. Metadata only audits outputs, not an input hint.',
  within_block_time_order_preserved=True,new_training=False,private_data=False,code_sha256=role.sha(Path(__file__))))
 inputs=[]
 for t in selected:
  cid=t['clip_id'];s=sources[cid];assert s['source_sha256']==t['source_sha256'];z=np.load(s['path']);ii=np.asarray(t['candidate']['query_to_source_frame_ids'])
  assert len(ii)==len(z['x']) and np.array_equal(np.sort(ii),np.arange(len(ii)))
  x=z['x'][ii].copy();ts=z['ts'].copy();p=W/'inputs'/f'{cid}.npz';p.parent.mkdir(exist_ok=True)
  fh,_=role.lc.be.ev.frozen_forward(base,x,{'WORD0739':0,'WORD1133':1,'WORD1567':2,'WORD2523':3}[t['word_id']])
  ob=role.features(context,x,ts,fh)
  np.savez_compressed(p,x=x,ts=ts,source_indices=ii,fh=fh,z=ob['z'])
  result=consumer.channels(x,ts,np.arange(len(x)),t['word_id']);q=W/'phase'/f'{cid}.json';write(q,result)
  inputs.append(dict(clip_id=cid,word_id=t['word_id'],path=str(p),sha256=role.sha(p),baseline=str(q),baseline_sha256=role.sha(q),target=t['candidate']))
 write(W/'INPUTS_COMPLETE.json',dict(rows=inputs))
 out=[]
 for seed in [42,43,44]:
  order=om.OrderedBinding();order.load_state_dict(torch.load(G/'order_runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));order.eval()
  for condition in ['source_role','continued_old_direct','reviewed_roles']:
   model=R/'runs'/str(seed)/'model.pt' if condition=='source_role' else U/'occurrence_supervision/runs'/condition/str(seed)/'model.pt'
   net=role.RoleAdapter(seed);net.load_state_dict(torch.load(model,weights_only=True,map_location='cpu'));net.eval()
   for s in inputs:
    z=np.load(s['path']);b=read(s['baseline']);word=s['word_id']
    with torch.inference_mode():p=net(torch.tensor(z['z']));p[:,10]=-1e4;rp=p.softmax(-1).numpy()
    cards=select_cards(rc.CARDS[word],rc.FAMILIES[word],np.asarray(b['phase_probability']),rp,np.asarray(b['observed_hand_mask']),z['ts'],cfg)
    pairs=[]
    for a,bb in order_observations(cards):
     aa,bs=a['span'],bb['span'];prob=None;state='association_unknown'
     if aa is not None and bs is not None:
      lo,hi=sorted([int(round(sum(aa)/2)),int(round(sum(bs)/2))]);ga=next(c['groups'][0][0] for c in rc.CARDS[word] if c['id']==a['id']);gb=next(c['groups'][0][0] for c in rc.CARDS[word] if c['id']==bb['id'])
      with torch.inference_mode():prob=float(order(torch.tensor(rc.observation(z['fh'],lo,hi)[None]),torch.tensor(rc.requirement('order',guides=[ga,gb])[None])).sigmoid()[0])
      state='supported' if aa[1]<bs[0] and prob>=.65 else 'reversed' if bs[1]<aa[0] and prob<=.35 else 'ambiguous'
     seam=[j['query_right'] for j in s['target']['splice_joins'] if j['synthetic_discontinuity']]
     spans=[v for v in [aa,bs] if v]
     crosses=any(a<j<=b for a,b in spans for j in seam)
     limited=word=='WORD0739' and [a['id'],bb['id']]==['close_chest','raise_forearm']
     mapped=None
     if limited and aa and bs:
      ba=s['target']['blocks']['B']['query_frame_span'];ca=s['target']['blocks']['C']['query_frame_span']
      mapped=bool(ba[0]<=aa[0]<=aa[1]<=ba[1] and ca[0]<=bs[0]<=bs[1]<=ca[1])
     pairs.append(dict(cards=[a['id'],bb['id']],spans=[aa,bs],state=state,probability=prob,
       limited_dynamic_target=limited,spans_within_target_occurrences=mapped,selected_span_crosses_splice=crosses,
       whole_sign_semantic_correctness=None))
    out.append(dict(condition=condition,seed=seed,clip_id=s['clip_id'],cards=cards,pairs=pairs,
     original_runtime_spans={c['id']:c.get('query_indices') for c in b['cards']}))
 write(W/'RESULTS.json',out)
 compact=[dict(condition=r['condition'],seed=r['seed'],clip=r['clip_id'],state=p['state'],mapped=p['spans_within_target_occurrences'],splice=p['selected_span_crosses_splice']) for r in out for p in r['pairs'] if p['limited_dynamic_target']]
 write(W/'COMPLETE.json',dict(limited_pairs=compact,web_changed=False,source_only_egypt_excluded_from_accuracy=True));print(compact)

if __name__=='__main__':main()
