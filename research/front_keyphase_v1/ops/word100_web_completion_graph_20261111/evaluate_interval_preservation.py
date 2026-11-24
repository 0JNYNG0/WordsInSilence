"""One frozen ablation: role evidence selects, never trims phase intervals."""
from graph import *
from prepare_linker import load
from occurrence_binding import select_occurrence, union_indices
from collections import Counter
import sys, numpy as np, torch
O=U.parent
G=O/'word100_guide_requirement_learning_20261111'
R=O/'word100_candidate_role_learning_20261111'
D=O/'word100_action_feedback_demo_20261108'


def main():
 assert not (U/'INTERVAL_PRESERVATION_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_interval_requirements',G/'requirement_core.py');sys.modules['requirement_core']=rc
 om=load('_interval_order',G/'order_binding.py')
 cfg=read(D/'FROZEN.json')['phase']
 src={(s['clip_id'],s['kind']):s for s in read(R/'EVALUATION_INPUTS.json')['rows']}
 saved=read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows']
 write(U/'INTERVAL_PRESERVATION_FROZEN.json',dict(
  reason='Independent audit found trimming removes normal direct labels and fast endpoint evidence.',
  generation='Unchanged phase threshold, hand validity, main minimum duration.',
  selection='Maximum joint phase-role confidence chooses whole source support interval; no desired chronology.',
  endpoint_child='At least two observed samples; no standalone action completion or duration requirement.',
  semantic_truth='Partial-core reversal is not a universally wrong action sequence; preserve exact source mapping and mark remaining semantic truth unknown.',
  final_data_access=False,new_training=False,web_changed=False,
  code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),U/'occurrence_binding.py']},
  comparison='Normal direct anchors and endpoint pairs, speed, missing, source-conflict audit; no 9/9 semantic error target.'))
 nets={}
 for seed in [42,43,44]:
  n=om.OrderedBinding();n.load_state_dict(torch.load(G/'order_runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));n.eval();nets[seed]=n
 cache={};out=[];memo={}
 for s in saved:
  cid,kind,seed=s['clip_id'],s['kind'],s['seed'];meta=src[cid,kind]
  if (cid,kind) not in cache:cache[cid,kind]={k:v for k,v in np.load(meta['path']).items()}
  c=cache[cid,kind];b=read(meta['baseline']);r=read(s['result'])
  phase=np.asarray(b['phase_probability']);role=np.asarray(r['role_probabilities']);ts=c['ts'];hand=np.asarray(b['observed_hand_mask']);word=cid.split('_')[2];cards=[]
  for spec in rc.FAMILIES[word]:
   card=next(k for k in rc.CARDS[word] if k['id']==spec['id']);f=spec['families'][0]
   valid=hand[:,[0 if h=='L' else 1 for h in card['hands']]].all(1);groups=[]
   for group in card['groups']:
    q=select_occurrence(phase[:,np.array(group)+1].sum(1),role[:,f],valid,ts,cfg['threshold'],cfg['min_evidence_ms'],bool(spec.get('parent')))
    groups.append(dict(group=group,**q))
   spans=[e['span'] for e in groups if e['span']]
   span=[min(e[0] for e in spans),max(e[1] for e in spans)] if len(spans)==len(groups) else None
   cards.append(dict(id=card['id'],family=f,span=span,evidence_intervals=spans,
                     observed_indices=union_indices(spans),groups=groups,required=spec['required'],parent=spec.get('parent'),semantic_completion=None))
  pairs=[];required=[k for k in cards if k['required'] and not k['parent']]
  for a,bb in zip(required,required[1:]):
   if a['family']==bb['family']:continue
   aa,bs=a['span'],bb['span'];p=None;st='association_unknown'
   if aa is not None and bs is not None:
    lo,hi=sorted([int(round(sum(aa)/2)),int(round(sum(bs)/2))])
    ga=next(k['groups'][0][0] for k in rc.CARDS[word] if k['id']==a['id']);gb=next(k['groups'][0][0] for k in rc.CARDS[word] if k['id']==bb['id'])
    key=(cid,kind,seed,lo,hi,ga,gb)
    if key not in memo:
     with torch.inference_mode():memo[key]=float(nets[seed](torch.tensor(rc.observation(c['fh'],lo,hi)[None]),torch.tensor(rc.requirement('order',guides=[ga,gb])[None])).sigmoid()[0])
    p=memo[key]
    st='supported' if aa[1]<bs[0] and p>=.65 else 'reversed' if bs[1]<aa[0] and p<=.35 else 'ambiguous'
   pairs.append(dict(cards=[a['id'],bb['id']],state=st,spans=[aa,bs],order_probability=p,semantic_completion=None))
  out.append(dict(seed=seed,clip_id=cid,kind=kind,cards=cards,pairs=pairs))
 write(U/'INTERVAL_PRESERVATION_RESULTS.json',out)
 summary={str(s):{k:dict(Counter(p['state'] for r in out if r['seed']==s and r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in out})} for s in [42,43,44]}
 write(U/'INTERVAL_PRESERVATION_COMPLETE.json',dict(summary=summary,web_changed=False,not_semantic_accuracy=True));print(summary)

if __name__=='__main__':main()
