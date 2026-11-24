"""Single new-model evaluation; unique filename avoids old generic imports."""
import time,traceback
from collections import Counter,defaultdict
from readout_model import *
from pair_contract import observe
import evaluate_episode as previous
NORMAL={'native','original','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest','seated_variation'}
CONFIRMED={'OBSERVED_A_BEFORE_B':1,'OBSERVED_B_BEFORE_A':0}

def query_rows(r,graph):
 out=[]
 for a,b in graph['precedences']:
  ia=np.flatnonzero(r['episode']==a);ib=np.flatnonzero(r['episode']==b)
  y=1 if len(ia) and len(ib) and ia.max()<ib.min() else 0 if len(ia) and len(ib) and ib.max()<ia.min() else -1
  reason='known_witness_order' if y>=0 else 'interleaved_witnesses' if len(ia) and len(ib) else 'witness_missing'
  out.append(dict(first=a,second=b,target=y,truth_reason=reason,first_witnesses=ia.tolist(),second_witnesses=ib.tolist()))
 return out

def summarize(pairs,anchors):
 known=[p for p in pairs if p['target']>=0];groups=defaultdict(list)
 for p in known:groups[(p['clip_id'],p['first'],p['second'])].append(p)
 def counts(pp):
  kk=[p for p in pp if p['target']>=0]
  return dict(known=len(kk),raw_correct=sum(p['raw_prediction']==p['target'] for p in kk),
   confirmed_correct=sum(p['confirmed_prediction']==p['target'] for p in kk),
   confirmed_wrong=sum(p['confirmed_prediction'] is not None and p['confirmed_prediction']!=p['target'] for p in kk),
   held_known=sum(p['confirmed_prediction'] is None for p in kk),unknown=len(pp)-len(kk),
   interleaved_confirmed=sum(p['truth_reason']=='interleaved_witnesses' and p['confirmed_prediction'] is not None for p in pp))
 norm=[p for p in pairs if p['kind'] in NORMAL];local=[p for p in pairs if 'local' in p['kind'] or 'short' in p['kind']]
 contrary=[p for p in pairs if p['target']==0];main_reverse=[p for p in pairs if p['kind']=='actual_reverse_main']
 def anchor_count(pp):
  return dict(positive_known=sum(p['target']==1 for p in pp),
   positive_confirmed=sum(p['available'] and p['target']==1 and p['effective_weight']>=.8 for p in pp),
   nonmain_known=sum(p['target']==0 for p in pp),
   nonmain_false_confirmed=sum(p['available'] and p['target']==0 and p['effective_weight']>=.8 for p in pp))
 normal_anchors=[p for p in anchors if p['kind'] in NORMAL]
 return dict(all=counts(pairs),normal=counts(norm),local=counts(local),
  local_evaluation_status='EVALUATED' if local else 'NOT_EVALUATED_NO_LOCAL_OR_SHORT_KIND',
  contrary=counts(contrary),main_reverse=counts(main_reverse),
  source_group_count=len(groups),source_group_macro_raw_accuracy=float(np.mean([np.mean([p['raw_prediction']==p['target'] for p in g]) for g in groups.values()])) if groups else None,
  source_group_macro_confirmed_correct_fraction=float(np.mean([np.mean([p['confirmed_prediction']==p['target'] for p in g]) for g in groups.values()])) if groups else None,
  by_kind={k:counts([p for p in pairs if p['kind']==k]) for k in sorted({p['kind'] for p in pairs})},
  by_person={str(k):counts([p for p in pairs if p['person']==k]) for k in sorted({p['person'] for p in pairs})},
  occurrence_positive_known=sum(p['target']==1 for p in anchors),
  occurrence_positive_confirmed=sum(p['available'] and p['target']==1 and p['effective_weight']>=.8 for p in anchors),
  occurrence_positive_raw_threshold=sum(p['target']==1 and p['probability']>=.8 for p in anchors),
  nonmain_negative_known=sum(p['target']==0 for p in anchors),
  nonmain_false_confirmed=sum(p['available'] and p['target']==0 and p['effective_weight']>=.8 for p in anchors),
  nonmain_raw_occurrence_false_confirmed=sum(p['available'] and p['target']==0 and p['probability']>=.8 for p in anchors),
  normal_anchors_by_family={str(k):anchor_count([p for p in normal_anchors if p['family']==k]) for k in sorted({p['family'] for p in normal_anchors})},
  normal_anchors_by_source={k:anchor_count([p for p in normal_anchors if p['clip_id']==k]) for k in sorted({p['clip_id'] for p in normal_anchors})},
  status_counts=dict(Counter(p['status'] for p in pairs)))

def main():
 assert read(Y/'TRAIN_PROGRESS.json')['status']=='COMPLETE' and not (Y/'EVALUATION.json').exists()
 torch.set_num_threads(2);plan=read(Y/'FROZEN.json');stats=read(D/'NORMALIZATION.json');flow=read(FLOW_UNIT/'FLOW_INPUTS.json');graphs=previous.graphs()
 splits={'fit':attach_queries(get_rows('fit')),'dev':get_rows('dev')}
 assert len(splits['fit'])==252 and len(splits['dev'])==88
 protocols={}
 for split,rr in splits.items():
  qq=[dict(q,clip_id=r['clip_id'],kind=r['kind']) for r in rr for q in query_rows(r,graphs[r['word_id']])]
  protocols[split]=dict(videos_and_variants=len(rr),unswapped_queries=len(qq),known=sum(q['target']>=0 for q in qq),
   held=sum(q['target']<0 for q in qq),source_relations=len({(q['clip_id'],q['first'],q['second']) for q in qq}))
 assert protocols['fit']['known']==288 and protocols['fit']['held']==27
 frozen=read(Y/'EVALUATION_PROTOCOL.json')
 assert protocols==frozen['splits'] and graphs==frozen['graphs']
 for p,h in frozen['pins'].items():assert sha(p)==h,p
 reports=[]
 for tag in plan['conditions']:
  dest=OUT/'runs'/tag;ckpt=dest/'model.pt';assert sha(ckpt)==read(dest/'COMPLETE.json')['checkpoint_sha256']
  m=ReadoutNet(tag);m.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True));m.eval()
  for split,rr in splits.items():
   pairs=[];anchors=[];seconds=[]
   for r in rr:
    x,im,l,_,_=batch_j([r],stats);fl=flow_batch([r],flow);t=time.perf_counter()
    with torch.inference_mode():ac,ro,oc=m(x,im,l,fl);pr=ac.softmax(-1)[0].numpy();role=ro.softmax(-1)[0].numpy();occ=oc.sigmoid()[0].numpy()
    seconds.append(time.perf_counter()-t)
    if split=='dev':
     old=np.load(DEST/'predictions/seated_variation'/f"{r['key']}.npz",allow_pickle=False)
     assert np.array_equal(pr,old['raw_action_probability']) and np.array_equal(r['current'],old['current_probability'])
    name=r['flow_key']
    if tag=='direct_readout':
     control=np.load(OUT/'predictions/tied_readout'/split/f'{name}.npz',allow_pickle=False)
     assert np.array_equal(pr,control['physical']) and np.array_equal(role,control['role'])
     assert np.array_equal(r['current'],control['current'])
    if 'times' in r:times=np.asarray(r['times'])
    else:
     assert sha(r['input'])==r['input_sha256']
     with np.load(r['input'],allow_pickle=False) as source:times=source['timestamp_us']/1e6
    assert len(times)==len(pr) and np.all(np.diff(times)>0)
    timestamp_kind='native_input_PTS'
    for q in query_rows(r,graphs[r['word_id']]):
     z=observe(pr,occ,r['available'],times,q['first'],q['second'],tag)
     p=z['forward_probability'];z.update(q,key=r.get('key',name),flow_key=name,clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],kind=r['kind'],timestamp_kind=timestamp_kind,
      raw_prediction=1 if p>.5 else 0 if p<.5 else None,confirmed_prediction=CONFIRMED.get(z['status']))
     pairs.append(z)
    ty=occurrence_targets([r],len(pr))[0].numpy()
    for frame,family in zip(*np.where(ty>=0)):
     anchors.append(dict(flow_key=name,clip_id=r['clip_id'],person=r['person'],kind=r['kind'],frame=int(frame),source_frame=int(r['mapping'][frame]),family=int(family),target=int(ty[frame,family]),probability=float(occ[frame,family]),available=bool(r['available'][frame]),effective_weight=float(occ[frame,family]*(pr[frame,family] if tag=='tied_readout' else 1))))
    target=OUT/'predictions'/tag/split/f'{name}.npz';target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
    np.savez_compressed(target,physical=pr,role=role,occurrence=occ,current=r['current'],image_available=r['available'],mapping=r['mapping'],times=times)
   summary=summarize(pairs,anchors);out=dict(condition=tag,split=split,summary=summary,pairs=pairs,anchors=anchors,head_median_seconds=float(np.median(seconds)))
   exclusive(OUT/'evaluations'/f'{tag}_{split}.json',out)
   reports.append({k:v for k,v in out.items() if k not in ['pairs','anchors']})
   print(tag,split,summary['all'],flush=True)
 for p,h in plan['pins'].items():assert sha(p)==h,p
 exclusive(Y/'EVALUATION.json',dict(at=clock(),reports=reports,new_model_clip_inferences=680,dev_new_inferences=176,old_model_reruns=0,whole_goal_complete=False))
 gate=compute_gates(reports)
 exclusive(Y/'GATES.json',dict(at=clock(),checks=gate,passed=all(gate.values()),failed=[k for k,v in gate.items() if not v],whole_goal_complete=False))

def compute_gates(reports):
 a=next(r['summary'] for r in reports if r['condition']=='tied_readout' and r['split']=='dev')
 b=next(r['summary'] for r in reports if r['condition']=='direct_readout' and r['split']=='dev')
 return dict(raw_correct_strictly_higher=b['all']['raw_correct']>a['all']['raw_correct'],
  source_macro_raw_strictly_higher=b['source_group_macro_raw_accuracy']>a['source_group_macro_raw_accuracy'],
  confirmed_correct_strictly_higher=b['all']['confirmed_correct']>a['all']['confirmed_correct'],
  source_macro_confirmed_strictly_higher=b['source_group_macro_confirmed_correct_fraction']>a['source_group_macro_confirmed_correct_fraction'],
  normal_confident_wrong_zero=b['normal']['confirmed_wrong']==0,
  normal_confirmed_correct_preserved=b['normal']['confirmed_correct']>=a['normal']['confirmed_correct'],
  nonmain_false_effective_not_increased=b['nonmain_false_confirmed']<=a['nonmain_false_confirmed'],
  normal_each_family_positive_preserved=all(b['normal_anchors_by_family'][k]['positive_confirmed']>=v['positive_confirmed'] for k,v in a['normal_anchors_by_family'].items()),
  contrary_raw_correct_preserved=b['contrary']['raw_correct']>=a['contrary']['raw_correct'],
  contrary_confirmed_correct_preserved=b['contrary']['confirmed_correct']>=a['contrary']['confirmed_correct'],
  contrary_confident_wrong_not_increased=b['contrary']['confirmed_wrong']<=a['contrary']['confirmed_wrong'],
  main_reverse_raw_correct_preserved=b['main_reverse']['raw_correct']>=a['main_reverse']['raw_correct'],
  main_reverse_confirmed_correct_preserved=b['main_reverse']['confirmed_correct']>=a['main_reverse']['confirmed_correct'],
  main_reverse_confident_wrong_not_increased=b['main_reverse']['confirmed_wrong']<=a['main_reverse']['confirmed_wrong'],
  known_interleaved_never_confirmed=b['all']['interleaved_confirmed']==0)

if __name__=='__main__':
 try:main()
 except BaseException:
  p=Y/'EVAL_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
