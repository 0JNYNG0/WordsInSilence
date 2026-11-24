"""Freeze score-free query manifests and numeric gates before training."""
from evaluate_readout import *

def main():
 assert not (Y/'FROZEN.json').exists() and not (Y/'EVALUATION_PROTOCOL.json').exists()
 graphs=previous.graphs();splits={};manifests={};pins={}
 for split in ['fit','dev']:
  rr=get_rows(split);qq=[]
  for r in rr:
   for q in query_rows(r,graphs[r['word_id']]):
    qq.append(dict(q,flow_key=r['flow_key'],clip_id=r['clip_id'],person=r['person'],kind=r['kind']))
   for key,hkey in [('cache','cache_sha256'),('input','input_sha256'),('base_path','base_sha256')]:
    if key in r:pins[r[key]]=r[hkey]
   pins.update(dict(r.get('extra_pins',[])))
  manifests[split]=qq;splits[split]=dict(videos_and_variants=len(rr),unswapped_queries=len(qq),known=sum(q['target']>=0 for q in qq),held=sum(q['target']<0 for q in qq),source_relations=len({(q['clip_id'],q['first'],q['second']) for q in qq}))
 assert splits['fit']['known']==288 and splits['fit']['held']==27 and splits['dev']['videos_and_variants']==88
 for p,h in pins.items():assert sha(p)==h,p
 gates=dict(raw_correct_strictly_higher='direct>tied',source_macro_raw_strictly_higher='direct>tied',
  confirmed_correct_strictly_higher='direct>tied',source_macro_confirmed_strictly_higher='direct>tied',
  normal_confident_wrong_zero=0,normal_confirmed_correct_preserved='direct>=tied',
  nonmain_false_effective_not_increased='direct<=tied',
  normal_each_family_positive_preserved='each direct>=tied',
  contrary_raw_correct_preserved='direct>=tied',contrary_confirmed_correct_preserved='direct>=tied',contrary_confident_wrong_not_increased='direct<=tied',
  main_reverse_raw_correct_preserved='direct>=tied',main_reverse_confirmed_correct_preserved='direct>=tied',main_reverse_confident_wrong_not_increased='direct<=tied',
  known_interleaved_never_confirmed=0)
 exclusive(Y/'EVALUATION_PROTOCOL.json',dict(at=clock(),status='FROZEN_BEFORE_TRAINING_NO_PREDICTIONS',splits=splits,query_manifest=manifests,
  graphs=graphs,pins=pins,numeric_gates=gates,no_model_forward=True,no_dev_selection=True,
  subgroups={split:dict(known_contrary=sum(q['target']==0 for q in qq),main_reverse_queries=sum(q['kind']=='actual_reverse_main' for q in qq),local_or_short_queries=sum('local' in q['kind'] or 'short' in q['kind'] for q in qq)) for split,qq in manifests.items()},
  absent_subgroups='DEV local/short0 is NOT_EVALUATED, never a zero-error pass; actual_reverse_main kept separate without claiming short local semantic error accuracy',
  contract='Evaluation targets do not enter runtime observe; interleaving held, missing witness is not absence gold; all known queries in denominator, including abstentions.',
  fit_queries_are_dependent=True,whole_goal_complete=False))
 print(splits,flush=True)

if __name__=='__main__':main()
