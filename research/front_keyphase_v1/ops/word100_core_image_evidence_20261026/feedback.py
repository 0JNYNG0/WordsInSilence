import prepare as R
import source_feedback as S
import numpy as np,traceback
from collections import Counter
from pathlib import Path
C=R.mod('preserved_feedback_consumer',R.O/'word100_feedback_evidence_consumer_20261026/consumer.py')
def fixtures():
 count=0
 for qs in [0,1,2,3,4,5]:
  for rs in [0,1,2,3,4,5]:
   shape=(3,2,23);q=dict(available=np.ones(shape,bool),quality_after=np.ones(shape,bool),state_after=np.full((3,2,15),5,np.uint8));r={k:v.copy() for k,v in q.items()};q['state_after'][1,0,0]=qs;r['state_after'][1,0,0]=rs;q['available'][1,0,0]=qs>0;r['available'][1,0,0]=rs>0;q['quality_after'][1,0,0]=qs==5;r['quality_after'][1,0,0]=rs==5;pair=dict(domain=np.ones(shape,bool),phase_old=np.arange(3,dtype=float),point_corroborated_after=q['quality_after']&r['quality_after']);got=S.states_on_pair(pair,q,r)[1,0,0]
   expected=(3 if rs==0 else 1) if qs==0 else (2 if rs==0 else (4 if 1 in (qs,rs) else (8 if qs==rs==5 else (6 if qs==5 else (5 if rs==5 else 7)))))
   assert got==expected,(qs,rs,got,expected);count+=1
 return count
def main():
 assert R.read(R.U/'INTEGRATION_VERIFIED.json')['pairs']==200;f=R.read(R.U/'FROZEN.json');p=R.read(R.U/'PREPARED.json');res=R.read(R.U/'PAIRS_DONE.json');R.check(p['outputs']);R.check(res['pins']);R.check(f['protected']);testcount=fixtures();R.write(R.U/'FEEDBACK_FROZEN.json',dict(at=R.now(),pins={str(p):R.sha(p) for p in [Path(__file__),Path(S.__file__),Path(C.__file__),R.U/'INTEGRATION_VERIFIED.json',R.U/'PREPARED.json',R.U/'PAIRS_DONE.json']},purpose='Research reusable source-aware receipts, no validated skill verdict/correction or operating update',fixtures=testcount));(R.U/'receipts').mkdir();samples={r['clip_id']:R.load(r['output']) for r in p['rows']};pins={};counts=Counter();rows=[]
 for i,row in enumerate(res['rows']):
  q=samples[row['query']];r=samples[row['reference']];z=R.load(row['pair']);a=S.build(z,q,r,row['query'],row['reference'],C,'before');b=S.build(z,q,r,row['query'],row['reference'],C,'after');assert a['whole_gesture_correct'] is b['whole_gesture_correct'] is None;before=[]
  for ha,hb in zip(a['hands'],b['hands']):
   for ga,gb in zip(ha['groups'],hb['groups']):
    for k in ['audit_cells','point_compared_cells','sequence_compared_cells','point_mean_difference','sequence_level_mean_difference','sequence_change_mean_difference','missing_point_intervals','sequence_context_unavailable_intervals','reference_missing_intervals','source_cells']:assert ga[k]==gb[k],k
    assert gb['corroborated_sequence_cells']>=ga['corroborated_sequence_cells'];before.append(dict(side=ha['side'],group=ga['group'],before_corroborated=ga['corroborated_sequence_cells'],after_corroborated=gb['corroborated_sequence_cells']));counts['preserved_groups']+=1
  states=S.states_on_pair(z,q,r);declared=Counter()
  for hand in b['source_separated_point_evidence']:
   for angle in hand['angles']:
    for e in angle['states']:
     assert e['gesture_error'] is None and e['correction_required'] is None;assert sum(s['samples'] for s in e['intervals'])==e['cells'];declared[e['code']]+=e['cells']
  actual=Counter({S.CODES[int(c)]:int(n) for c,n in zip(*np.unique(states[states>0],return_counts=True))});assert actual==declared;assert sum(actual.values())==int(z['domain'][:,:,:15].sum());counts.update(actual);b['before_after_evidence']=before;b['source_context']=dict(word_code=row['word_code'],label=row['label'],split='publicTRAIN_development',anatomical_truth=False);op=R.U/'receipts'/Path(row['pair']).with_suffix('.json').name;R.write(op,b);pins[str(op)]=R.sha(op);rows.append(dict(query=row['query'],reference=row['reference'],receipt=str(op)));counts['pairs']+=1
  if (i+1)%50==0:print('SOURCE_FEEDBACK',i+1,200,flush=True)
 assert counts['pairs']==200 and counts['preserved_groups']==2000;R.check(f['protected']);R.write(R.U/'FEEDBACK_VERIFIED.json',dict(at=R.now(),status='PASS_SOURCE_AWARE_RESEARCH_RECEIPTS200',rows=rows,pins=pins,counts=dict(counts),state_combination_fixtures=testcount,method='Source code6x6 observed/unobserved fixtures; per-source interval counts checked against saved masks; all2000raw group metrics before/after exactly equal. SameCodex verifier,not independent human truth. No sign correctness or normative correction claims.',operating_changed=False));print('FEEDBACK_VERIFIED',dict(counts),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=R.U/'FEEDBACK_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
