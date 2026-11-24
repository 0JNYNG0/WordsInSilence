"""Read saved public evidence; explain missing corroboration without changing it."""
import run as R
import numpy as np
from collections import Counter

def main():
 f=R.read(R.U/'FROZEN.json');r=R.read(R.U/'RESULT.json');v=R.read(R.U/'VERIFIED.json');assert v['status']=='PASS_REAL600_EVIDENCE_PRESERVATION';R.check(f['pins']);R.check(r['pins'])
 samples={x['clip_id']:R.load(x['output']) for x in f['rows']};meta={x['clip_id']:x for x in f['rows']};base=Counter();counts=Counter();rows=[];pins={};out=R.U/'causes';out.mkdir()
 for cid,z in samples.items():
  d=z['core'][:,None,None]*np.ones((1,2,15),bool);st=z['angle_state'];assert st.shape==d.shape
  base['reference_audit_angle_cells']+=int(d.sum())
  for s in range(5):base['state_'+str(s)]+=int((d&(st==s)).sum())
  base['eligible']+=int((d&z['quality_eligible'][:,:,:15]).sum())
 for i,row in enumerate(r['rows']):
  z=R.load(row['pair']);q=samples[row['query']];ref=samples[row['reference']];n=len(ref['core']);stq=q['angle_state'];strf=ref['angle_state'];so=z['sequence_ok'][:,:,:15];qual=z['sequence_quality'][:,:,:15]
  no_second=np.zeros_like(so);disagreement=np.zeros_like(so);identity=np.zeros_like(so);causelo=np.full(so.shape,-1,int);causehi=causelo.copy();point_no_second=np.zeros_like(so);point_disagreement=np.zeros_like(so)
  for j in range(n):
   a,b=int(np.floor(z['phase_old'][j])),int(np.ceil(z['phase_old'][j]));point_no_second[j]=(stq[a]==2)|(stq[b]==2)|(strf[j]==2);point_disagreement[j]=(stq[a]==3)|(stq[b]==3)|(strf[j]==3)
  for j in range(2,n-2):
   for h in range(2):
    for a in range(15):
     if not so[j,h,a]:continue
     lo=int(z['span_first'][j-2]);hi=int(z['span_last'][j+2])
     if z['source'][j,h,a]==1:lo=min(lo,int(np.floor(z['phase_new'][j-2])));hi=max(hi,int(np.ceil(z['phase_new'][j+2])))
     qs=stq[lo:hi+1,h,a];rs=strf[j-2:j+3,h,a];no_second[j,h,a]=bool((qs==2).any() or (rs==2).any());disagreement[j,h,a]=bool((qs==3).any() or (rs==3).any());identity[j,h,a]=bool(q['identity_conflict'][lo:hi+1,h].any() or ref['identity_conflict'][j-2:j+3,h].any());causelo[j,h,a]=lo;causehi[j,h,a]=hi
  cause=no_second|disagreement|identity
  assert np.array_equal(so&~qual,so&cause),row['name']
  assert not(qual&cause).any()
  stats=dict(sequence_computable=int(so.sum()),sequence_corroborated=int(qual.sum()),uncorroborated=int((so&~qual).sum()),no_second_prediction=int((so&no_second).sum()),prediction_disagreement=int((so&disagreement).sum()),identity_disagreement=int((so&identity).sum()),only_no_second_prediction=int((so&no_second&~disagreement&~identity).sum()),both_prediction_and_identity_or_missing=int((so&disagreement&(identity|no_second)).sum()))
  counts.update(stats);path=out/(row['name']+'.npz');np.savez_compressed(path,sequence_no_second=no_second,sequence_prediction_disagreement=disagreement,sequence_identity_disagreement=identity,query_interval_start=causelo,query_interval_end=causehi,point_no_second=point_no_second,point_prediction_disagreement=point_disagreement);pins[str(path)]=R.sha(path);rows.append(dict(name=row['name'],path=str(path),**stats))
  if (i+1)%100==0:print('CAUSES',i+1,600,flush=True)
 assert counts['sequence_corroborated']==v['groups']['finger_angles']['quality'] and counts['sequence_computable']==v['groups']['finger_angles']['sequence']
 R.check(f['pins']);R.check(r['pins']);R.check(f['protected']);R.write(R.U/'CAUSE_AUDIT.json',dict(at=R.now(),status='EXPLAINED_ALL_UNCORROBORATED_SEQUENCE_ANGLES',base_core_states=dict(base),pair_counts=dict(counts),rows=rows,pins=pins,code=R.sha(__file__),
  states={'0':'not_observed','1':'degenerate_geometry','2':'no_second_assessment','3':'predictions_disagree','4':'assessed_no_conflict'},
  meaning='State2 is missing second evidence,NOT false/incorrect hand. State3 is disagreement,NOT anatomical truth. State4 is agreement,NOT anatomical truth. Causes may overlap. Coordinates/differences/status preserved. Quality false is explicitly explained rather than called gesture error.',
  method='Exhaustive saved source interval audit; cause union exactly equals previous uncorroborated set,all600. Same Codex arithmetic,not independent anatomical ground truth.',training=0,inference=0,production_changed=False));print('CAUSE_AUDIT_COMPLETE',dict(counts),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  p=R.U/'CAUSE_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
