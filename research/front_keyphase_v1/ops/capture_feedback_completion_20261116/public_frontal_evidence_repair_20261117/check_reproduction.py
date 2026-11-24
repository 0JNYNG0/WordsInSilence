"""Explicit public-source fault injection through the real aggregation path."""
from check_evidence import *
import apple_candidate,evidence_candidate

def main():
 torch.set_num_threads(2);old=RoleEvidenceCandidate();new=EvidenceLesson();rows=[]
 original1=apple_candidate.hand_evidence;original2=evidence_candidate.hand_evidence
 try:
  for r in read(N/'INPUTS.json')['rows']:
   require_front(r)
   if r['word']!='WORD0099' or r.get('role')=='fit':continue
   raw=dict(np.load(r['raw']));base=old.analyze(raw);ev=next(e for e in base['events']if e['region']==2);lo,hi=ev['context'];ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));s=new.surface.sequence(ch['tcn280'],'R');f=original1(ch,'R');centers=[i for i in range(lo+4,hi-4)if np.all(s[i-3:i+4]==0)and not f['uncertain'][i-3:i+4].any()]
   if not centers:continue
   mid=centers[len(centers)//2]
   for kind in ['short_projected_overlap','actual_missing','long_overlap']:
    a,b=mid-2,mid+2
    if kind=='long_overlap':a,b=lo+1,hi-1
    def injected(ch,hand):
     facts=original1(ch,hand)
     if hand=='R':
      facts['projected_overlap'][a:b+1]=True
      if kind=='actual_missing':facts['own_missing'][a:b+1]=True
      facts['uncertain']=facts['projected_overlap']|facts['own_missing']
     return facts
    apple_candidate.hand_evidence=evidence_candidate.hand_evidence=injected
    before=old.analyze(raw);after=new.analyze(raw);aa=next(c for c in before['criteria']if c['id']=='wipe_shape');bb=next(c for c in after['criteria']if c['id']=='wipe_shape')
    passed=aa['status']=='unknown' and bb['status']==('supported' if kind=='short_projected_overlap' else 'unknown')
    rows.append(dict(id=r['id'],kind=kind,frame_span=[a,b],before=aa['status'],after=bb['status'],passed=passed,synthetic_observation_mask=True));print('REPRO',r['id'],kind,aa['status'],bb['status'],flush=True)
    apple_candidate.hand_evidence=original1;evidence_candidate.hand_evidence=original2
 finally:apple_candidate.hand_evidence=original1;evidence_candidate.hand_evidence=original2
 write(D/'checks/AGGREGATION_REPRODUCTION.json',dict(rows=rows,passed=bool(rows)and all(r['passed']for r in rows),scope='Public observed sequences with injected projected-overlap/missing flags; no modified source frames or private footage',private_data=False))
 assert rows and all(r['passed']for r in rows)
if __name__=='__main__':main()
