from probe_short import *
import hashlib

def main():
 torch.set_num_threads(2);a=FrontalLesson();b=ShortArrivalLesson();rows=[]
 truth=read(O/'basic_demo_feedback_20261115/acquisition_motion_followup/MOTION_TRUTH.json')['rows']
 for r in sources():
  if r['word']!='WORD1290':continue
  raw=dict(np.load(r['raw']));n=len(raw['xy_px']);base=a.analyze(raw);normal=b.analyze(raw)
  aa={c['id']:c['status']for c in base['criteria']};bb={c['id']:c['status']for c in normal['criteria']}
  rows.append(dict(id=r['id'],kind='normal',before=len(base['events']),after=len(normal['events']),passed=aa==bb and len(base['events'])==len(normal['events'])))
  for factor in [1.25,1.5,2]:
   ids=np.unique(np.rint(np.arange(0,n-1,factor)).astype(int));z=subset(raw,ids);qa=a.analyze(z);qb=b.analyze(z)
   # Temporal decimation is a derived stress test, not a newly observed person.
   passed=len(qb['events'])>=len(qa['events']) and len(qb['events'])<=2
   rows.append(dict(id=r['id'],kind='speed_'+str(factor),before=len(qa['events']),after=len(qb['events']),passed=passed,source_indices=ids.tolist(),events=qb['events']))
  if r['person']in [1,2,11,12,13]:
   intervals=sorted([t for t in truth if t['clip_id']==r['id'] and t['label']==1],key=lambda t:t['start'])
   if len(intervals)<2:continue
   end=intervals[0]['end'];part=intervals[1]['start'];tail=list(range(n-10,n))
   controls={'one_then_hold':list(range(end+1))+[end]*30+tail,'second_approach_only':list(range(part+1))+[part]*30+tail,'constant_near':[end]*90}
   for kind,ids in controls.items():
    z=subset(raw,np.array(ids));qa=a.analyze(z);qb=b.analyze(z);limit=0 if kind=='constant_near'else 1
    rows.append(dict(id=r['id'],kind=kind,before=len(qa['events']),after=len(qb['events']),passed=len(qb['events'])<=limit,source_intervals=intervals[:2],source_indices=ids,events=qb['events']))
   z={k:v.copy()for k,v in raw.items()};start=intervals[1]['start'];end=intervals[1]['end'];z['xy_px'][start:end+1,112:133]=np.nan;z['score'][start:end+1,112:133]=0
   qb=b.analyze(z);qa=a.analyze(z);rows.append(dict(id=r['id'],kind='second_hand_missing',before=len(qa['events']),after=len(qb['events']),passed=len(qb['events'])<=1))
  print('SHORT',r['id'],'failures',[(x['kind'],x['before'],x['after'])for x in rows if x['id']==r['id']and not x['passed']],flush=True)
 speed=[r for r in rows if r['kind'].startswith('speed_')]
 result=dict(passed=all(r['passed']for r in rows),rows=rows,improved=sum(r['after']>r['before']for r in speed),private_data=False,scope='Frontal public normal and derived temporal/partial/dropout controls. Not novice accuracy or verified physical contact.',training=False)
 write(D/'checks/SHORT_CHECK.json',result)
 print('SUMMARY',result['passed'],len(rows),result['improved'],flush=True)
if __name__=='__main__':main()
