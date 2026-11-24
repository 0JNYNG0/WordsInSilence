from endpoint_candidate import *
sys.path.insert(0,str(P));from probe_short import subset
def main():
 torch.set_num_threads(2);a=ShortArrivalLesson();b=EndpointLesson();rows=[]
 states=read(O/'basic_demo_feedback_20261115/source_corrected/SOURCE_TRUTH.json')['rows'];motion=read(O/'basic_demo_feedback_20261115/acquisition_motion_followup/MOTION_TRUTH.json')['rows']
 for r in sources():
  if r['word']!='WORD1290':continue
  approach=sorted([x for x in motion if x['clip_id']==r['id']and x['label']==1],key=lambda x:x['start'])
  if len(approach)!=2:continue
  checks=[x for x in states if x['clip_id']==r['id']and x['state']==1 and approach[0]['end']<x['frame']<approach[1]['end']]
  raw=dict(np.load(r['raw']));n=len(raw['xy_px'])
  for anchor in checks:
   cut=anchor['frame'];ids=np.array(list(range(cut+1))+[cut]*30+list(range(n-10,n)));z=subset(raw,ids);qa=a.analyze(z);qb=b.analyze(z)
   rows.append(dict(id=r['id'],cut=cut,source_truth=anchor,before=len(qa['events']),after=len(qb['events']),passed=len(qb['events'])<=1))
 print('REVIEWED PARTIALS',len(rows),'fails',[(r['id'],r['cut'],r['before'],r['after'])for r in rows if not r['passed']],flush=True)
 write(D/'checks/REVIEWED_PARTIAL_CHECK.json',dict(passed=bool(rows)and all(r['passed']for r in rows),rows=rows,scope='Truncated at directly reviewed separated-hand source frames before second approach endpoint; held and then tail. Derived control, not a filmed human mistake.',private_data=False))
if __name__=='__main__':main()
