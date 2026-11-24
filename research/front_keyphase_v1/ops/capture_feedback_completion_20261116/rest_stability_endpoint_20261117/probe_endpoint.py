from base import *
sys.path.insert(0,str(P))
from short_arrival import ShortArrivalLesson,adapter,torch
def main():
 torch.set_num_threads(2);net=ShortArrivalLesson();rows=[]
 for p in sorted((P/'probe').glob('NIA_SL_WORD1290*.npz')):
  z=dict(np.load(p));q=net.analyze(z)
  row=dict(id=p.stem,events=q['events'],criteria={c['id']:c['status']for c in q['criteria']},near=sum(x==2 for x in q['relation_states']),inspection_near=sum(x==2 for x in q['inspection_relation_states']),source=str(p))
  write(D/'probe'/f'{p.stem}.json',q);rows.append(row);print('ENDPOINT',p.stem,len(q['events']),row['near'],row['inspection_near'],flush=True)
 write(D/'probe/ENDPOINT_BASELINE.json',rows)
if __name__=='__main__':main()
