"""Remove irrelevant opposite-hand dependency from one-hand region finding."""
from train_observer import *
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not(U/'ROLE_REGION_FROZEN.json').exists()
 inputs={r['person']:r for r in read(U/'INPUTS.json')['rows']if r['role']=='fit'};records=read(U/'REGION_REPAIR_FROZEN.json')['records'];cache={p:relation(dict(np.load(r['channels'])))for p,r in inputs.items()}
 write(U/'ROLE_REGION_FROZEN.json',dict(cause='Public capture crop: right hand/mouth observed, irrelevant lowered left hand outside frame; shared region feature made mouth region disappear.',training_people=sorted(inputs),records=records,heads={'mouth_role':'right hand relative to mouth only','pair_role':'right hand relative to left palm only'},steps=500,seeds=[42,43,44],confidence=.70,forbidden=['User data','DEV/held fit','Missing-coordinate imputation','Word-specific cutoff'],gate='Original normal/control regression unchanged; upper-body source capture recovers without inventing hidden points. People16/17 now diagnostic since inspected.'))
 runs={}
 for k,name in[(1,'mouth_role'),(2,'pair_role')]:
  x=[];y=[]
  for r in records:
   rel,av=cache[r['person']];i=r['frame']
   if av[i,k-1]:x.append(rel[i,(k-1)*2:k*2]);y.append(int(r['label']==k))
  runs[name]=train(name,np.array(x),np.array(y))
 write(U/'ROLE_REGION_TRAINED.json',dict(runs=runs));print('ROLE_HEADS_TRAINED')
if __name__=='__main__':main()
