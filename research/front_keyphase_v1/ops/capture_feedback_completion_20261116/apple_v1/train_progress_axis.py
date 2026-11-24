from train_observer import *
def axis_input(values,axis):
 v=np.asarray(values)[:,axis];ix=np.linspace(0,len(v)-1,7).round().astype(int)
 return np.r_[v[ix]-v[0],v[0],v[-1]].astype('float32')
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not(U/'AXIS_FROZEN.json').exists()
 xx=[];yy=[];truth={r['person']:r for r in read(U/'SOURCE_TRUTH.json')['rows']};tr=[]
 for r in read(U/'INPUTS.json')['rows']:
  if r['role']!='fit':continue
  z=np.load(U/'features'/f'p{r["person"]:02d}.npz');t=truth[r['person']]
  for axis,(a,b) in enumerate([t['mouth_sweep'],t['wipe_sweeps'][0]]):
   # Rightward in a frontal image is decreasing x; wiping down increases y.
   v=z['mouth'if axis==0 else'pair'][a:b+1];label=1 if axis==0 else 2
   tr.append(dict(person=r['person'],axis=axis,span=[a,b],label=label))
   for sc in [.8,1,1.2]:
    for offset in[-.06,0,.06]:
     w=v*sc+offset
     variants=[(w,label),(w[::-1],3-label),(np.repeat(w[0:1],len(w),0),0),(np.repeat(w[-1:],len(w),0),0)]
     for ratio in[.25,.5]:variants.append((w[:max(2,round(len(w)*ratio))],3))
     for q,c in variants:xx.append(axis_input(q,axis));yy.append(c)
 write(U/'AXIS_FROZEN.json',dict(cause='Preparation/orthogonal movement can dominate full-vector classifier; guide specifies axis, completion keeps start and end relative to body target.',
  classes=['hold','decreasing_complete','increasing_complete','partial_observed'],steps=500,seeds=[42,43,44],confidence=.75,
  truth=tr,derived_partial='Public FIT source prefixes only, not actual learner-error accuracy',no_goal_label_input=True,word_specific_threshold=False,
  acceptance='Normal full movement, partial prefixes, reversed and holds; no additional true-source difference claims.'))
 write(U/'AXIS_TRAINED.json',dict(runs=train('axis',np.array(xx),np.array(yy))));print('AXIS',len(xx))
if __name__=='__main__':main()
