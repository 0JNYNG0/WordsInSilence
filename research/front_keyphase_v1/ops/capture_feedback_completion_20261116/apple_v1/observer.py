from train_observer import *
from visible_lesson import contiguous
class Ensemble:
 def __init__(self,name):
  self.models=[]
  for seed in[42,43,44]:
   c=torch.load(U/'runs'/f'{name}_{seed}.pt',map_location='cpu',weights_only=False);m=Head(len(c['mean']),c['out']);m.load_state_dict(c['state']);m.eval();self.models.append((m,c['mean'],c['std']))
 def predict(self,x,confidence):
  p=[]
  with torch.inference_mode():
   for m,mu,sd in self.models:p.append(m(torch.tensor((x-mu)/sd,dtype=torch.float32)).softmax(-1).numpy())
  p=np.array(p);q=p.argmax(-1);q[p.max(-1)<confidence]=-1;return np.where((q==q[0]).all(0),q[0],-1),p
def regions(s,value):
 # A brief unresolved observed frame does not erase either adjoining span.
 ids=np.flatnonzero(s==value);runs=np.split(ids,np.flatnonzero(np.diff(ids)>4)+1)if len(ids)else[]
 return [[int(a[0]),int(a[-1])]for a in runs if len(a)>=3]
class AppleObserver:
 def __init__(self,local=False):self.feature=Features();self.shape=ViewConditionedShape();self.local=local;self.region=Ensemble('region_local'if local else'region');self.progress=Ensemble('progress')
 def analyze(self,raw):
  st=time.perf_counter();ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  f,q,_=self.feature(ch);rel,av=relation(ch);x=np.c_[np.nan_to_num(rel),av]if self.local else np.c_[np.nan_to_num(rel),av,f];s,p=self.region.predict(x,.70)
  s[(s==1)&~av[:,0]]=-1;s[(s==2)&~av[:,1]]=-1;self.last_features=f
  sh={h:self.shape.sequence(ch,h)for h in['L','R']};spans={str(k):regions(s,k)for k in[1,2]};progress=[]
  for k in[1,2]:
   for a,b in spans[str(k)]:
    if b-a<3:continue
    v=rel[a:b+1,(k-1)*2:k*2];good=av[a:b+1,k-1]
    state,prob=self.progress.predict(progress_input(v)[None],.75)
    progress.append(dict(region=k,span=[a,b],direction=int(state[0]),observed=bool(good.all()),delta=(v[-1]-v[0]).tolist(),prob=prob[:,0].round(3).tolist()))
  return dict(regions=spans,region_states=s.tolist(),progress=progress,shape={h:{k:v.tolist()for k,v in sh[h].items()if k in['states','fingers','available']}for h in['L','R']},rel=rel.tolist(),available=av.tolist(),seconds=time.perf_counter()-st)
def main():
 torch.set_num_threads(2);local='--local'in sys.argv;net=AppleObserver(local);out=[]
 for r in read(U/'INPUTS.json')['rows']:
  z=net.analyze(dict(np.load(r['raw'])));write(U/('local_sequence'if local else'baseline_sequence')/f'p{r["person"]:02d}.json',z)
  out.append(dict(person=r['person'],regions=z['regions'],progress=[{k:v for k,v in x.items()if k!='prob'}for x in z['progress']]))
 write(U/('SEQUENCE_LOCAL.json'if local else'SEQUENCE_BASELINE.json'),dict(rows=out));print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
