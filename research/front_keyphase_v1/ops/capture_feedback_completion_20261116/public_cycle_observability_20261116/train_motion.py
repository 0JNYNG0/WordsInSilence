from cycle_common import *
import motion_model as mm

T=B/'acquisition_motion_followup'
def motion_input(ch,a,b):
 x=mm.feature(ch,a,b)
 if x is None:return None
 short={k:v[a:b+1]if isinstance(v,np.ndarray)and v.ndim and len(v)==len(ch['frame_id'])else v for k,v in ch.items()};d=palm_diameter_distance(short)
 if not np.isfinite(d).all():return None
 dv=np.diff(d);return np.r_[x,d[-1]-d[0],np.ptp(d),dv.mean(),dv.std(),dv.min(),dv.max(),np.abs(dv).sum()].astype(np.float32)

class RegionMotion(mm.Observer):
 def __init__(self):super().__init__('mlp32',M/'motion')
 def sequence(self,ch,radius=3):return np.array([self.predict(motion_input(ch,i-radius,i+radius))for i in range(len(ch['frame_id']))])

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);out=M/'motion';out.mkdir(exist_ok=True);assert not(out/'TRAINED.json').exists();rows=read(T/'MOTION_TRUTH.json')['rows'];by={r['id']:r for r in all_sources()};cache={};fit=[];xx=[];yy=[];excluded=[]
 def get(cid):
  if cid not in cache:cache[cid]=channels(dict(np.load(by[cid]['raw'])))
  return cache[cid]
 for r in rows:
  if r['split']!='fit':continue
  x=motion_input(get(r['clip_id']),r['start'],r['end'])
  if x is None:excluded.append(r);continue
  xx.append(x);yy.append(r['label']);fit.append(r)
 x=np.array(xx,np.float32);y=torch.tensor(yy);mu=x.mean(0);sd=np.maximum(x.std(0),1e-4);cw=torch.tensor(len(y)/(3*np.bincount(yy,minlength=3)),dtype=torch.float32);runs=[]
 write(out/'FROZEN.json',dict(rows=fit,excluded_missing=excluded,features='Existing10 palm-centre trend +7 projected-palm-diameter-normalized hand-edge/other-palm distance trends',seeds=[42,43,44],steps=400,lr=.003,confidence=.7,base=str(T/'runs'),split='Existing reviewed FIT people1-9 only; no DEV threshold/epoch selection; no private data.',reason='Short approach can be poorly represented by palm-centre motion when hand edges return to target. Supplemental observer only, not full backbone replacement.'))
 for seed in [42,43,44]:
  ck=torch.load(T/'runs'/f'mlp32_{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);net=mm.Net('mlp32',17);state=net.state_dict()
  for k,v in ck['state'].items():
   if state[k].shape==v.shape:state[k]=v
   else:state[k].zero_();state[k][:,:10]=v
  net.load_state_dict(state);mean=np.r_[ck['mean'],mu[10:]];std=np.r_[ck['std'],sd[10:]];tx=torch.tensor((x-mean)/std);opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001)
  for _ in range(400):
   loss=torch.nn.functional.cross_entropy(net(tx),y,weight=cw);opt.zero_grad();loss.backward();opt.step()
  p=out/f'mlp32_{seed}.pt';torch.save(dict(state=net.state_dict(),mean=mean,std=std),p);runs.append(dict(seed=seed,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 write(out/'TRAINED.json',dict(runs=runs,fit_intervals=len(fit),excluded_missing=len(excluded),private_data=False));old=mm.Observer('mlp32');new=RegionMotion();checks=[]
 for r in rows:
  if r['split']=='fit':continue
  ch=get(r['clip_id']);a=old.predict(mm.feature(ch,r['start'],r['end']));b=new.predict(motion_input(ch,r['start'],r['end']));c=a if a>=0 else b;checks.append(dict(**r,before=a,new=b,combined=c))
 write(out/'CHECK.json',dict(rows=checks,added_correct=sum(r['before']<0 and r['combined']==r['label']for r in checks),added_wrong=sum(r['before']<0 and r['combined']>=0 and r['combined']!=r['label']for r in checks)));print('MOTION',len(fit),read(out/'CHECK.json')['added_correct'],read(out/'CHECK.json')['added_wrong'],flush=True)
if __name__=='__main__':main()
