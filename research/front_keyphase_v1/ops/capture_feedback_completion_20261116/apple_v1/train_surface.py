"""New semantic target: usable open/curved palm versus clenched/other form.

This is NOT a relabeling of all bent fingers as a fist, and does not overwrite
the existing finger-extension or coarse-flexion models.
"""
from train_observer import *
from local_shape import HandHead
from finger_model import features,windows
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not(U/'SURFACE_FROZEN.json').exists();rows=read(U/'INPUTS.json')['rows'];tt={r['person']:r for r in read(U/'SOURCE_TRUTH.json')['rows']};xx=[];yy=[];meta=[]
 for r in rows:
  if r['role']!='fit':continue
  t=tt[r['person']];ch=np.load(r['channels']);b=t['pair_anchors'];a=t['mouth_anchors']
  # Source-visible right palm, separate left fist BEFORE overlap, index form.
  for h,ids,label in [('R',[b[0],b[2],b[4],b[-2]],0),('L',[b[0]-3],1),('R',[a[0],a[2],a[-1]],2)]:
   f,av=features(ch['tcn280'],h)
   for i in ids:
    if not av[i].all():continue
    xx.append(windows(f,[i])[0]);yy.append(label);meta.append(dict(person=r['person'],frame=i,hand=h,label=label,source=r['source_sha256']))
 # Prior reviewed public digit forms: four open fingers is a palm form;
 # numeral two/six are distinct finger patterns. No symbolic number target.
 demo=read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']
 for r in demo:
  if r['role']!='fit'or r['word']not in['WORD2633','WORD2635','WORD2637']:continue
  z=np.load(r['channels']);f,a=features(z['tcn280'],'R')
  for i in r['anchors']:
   if not a[i].all():continue
   label=0 if r['word']=='WORD2635'else 2;xx.append(windows(f,[i])[0]);yy.append(label);meta.append(dict(person=r['person'],frame=i,hand='R',label=label,source=r['source_sha256']))
 write(U/'SURFACE_FROZEN.json',dict(labels=['open_or_curved_palm','clenched_hand','distinct_finger_pattern'],rows=meta,steps=500,seeds=[42,43,44],confidence=.85,
   limitation='Observed coarse shape, no contact/hidden-finger inference. Existing finger model and flexion labels unchanged.',
   cause='Normal curved palm was incorrectly equated with clenched fist by coarse all-flexed output. New semantic target is source-visible shape, not individual extension.',
   scope='Standalone shared hand-local head; inputs exclude word, body position, person and requested target. All original model outputs retained.'))
 x=np.array(xx);y=torch.tensor(yy);runs=[]
 for seed in[42,43,44]:
  ck=torch.load(O/'basic_demo_feedback_20261115/local_shape/temporal'/f'{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);m=HandHead('temporal');m.load_state_dict(ck['state']);tx=torch.tensor((x-ck['mean'])/ck['std']);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.005)
  for _ in range(500):
   loss=F.cross_entropy(m(tx),y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
  p=U/'runs'/f'surface_{seed}.pt';torch.save(dict(state=m.state_dict(),mean=ck['mean'],std=ck['std']),p);runs.append(dict(seed=seed,sha256=sha(p)))
 write(U/'SURFACE_TRAINED.json',dict(runs=runs));print('SURFACE_TRAINED',len(xx))
class Surface:
 def __init__(self):
  self.models=[]
  for seed in[42,43,44]:
   c=torch.load(U/'runs'/f'surface_{seed}.pt',map_location='cpu',weights_only=False);m=HandHead('temporal');m.load_state_dict(c['state']);m.eval();self.models.append((m,c['mean'],c['std']))
 def sequence(self,x,hand):
  f,a=features(x,hand);w=windows(f,list(range(len(x))));qq=[]
  with torch.inference_mode():
   for m,mu,sd in self.models:
    p=m(torch.tensor((w-mu)/sd)).softmax(-1).numpy();q=p.argmax(-1);q[p.max(-1)<.85]=-1;qq.append(q)
  qq=np.array(qq);q=np.where((qq==qq[0]).all(0),qq[0],-1);q[~a.all(1)]=-1;return q
if __name__=='__main__':main()
