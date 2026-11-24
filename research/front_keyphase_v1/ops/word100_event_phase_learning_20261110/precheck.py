from phase_core import *
import ast
torch.set_num_threads(4);torch.set_num_interop_threads(1);cfg=read(U/'FROZEN.json');checks=[]
def ck(n,x):checks.append(dict(name=n,passed=bool(x)));assert x,n
for p,h in {**cfg['pins'],**cfg['code_pins'],**cfg['protections']}.items():ck('pin:'+p,sha(p)==h)
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'))
rr=rows();fit=[r for r in rr if r['role']=='fit'];dev=[r for r in rr if r['role']=='development'];ck('100words899FIT300DEV',len(fit)==899 and len(dev)==300 and len(set(r['class_index'] for r in fit))==100);ck('recordIDsdisjoint',not {r['person'] for r in fit}&{r['person'] for r in dev})
for r in fit:
 z=np.load(U/'cache'/f"{r['clip_id']}.npz");ck('all100cache:'+r['clip_id'],z['g_full'].shape[0]==len(r['frame_id']) and z['q_full'].shape==(len(r['frame_id']),10,10));ck('labels_finite',np.isfinite(z['phase_y']).all() and np.all(z['phase_w']>=0))
for seq,ncomplete in [([1,1,2,2,3,3],1),([3,3,2,2,1,1],0),([1,1,2,2],0),([1,2,3],0),([1,1,2,2,3,3,1,1,2,2,3,3],2),([1,1,2,2,1,1,2,2,3,3],1)]:
 p=np.eye(5)[seq];events,parts=proposals(p,np.ones(len(p),bool));ck('no_force_or_reorder:'+str(seq),len(events)==ncomplete);covered={j for e in events+parts for j in range(e['a'],e['b']+1)};ck('partials_not_removed',covered==set(range(len(seq))))
r=next(r for r in fit if r['pairs']);z=np.load(U/'cache'/f"{r['clip_id']}.npz");r.update({k:z[k] for k in z.files});net=PhaseNet();g,q,h,l,y,w=batch([r]);p,word,_=net(g,q,h,l);ck('outputs',p.shape==(1,len(r['frame_id']),10,5) and word.shape==(1,100));loss=(-(p.log_softmax(-1)*y).sum(-1)*w).sum()/w.sum();loss.backward();ck('gradients_and_finite',torch.isfinite(loss) and net.phase[0].weight.grad.abs().sum()>0)
write(U/'PRETRAIN_CHECK.json',dict(passed=True,checks=checks,count=len(checks),reviewer='Same Codex code/data contracts, not independent expert boundary labels.'));print(json.dumps(dict(passed=True,count=len(checks))))
