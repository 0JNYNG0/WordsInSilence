from episode_core import *
import ast
torch.set_num_threads(4);checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
ck('module_isolation',U.name=='word100_action_episode_supervision_20261110')
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'));ck('syntax:'+p.name,True)
ck('parent_arrival_same',MAP[4]==MAP[5]);ck('crossing_same_family',MAP[6]==MAP[7])
net=EpisodeNet().eval();g=torch.zeros((2,15,66));q=torch.zeros((2,15,10,10));fh=torch.randn((2,15,64));ll=torch.tensor([15,11])
with torch.inference_mode():
 r,b,w,h=net(g,q,fh,ll);ck('output_shapes',r.shape==(2,15,8) and b.shape==(2,15,10,3) and w.shape==(2,100))
 r1,b1,_,_=net(g,q,fh,ll,True);r2,b2,_,_=net(g,q,fh*100,ll,True);ck('appearance_unavailable_path',torch.equal(r1,r2) and torch.equal(b1,b2))
bp=np.ones((10,10,3));qp=np.zeros((10,10,10));rp=np.full((10,8),1/8);ee,pp=decode(rp,bp,qp);ck('unobserved_not_completed',not ee and not pp)
eps=read(U/'ACTION_EPISODES.json')['rows']
for e in eps:ck('reviewed_start_middle_end',e['onset'][0]<=e['onset'][1]<=e['offset'][0]<=e['offset'][1])
cfg=read(U/'FROZEN.json')
for p,h in {**cfg['pins'],**cfg['protections']}.items():ck('protected:'+p,sha(p)==h)
write(U/'PREFLIGHT.json',dict(passed=True,count=len(checks),checks=checks));write(U/'CODE_PINS.json',{str(p):sha(p) for p in [U/'episode_core.py',U/'train.py']});print(json.dumps(dict(passed=True,checks=len(checks))))
