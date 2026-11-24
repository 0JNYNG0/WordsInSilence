from triplet_core import *
torch.set_num_threads(4)
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
net=TripletNet().eval();q=torch.randn(1,30,10,10);q[...,-2:]=1;h=torch.randn(1,30,64);ts=np.arange(30)*33333
with torch.inference_mode():
 a,b=net.segment(h,q,[(0,10,18)],[ts]);q2=q.clone();q2[:,:10]*=99;q2[:,19:]*=99;h2=h*99
 _,b2=net.segment(h2,q2,[(0,10,18)],[ts]);ck('core_cannot_read_neighbours_or_biGRU',torch.equal(b,b2))
 _,z=net.segment(h,q,[(0,10,18)],[ts],True);_,z2=net.segment(h2,q2,[(0,10,18)],[ts],True);ck('masked_core_independent',torch.equal(z,z2))
ck('no_future_inside_center',all(10<=lo<hi<=19 for lo,hi in windows(10,18,30,ts)[1:4]))
ck('overlap_not_convenient_order',lc.order_state([dict(a=0,b=10)],[dict(a=5,b=15)])=='ambiguous')
ck('reverse_not_forced_forward',lc.order_state([dict(a=10,b=15)],[dict(a=0,b=5)])=='contradicted')
ck('missing_not_omission',lc.order_state([],[])=='unknown')
ck('unobserved_not_accepted',not accepted_spans([dict(a=0,b=4,context=3,confidence=1.,observed=False)]))
ck('background_not_accepted',not accepted_spans([dict(a=0,b=4,context=0,confidence=1.,observed=True)]))
ck('partial_identity_allowed',len(accepted_spans([dict(a=0,b=4,context=3,confidence=1.,observed=True,core_state='progress')]))==1)
cfg=read(U/'FROZEN.json')
for p,v in {**cfg['protections'],**cfg['pins']}.items():ck('protected:'+p,sha(p)==v)
write(U/'PREFLIGHT.json',dict(passed=True,count=len(checks),checks=checks))
write(U/'CODE_PINS.json',{str(p):sha(p) for p in U.glob('*.py') if p.name in ['triplet_core.py','train.py','connector.py']})
print(json.dumps(dict(passed=True,checks=len(checks))))
