"""Logic and opt-in integration tests; synthetic probabilities, no model forward/camera."""
from core import *
import copy
guard=load('_test_guard',E/'completion_guard_v2.py');adapter=load('_test_adapter',E/'runtime_adapter.py')
def decision(group):return dict(cards=[dict(id='motion',groups=[group],hands=['R'],features={},query_indices=[1,6],phase_evidence=[dict(group=group,span=[1,6],available_runs=[[1,6]])],status='supported',display_status='similar',message='same',educational_wrong=False)],path=[[0,0],[1,2]],tcn_top3=[dict(word='unchanged')],frame_id=list(range(8)))
raw=np.ones((8,280),np.float32);ts=np.arange(8)*33333;p=np.zeros((8,19),np.float32);p[:,4]=1
checks=[]
def check(ok,label):assert ok,label;checks.append(label)
a=decision([2]);b=guard.apply_guard(a,p,raw,ts,{})
check(b['cards'][0]['display_status']=='uncertain','earlier core cannot complete later');check(a['cards'][0]['display_status']=='similar','input immutable')
p[:]=0;p[:,6]=1;check(guard.apply_guard(a,p,raw,ts,{})['cards'][0]['display_status']=='similar','same event retained')
p[:]=1/19;check(guard.apply_guard(a,p,raw,ts,{})['cards'][0]['display_status']=='similar','uncertain new head cannot veto normal alone')
p[:]=0;p[:,10]=1;a=decision([3]);check(guard.apply_guard(a,p,raw,ts,{'motion':{'phase_groups':[3,4]}})['cards'][0]['display_status']=='similar','optional endpoint belongs to same stroke')
p[:]=0;p[:,14]=1;a=decision([5,7]);a['cards'][0]['phase_evidence'][0]['group']=[5,6,7,8,9];check(guard.apply_guard(a,p,raw,ts,{})['cards'][0]['display_status']=='similar','both sides of shared alternation supported')
a=decision([2]);a['cards'][0]['display_status']='difference';a['cards'][0]['status']='difference_candidate';p[:]=0;p[:,4]=1;check(guard.apply_guard(a,p,raw,ts,{})['cards'][0]['display_status']=='difference','existing difference retained')
a=decision([2]);a['cards'][0]['display_status']='uncertain';a['cards'][0]['status']='unknown';raw[:]=0;check(guard.apply_guard(a,p,raw,ts,{})['cards'][0]['display_status']=='uncertain','missing remains unknown')
raw[:]=1
class Replay:
 def channels(self,x,ts,fid,word):return decision([2])
 def raw(self,x,ts,fid,word):
  out=self.channels(x,ts,fid,word);out['quarantine_tail_called']=True;return out
class Auditor:
 def apply(self,d,x,ts):return guard.apply_guard(d,p,x,ts,{})
obj=adapter.attach_to_isolated_consumer(Replay(),Auditor());out=obj.raw(raw,ts,np.arange(8),'arbitrary_word')
check(out['quarantine_tail_called'],'final observation guard remains reachable');check(out['cards'][0]['display_status']=='uncertain','raw calls augmented channels');check(out['path']==decision([2])['path'] and out['tcn_top3']==decision([2])['tcn_top3'],'TCN and DTW preserved')
for factor in [.75,1.25]:check(guard.apply_guard(decision([2]),p,raw,(ts/factor).astype(np.int64),{})['cards'][0]['display_status']=='uncertain','conflict stable at tempo '+str(factor))
write(E/'CONTRACT_TESTS.json',dict(status='PASS',checks=checks,model_forward=False,camera=False,live_web=False));print('CONTRACT_TESTS_PASS',len(checks))
