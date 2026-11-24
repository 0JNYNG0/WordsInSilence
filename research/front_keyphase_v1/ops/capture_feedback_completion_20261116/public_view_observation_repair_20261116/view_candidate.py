from view_common import *
from plane_model import PlaneHead,PlaneObserver,hand_plane_features
from train_palm import SupplementalPalm,HandHead
from visible_lesson import shape_or_plane,contiguous

class ViewPlane(PlaneObserver):
 def __init__(self):
  self.models=[]
  for seed in[42,43,44]:
   ck=torch.load(J/'plane'/f'{seed}.pt',map_location='cpu',weights_only=False);m=PlaneHead();m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))
class ViewPalm(SupplementalPalm):
 def __init__(self):
  self.models=[]
  for seed in[42,43,44]:
   ck=torch.load(J/'palm'/f'{seed}.pt',map_location='cpu',weights_only=False);m=HandHead('temporal');m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))

class ViewLesson(PalmMotionLesson):
 def __init__(self):super().__init__();self.view_plane=ViewPlane();self.palm=ViewPalm()
 def form_states(self,ch,hand):
  s,recovered,priorform=super().form_states(ch,hand);obs=self.shape.observations[hand];new=self.palm.sequence(ch['tcn280'],hand);p=self.view_plane.sequence(ch,hand)
  admissible=(s<0)&(priorform==0)&(new==0)&(p==0)&obs['available'].all(1)&~obs['projection_conflict']
  s[admissible]=0;recovered|=admissible
  return s,recovered,priorform
 def analyze(self,raw,lesson=None):
  start=time.perf_counter();q=super().analyze(raw,lesson);ch=channels(raw);ts=ch['timestamp_us'];s=np.array(q['relation_states'])
  context=[dict(start=int(r[0]),end=int(r[-1]))for r in contiguous(np.flatnonzero(s>0))]+q.get('additional_body_events',[])
  q['previous_plane_states']=q['plane_states'].copy();q['view_plane_states']={}
  for hand,side in [('L','left'),('R','right')]:
   states=self.view_plane.sequence(ch,hand);prior=next(c for c in q['criteria']if c['id']==side+'_orientation');c=shape_or_plane(prior['id'],hand,prior['expected'],states,q['events'],ts,'projected_plane',context)
   q['criteria']=[c if r['id']==c['id']else r for r in q['criteria']];q['view_plane_states'][hand]=states.tolist();q['plane_states'][hand]=states.tolist()
  q['seconds']=time.perf_counter()-start
  return q
