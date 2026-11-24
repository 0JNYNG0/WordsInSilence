"""Test hand-form evidence separately from precise individual extension.

Public source hands allow normal slight flexion. No inference of hidden joints,
and known closed/distinct-form outputs are not converted into open hands.
"""
from audit import *
from thanks_candidate import ShapeRecovery
sys.path.insert(0,str(G/'apple_v1'))
from train_surface import Surface
from visible_lesson import shape_or_plane,contiguous

class RecordedShape(ShapeRecovery):
 def __init__(self):super().__init__();self.observations={}
 def sequence(self,ch,hand,plane=None):
  q=super().sequence(ch,hand,plane);self.observations[hand]=q;return q

class FormLesson(ThanksCandidate):
 def __init__(self):
  super().__init__();self.shape=RecordedShape();self.plane=self.shape.plane;self.form=Surface()
 def form_states(self,ch,hand):
  q=self.shape.observations[hand];s=q['states'].copy();form=self.form.sequence(ch['tcn280'],hand);projected,_=self.shape.projection.sequence(ch['tcn280'],hand)
  compatible=(s<0)&(form==0)&(projected==0)&q['available'].all(1)&~q['projection_conflict']
  s[compatible]=0
  return s,compatible,form
 def analyze(self,raw,lesson=None):
  result=super().analyze(raw,lesson);ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  ts=ch['timestamp_us'];context=[dict(start=int(r[0]),end=int(r[-1]))for r in contiguous(np.flatnonzero(np.array(result['relation_states'])>0))]
  result['hand_form_evidence']={}
  for hand,side in [('R','right'),('L','left')]:
   prior=next(c for c in result['criteria']if c['id']==side+'_shape')
   s,recovered,form=self.form_states(ch,hand);result['hand_form_evidence'][hand]=dict(states=s.tolist(),recovered=recovered.tolist(),form=form.tolist(),individual_fingers_inferred=False)
   if prior['expected']!=0 or prior['status']=='difference':continue
   c=shape_or_plane(side+'_shape',hand,0,s,result['events'],ts,'hand_form',context)
   c['semantic_target']='source_compatible_open_palm; precise finger extension remains separate'
   # A genuine observation conflict still has to be shown as uncertain.
   if prior.get('uncertain_spans'):
    c['status']='unknown';c['uncertain_spans']=prior['uncertain_spans'];c['uncertainty_reason']=prior['uncertainty_reason']
   result['criteria']=[c if x['id']==c['id']else x for x in result['criteria']]
  return result
