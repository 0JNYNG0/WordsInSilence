from common import *
from basic_runtime import RelationObserver
class FallbackRelation:
 def __init__(self):self.prior=RelationObserver();self.geometry=GeometryRelation()
 def __call__(self,f,quality,ts):
  a,p=self.prior(f,quality,ts);b,_=self.geometry(f,quality,ts)
  # Preserve already observed separated/near states, including short reversals.
  # New body evidence can recover a suppressed interaction, never erase one.
  return np.where((a<=0)&(b>0),b,a),p
class FallbackLesson(QualifiedFormLesson):
 def __init__(self):super().__init__();self.relation=FallbackRelation()
