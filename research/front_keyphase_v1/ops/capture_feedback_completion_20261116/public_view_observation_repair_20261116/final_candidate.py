"""Adopt only the view-trained posture head with demonstrated benefit.

Keep the deployed palm/finger and movement heads unchanged. The separately
trained palm/relation experiments did not improve their scoped public results.
"""
from view_candidate import ViewLesson,ViewPlane,PalmMotionLesson

class PlaneViewLesson(ViewLesson):
 def __init__(self):
  PalmMotionLesson.__init__(self)
  self.view_plane=ViewPlane()
 def form_states(self,ch,hand):
  return PalmMotionLesson.form_states(self,ch,hand)
