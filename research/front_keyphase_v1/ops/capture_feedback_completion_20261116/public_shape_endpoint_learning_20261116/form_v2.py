"""Separate coarse palm form from individual flexion, only in validated view.

The projection head was trained for upright hand views. It must not provide
positive recovery in an unvalidated horizontal/foreshortened projection.
This does not override any known closed/patterned state or infer hidden joints.
"""
from form_candidate import FormLesson
class FinalFormLesson(FormLesson):
 def form_states(self,ch,hand):
  prior=self.shape.observations[hand]['states'];s,recovered,form=super().form_states(ch,hand)
  recovered &= self.plane.sequence(ch,hand)==0
  s=prior.copy();s[recovered]=0
  return s,recovered,form
