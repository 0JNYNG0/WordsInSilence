"""Keep visible shape contradictions inside an observed interaction envelope.

A shape change can make the learned relation class temporarily become zero.
That cannot discard an observed contradiction between two interaction anchors.
The envelope only extends error inspection, never creates events or support.
It stops at the last observed interaction and never fills missing hand joints.
"""
from form_v2 import FinalFormLesson
from form_candidate import np,adapter,contiguous,shape_or_plane
import time
class GuardFormLesson(FinalFormLesson):
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();q=super().analyze(raw,lesson)
  anchors=np.flatnonzero(np.array(q['relation_states'])>0)
  if len(anchors)<2:return q
  a,b=int(anchors[0]),int(anchors[-1]);q['shape_inspection_envelope']=[a,b]
  for hand,side in [('R','right'),('L','left')]:
   prior=next(c for c in q['criteria']if c['id']==side+'_shape')
   if prior['expected']!=0 or prior['status']=='difference':continue
   available=self.shape.observations[hand]['available'].all(1)
   ids=np.flatnonzero(available[a:b+1])+a
   context=[dict(start=int(r[0]),end=int(r[-1]))for r in contiguous(ids)]
   states=np.array(q['hand_form_evidence'][hand]['states'])
   probe=shape_or_plane(side+'_shape',hand,prior['expected'],states,[],np.array(q['timestamp_us']),'hand_form',context)
   if probe['status']=='difference':
    updated=dict(prior,status='difference',difference_spans=probe['difference_spans'],inspection='Visible contradiction between interaction anchors; no new completed event')
    q['criteria']=[updated if c['id']==prior['id']else c for c in q['criteria']]
  q['seconds']=time.perf_counter()-started
  return q
