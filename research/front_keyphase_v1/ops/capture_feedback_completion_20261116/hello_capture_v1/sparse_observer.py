"""Reuse the trained prefix context only when an actual movement needs checking."""
from batch_observer import *
class SparsePrefixAction(ActionObserver):
 def prepare(self,raw):
  self.reset();self.raw=raw;self.cached_prefix=None
  self.channels=self.adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));self.x=self.channels['tcn280'].astype(np.float32)
 def movement(self,start_us,end_us,axis,moving='R',anchor='L',min_samples=None):
  if start_us is None:return dict(eligible=False,state=-1,reason='no_observed_start')
  raw=self.raw;end=int(np.searchsorted(raw['timestamp_us'],end_us));hi=min(len(raw['frame_id']),end+4);lo=max(0,hi-self.max_frames)
  if self.cached_prefix!=(lo,hi):
   self.channels=self.adapter.convert(raw['xy_px'][lo:hi],raw['score'][lo:hi],raw['timestamp_us'][lo:hi],raw['frame_id'][lo:hi],int(raw['source_width']),int(raw['source_height']));self.x=self.channels['tcn280'].astype(np.float32);pose=self.rc.prior.shared.normalize(self.x)
   with torch.inference_mode():
    h=torch.nn.functional.gelu(self.base.input(torch.from_numpy(pose[None]))).transpose(1,2)
    for layer in self.base.temporal:h=layer(h)
    self.h,_=self.base.context(h.transpose(1,2))
   self.calls+=1;self.cached_prefix=(lo,hi)
  self.assessed_timestamp_us=int(end_us)
  return super().movement(start_us,end_us,axis,moving,anchor,min_samples)
class SparseLesson(BatchLesson):
 def __init__(self):super().__init__(SparsePrefixAction())
