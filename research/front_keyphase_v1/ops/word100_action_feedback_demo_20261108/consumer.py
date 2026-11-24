"""Shared RTMW coordinates -> word classifier, learned evidence, and display alignment."""
from task import *
import numpy as np,torch
from model import EvidenceHeads
from evidence import assess
from runtime_candidate import Candidate,A
from prepare_feedback import normalize
from alignment import dtw
sys.path.insert(0,str(O/'word100_observation_guard_20261108'))
from guard import quarantine
sys.path.insert(0,str(O/'word100_positive_direction_20261109'))
from positive_evidence import refine
sys.path.insert(0,str(O/'word100_motion_components_20261109'))
from component_coverage import apply_coverage

class FeedbackConsumer:
 def __init__(self):
  torch.set_num_threads(4);self.base=Candidate();self.prep=read(U/'PREPARED.json');self.head=EvidenceHeads(self.prep['guide_words']);self.head.load_state_dict(torch.load(U/'runs/42/model.pt',map_location='cpu',weights_only=True));self.head.eval();self.ref=read(U/'REFERENCES.json')
 def raw(self,raw,word):
  started=time.perf_counter()
  observed=quarantine(raw['xy_px'],raw['score'],raw['timestamp_us'],int(raw['source_width']),int(raw['source_height']))
  try:a=A.convert(observed['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  except A.PilotDataError as exc:raise ValueError('몸·손의 기준점을 충분히 찾지 못했어요. 어깨와 팔이 보이는지 확인하고 다시 촬영해 주세요.') from exc
  out=self.channels(a['tcn280'],a['timestamp_us'],a['frame_id'],word)
  # A quarantined estimate is missing evidence, never a corrected hand pose.
  # Use the identical observation contract for the model and the displayed skeleton.
  out['query_xy']=np.nan_to_num(observed['xy_px'],nan=-1,posinf=-1,neginf=-1).tolist()
  out['observation_guard']=dict(version=observed['version'],hand_quarantined=observed['hand_quarantined'].tolist(),wrist_quarantined=observed['wrist_quarantined'].tolist(),reason_bits=observed['reason_bits'].tolist(),raw_preserved=True,repaired_coordinates=False)
  for card in out['cards']:
   span=card.get('query_indices');sides=[0 if s=='L' else 1 for s in card['hands']]
   if span and observed['hand_quarantined'][span[0]:span[1]+1,sides].any():
    card.update(status='unknown',display_status='uncertain',message='이 구간의 손을 일부 확인하기 어려워요. 참고와 함께 살펴보세요.',observation_quarantine_overlap=True,educational_wrong=False)
  out['width']=int(raw['source_width']);out['height']=int(raw['source_height']);out['compute_seconds']=time.perf_counter()-started
  return out
 def channels(self,x,ts,fid,word):
  if len(x)<4:raise ValueError('촬영이 너무 짧아요. 동작을 마친 다음 종료해 주세요.')
  t=time.perf_counter();wi=self.base.f['feedback']['words'].index(word);z=normalize(x,self.base.stats)
  with torch.inference_mode():
   logits=self.base.tcn(torch.from_numpy(x[None]),torch.ones(1,len(x)))[0];prob=logits.softmax(0).numpy()
   o=self.base.feedback(torch.from_numpy(z[None]),torch.tensor([len(z)]),torch.tensor([wi]));h=o['latent'];p=o['attributes'].sigmoid();valid=torch.ones((1,len(x)),dtype=torch.bool)
   ph,d0=self.head(h,p,torch.tensor([wi]),valid,torch.zeros_like(p));_,d1=self.head(h,p,torch.tensor([wi]),valid,torch.ones_like(p))
  neural=time.perf_counter()-t;t=time.perf_counter();phase=ph[0].softmax(-1).numpy();e=assess(word,phase,d0[0].sigmoid().numpy(),d1[0].sigmoid().numpy(),x,ts,fid)
  e=refine(e,p[0].numpy(),x,ts,fid,phase)
  e=apply_coverage(e)
  prototype=self.base.feedback.prototype[wi,:self.base.stats['prototype_lengths'][wi]].detach().numpy();alignment=dtw(h[0].numpy(),prototype);rank=np.argsort(-prob,kind='stable')[:3]
  # The DTW path is for playback only; it cannot turn a missing stage into evidence.
  e.update(word=word,label=read(U/'FROZEN.json')['labels'][word],tcn_top3=[dict(word=self.base.f['classes'][i],label=self.base.f['labels'][i],probability=float(prob[i])) for i in rank],path=alignment['path'].tolist(),timestamp_us=ts.tolist(),frame_id=fid.tolist(),prototype_length=len(prototype),timing=dict(neural_seconds=neural,evidence_and_alignment_seconds=time.perf_counter()-t),raw_model_seed=42,normal_or_wrong=None,overall_pass=False,correction_validated=False,reference_video_reads=0,rgb_model_calls=0,notice='연구용 비교입니다. 비슷한 부분이 보여도 동작 완수·수어 정답을 보장하지 않아요.')
  # Component-specific observations are review cues, never a whole-sign verdict.
  # refine keeps uncertainty separate from observed shape/direction contradictions.
  return e
