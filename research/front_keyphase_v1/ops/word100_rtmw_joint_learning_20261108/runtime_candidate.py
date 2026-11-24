"""Research-only coordinate consumer. No source-video/reference-image access."""
from common import *
import numpy as np,torch,importlib.util
from pilot.model_masked_v1 import MaskedSignTCN
from feedback_model import PoseGuideNet
from prepare_feedback import normalize
from feedback_controls import gate
sys.path.insert(0,str(O/'word100_learned_guide_feedback_20261102'))
from alignment import dtw,representative,stage_at
spec=importlib.util.spec_from_file_location('_rtmw_candidate_shared',O/'word100_rtmw_shared_input_20261108/adapter.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)

def insufficient(ts,fid,reason,hand_frames=0):
 return dict(status='INSUFFICIENT_COORDINATE_INPUT',tcn_top3=[],feedback_status='INSUFFICIENT_OBSERVATION',normal_or_wrong=None,
  timing={},hand_estimate_frames=hand_frames,reason=reason,word_classifier_uses_requested_target=False,educational_correctness_validated=False,
  frame_id=np.asarray(fid).tolist(),timestamp_us=np.asarray(ts).tolist(),reference_video_reads=0,rgb_forward_calls=0)

class Candidate:
 def __init__(self,tcn_condition='finetune_C',feedback_condition='pose_transfer'):
  self.feedback_root=U/'feedback_viewfix'
  self.f=read(U/'FROZEN.json');self.stats=read(self.feedback_root/'FEEDBACK_NORMALIZATION.json');self.guides=read(O/'word100_learned_guide_feedback_20261102/GUIDE_SUPPORT.json')['guides']
  self.tcn_condition=tcn_condition;self.feedback_condition=feedback_condition
  p=U/'tcn_runs'/f'{tcn_condition}_20261013';assert sha(p/'model.pt')==read(p/'COMPLETE.json')['checkpoint_sha256'];ck=torch.load(p/'model.pt',map_location='cpu',weights_only=True)
  self.tcn=MaskedSignTCN(**ck['model_config']);self.tcn.load_state_dict(ck['state_dict']);self.tcn.eval()
  p=self.feedback_root/'feedback_runs'/f'{feedback_condition}_42';assert sha(p/'model.pt')==read(p/'COMPLETE.json')['checkpoint_sha256']
  self.feedback=PoseGuideNet(self.stats['guide_vectors'],self.stats['guide_words'],self.stats['prototype_lengths']);self.feedback.load_state_dict(torch.load(p/'model.pt',map_location='cpu',weights_only=True));self.feedback.eval()
 def from_raw(self,raw,target_word):
  t=time.perf_counter()
  try:a=A.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  except A.PilotDataError as exc:
   return insufficient(raw['timestamp_us'],raw['frame_id'],str(exc))
  prep=time.perf_counter()-t
  out=self.from_channels(a['tcn280'],a['timestamp_us'],a['frame_id'],target_word);out['timing']['shared_input_seconds']=prep;return out
 def from_channels(self,x,ts,fid,target_word):
  hand_frames=int((x[:,237:279].sum(1)>0).sum())
  if len(x)<4 or hand_frames<4:
   return insufficient(ts,fid,'Fewer than four frames with hand coordinate estimates',hand_frames)
  # Requested target is intentionally absent from word-classifier call.
  timing={};t=time.perf_counter()
  with torch.inference_mode():
   logit=self.tcn(torch.from_numpy(x[None]),torch.ones(1,len(x)))[0];prob=logit.softmax(0).numpy()
  timing['tcn_seconds']=time.perf_counter()-t;rank=np.argsort(-prob,kind='stable')[:3]
  result=dict(tcn_top3=[dict(word=self.f['classes'][i],label=self.f['labels'][i],probability=float(prob[i])) for i in rank],timing=timing,word_classifier_uses_requested_target=False,
              status='RESEARCH_CANDIDATE',hand_estimate_frames=hand_frames,feedback_input_revision='full_upper_fit_viewfix',educational_correctness_validated=False,
              normal_or_wrong=None,reference_video_reads=0,rgb_forward_calls=0)
  if target_word not in self.f['feedback']['words']:
   result['feedback_status']='UNSUPPORTED_WORD';return result
  wi=self.f['feedback']['words'].index(target_word);t=time.perf_counter();z=normalize(x,self.stats)
  with torch.inference_mode():o=self.feedback(torch.from_numpy(z[None]),torch.tensor([len(x)]),torch.tensor([wi]))
  timing['feedback_seconds']=time.perf_counter()-t;t=time.perf_counter()
  latent=o['latent'][0].numpy();prototype=self.feedback.prototype[wi,:self.stats['prototype_lengths'][wi]].detach().numpy();aligned=dtw(latent,prototype)
  timing['dtw_seconds']=time.perf_counter()-t;states=gate(x,ts);sp=o['stage'][0].softmax(-1).numpy();ap=o['attributes'][0].sigmoid().numpy()
  result.update(feedback_status='MODEL_OBSERVATIONS_ONLY_NOT_VALIDATED_CORRECTIONS',frame_id=fid.tolist(),timestamp_us=ts.tolist(),attributes=ap.tolist(),
   direct_stage=[self.stats['guide_ids'][i] for i in sp.argmax(-1)],dtw_stage=[stage_at(representative(aligned['path'],i),self.guides,target_word) for i in range(len(x))],
   path=aligned['path'].tolist(),path_axes=['query_frame_index','learned_pattern_index'],hand_evidence=states.tolist(),
   normal_or_wrong=None,mean_alignment_cost=aligned['mean_cost'],reference_video_reads=0,rgb_forward_calls=0)
  return result
