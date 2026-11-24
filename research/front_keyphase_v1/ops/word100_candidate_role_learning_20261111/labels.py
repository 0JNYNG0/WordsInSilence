from role_core import *
from collections import Counter
assert not (U/'LABELS_FROZEN.json').exists()
# Only frames actually inspected in expanded original RGB sheets. These are
# observed entry/return examples, not thresholds or word-ID branches at runtime.
extra={
 '0739':{2:([],[101]),4:([],[131,136]),6:([],[]),8:([],[]),9:([32],[86,90])},
 '1133':{2:([39,42],[80,83]),4:([66,69],[107,110,113]),6:([34,36],[65,68,70]),8:([61,64],[]),9:([],[71,73,75,77])},
 '1567':{2:([36,41],[105]),4:([53,58],[123,128]),6:([41,45],[96,100]),8:([57,61],[117,121]),9:([49,53],[107,111])},
 '2523':{2:([47],[]),4:([66,69],[108,111,115,118]),6:([52],[85,88,90,93]),8:([38,41],[78,81,84]),9:([50,53],[86,89,92])}}
inputs={r['clip_id']:r for r in read(U/'REVIEW_INPUTS.json')['rows']};ann=read(jc.ep.E/'REVIEW_LABELS.json')['rows'];out=[];count=Counter();changes=[]
for a in ann:
 if not 1<=a['person']<=9:continue
 p=P/'labels'/f"{a['clip_id']}.npz";assert p.exists();z=np.load(p);y=z['y'].copy();w=z['w'].copy();direct=[]
 # Existing zero labels do not automatically mean actual entry/return or idle.
 # They remain generic background unless explicitly re-reviewed below.
 if a['clip_id'] in inputs:
  r=inputs[a['clip_id']];assert sha(r['expanded_sheet'])==r['expanded_sheet_sha256']
  word=a['clip_id'][11:15];en,ex=extra[word][a['person']]
  for cls,frames in [(0,[0,r['source_frames']-1]),(8,en),(9,ex)]:
   for f in frames:
    assert 0<=f<len(y);changes.append(dict(clip=a['clip_id'],frame=f,old=int(y[f]),new=cls));y[f]=cls;w[f]=1;direct.append(dict(frame=f,role=cls))
  # Equal neighboring newly checked roles may supervise intermediate frames
  # weakly, never claim a fully reviewed exact boundary.
  for cls,frames in [(8,en),(9,ex)]:
   for a0,b0 in zip(frames,frames[1:]):
    if b0-a0<=6:
     for f in range(a0+1,b0):y[f]=cls;w[f]=.2
  provenance=r['expanded_sheet']
 else:provenance=a['sheet']
 count.update(int(v) for v in y[w==1]);out.append(dict(clip_id=a['clip_id'],person=a['person'],source_sha256=a['source_sha256'],y=y.tolist(),w=w.tolist(),new_anchors=direct,sheet=provenance,sheet_sha256=sha(provenance),moving_hands=['R'] if a['clip_id'][7:15]=='WORD0739' else ['R','L'],precision='Reviewed anchors and inherited reviewed boundary intervals; interpolated role weights0.2 kept distinct'))
write(U/'ROLE_LABELS.json',dict(rows=out,roles=ROLE_NAMES,reviewer='same Codex original RGB review, not expert or independent human',changes=changes,transition_class_no_new_confirmed_labels=True,completion='Role/hold labels never prove handshape correctness or semantic completion'))
write(U/'LABELS_FROZEN.json',dict(sha256=sha(U/'ROLE_LABELS.json'),role_videos=len(out),new_review_videos=20,new_reviewed_anchor_count=len(changes),direct_counts=dict(count),role_baseline=str(P/'labels'),no_new_videos_claim=True))
cfg=dict(seeds=[42,43,44],primary_seed=42,epochs=40,lr=.001,batch=128,confidence=.65,role_span_min_us=60000,candidate_role_coverage=.65,quality_confidence=.65,old_quality_preserved=False,common_encoder_frozen=True,shape_direction_frozen=True,role_classes=ROLE_NAMES,training='Existing 36 public role clips +30 boundary clips union; full/upper views and core-time-reversal identity augmentation; no word-ID or desired-direction inputs.',evaluation='Historical78 regression conditions, reused direct public annotations; no fresh independent result claimed. No evaluation-based epochs/threshold/seed choice.',adoption='All3seeds:normal12+speed24 no extra warning,old9 errors retained,no unknown promoted normal,reverse9 more than5 valid warnings,false similar1 reduced,actual missing versus mask improved; otherwise no web adoption.',pins={str(p):sha(p) for p in [U/'ROLE_LABELS.json',A/'FIT_LABELS.json',P/'ACTION_EPISODES.json',A/'guide_warning_gate/VERIFIED_FULLPATH.json',J/'runs/42/model.pt',ac.V/'runs/42/model.pt']})
write(U/'FROZEN.json',cfg);event('labels_frozen',videos=len(out),new_anchors=len(changes),counts=dict(count))
