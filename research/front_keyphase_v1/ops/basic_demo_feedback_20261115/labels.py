"""Sparse, directly inspected public RGB labels, never predicted labels."""
from prepare import *

# 0 outside reviewed interaction, 1 separated, 2 near in image projection.
# No unviewed frame is filled from interpolation; transitions remain -1.
LABELS={
 1:{0:[41,86,89],1:[47,56,68,80],2:[50,62,74]},
 2:{0:[43,85,88],1:[49,58,70,82],2:[52,64,79]},
 3:{0:[44,47,50,89,92,95],1:[56,65,68,80,83],2:[62,77]},
 4:{0:[75,78,123,126,129],1:[84,87,96,105,108,117],2:[90,99,114]},
 5:{0:[35,38,77,80,83],1:[41,50,59,62,68],2:[44,53,65]},
 6:{0:[33,36,39,75],1:[48,51,57,60,66,69],2:[45,54,63]},
 7:{0:[42,84,87],1:[48,51,57,60,69,72,78],2:[54,63]},
 8:{0:[36,39,87,90,93],1:[48,51,60,63,72,75,78,84],2:[54,66,81]},
 9:{0:[42,45,84,87],1:[51,60,63,72,75,78],2:[57,66,81]},
 11:{0:[40,43],1:[49,58,61,70,76],2:[52,64]},
 12:{0:[31,76],1:[37,40,46,49,58,61,64,70,73],2:[43,52,67]},
 13:{0:[49,52,91,94],1:[55,64,73,76,85],2:[58,67,82]}}

def main():
 rows=read(U/'INPUTS.json')['rows']; gt=[]
 for r in rows:
  cid=r['clip_id'];labels=LABELS[r['person']]; seen=set()
  for c,frames in labels.items():
   for f in frames:
    assert f not in seen and 0<=f<r['frames'];seen.add(f)
    gt.append(dict(clip_id=cid,person=r['person'],split=r['role'],frame=f,state=c,
      state_name=['outside_interaction','vertically_separated','near_in_projection'][c],
      source='direct RGB review of native frame in review sheets',source_sha256=r['source_sha256'],
      physical_contact=None,whole_word_correct=None))
 source=read(O/'word100_linguistic_feedback_pipeline_20261027/source_audit/SOURCE_ENTRY_DETAILS.json')
 def search(v):
  if isinstance(v,dict):
   if v.get('query_word_id')=='WORD1290' and v.get('entry_id')=='2372':return v
   for a in v.values():
    found=search(a)
    if found:return found
  if isinstance(v,list):
   for a in v:
    found=search(a)
    if found:return found
 guide=search(source);assert guide
 write(U/'SOURCE_TRUTH.json',dict(rows=gt,reviewer='Codex visual review; no independent signer/expert review',
  source_guide=guide,guide_video_relation='Selected gratitude sense matches right open edge approaching left hand; not inspection sense12468',
  clauses=[dict(id='handshape',text='양손을 펴고 오른손을 세운다',targets=['right_open','left_open','right_edge_orientation'],status='source supported, observer qualification pending'),
           dict(id='repeated_approach',text='오른손 옆면을 왼 손등 쪽에 반복해서 댄다',targets=['separated','near','separated','near'],status='visible projected approach can be supervised; exact physical contact not established')],
  repetition='Dictionary says twice. Reviewed source clips include three approaches. Do not enforce exactly two or declare extra cycle wrong.',
  boundaries='Only listed frames supervised, no source interval/DTW transferred frame promoted to direct truth.',
  unknown='Unlisted transitions and ambiguous proximity excluded; finger and orientation truth not inferred from class label.'))
 write(U/'TRAIN_FROZEN.json',dict(seeds=[42,43,44],steps=400,lr=.001,weight_decay=.0001,
  architectures=['frame_mlp','context_gru'],hidden=48,dropout=.1,normalization='FIT frames only',
  input='body-relative RTMW geometry, masks and frozen100 Conv+GRU64 context; no word ID, target class or normalized clip time',
  targets='Sparse RGB state CE, masked unreviewed frames. Whole-action correctness is NOT trained from word name.',
  comparison='Same labels and pretrained input: frame MLP vs local temporal conv + bidirectional GRU head, three fixed seeds, no dev-selected epoch',
  acceptance='Require held-person reliable near/separated states, preserve normal cycles, never fabricate cycle across missing observation; no web full pass until all guide requirements qualified',
  split=read(U/'PLAN.json')['split'],diagnostics=['original','speed_timestamp','prefix_after_first_approach','frozen_pose','missing_hands','uniform_scale_translation'],
  reserved='Not open until development decision frozen. Existing task-unrelated public exposure must be disclosed.'))
 print('DIRECT ANCHORS',len(gt),'FIT',sum(g['split']=='fit' for g in gt))
if __name__=='__main__':main()
