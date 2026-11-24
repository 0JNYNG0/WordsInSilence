from pathlib import Path
import json,sys,collections,time
import numpy as np,torch
from lesson_contract import assess
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
sys.path.insert(0,str(O/'word100_rtmw_joint_learning_20261108'))
from runtime_candidate import Candidate
torch.set_num_threads(2);model=Candidate();rows=[]
manifest=read(O/'word100_clause_finger_diagnostics_20261114/INTEGRATED_RESULTS.json')['rows']
previous=read(O/'word100_finger_repair_20261115/REPLAY_RESULTS.json')['rows']
sources={r['clip_id']:r for r in read(O/'word100_public_g4_20261114/EXTRACTED.json')['rows']}
labels=read(O/'word100_public_g4_20261114/LABELS_FROZEN.json')['actions']
guides=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
for r,pr in zip(manifest,previous):
 assert(r['clip_id'],r['kind'])==(pr['clip_id'],pr['kind'])
 if r['dataset']=='development82':x=np.load(r['source'])['x'].copy()
 else:
  ch=np.load(sources[r['clip_id']]['channel_path']);n=len(ch['tcn280']);ids=np.arange(n)
  if r['kind']=='prefix_before_goal':n=next(a['partial_end']for a in labels if a['clip_id']==r['clip_id']and a['card']=='close_chest')+1;ids=np.arange(n)
  if r['kind']=='continuous_full_reverse':ids=ids[::-1].copy()
  x=ch['tcn280'][ids].copy();ts=np.round(ch['timestamp_us'][:n]/{'speed075':.75,'speed125':1.25}.get(r['kind'],1)).astype(np.int64);x[:,279]=np.r_[0,np.diff(ts)/1e6]
  if r['kind']=='all_hands_missing':x[:,18:186]=0;x[:,195:279]=0
 word=r['clip_id'].split('_')[2];st=time.perf_counter()
 with torch.inference_mode():p=model.tcn(torch.tensor(x[None]),torch.ones(1,len(x)))[0].softmax(0).numpy()
 rank=np.argsort(-p,kind='stable')[:3];top=[dict(word=model.f['classes'][int(i)],probability=float(p[i]))for i in rank]
 z=read(pr['output'])['after'];outcome=assess(z,[g['id']for g in guides[word]if g['required']]);rows.append(dict(clip_id=r['clip_id'],kind=r['kind'],normal=r['normal'],dataset=r['dataset'],tcn_top3=top,recognition_matches=top[0]['word']==word,lesson=outcome,source_result=pr['output'],tcn_seconds=time.perf_counter()-st))
summary=dict(cases=len(rows),normal=sum(r['normal']for r in rows),recognition_matches=sum(r['recognition_matches']for r in rows),recognition_match_with_correction=sum(r['recognition_matches']and r['lesson']['status']=='correct'for r in rows),recognition_match_without_complete_evidence=sum(r['recognition_matches']and not r['lesson']['can_advance']for r in rows),outcomes=dict(collections.Counter(r['lesson']['status']for r in rows)),normal_feature_match=sum(r['normal']and r['lesson']['features_match']for r in rows),control_feature_match=[dict(clip_id=r['clip_id'],kind=r['kind'],recognition_matches=r['recognition_matches'])for r in rows if not r['normal']and r['lesson']['features_match']],no_completion_witness='Current saved output has no qualified whole-action completion contract. Do not turn all unknowns into pass.')
write(U/'PUBLIC_ACCEPTANCE_AUDIT.json',dict(summary=summary,rows=rows,scope='Current unchanged TCN inference plus archived latest feedback on exposed122public conditions. Not physical webcam, not fresh model accuracy. Non-normal means a test condition, not automatically proven learner error. No model selection/training.'))
print(summary)
