from pathlib import Path
import json,hashlib
from collections import Counter
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n','utf-8')
def main():
 direct=read(U/'DIRECT_SCORES.json');issues=[];paired=[]
 for r in direct:
  if r['mode']!='position':continue
  q=r['previous'];b=q['previous'];p=r['after'];t=b['target']
  if (t==2)!=p['physical_event_complete']:issues.append(dict(clip_id=b['clip_id'],seed=b['seed'],event=b['event'],kind=b['kind'],target=t,new_complete=p['physical_event_complete'],previous_complete=q['after']['complete_joint'],probability=p['quality_probability'],relative_start=p['body_relative_position'][4:8],relative_end=p['body_relative_position'][8:12]))
  if b['kind'] in ['whole','middle','suffix'] and b['event']['axis']==2:paired.append(dict(clip_id=b['clip_id'],seed=b['seed'],kind=b['kind'],event=b['event'],target=t,new_complete=p['physical_event_complete'],relative_start=p['body_relative_position'][4:8],relative_end=p['body_relative_position'][8:12]))
 counts=Counter();changed=[]
 for r in read(U/'FULLPATH_COMPLETE.json')['rows']:
  if r['condition']!='position_and_binding':continue
  o=read(r['result']);old=read(o['previous']);original=read(old['previous']);counts['candidates']+=len(o['ledger'])
  for a,b in zip(o['ledger'],original['ledger']):
   if a['family']!=b['family']:counts['changed_family']+=1;changed.append(dict(clip=r['clip_id'],kind=r['kind'],seed=r['seed'],event=a['event'],span=[a['a'],a['b']],before=b['family'],after=a['family']))
  counts['feedback_conditions']+=1;counts['same_feature_verdict']+=r['before']['display']==r['after']['display']
 labels=read(O/'word100_completion_boundary_targets_20261111/LABELS.json')['rows'];fit=[a for a in labels if a['role']=='fit'];coverage={}
 for a in fit:coverage.setdefault(f"{a['axis']}/{a['moving']}",set()).add(a['clip_id'])
 report=dict(issues=issues,along_arm_whole_partial_same_input_evidence=paired,assignment_changes=changed,counts=dict(counts),fit_axis_videos={k:len(v) for k,v in coverage.items()},confirmed=['Normalized body-relative start/end values reach the new model; scale/translation and missing-mask tests passed.','Same physical interval classifier still confuses some middle/suffix spans with whole spans in all seeds.','Sequence residual preserves64/67 common-role anchors after correcting legacy class mapping; it has not improved final verdicts on the78conditions.','The physical completion model has no guide requirement input. Guide roles are attached afterward. Current labeling does not directly supervise how much of a specific guide action was performed.'],not_proven=['Which new label or architecture will generalize to natural learner omissions.','That all video failures originate from position information loss.','That two models agreeing is independent correctness evidence.'],scope='Post-score diagnosis only. No labels, weights, thresholds or inference changed.')
 write(U/'ERROR_AUDIT.json',report);print(json.dumps(dict(counts=dict(counts),issue_count=len(issues),fit_axis_videos=report['fit_axis_videos']),ensure_ascii=False))
if __name__=='__main__':main()
