from pathlib import Path
import json,hashlib,collections
from component_feedback import describe_component,combine
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2,default=lambda v:v.item())
plan=dict(reason='Combined location/posture prototypes failed on normal exposed records. Preserve first results; separate shape evidence from unresolved body relation instead of declaring incompletion.',
 target={'start':{'shape':'four_open'},'goal':{'shape':'four_closed'}},
 unchanged=['weights','threshold .8','labels','source intervals','all original shape/direction/comparison outputs'],
 no_retraining=True,no_new_inference=True,no_whole_action_accuracy_claim=True,
 exposed_development_followup=True,not_new_holdout=True,
 source_sha256=sha(U/'EVALUATION.json'),core_sha256=sha(U/'core.py'),feedback_sha256=sha(U/'component_feedback.py'))
write('DECOMPOSITION_FROZEN.json',plan)
data=read(U/'EVALUATION.json');out=[]
for r in data['rows']:
 if r['requirement']!='hand_close_to_upper_body':continue
 part=describe_component({'shape':r['states']['shape']},{'shape':r['available']['shape']},r['timestamps_us'],plan['target'])
 # Region results are never converted into an educational error. Even a
 # positive shape trace cannot claim this source action's whole completion.
 remaining=dict(hand_body_relation='unqualified_region_model',automatic_action_association='not_evaluated_here')
 out.append(dict(group_id=r['group_id'],clip_id=r['clip_id'],split=r['split'],variant=r['variant'],
   source_frames=r['source_frames'],old_combined_status=r['result']['status'],
   result=combine([part],remaining),edited_public_control=True))
summary=[]
for split in ['fit','exposed_probe']:
 for variant in sorted({r['variant']for r in out}):
  rr=[r for r in out if(r['split'],r['variant'])==(split,variant)]
  summary.append(dict(split=split,variant=variant,n=len(rr),statuses=dict(collections.Counter(r['result']['parts'][0]['evidence']['status']for r in rr))))
write('DECOMPOSITION.json',dict(rows=out,summary=summary,
  claim='Separate shape-change evidence only. Not improved whole-action accuracy by dropping location.',
  independent_holdout=False,web_changed=False))
print(json.dumps([r for r in summary if r['split']=='exposed_probe'],ensure_ascii=False))
