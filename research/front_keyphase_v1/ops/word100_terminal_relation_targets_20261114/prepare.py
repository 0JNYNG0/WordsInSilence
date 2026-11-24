from pathlib import Path
import json,hashlib,collections,shutil
import numpy as np
from core import features,CLASSES
U=Path(__file__).resolve().parent;O=U.parent;S=O/'word100_goal_state_learning_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)

# These are manual RGB contact-sheet observations made before fitting. Slots
# index reviewed images only, and are NEVER model inputs or inference rules.
# Ambiguous borders/oblique arms/thumbs/pinch are deliberately not forced.
catalog={}
for record in range(2,10):
 catalog[record]={
  'WORD0739':{
   'shape_R':['four_open','four_closed','four_closed','four_closed','four_closed','four_closed'],
   'region_R':['face_front',None,'upper_body','upper_body',None,None],
   'forearm_R':[None,None,'across','across',None,'upright']},
  'WORD1133':{'shape_R':['four_open']*3,'shape_L':['four_closed']*3,
    'region_R':['upper_body']*3,'region_L':['upper_body']*3},
  'WORD1567':{'shape_R':['four_closed',None,'four_closed'],'shape_L':['four_closed',None,'four_closed']},
  'WORD2523':{'shape_R':[None]*3,'shape_L':[None]*3,
    'region_R':['face_front',None,'upper_body'],'region_L':['face_front',None,'upper_body']}}
# Record08 visibly closes beside the lower face/shoulder without substantial
# palm lowering. Broad shoulder/upper-body observation; not exact chest contact.
catalog[8]['WORD0739']['region_R'][1]='upper_body'
catalog[6]['WORD0739']['region_R'][1]='upper_body'
catalog[9]['WORD0739']['region_R'][1]='upper_body'
src=read(S/'REVIEW_INPUTS.json')['rows'];arr=np.load(S/'source_arrays.npz')
group=collections.defaultdict(list)
for r in src:group[(r['person'],r['word_id'])].append(r)
rows=[];seen={}
for (record,word),rr in sorted(group.items()):
 rr=sorted(rr,key=lambda r:(r['family'],r['review_slot']))
 for key,labels in catalog[record][word].items():
  domain,hand=key.split('_');assert len(labels)==len(rr)
  for r,label in zip(rr,labels):
   unique=(r['clip_id'],r['frame'],domain,hand)
   if unique in seen:
    assert seen[unique]==label;continue
   seen[unique]=label;fid=arr[r['clip_id']+'_frame_id'];ii=np.flatnonzero(fid==r['frame']);assert len(ii)==1
   f,ok=features(arr[r['clip_id']+'_x'][ii],domain,hand)
   rows.append(dict(index=len(rows),clip_id=r['clip_id'],record_id=record,word=word,
       split='fit'if record<=7 else'exposed_probe',frame=r['frame'],domain=domain,hand=hand,
       label=label,target=CLASSES[domain].index(label)if label else -1,
       available=bool(ok[0]),feature=f[0].tolist(),array_index=int(ii[0]),
       source=r['source'],source_sha256=r['source_sha256'],
       review_sheet=str(S/f'review_record{record:02d}.jpg'),reviewer='Codex RGB review; not independent KSL expert',
       whole_action_correctness=None))
write('LABELS.json',dict(rows=rows,definitions={
 'shape':'Only four non-thumb fingers visibly extended vs folded; not exact fist/thumb/pinch.',
 'region':'Clearly central face-front vs broad upper torso/shoulder. Border states unknown; no exact contact.',
 'forearm':'Clearly across-body vs upright configurations. Oblique poses unlabelled, not wrong.'},
 catalog=catalog,source_pins={str(S/n):sha(S/n)for n in ['REVIEW_INPUTS.json','source_arrays.npz']+[f'review_record{i:02d}.jpg'for i in range(2,10)]},
 no_probe_score_used_for_labels=True,prior_exposed_probe=True,private_data=False))
audit=[]
for d in CLASSES:
 rr=[r for r in rows if r['domain']==d and r['split']=='fit'and r['available']and r['target']>=0]
 for c in CLASSES[d]:
  sub=[r for r in rr if r['label']==c];records=sorted({r['record_id']for r in sub})
  audit.append(dict(domain=d,label=c,n=len(sub),records=records,words=sorted({r['word']for r in sub})))
  assert len(records)>=4,(d,c,records)
write('LABEL_VALIDATION.json',dict(passed=True,fit_coverage=audit,
 unique_frames=len({(r['clip_id'],r['frame'])for r in rows}),rows=len(rows),
 known=sum(r['target']>=0 for r in rows),unknown=sum(r['target']<0 for r in rows),
 warning='Forearm prototypes still from one expression, six FIT records. Cross-word generalization not established.',
 annotator='Same Codex separate data audit; not human-expert agreement'))
requirements={
 'hand_close_to_upper_body':{'start':{'shape':'four_open','region':'face_front'},'goal':{'shape':'four_closed','region':'upper_body'}},
 'raise_closed_forearm':{'start':{'shape':'four_closed','forearm':'across'},'goal':{'shape':'four_closed','forearm':'upright'}}}
write('FROZEN.json',dict(purpose='Visible terminal traits and state sequence, not word correctness or new action segmentation',
 fit_records=list(range(2,8)),exposed_probe_records=[8,9],seeds=[42,43,44],steps=800,lr=.01,weight_decay=.001,
 model='Three independent small MLP heads, each Linear ->16 Tanh ->2; existing common100 Conv+GRU unchanged',
 confidence=.8,all_three_seeds=True,no_checkpoint_selection=True,requirements=requirements,
 cases=['normal','short_goal_prefix','speed075','speed125','held_start','goal_only','reverse','missing_tail','missing_all','middle_gap2','middle_long_gap'],
 expected_scope='Edited public observations only; neither true learner errors nor educational tolerance ground truth',
 training_pins={str(U/n):sha(U/n)for n in ['core.py','LABELS.json','LABEL_VALIDATION.json']},
 admission='All known probe states supported, normal/short/speed preserved, no false feature-sequence support on static/reverse/missing controls; otherwise diagnostic only',
 protected_manifest=str(O/'word100_web_completion_graph_20261111/PROTECTIONS.json'),
 no_final8=True,no_web_change=True,no_new_download=True))
write('CURRENT_EXECUTION.json',dict(status='LABELS_VALIDATED_READY_TO_TRAIN',training_running=False,evaluation_running=False))
print(json.dumps(dict(rows=len(rows),audit=audit),ensure_ascii=False))
