"""Create an isolated, single-intervention experiment; never edits a prior unit."""
from pathlib import Path
import json, shutil, hashlib, datetime
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
FACT=O/'word100_assignment_scale_factorial_20261019'
BASE=O/'word100_body_boundary_20261017'
def put(name,text):
 p=U/name; assert not p.exists(),str(p); p.write_text(text,encoding='utf-8')
def js(name,data):put(name,json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def replace_once(s,old,new):
 assert s.count(old)==1,old[:100];return s.replace(old,new,1)

for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
 (U/'before').mkdir(exist_ok=True); assert not (U/'before'/p.name).exists();shutil.copyfile(p,U/'before'/p.name)
common=(FACT/'common.py').read_text(encoding='utf-8')
common=common.replace('Local finite hand-scale factorial experiment.','Balanced training-batch experiment.')
common=common.replace("ARMS=('R10','R01')","ARMS=('BAL31',)")
put('common.py',common)
put('balanced.py','''"""Divide an ordered sequence into ceil(N/max_size) near-equal batches."""
def balanced_batches(order, max_size):
 if not isinstance(max_size,int) or isinstance(max_size,bool) or max_size<1:
  raise ValueError('max_size must be a positive integer')
 n=len(order)
 if n==0:return []
 count=(n+max_size-1)//max_size
 small,extra=divmod(n,count);out=[];start=0
 for i in range(count):
  size=small+(i<extra);out.append(order[start:start+size]);start+=size
 assert start==n
 return out
''')
s=(FACT/'train.py').read_text(encoding='utf-8')
s=replace_once(s,'Finite-scale factorial paired runs. Saved C/Q remain read-only.','Balanced minibatch paired runs. Original C remains read-only.')
s=replace_once(s,"ARMS=('R10','R01');VIEWS=('masked','sensor')","from balanced import balanced_batches\nARMS=('BAL31',);VIEWS=('masked','sensor')")
start=s.index('def loadrows(');end=s.index('\ndef scores(',start)
s=s[:start]+'''def loadrows(context):
 original=read(BASE/'training_CONTEXT.json');assert original['ordered_rows']==context['rows']
 rows=[]
 for r in context['rows']:
  q=copy.deepcopy(r);q['arrays']={'BAL31':{}}
  for v in VIEWS:
   info=r['views'][v];path=info['channel_file']
   if path is None: q['arrays']['BAL31'][v]=None;assert r['role']!='train_candidate'
   else:
    assert sha(path)==info['sha256'];z=load(path)
    keys=('x_whole','x_context') if r['role']=='train_candidate' else ('x_whole',)
    q['arrays']['BAL31'][v]={k:z[k] for k in keys}
  rows.append(q)
 return [r for r in rows if r['role']=='train_candidate'],[r for r in rows if r['role']=='development_only']
''' +s[end:]
s=replace_once(s,'for step,j in enumerate(range(0,899,32),1):\n   rr=[train[i] for i in order[j:j+32]];',
 'for step,indices in enumerate(balanced_batches(order,TRAIN[\'batch_size\']),1):\n   rr=[train[i] for i in indices];')
s=replace_once(s,"'masked_top1':metrics['masked']['top1_correct'],'trace_sha256'","'masked_top1':metrics['masked']['top1_correct'],'sensor_ce':metrics['sensor']['mean_ce'],'masked_ce':metrics['masked']['mean_ce'],'trace_sha256'")
s=replace_once(s,"'feature_schema':'assignment_scale_factorial_280'","'feature_schema':'original_C_280_balanced_batches'")
s=replace_once(s,"'train.py','common.py','FROZEN_PROTOCOL.json'","'train.py','common.py','balanced.py','FROZEN_PROTOCOL.json'")
s=replace_once(s,"runs=6)","runs=3)")
put('train.py',s)
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
js('AUTHORIZATION.json',{'at':at,'operator':'Codex local','user_request':'899개를 31개씩 29묶음으로 균등하게 나누는 조건부터 검증하고, 문제없으면 3회 학습해서 기존 C와 비교. 다른 조건과 기존 C 유지. 사용자 촬영 학습/튜닝 사용 금지. 인식률과 학습 후반 흔들림 비교.','allowed':['local input and partition verification','three paired fresh training runs','saved scores and traces independent arithmetic','reports and root status synchronization'],'excluded':['user/mentor recordings','production replacement','new extraction','additional experimental arms','external transfer','Git']})
js('FROZEN_PROTOCOL.json',{'at':at,'arm':'BAL31','intervention':'Only training minibatch boundaries: ceil(N/max_batch) near-equal contiguous groups of the same shuffled order; 899 -> 29 x 31. Evaluation remains batch32.','seeds':[20261013,20261014,20261015],'baseline':'Original C stored selected scores and epoch histories; no retraining or reinference.','fixed':['original C input bytes and order','899 TRAIN / 300 DEV / held-out excluded','original C model and fresh initial states','view and temporal schedule','AdamW lr0.001 wd0.001; clip1; max80; patience12','mean masked/sensor DEV CE minimum, strict delta1e-6, earliest ties','29 optimizer steps per epoch; each source once'],'late_window':{'first_epoch':61,'last_epoch':80,'metrics':['sensor/masked top1 adjacent absolute delta mean and maximum','accuracy population SD and range','accuracy mean','sensor/masked/selection CE mean and adjacent change'],'incomplete_window':'Report actual coverage; do not fill, extend, or change stopping.'},'primary':'paired selected sensor DEV accuracy mean/range and delta percentage points','secondary':['masked accuracy and top5','per signer and frozen missingness group','paired gains/losses','late stability separate from selected accuracy'],'review_gate':'mean sensor higher, better in >=2 seeds, masked mean and top5 maintained, no mean per-signer/group regression in either view; engineering gate only, no automatic deployment','limits':['Batch grouping changes gradient composition, padding and stochastic dropout draws as well as BN statistics; not BN-only causal identification.','Repeatedly used public DEV is not blind test or real-webcam accuracy.','Three seeds measure initialization variation, not three new users.'],'training_runs_max':3,'user_data_used':False,'production_replacement':False})
print(str(U))
