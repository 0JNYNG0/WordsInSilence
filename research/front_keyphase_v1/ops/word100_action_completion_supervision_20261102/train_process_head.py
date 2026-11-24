"""One fixed additive supervised candidate; requires frozen evaluation first."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import json,hashlib,time,traceback,os,random
import numpy as np
import torch
from torch.nn import functional as F
from process_head import ProcessEvidenceNet,QUERIES,make_input,batch

U=Path(__file__).resolve().parent
OUT=Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102/process_run/seed42')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def now():return datetime.now(timezone.utc).isoformat()

def main():
 assert not OUT.exists(), 'Completed or interrupted candidate must not silently restart'
 contract=read(U/'PROCESS_COMPARISON_FROZEN.json')
 assert contract['status']=='READY_FOR_ONE_FIXED_TRAINING_RUN'
 assert contract['seed']==42 and contract['epochs']==80 and contract['candidates']==1
 for p,h in contract['pins'].items():assert sha(p)==h,p
 targets=read(U/'PROCESS_VIEW_TARGETS.json')['rows']
 inputs={r['pixel_view_key']:r for r in read(U/'PROCESS_INPUT_MANIFEST.json')['rows']}
 cache={}
 for k,r in inputs.items():
  assert r['role']=='fit' and r['person'] in range(1,10) and sha(r['input'])==r['input_sha256']
  with np.load(r['input'],allow_pickle=False) as z:cache[k]={a:z[a].copy() for a in z.files}
 assert len(cache)==216 and len(targets)==217
 # Unique materialized FIT views only; no labels, target-word, DEV or private data
 # enter normalization. Blank pixels are excluded from RGB statistics.
 valid=np.concatenate([a['rgb'][a['image_available']] for a in cache.values()],axis=0)
 stats=dict(mean=valid.mean(0,dtype=np.float64).tolist(),std=np.maximum(valid.std(0,dtype=np.float64),.001).tolist(),
  source='FIT216 unique actual pixel/time views; blank rows excluded; no DEV',frames=len(valid))
 seq={k:make_input(a,stats) for k,a in cache.items()}
 groups=defaultdict(list)
 for r in targets:
  assert r['role']=='fit' and r['recorded_person_id'] in range(1,10)
  assert r['physical_query'] in QUERIES and r['pixel_view_key'] in cache
  source=inputs[r['pixel_view_key']]
  assert source['clip_id']==r['clip_id'] and source['person']==r['recorded_person_id']
  assert source['source_sha256']==r['source_sha256'] and r['view_id'] in source['aliases']
  a=cache[r['pixel_view_key']];fm=r['ordered_frame_map']
  assert len(a['frame_id'])==len(fm)
  assert a['frame_id'].tolist()==[f['output_index'] for f in fm]==list(range(len(fm)))
  assert source['source_frame_indices']==[f['source_native_frame'] for f in fm]
  assert source['source_timestamps_us']==[f['source_native_timestamp_us'] for f in fm]
  assert a['timestamp_us'].tolist()==[f['view_timestamp_us'] for f in fm]
  assert a['image_available'].tolist()==[f['image_available'] for f in fm]
  for v in r['sparse_relation_targets']:
   f=fm[v['output_index']]
   assert f['source_native_frame']==v['source_frame'] and f['image_available']
  for v in r['fully_retained_motion_interval_targets']:
   indices=v['output_indices'];native=v['source_frames']
   assert len(indices)>=2 and indices==list(range(indices[0],indices[-1]+1))
   assert native==[fm[j]['source_native_frame'] for j in indices]
   assert native==list(range(native[0],native[-1]+1))
   assert all(fm[j]['image_available'] for j in indices)
  groups[(r['recorded_person_id'],r['physical_query'])].append(r)
 assert len(groups)==45
 # Repeated sparse point/interval labels across edited views receive inverse
 # multiplicity within their source/query, so copies don't manufacture subjects.
 multiplicities={}
 for key,rs in groups.items():
  points=defaultdict(int);intervals=defaultdict(int)
  for r in rs:
   for v in r['sparse_relation_targets']:
    for dim,field in enumerate(['setup_relation','terminal_relation']):
     if v[field] is not None:points[(v['source_frame'],dim)]+=1
   for v in r['fully_retained_motion_interval_targets']:
    if v['label'] is not None:intervals[tuple(v['source_frames'])]+=1
  multiplicities[key]=(points,intervals)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
 torch.use_deterministic_algorithms(True)
 model=ProcessEvidenceNet();optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
 OUT.mkdir(parents=True,exist_ok=False)
 write(OUT/'STARTED.json',dict(at=now(),pid=os.getpid(),contract_sha256=sha(U/'PROCESS_COMPARISON_FROZEN.json'),seed=42,epochs=80))
 write(OUT/'NORMALIZATION.json',stats)
 keys=sorted(groups);history=[];started=time.perf_counter()
 for epoch in range(80):
  random.shuffle(keys);model.train();epoch_losses=[]
  for offset in range(0,len(keys),12):
   selected=keys[offset:offset+12]
   records=[(key,r) for key in selected for r in groups[key]]
   x,lengths=batch([seq[r['pixel_view_key']] for _,r in records])
   qi=torch.tensor([QUERIES.index(r['physical_query']) for _,r in records],dtype=torch.long)
   pred=model(x,lengths,qi);by_group=defaultdict(list)
   interval_sequences=[];interval_queries=[];interval_meta=[]
   for i,(key,r) in enumerate(records):
    by_group[key].append(i)
    for v in r['fully_retained_motion_interval_targets']:
     if v['label'] is None:continue
     indices=v['output_indices'];assert len(indices)>=2
     assert all(r['ordered_frame_map'][j]['image_available'] for j in indices)
     interval_sequences.append(seq[r['pixel_view_key']][indices]);interval_queries.append(QUERIES.index(r['physical_query']))
     interval_meta.append((key,float(v['label']),tuple(v['source_frames'])))
   if interval_sequences:
    ix,il=batch(interval_sequences);ip=model.interval_logits(ix,il,torch.tensor(interval_queries,dtype=torch.long))
   group_losses=[]
   for key,positions in by_group.items():
    zero=pred['process_logits'][positions].sum()*0
    class_losses=[]
    for label in [0,1]:
     ix=[i for i in positions if records[i][1]['observed_process_evidence']==label]
     if ix:class_losses.append(F.binary_cross_entropy_with_logits(pred['process_logits'][ix],torch.full((len(ix),),float(label))))
    view_loss=torch.stack(class_losses).mean() if class_losses else zero
    point_weights,interval_weights=multiplicities[key]
    point_losses=[];point_total=0.
    for i in positions:
     for v in records[i][1]['sparse_relation_targets']:
      for dim,field in enumerate(['setup_relation','terminal_relation']):
       if v[field] is None:continue
       weight=1./point_weights[(v['source_frame'],dim)]
       point_losses.append(F.binary_cross_entropy_with_logits(pred['point_logits'][i,v['output_index'],dim],torch.tensor(float(v[field])))*weight)
       point_total+=weight
    point_loss=torch.stack(point_losses).sum()/point_total if point_losses else zero
    motion_losses=[];motion_total=0.
    for i,(g,label,identity) in enumerate(interval_meta):
     if g!=key:continue
     weight=1./interval_weights[identity]
     motion_losses.append(F.binary_cross_entropy_with_logits(ip[i],torch.tensor(label))*weight);motion_total+=weight
    motion_loss=torch.stack(motion_losses).sum()/motion_total if motion_losses else zero
    group_losses.append(view_loss+.25*point_loss+.25*motion_loss)
   loss=torch.stack(group_losses).mean();assert torch.isfinite(loss)
   optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True);optimizer.step()
   epoch_losses.append(float(loss.detach()))
  history.append(dict(epoch=epoch+1,training_loss=float(np.mean(epoch_losses))))
  write(U/'PROCESS_TRAIN_PROGRESS.json',dict(at=now(),status='RUNNING',pid=os.getpid(),epoch=epoch+1,total_epochs=80,last_loss=history[-1]['training_loss'],elapsed_seconds=time.perf_counter()-started))
  if epoch==0 or (epoch+1)%10==0:print('process training',epoch+1,history[-1]['training_loss'],flush=True)
 for p,h in contract['pins'].items():assert sha(p)==h,p
 torch.save(model.state_dict(),OUT/'model.pt')
 write(OUT/'HISTORY.json',history)
 write(U/'PROCESS_TRAIN_RESULT.json',dict(at=now(),status='FIXED_CANDIDATE_TRAINED_NOT_EVALUATED',checkpoint=str(OUT/'model.pt'),
  checkpoint_sha256=sha(OUT/'model.pt'),normalization=str(OUT/'NORMALIZATION.json'),normalization_sha256=sha(OUT/'NORMALIZATION.json'),
  parameters=sum(p.numel() for p in model.parameters()),epochs=80,seed=42,candidates=1,source_queries=45,source_clips=36,
  registered_fit_ids=list(range(1,10)),views=217,seconds=time.perf_counter()-started,history=history,
  contract_sha256=sha(U/'PROCESS_COMPARISON_FROZEN.json'),old_weights_changed=False,development_used_for_optimization=False,
  unknown_as_negative=False,new_accuracy_claim=False,whole_goal_complete=False))
 write(U/'PROCESS_TRAIN_PROGRESS.json',dict(at=now(),status='COMPLETE_NOT_EVALUATED',active=None,epoch=80,total_epochs=80))

if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'PROCESS_TRAIN_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
