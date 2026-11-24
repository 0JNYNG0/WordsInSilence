"""One stored forward per fixed FIT/DEV query-view; no selection or old model run.

These raw .5/.8/.2 diagnostics are NOT the eventual qualified feedback claim.
Automatic-proposal/traceable-chain qualification is evaluated separately.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,time,traceback
import numpy as np
import torch
from process_head import ProcessEvidenceNet,QUERIES,make_input,batch
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def check_pins(c):
 for p,h in c['pins'].items():assert sha(p)==h,p
def score_summary(rows):
 known=[r for r in rows if r['truth'] is not None]
 counts=Counter(str(r['truth']) for r in rows)
 raw_correct=sum((r['probability']>=.5)==bool(r['truth']) for r in known)
 class_rows={str(y):[r for r in known if r['truth']==y] for y in [0,1]}
 class_counts={k:dict(total=len(rs),raw_correct=sum((r['probability']>=.5)==bool(int(k)) for r in rs),
   confident_correct=sum(r['raw_display']==int(k) for r in rs),confident_wrong=sum(r['raw_display']==1-int(k) for r in rs),
   abstained=sum(r['raw_display'] is None for r in rs)) for k,rs in class_rows.items()}
 recalls=[v['raw_correct']/v['total'] for v in class_counts.values() if v['total']]
 return dict(counts=dict(counts),known=len(known),raw_correct=raw_correct,raw_accuracy=raw_correct/len(known) if known else None,
  class_counts=class_counts,raw_balanced_accuracy=sum(recalls)/2 if len(recalls)==2 else None,
  unknown_raw_confident=sum(r['truth'] is None and r['raw_display'] is not None for r in rows),
  disclaimer='Raw confidence is not traceable observed evidence. Unknown truth remains unknown even if raw probability is high.')

def load_case(r,source):
 assert source['role']==r['role'] and source['clip_id']==r['clip_id'] and source['person']==r['recorded_person_id']
 assert source['source_sha256']==r['source_sha256'] and r['view_id'] in source['aliases']
 assert sha(source['input'])==source['input_sha256']
 with np.load(source['input'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 fm=r['ordered_frame_map']
 assert a['frame_id'].tolist()==[f['output_index'] for f in fm]==list(range(len(fm)))
 assert a['timestamp_us'].tolist()==[f['view_timestamp_us'] for f in fm]
 assert a['image_available'].tolist()==[f['image_available'] for f in fm]
 assert source['source_frame_indices']==[f['source_native_frame'] for f in fm]
 assert source['source_timestamps_us']==[f['source_native_timestamp_us'] for f in fm]
 for t in r['sparse_relation_targets']:
  assert fm[t['output_index']]['image_available'] and fm[t['output_index']]['source_native_frame']==t['source_frame']
 for t in r['fully_retained_motion_interval_targets']:
  ii=t['output_indices'];ff=t['source_frames']
  assert len(ii)>=2 and ii==list(range(ii[0],ii[-1]+1)) and ff==list(range(ff[0],ff[-1]+1))
  assert ff==[fm[i]['source_native_frame'] for i in ii] and all(fm[i]['image_available'] for i in ii)
 return a

def main():
 result=U/'PROCESS_EVALUATION.json';started_file=U/'PROCESS_EVALUATION_STARTED.json'
 assert not result.exists() and not started_file.exists(),'No silent repeat of a completed/interrupted evaluation'
 contract=read(U/'PROCESS_COMPARISON_FROZEN.json');check_pins(contract)
 train=read(U/'PROCESS_TRAIN_RESULT.json')
 assert train['status']=='FIXED_CANDIDATE_TRAINED_NOT_EVALUATED'
 assert sha(train['checkpoint'])==train['checkpoint_sha256'] and sha(train['normalization'])==train['normalization_sha256']
 assert train['contract_sha256']==sha(U/'PROCESS_COMPARISON_FROZEN.json')
 stats=read(train['normalization']);torch.set_num_threads(4)
 model=ProcessEvidenceNet();model.load_state_dict(torch.load(train['checkpoint'],map_location='cpu',weights_only=True));model.eval()
 prepared=[]
 for role,tn,im,expected,persons in [
  ('fit','PROCESS_VIEW_TARGETS.json','PROCESS_INPUT_MANIFEST.json',217,set(range(1,10))),
  ('development','DEV_PROCESS_VIEW_TARGETS.json','DEV_PROCESS_INPUT_MANIFEST.json',80,{11,12,13})]:
  targets=read(U/tn)['rows'];inputs={r['pixel_view_key']:r for r in read(U/im)['rows']}
  assert len(targets)==expected
  for r in targets:
   assert r['role']==role and r['recorded_person_id'] in persons and r['physical_query'] in QUERIES
   a=load_case(r,inputs[r['pixel_view_key']]);prepared.append((r,a,make_input(a,stats)))
 write(started_file,dict(at=datetime.now(timezone.utc).isoformat(),cases=len(prepared),contract_sha256=sha(U/'PROCESS_COMPARISON_FROZEN.json')))
 rows=[];t0=time.perf_counter()
 with torch.no_grad():
  for r,a,seq in prepared:
   start=time.perf_counter();x,l=batch([seq]);q=torch.tensor([QUERIES.index(r['physical_query'])])
   pred=model(x,l,q);p=float(pred['process_logits'].sigmoid()[0]);point=pred['point_logits'].sigmoid()[0].tolist()
   process_seconds=time.perf_counter()-start;interval_start=time.perf_counter()
   intervals=[]
   targets=r['fully_retained_motion_interval_targets']
   if targets:
    ix,il=batch([seq[v['output_indices']] for v in targets]);iq=q.expand(len(targets))
    ip=model.interval_logits(ix,il,iq).sigmoid().tolist()
    intervals=[dict(source_frames=v['source_frames'],output_indices=v['output_indices'],truth=v['label'],probability=float(pp)) for v,pp in zip(targets,ip)]
   interval_seconds=time.perf_counter()-interval_start
   point_rows=[dict(output_index=v['output_index'],source_frame=v['source_frame'],setup_truth=v['setup_relation'],
    terminal_truth=v['terminal_relation'],setup_probability=point[v['output_index']][0],terminal_probability=point[v['output_index']][1]) for v in r['sparse_relation_targets']]
   rows.append(dict(view_id=r['view_id'],query_id=r['query_id'],clip_id=r['clip_id'],role=r['role'],person=r['recorded_person_id'],
    physical_query=r['physical_query'],pixel_view_key=r['pixel_view_key'],kind=r['construction_kind_audit_only'],
    truth=r['observed_process_evidence'],probability=p,raw_display=1 if p>=.8 else 0 if p<=.2 else None,
    frames=len(seq),image_available=a['image_available'].tolist(),source_frames=[f['source_native_frame'] for f in r['ordered_frame_map']],
    source_timestamps_us=[f['source_native_timestamp_us'] for f in r['ordered_frame_map']],view_timestamps_us=a['timestamp_us'].tolist(),
    point_probabilities=point,sparse_point_scores=point_rows,interval_scores=intervals,
    inference_seconds=time.perf_counter()-start,process_forward_seconds=process_seconds,
    annotation_interval_diagnostic_seconds=interval_seconds,timing_scope='Diagnostic head forwards including annotation-selected intervals; not runtime or end-to-end latency',
    whole_word_correctness=None,learner_error=None))
 summaries={}
 for role in ['fit','development']:
  rr=[r for r in rows if r['role']==role]
  per_query={q:score_summary([r for r in rr if r['physical_query']==q]) for q in QUERIES}
  ba=[x['raw_balanced_accuracy'] for x in per_query.values() if x['raw_balanced_accuracy'] is not None]
  per_source={q:score_summary([r for r in rr if r['query_id']==q]) for q in sorted({r['query_id'] for r in rr})}
  source_ba=[x['raw_balanced_accuracy'] for x in per_source.values() if x['raw_balanced_accuracy'] is not None]
  summaries[role]=dict(overall=score_summary(rr),per_query=per_query,macro_query_balanced_accuracy=sum(ba)/len(ba) if ba else None,
   per_source_query=per_source,macro_source_query_balanced_accuracy=sum(source_ba)/len(source_ba) if source_ba else None,
   source_query_count=len(per_source),independent_case_count_claim=False,
   per_kind={k:score_summary([r for r in rr if r['kind']==k]) for k in sorted({r['kind'] for r in rr})},
   per_person={str(p):score_summary([r for r in rr if r['person']==p]) for p in sorted({r['person'] for r in rr})},
   short=score_summary([r for r in rr if r['frames']<16]),long=score_summary([r for r in rr if r['frames']>=16]))
 check_pins(contract)
 write(result,dict(status='ONE_FIXED_RAW_EVALUATION_SAVED_NOT_INTEGRATION_APPROVAL',rows=rows,summaries=summaries,
  seconds=time.perf_counter()-t0,checkpoint_sha256=train['checkpoint_sha256'],contract_sha256=sha(U/'PROCESS_COMPARISON_FROZEN.json'),
  automatic_path_evaluated=False,old_models_executed=0,heldout_personal_inputs=0,whole_goal_complete=False,
  limitations=['Same-Codex physical labels, not expert linguistic correctness','Repeated DEV, not independent final test',
   'Edited views from the same source are correlated','Manual bounded-window diagnostics cannot authorize runtime adoption']))
 print(json.dumps(summaries['development']['overall'],ensure_ascii=False))

if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'PROCESS_EVALUATION_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
