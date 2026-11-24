from pathlib import Path
import sys,json,hashlib,time,os
import numpy as np
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_context_sequence_match_20261031';REL=U.parent/'word100_relational_sequence_compare_20261031';PHASE=U.parent/'word100_core_stage_evidence_20261031'
sys.path.insert(0,str(REL))
from relational import GROUPS,features,observed_summary
from split_match import match,feedback,window_dp
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
def init():
 d=read(OLD/'AUTHORIZATION.json');d.update(request='2026-11-01 user confirms generic all-word design and approves all4 steps without stopping: core/context split, conflict evidence, public controls, verified diagnostic feedback integration.',unit=U.name,algorithm_word_specific=False,validated_words=4,all_words_performance_verified=False,neural_training=False)
 write(U/'AUTHORIZATION.json',d);write(U/'CURRENT_EXECUTION.json',dict(status='PLAN_REVIEW_IMPLEMENTATION',active=None))
def freeze():
 files=[U/n for n in ['AUTHORIZATION.json','PLAN_KO.txt','split_match.py','run.py','summarize.py','review/PLAN_REVIEW.json','review/SEMANTIC_FEEDBACK_CONTRACT_REVIEW.json','review/SYNTHETIC_CHECK.json']]
 files += [OLD/'context_match.py',OLD/'COMPLETION.json',OLD/'RESULT_MANIFEST.json',REL/'relational.py',PHASE/'PHASE_DATA.json']
 files += [Path(r['input']) for r in read(U/'AUTHORIZATION.json')['rows']]
 files += [Path(x['file']) for x in read(OLD/'RESULT_MANIFEST.json')['results']]
 files += sorted((REL/'controls').glob('*.json'))+sorted((REL/'controls').glob('*.npz'))
 write(U/'FROZEN.json',dict(status='FROZEN_BEFORE_NEW_RESULTS',conditions='saved mean-only + saved context-combined + new core-first/separate anchored context',new_candidate_count=1,new_runs=162,core_cost='unchanged previousframecost/weightedDTW/allwindows [.5,2]framecount/run<=2',context='separate entry andexit ±.20referencePTS; query extent multipliedby corePTSdurationratio; never changescorepath',selection='exact minimalcore candidates EPS1e-9; context selects only if ALLmin candidates contextcomplete and unique minimumcontextcost. Otherwise rawrepresentative+disjoint ambiguity. Contextneveroverridesuniquecore.',source_guide='original reference anchors only; not query labels',feedback='physical evidence/missing/conflict only; no correct/wrong KSL/correction',word_blind='no word/person/alias passed to calculations; segment_id is outputidentity only; reference physicalstage drafts remain suppliedinputs not automaticallwordsegmentation',new_neural_training=False,production_adoption=False,pins={str(p):sha(p) for p in files}))
def numerical(result,indices,key):
 indices=np.asarray(indices);out=[]
 for a in result['anchors']:
  lo,hi=a['reference_core'];expected=set(range(lo,hi+1));p=a[key]
  d=dict(segment_id=a['segment_id'],state=a['state'],expected_original_frames=len(expected),present_unique_original_frames=len(set(indices.tolist())&expected),missing_original_frames=sorted(expected-set(indices.tolist())))
  if p is None:d.update(center_hit=False,center_original=None,recall=0.,precision=0.,path_mean_error=None)
  else:
   x,y=p['query_core'];got=set(indices[x:y+1].tolist());path=np.asarray(p['core_path']);err=np.abs(indices[path[:,0]]-path[:,1]);center=int(indices[(x+y)//2]);d.update(center_hit=lo<=center<=hi,center_original=center,recall=len(expected&got)/len(expected),precision=len(expected&got)/len(got),path_mean_error=float(err.mean()))
  out.append(d)
 return out
def execute():
 f=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in f['pins'].items());write(U/'RUNNING.json',dict(pid=os.getpid(),started=time.time()))
 rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']};clips={r['alias']:r for r in read(PHASE/'PHASE_DATA.json')['clips']};arr={a:load(r['input']) for a,r in rows.items()};feat={a:features(v) for a,v in arr.items()}
 window_dp(np.zeros((4,4)),0,4);manifest=[];folder=U/'results';folder.mkdir(exist_ok=True)
 for row in read(OLD/'RESULT_MANIFEST.json')['results']:
  assert sha(row['file'])==row['sha256'];old=read(row['file']);a,b=old['query'],old['reference'];kind=old['kind'];path=REL/'controls'/(old['name']+'.npz') if kind=='control' else Path(rows[a]['input']);qarr=load(path)
  t=time.perf_counter();d=match(features(qarr),feat[b],clips[b]['segments'],qarr['timestamp_us'],arr[b]['timestamp_us'],old,GROUPS,observed_summary);d['seconds_incremental_core_split']=time.perf_counter()-t
  d.update(name=old['name'],kind=kind,query=a,reference=b,role=rows[a]['role'],control=old.get('control'),old_result=row['file'],old_result_sha256=row['sha256'],query_array=str(path),query_array_sha256=sha(path),frozen_sha256=sha(U/'FROZEN.json'),old_context_seconds_saved=old['seconds'])
  if kind=='control':
   prior=read(REL/'controls'/(old['name']+'.json'));indices=prior['metadata'].get('original_indices',list(range(len(qarr['x']))));d['numerical_raw']=numerical(d,indices,'raw');d['numerical_selected']=numerical(d,indices,'selected')
  for anchor in d['anchors']:
   anchor['feedback']=feedback(anchor)
   # Source guides remain attached to the original reference/anchor, never the query.
   segment=next(x for x in clips[b]['segments'] if x['segment_id']==anchor['segment_id'])
   anchor['reference_guidance']=dict(description=segment.get('description_ko'),source_links=segment.get('source_links',[]),query_label_inferred=False)
  occupied=set()
  for anchor in d['anchors']:
   if anchor['selected']:occupied.update(range(anchor['selected']['query_core'][0],anchor['selected']['query_core'][1]+1))
  d['outside_core_candidates']=[i for i in range(len(qarr['x'])) if i not in occupied];d['outside_is_not_normal_or_discarded']=True
  dest=folder/(old['name']+'.json');write(dest,d);manifest.append(dict(file=str(dest),sha256=sha(dest),kind=kind,name=old['name']))
  (U/'STATUS.json').write_text(json.dumps(dict(done=len(manifest),total=162,last=old['name'])),encoding='utf8');print(len(manifest),old['name'],round(d['seconds_incremental_core_split'],3),flush=True)
 write(U/'RESULT_MANIFEST.json',dict(status='COMPLETE_PENDING_VERIFICATION',results=manifest));(U/'RUNNING.json').unlink()
if __name__=='__main__':{'init':init,'freeze':freeze,'run':execute}[sys.argv[1]]()
