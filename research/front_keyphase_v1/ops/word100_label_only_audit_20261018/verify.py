"""Independent source enumeration/count/preservation check; no model or audit imports.

This verifies audit coverage/arithmetic, not the human truth of anatomical labels.
"""
import json, hashlib, time
from pathlib import Path
from collections import Counter
U=Path(__file__).resolve().parent;P=U.parent/'word100_assignment_conflict_20261018'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
start=time.monotonic();c=read(U/'CONTEXT.json');old=read(P/'CONTEXT.json')
ev=read(U/'EVENTS.json');rv=read(U/'VISUAL_REVIEW.json');s=read(U/'SUMMARY.json')
assert len(old['rows'])==899
removed={(x['clip_id'],x['view'],x['index'],x['raw_index']) for x in read(P/'CHANGES.json')}
assert len(removed)==1
source={};all_label=0
for r in old['rows']:
 assert r['role']=='train_candidate' and 1<=r['signer_id']<=9
 for v in ('masked','sensor'):
  traces=read(Path(r['views'][v]['assignment']).with_name('trace.json'))
  for f,tr in enumerate(traces):
   for raw_index,why in enumerate(tr['reasons']):
    if why!='label_without_close_pose':continue
    all_label+=1;key=(r['clip_id'],v,f,raw_index)
    if key not in removed:source[key]=tr['assigned_sides'][raw_index]
assert all_label==457 and len(source)==456
observed={(e['clip_id'],e['view'],e['index'],e['raw_index']):e['side'] for e in ev}
assert observed==source and len(ev)==len(observed)
records=rv['records'];assert len(records)==456
assert [r['id'] for r in records]==list(range(1,457))
assert rv['reviewed_events']==456 and rv['reviewed_sheets']==38 and rv['detailed_cases']==23
details=read(U/'DETAILS.json');assert len(details)==23
assert {d['id'] for d in details}=={r['id'] for r in records if r['detail']}
for r,e in zip(records,ev):
 for k in ('id','clip_id','word','view','index','frame_id','raw_index','side','first_assignment_for_side','label_score'):assert r[k]==e[k]
 assert r['basis'] and r['verdict'] in ('normal_identity','error_identity','uncertain')
 if r['verdict']=='normal_identity':assert r['actual_side']==r['side']
 if r['verdict']=='error_identity':assert r['actual_side']==1-r['side']
 if r['verdict']=='uncertain':assert r['actual_side'] is None
 for k in ('panel','sheet','detail'):
  if r[k]:assert (U/r[k]).is_file()
counts=dict(Counter(r['verdict'] for r in records));assert counts==s['counts']
assert sum(counts.values())==456
assert dict(Counter(r['category'] for r in records))==s['categories']
for k,n in counts.items():assert abs(s['percent'][k]-n/456*100)<1e-12
assert len({r['clip_id'] for r in records})==s['clips']==353
assert len({r['word'] for r in records})==s['words']==100
assert len({r['clip_id'] for r in records if r['verdict']=='error_identity'})==s['error_clips']==12
for v in ('masked','sensor'):assert dict(Counter(r['verdict'] for r in records if r['view']==v))==s['by_view'][v]
for v in (True,False):assert dict(Counter(r['verdict'] for r in records if r['first_assignment_for_side']==v))==s['by_first_entry'][str(v)]
assert len(list((U/'frames').glob('*.jpg')))==456 and len(list((U/'sheets').glob('*.jpg')))==38
for n,(path,digest) in enumerate(c['pins'].items(),1):
 assert sha(path)==digest,path
 if n%3000==0:print('Protected hash pins verified',n,flush=True)
for r in c['rows']:assert sha(U/'videos'/f"{r['clip_id']}.mp4")==r['source']['sha256']
assert not (U/'RUNNING.json').exists()
assert s['new_training']==s['new_classifier_inference']==s['user_mentor_dev_heldout_reads']==0
assert not s['assignment_code_changes'] and not s['production_changed']
result={'status':'PASS','source_rows_enumerated':899,'source_label_only_before_guard':457,'previous_withheld_reused':1,'remaining_exact_event_identity_matched':456,'unique_clips':353,'counts':counts,'categories':s['categories'],'annotation_coverage':456,'evidence_panels':456,'sheets':38,'neighbor_sequences':23,'original_video_hashes':353,'protected_hashes':len(c['pins']),'same_operator_independent_calculation':True,'independent_human_ground_truth':False,'visual_verdict_truth_verified_by_code':False,'seconds':time.monotonic()-start}
with (U/'INDEPENDENT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False),flush=True)
