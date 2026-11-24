"""Mapping/hash checks only; does not independently judge the hand movement."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0
def ck(v):
 global checks
 assert v
 checks+=1
p=U/'REVIEW_FIRST_2523.json';d=read(p);mp=U/'FIRST_RUNTIME_INTERVAL_REVIEW_MANIFEST.json';m=read(mp)
ck(d['manifest']['sha256']==sha(mp))
for path,h in m['pins'].items():ck(sha(path)==h)
lookup={r['review_id']:r for r in m['rows']}
ck(len(d['rows'])==9)
for r in d['rows']:
 x=lookup[r['review_id']];axis={a['source_frame']:a for a in x['display_axis']}
 ck(r['role']=='fit' and r['person'] in range(1,10) and r['word_id']=='WORD2523')
 ck(r['case_id']==x['case_id'])
 ck(r['source_sha256']==x['source_sha256'] and r['input_sha256']==x['input_sha256'])
 ck(r['source_frames']==x['nominal_source_frames'])
 ck(r['source_timestamps_us']==x['nominal_source_timestamps_us'])
 ck(r['output_indices']==x['nominal_output_indices'])
 ck({e['path']:e['sha256'] for e in r['reviewed_pages']}=={e['path']:e['sha256'] for e in x['review_pages']})
 for e in r['reviewed_pages']:
  ck(e['actually_viewed']);ck(sha(e['path'])==e['sha256'])
 for e in r['additional_native_images_reviewed']:
  ck(e['actually_viewed']);ck(sha(e['path'])==e['sha256']);ck(e['frame'] in r['source_frames'])
 for e in r['setup_witnesses']+r['terminal_witnesses']:
  a=axis[e['source_frame']];ck(a['nominal'])
  ck(e['source_timestamp_us']==a['source_timestamp_us'] and e['output_index']==a['proposal_output_index'])
 for e in r['motion_witnesses']:
  ck(e['source_frames']==list(range(e['source_frames'][0],e['source_frames'][-1]+1)))
  ck(e['output_indices']==list(range(e['output_indices'][0],e['output_indices'][-1]+1)))
  ck(all(axis[f]['nominal'] for f in e['source_frames']))
  ck(e['output_indices']==[axis[f]['proposal_output_index'] for f in e['source_frames']])
  ck(e['source_timestamps_us']==[axis[f]['source_timestamp_us'] for f in e['source_frames']])
 ck(r['observed_process_evidence'] in (0,1,None))
 ck(r['label']==r['observed_process_evidence'])
 ck(r['label_scope']=='nominal_visible_process_only' and not r['label_inferred_from_whole_view'])
 ck(not r['context_frames_used_to_supply_missing_process'])
 ck(all(r[k] is None for k in ('whole_word_correctness','learner_error','required_repetitions','exact_completion_boundary')))
result=dict(status='PASS_MAPPING_HASH_SCOPE_AUDIT',checks=checks,review_sha256=sha(p),manifest_sha256=sha(mp),
 counts=d['counts'],all_nominal_witnesses_in_selected_span=True,model_calls=0,training_steps=0,
 independent_semantic_accuracy_claim=False,independent_human_expert=False,
 limitation='Verifies recorded provenance and sparse source/local/time mappings; does not re-perform or independently validate visual semantics.')
with (U/'REVIEW_FIRST_2523_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result))
