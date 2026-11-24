"""Independent saved-target lineage/axis audit plus actual FIT data binding.

No detector, encoder, model checkpoint, model forward or training is permitted.
Run only after the root produces the complete NOMINAL_PROCESS_TARGETS.json.
"""
from pathlib import Path
from collections import Counter
import json,hashlib,traceback
import numpy as np

U=Path(__file__).resolve().parent
AB=U.parent/'word100_action_completion_supervision_20261102'
CHECKS=0
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ck(value,note):
 global CHECKS
 assert value,note
 CHECKS+=1

def main():
 dest=U/'NOMINAL_TARGETS_VERIFIED.json';ck(not dest.exists(),'No overwrite of completed audit')
 targetpath=U/'NOMINAL_PROCESS_TARGETS.json';d=read(targetpath)
 ck(d['status']=='DIRECT_NOMINAL_PROCESS_TARGETS_READY','ready status')
 M=read(U/'FIRST_RUNTIME_INTERVAL_REVIEW_MANIFEST.json')
 originals={r['id']:r for r in read(AB/'PROCESS_TARGETS.json')['rows']}
 runtime={r['query_id']+'__runtime_proposal':r for r in read(AB/'PROCESS_RUNTIME_SUPERVISION_BINDING.json')['rows'] if r['role']=='fit'}
 views={r['view_id']:r for r in read(U/'FIT_VIEW_TARGETS_V2.json')['rows']}
 inputs={r['pixel_view_key']:r for r in read(AB/'PROCESS_INPUT_MANIFEST.json')['rows']}
 rgb=read(U/'RGB_SUPPORT_MANIFEST.json')['inputs']
 saved=read(AB/'PROCESS_AUTO_EVALUATION.json')
 auto={r['query_id']:r for r in saved['automatic_rows'] if r['role']=='fit'}
 savedviews={r['view_id']:r for r in saved['view_rows'] if r['role']=='fit'}
 ck(len(runtime)==45 and len(views)==217 and len(inputs)==216,'FIT manifest counts')
 pins={str(p.resolve()):sha(p) for p in [Path(__file__),targetpath,U/'process_data.py',U/'process_head.py',
    U/'build_nominal_process_targets.py',U/'process_rgb_support.py']}
 for source in (d,M):
  for path,h in source['pins'].items():
   ck(sha(path)==h,'pinned source '+path);pins[str(Path(path).resolve())]=h
 ck(all(r['role']=='fit' and r['person'] in range(1,10) for r in d['rows']),'no DEV targets')
 byid={r['span_id']:r for r in d['rows']};ck(len(byid)==len(d['rows']),'unique span identifiers')
 direct={}
 for name in ('REVIEW_FIRST_1133.json','REVIEW_FIRST_2523.json','REVIEW_FIRST_0739_1567.json'):
  p=U/name;review=read(p);pins[str(p.resolve())]=sha(p)
  for r in review['rows']:
   ck(r['review_id'] not in direct,'unique directly reviewed query')
   direct[r['review_id']]=(r,p)
 ck(len(direct)==45,'45 direct nominal reviews')
 expected={};direct_unknown=[]
 for m in M['rows']:
  candidates=auto[m['query_id']]['qualification']['interval_candidates']
  ck(bool(candidates),'prepared candidate exists')
  ck([(x['end'],x['start']) for x in candidates]==sorted((x['end'],x['start']) for x in candidates),'chronological selection')
  ck(m['nominal_output_indices']==candidates[0]['output_indices'],'candidate0 never score choice')
  r,p=direct[m['review_id']]
  ck(r['case_id']==m['case_id'],'direct case binding')
  ck(r['source_frames']==m['nominal_source_frames'] and r['source_timestamps_us']==m['nominal_source_timestamps_us'],'direct exact frame/PTS scope')
  ck({x['path']:x['sha256'] for x in r['reviewed_pages']}=={x['path']:x['sha256'] for x in m['review_pages']},'all pages reviewed')
  for page in r['reviewed_pages']:
   ck(sha(page['path'])==page['sha256'],'reviewed page immutable');pins[str(Path(page['path']).resolve())]=page['sha256']
  expected[m['case_id']+'__frozen_first_review']=(m['case_id'],m['nominal_output_indices'],r['observed_process_evidence'],'direct_nominal_review')
  if r['observed_process_evidence'] is None:direct_unknown.append(m['review_id'])
  for j,s in enumerate(m['existing_direct_positive_span_candidates']):
   original=originals[m['query_id']];seq=original['sequence_links'][s['original_sequence_index']]
   ck(seq==s['original_sequence'] and seq['visible_chain']=='yes','explicit original sequence equality')
   ck(sha(s['source_annotation'])==s['source_annotation_sha256']==original['source_annotation_sha256'],'direct original annotation hash')
   pins[str(Path(s['source_annotation']).resolve())]=s['source_annotation_sha256']
   linked=runtime[m['case_id']]['sequence_bindings'][s['original_sequence_index']]
   ck(linked['supports_observed_process_positive'] and linked['original_annotation']==seq,'verified runtime chain binding')
   expected[m['case_id']+'__existing_direct_chain_'+str(j)]=(m['case_id'],s['output_indices'],1,'explicit_direct_chain')
 expected_excluded=set()
 for key,v in savedviews.items():
  if v['kind']!='middle_splice' or not v['qualification']['interval_candidates']:continue
  ck(v['truth']==views[key]['observed_process_evidence'],'saved truth matches direct source')
  if views[key]['observed_process_evidence']!=0:
   expected_excluded.add(key);continue
  expected[key+'__contained_absence_first_candidate']=(key,v['qualification']['interval_candidates'][0]['output_indices'],0,'contained_reviewed_absence')
 ck(set(expected)==set(byid),'all expected and only expected spans')
 ck({r['case_id'] for r in d['excluded']}==expected_excluded,'positive/unknown parent exclusions preserved')
 outside=0
 for sid,(case,ii,label,kind) in expected.items():
  t=byid[sid]
  ck(t['case_id']==case and t['output_indices']==ii and t['observed_process_evidence']==label and t['binding_kind']==kind,'exact target '+sid)
  if case in runtime:
   r=runtime[case];axis=r['proposal_axis'];inputsha=r['input_sha256'];sourcehash=r['source_sha256'];person=r['person']
  else:
   r=views[case];src=inputs[r['pixel_view_key']];inputsha=src['input_sha256'];sourcehash=r['source_sha256'];person=r['recorded_person_id']
   axis=[dict(output_index=a['output_index'],source_frame=a['source_native_frame'],source_timestamp_us=a['source_native_timestamp_us'],image_available=a['image_available']) for a in r['ordered_frame_map']]
  ck(ii==list(range(ii[0],ii[-1]+1)) and 0<=ii[0]<ii[-1]<len(axis),'bounded contiguous local indices')
  aa=[axis[j] for j in ii]
  ck(t['source_frames']==[a['source_frame'] for a in aa],'actual source frames')
  ck(t['source_timestamps_us']==[a['source_timestamp_us'] for a in aa],'actual source time')
  ck(t['image_available']==[a['image_available'] for a in aa],'actual availability')
  ck(t['input_sha256']==inputsha and t['source_sha256']==sourcehash and t['person']==person,'identity binding')
  ck(t['predicate_definitions']==r['predicate_definitions'],'physical query definitions')
  ck(t['label_scope']=='nominal_visible_process_only','narrow label scope')
  ck(all(t[k] is None for k in ('whole_word_correctness','learner_error','required_repetitions','exact_completion_boundary')),'semantic unknown preserved')
  review=t['direct_review'];ck(sha(review['path'])==review['sha256'] and bool(review['reason']),'direct provenance')
  pins[str(Path(review['path']).resolve())]=review['sha256']
  support=rgb[inputsha]['support'];sf=support['source_frame_ids'];mapper={f:i for i,f in enumerate(sf)}
  ci=[mapper[f] for f in t['source_frames']];ctx=t['rgb_context_support']
  ck(ctx['nominal_cache_rows']==ci,'RGB nominal row mapping')
  ck(ctx['nominal_source_frames']==t['source_frames'] and ctx['nominal_source_timestamps_us']==t['source_timestamps_us'],'RGB nominal source/time')
  # Independently union saved encoder windows and strictly positive interpolation
  # coefficients; this is structural context, never direct process truth.
  encoded=set();consumed=set()
  for i in ci:
   row=set()
   for w,n,den in support['row_window_terms'][i]:
    ck(den>0 and n>0,'only positive interpolation support')
    row.update(support['windows'][w]['slot_rows'])
   encoded.update(row)
   if support['image_available'][i] and support['rgb_finite'][i]:consumed.update(row)
  ck(ctx['encoded_context_cache_rows']==sorted(encoded),'actual encoded support union')
  ck(ctx['consumed_context_cache_rows']==sorted(consumed),'actual consumed support union')
  ck(ctx['outside_nominal_consumed_rows']==sorted(consumed-set(ci)),'outside nominal is explicit')
  ck(ctx['observed_context_source_frames']==[sf[j] for j in sorted(consumed) if support['image_available'][j]],'observed RGB support')
  ck(ctx['synthetic_context_source_coordinates']==[sf[j] for j in sorted(consumed) if not support['image_available'][j]],'blank RGB context preserved')
  ck(ctx['structural_context_is_direct_nominal_evidence'] is False and ctx['local_process_truth'] is None,'RGB support not target truth')
  outside+=bool(consumed-set(ci))
  if kind=='explicit_direct_chain':
   seq=t['original_sequence'];required=set(seq['setup_frames']+seq['terminal_frames']);lo,hi=seq['motion_interval'];required.update(range(lo,hi+1))
   local={a['source_frame']:a['output_index'] for a in axis}
   ck(t['required_source_frames']==sorted(required),'explicit required source evidence')
   ck(t['explicit_chain_witness_output_indices']==[local[f] for f in sorted(required)],'explicit witness local axis')
   ck(required<=set(t['source_frames']) and all(t['image_available']),'full exact visible chain retained')
   ck(t['label_inferred_from_whole_view'] is False,'no positive whole-copy')
  elif kind=='contained_reviewed_absence':
   ck(r['observed_process_evidence']==0 and t['parent_whole_process_truth']==0,'same edited parent direct absence')
   ck(rgb[inputsha]['route']=='actual_materialized_view_cache','no original unedited neighbor cache')
   ck(set(ci)<=consumed<=set(range(len(axis))),'entire RGB context contained in reviewed edited parent')
   ck(t['support_output_indices']==sorted(consumed) and t['parent_reviewed_output_indices']==list(range(len(axis))),'absence scope support indices')
   ck(t['label_inferred_from_whole_view'] is True and t['new_motion_zero_labels']==0,'absence inference disclosed; no net-motion0 invention')
  else:ck(t['label_inferred_from_whole_view'] is False,'direct label not copied')
 ck(dict(Counter(t['binding_kind'] for t in d['rows']))==d['counts'],'saved binding counts')
 ck(dict(Counter(str(t['observed_process_evidence']) for t in d['rows']))==d['labels'],'saved label counts')
 # Execute only the data adapter and feature normalization. Explicitly reject
 # all torch Module calls while verifying that all262 FIT records consume the
 # saved nominal supervision with the exact source/local mappings above.
 import torch
 original_call=torch.nn.Module._call_impl
 def forbidden(*args,**kwargs):raise AssertionError('Model forward forbidden in saved-target verification')
 torch.nn.Module._call_impl=forbidden
 try:
  from process_data import build_fit_dataset,multiplicities
  dataset=build_fit_dataset(targetpath,require_spans=True)
  mult=multiplicities(dataset['records'])
 finally:torch.nn.Module._call_impl=original_call
 ck(len(dataset['records'])==262 and len(dataset['groups'])==45,'262 FIT records/45 groups consume adapter')
 ck(dataset['development_arrays_opened']==0 and dataset['model_forwards']==0 and dataset['training_steps']==0,'no DEV inference or training')
 consumedtargets=[t for r in dataset['records'] for t in r['nominal_process_targets']]
 ck({t['span_id'] for t in consumedtargets}==set(byid) and len(consumedtargets)==len(byid),'every target consumed once')
 for t in consumedtargets:ck(t==byid[t['span_id']],'consumed target equals saved target')
 ck(len(mult)==45,'duplicate supervision weighting source/query groups')
 pins.update(dataset['pins'])
 for path,h in pins.items():ck(sha(path)==h,'final pinned artifact '+path)
 result=dict(status='PASS_SAVED_NOMINAL_TARGETS_AND_ACTUAL_FIT_BINDING',checks=CHECKS,
  target_sha256=sha(targetpath),counts=d['counts'],labels=d['labels'],
  directly_reviewed_first_intervals=45,explicit_existing_positive_chains=44,
  direct_unknown_review_ids=direct_unknown,spans_with_RGB_support_outside_nominal=outside,
  actual_fit_records_consumed=262,source_query_groups=45,actual_nominal_target_rows_consumed=len(consumedtargets),
  actual_data_adapter_status=dataset['nominal_span_status'],freezable_pins=pins,
  model_calls=0,training_steps=0,new_detector_calls=0,new_encoder_calls=0,
  independent_human_expert=False,independent_visual_label_validation=False,
  limitations=['Separate saved-value implementation audit by Codex; not independent human semantic ground truth.',
   'RGB structural support may extend beyond the nominal target; it does not transfer process labels.',
   'Labels concern a narrow physically visible process, not full-sign correctness or learner error.',
   'No validation score, new model output, or training success is asserted.'])
 with dest.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ['status','checks','counts','labels','actual_fit_records_consumed','actual_nominal_target_rows_consumed']}))

if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'NOMINAL_TARGETS_VERIFY_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
