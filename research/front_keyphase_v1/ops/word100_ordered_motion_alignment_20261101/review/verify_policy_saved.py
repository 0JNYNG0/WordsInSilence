"""Independent saved-policy audit. Run only after root confirms execution.
No flow_guard/matcher/DP/model imports and no source/result writes.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,math,sys
U=Path(__file__).resolve().parents[1];O=U.parent
OLD=O/'word100_core_context_split_20261101';REL=O/'word100_relational_sequence_compare_20261031';AUD=O/'word100_crossperson_stage_audit_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0;failures=[]
def ck(value,message):
 global checks
 checks+=1
 if not value:failures.append(message)
def eq(actual,expected,label):
 if isinstance(expected,dict):
  ck(isinstance(actual,dict) and actual.keys()==expected.keys(),label+' keys')
  if isinstance(actual,dict):
   for k in expected:
    if k in actual:eq(actual[k],expected[k],label+'/'+str(k))
 elif isinstance(expected,list):
  ck(isinstance(actual,list) and len(actual)==len(expected),label+' length')
  if isinstance(actual,list):
   for i,(a,b) in enumerate(zip(actual,expected)):eq(a,b,label+'/'+str(i))
 else:ck(type(actual)==type(expected) and actual==expected,label+' exact')
def main():
 frozen=read(U/'FROZEN.json');manifest=read(U/'RESULT_MANIFEST.json');summary=read(U/'SUMMARY.json')
 oldrows=read(OLD/'RESULT_MANIFEST.json')['results'];old={x['name']:x for x in oldrows}
 new={x['name']:x for x in manifest['results']}
 ck(len(oldrows)==len(old)==len(new)==len(manifest['results'])==162,'162 unchanged records')
 ck(set(old)==set(new),'all previous rows retained')
 before={p:sha(p) for p in frozen['pins']}
 for p,h in frozen['pins'].items():ck(before[p]==h,'frozen hash '+p)
 previous=read(AUD/'ASSESSMENT.json');audit={(e['case'],e['segment_id']):e for e in previous['events']}
 saved=[];opportunities=Counter();diagnostic=defaultdict(Counter);strata=Counter();allmapped=Counter()
 for name,entry in old.items():
  nentry=new[name];dest=Path(nentry['file']);ck(sha(dest)==nentry['sha256'],'new result sha '+name)
  d=read(dest);localpath=Path(entry['file']);local=read(localpath)
  fullpath=REL/('pairs' if entry['kind']=='person_pair' else 'controls')/(name+'.json');base=read(fullpath)
  ck(sha(localpath)==entry['sha256'],'local saved hash '+name)
  ck(d['source_full_path']==str(fullpath) and d['source_full_sha256']==sha(fullpath),'full source provenance '+name)
  ck(d['source_local_result']==entry['file'] and d['source_local_sha256']==entry['sha256'],'local source provenance '+name)
  ck(d['frozen_sha256']==sha(U/'FROZEN.json'),'frozen ancestry '+name)
  for field,value in [('name',name),('kind',entry['kind']),('query',local['query']),('reference',local['reference']),('role',local['role']),('control',local.get('control'))]:eq(d[field],value,'identity '+name+'/'+field)
  ck(math.isfinite(d['incremental_policy_seconds']) and d['incremental_policy_seconds']>=0,'valid policy time '+name)
  eq(d['path'],base['path'],'unchanged full path '+name)
  nq,nr=base['query_frames'],base['reference_frames'];path=base['path']
  ck(d['query_scope_retained']==nq and d['reference_scope_retained']==nr,'full scopes '+name)
  ck(path[0]==[0,0] and path[-1]==[nq-1,nr-1],'path endpoints '+name)
  ck(len(path)==len(set(tuple(p) for p in path)),'unique links '+name)
  for x,y in zip(path,path[1:]):ck((y[0]-x[0],y[1]-x[1]) in ((1,0),(0,1),(1,1)),'valid monotone saved path '+name)
  by=defaultdict(list)
  for qi,ri in path:
   ck(type(qi) is int and type(ri) is int and 0<=qi<nq and 0<=ri<nr,'original link bounds '+name);by[ri].append(qi)
  ck(set(by)==set(range(nr)) and {qi for qi,ri in path}==set(range(nq)),'no frame removed '+name)
  ck(len(d['mapping'])==nr,'all reference rows mapped '+name)
  for ri,row in enumerate(d['mapping']):
   qs=by[ri];expected=dict(reference_frame=ri,query_frames=qs,query_range=[min(qs),max(qs)],display_representative=sorted(qs)[(len(qs)-1)//2],representative_is_display_only=True)
   eq(row,expected,'whole many-to-one range '+name+'/'+str(ri))
   ck(sum(qs[0]<=q<=qs[-1] for q in qs)==len(qs),'mapped query range covers links')
  allmapped['reference_frames']+=nr;allmapped['query_frames']+=nq;allmapped['path_links']+=len(path)
  la={a['segment_id']:a for a in local['anchors']};ga={a['segment_id']:a for a in base['stages']};oa={a['segment_id']:a for a in d['anchors']}
  ck(len(la)==len(local['anchors']) and len(oa)==len(d['anchors']) and set(la)==set(ga)==set(oa),'original anchors retained '+name)
  calc={}
  for sid,anc in la.items():
   lo,hi=anc['reference_core'];links=[p for p in path if lo<=p[1]<=hi];gs={p[0] for p in links};full=[min(gs),max(gs)]
   ck(anc['reference_core']==ga[sid]['reference_core'],'same source core '+name+'/'+sid)
   selected=anc.get('selected');span=selected['query_core'] if selected else None;flags=set();supported=False
   if span is None:relation='local_unavailable';flags.add('local_unavailable')
   else:
    ck(type(span[0]) is int and type(span[1]) is int and 0<=span[0]<=span[1]<nq,'local range valid')
    localset=set(range(span[0],span[1]+1));globalset=set(range(full[0],full[1]+1))
    relation='disjoint' if not localset&globalset else 'contained' if localset<=globalset else 'overlap_extends'
    expected_groups=[v for v in selected['groups'].values() if v['expected']>0]
    supported=len(expected_groups)>0 and all(v['sufficient'] is True for v in expected_groups)
    if relation=='disjoint':flags.add('local_global_location_conflict')
    if relation=='overlap_extends':flags.add('local_global_boundary_disagreement')
    if not supported:flags.add('local_observation_insufficient')
    if anc.get('state')=='ambiguous_occurrence':flags.add('local_occurrence_ambiguous')
    elif anc.get('state')!='supported_core_candidate':flags.add('local_source_state_not_supported')
   calc[sid]=dict(segment_id=sid,reference_core=[lo,hi],global_query_range=full,local_query_range=span,range_relation=relation,local_observation_supported=supported,original_local_state=anc.get('state'),flags=flags,selected_path=links,core_reference_opportunities=hi-lo+1,local_not_used_to_move_global=True)
  sequence=sorted(calc.values(),key=lambda x:tuple(x['reference_core']));rels=[]
  for first,second in zip(sequence,sequence[1:]):
   x,y=first['local_query_range'],second['local_query_range']
   if first['reference_core'][1]>=second['reference_core'][0]:rel='reference_intervals_overlap'
   elif x is None or y is None:rel='local_unavailable'
   elif max(x)<min(y):rel='forward_separated'
   elif max(y)<min(x):rel='reverse_separated'
   else:rel='overlap'
   rels.append(dict(first=first['segment_id'],second=second['segment_id'],state=rel))
   if rel in ('reverse_separated','overlap'):
    first['flags'].add('local_order_unresolved');second['flags'].add('local_order_unresolved')
  eq(d['local_relationships'],rels,'local order independent '+name)
  for sid,expected in calc.items():
   expected['flags']=sorted(expected['flags']);expected['status']='local_corroboration_unresolved' if expected['flags'] else 'window_compatible_not_exact_phase_proof'
   expected.update(sign_correct=None,sign_error=None,correction_instruction=None,unchanged_global_groups=ga[sid]['groups'])
   if entry['kind']=='person_pair':
    prior=audit[(name,sid)];expected['saved_physical_audit_global']=prior['methods']['global_dtw'];expected['saved_physical_audit_local']=prior['methods']['core_first']
    role='fit' if d['role']=='fit' else 'dev';opportunities[role]+=1;loc=prior['methods']['core_first'];held=bool(expected['flags'])
    diagnostic[role]['total']+=1;diagnostic[role]['held']+=held;diagnostic[role]['compatible_window_only']+=not held
    diagnostic[role]['held_local_'+loc['state']]+=held
    diagnostic[role]['held_local_occurrence_'+loc['occurrence']]+=held
    diagnostic[role]['compatible_local_'+loc['state']]+=not held
    diagnostic[role]['compatible_local_occurrence_'+loc['occurrence']]+=not held
   eq(oa[sid],expected,'policy anchor '+name+'/'+sid)
   ck(oa[sid]['sign_correct'] is None and oa[sid]['sign_error'] is None and oa[sid]['correction_instruction'] is None,'no normal/error promotion')
   if not expected['flags']:ck(oa[sid]['status']=='window_compatible_not_exact_phase_proof','zero flags no correctness')
  ck(d['new_alignment'] is False and d['numeric_difference_or_support_recomputed'] is False,'no new alignment/support')
  strata[entry['kind']]+=1;saved.append(d)
 ck(strata==Counter(person_pair=36,control=126),'36 pairs126 controls')
 ck(opportunities==Counter(fit=78,dev=39),'117 person anchors no None excluded')
 groupnames=['fit','dev']+sorted({d['control'] for d in saved if d.get('control')})
 recalc={}
 for group in groupnames:
  cases=[d for d in saved if (d['kind']=='person_pair' and ('fit' if d['role']=='fit' else 'dev')==group)] if group in ('fit','dev') else [d for d in saved if d.get('control')==group]
  anchors=[a for d in cases for a in d['anchors']]
  recalc[group]=dict(cases=len(cases),anchors=len(anchors),range_relation=dict(Counter(a['range_relation'] for a in anchors)),status=dict(Counter(a['status'] for a in anchors)),flags=dict(Counter(f for a in anchors for f in a['flags'])),path_unchanged=True)
 eq(summary['summary'],recalc,'summary fixed strata counts')
 eq(summary['old_physical_summary'],previous['summary']['dev'],'prior DEV annotation counts unchanged')
 ck(summary['full_path_replacement'] is False and summary['alignment_accuracy_improved'] is False and summary['no_local_agreement_is_correctness'] is True,'report no accuracy/correctness promotion')
 # Verify the final report's join without changing any physical labels/scores.
 oldreport=read(AUD/'REPORT_DATA.json');report=read(U/'REPORT_DATA.json');evidence=read(U/'POLICY_EVIDENCE_AUDIT.json')
 sources={d['name']:d for d in saved if d['kind']=='person_pair'};evidence_counts={};evidence_rows=[];escaped=[]
 for role in ('fit','development'):
  c=Counter()
  for case in oldreport['cases']:
   if case['role']!=role:continue
   aa={x['segment_id']:x for x in sources[case['id']]['anchors']}
   for e in case['events']:
    g=aa[e['segment_id']];v=e['methods']['core_first'];held=bool(g['flags'])
    if v['state']=='same_kind_core':category={'same_documented_ordinal':'same_kind_same_ordinal','different_documented_ordinal':'different_ordinal'}.get(v['occurrence'],'same_kind_ordinal_unknown')
    else:category='different_kind' if v['state']=='different_kind_core' else v['state']
    c['all']+=1;c[category]+=1;c[category+('_held' if held else '_compatible')]+=1
    c['global_observation_insufficient']+=bool(e['methods']['global_dtw']['insufficient_groups'])
    row=dict(case=case['id'],segment=e['segment_id'],role=role,local_category=category,local_research_supported=v.get('saved_state')=='supported_core_candidate',flags=g['flags'],policy_status=g['status'],original_global_state=e['methods']['global_dtw']['state'])
    evidence_rows.append(row)
    if not held and category in ('different_kind','different_ordinal'):escaped.append(row)
  evidence_counts[role]=dict(c)
 eq(evidence['counts'],evidence_counts,'evidence count no hidden withholding')
 eq(evidence['rows'],evidence_rows,'117 evidence rows');eq(report['evidence_counts'],evidence_counts,'report evidence counts')
 eq(report['policy_summary'],summary,'report policy summary')
 for field in ('summary','changes','annotation_scope','kinds'):eq(report[field],oldreport[field],'original physical report retained/'+field)
 expected_cases=[]
 for case in oldreport['cases']:
  clone=json.loads(json.dumps(case));aa={x['segment_id']:x for x in sources[case['id']]['anchors']}
  for e in clone['events']:
   g=aa[e['segment_id']];e['guard']={k:v for k,v in g.items() if k not in ('unchanged_global_groups','saved_physical_audit_global','saved_physical_audit_local')}
   e['methods']={k:v for k,v in e['methods'].items() if k in ('global_dtw','core_first')}
  expected_cases.append(clone)
 eq(report['cases'],expected_cases,'report case methods and guard exact')
 protected=read(O/'word100_semantic_episode_pilot_20261031/FROZEN.json')['protected'];ck(len(protected)==32,'protected count32')
 for p,h in protected.items():ck(sha(p)==h,'protected unchanged '+p)
 cause=read(U/'review/BASELINE_CAUSE_AUDIT.json')
 for p,h in cause['source_pins'].items():ck(sha(p)==h,'cause source unchanged '+p)
 for p,h in frozen['pins'].items():ck(sha(p)==h==before[p],'frozen source unchanged '+p)
 ck(not (U/'RUNNING.json').exists(),'finished no running marker')
 out=dict(status='PASS_SAVED_POLICY_VERIFICATION' if not failures else 'FAIL_SAVED_POLICY_VERIFICATION',reviewer='same Codex A independent arithmetic against root policy; not human semantic truth',checks=checks,failures=failures,rows=len(saved),strata=dict(strata),person_anchor_opportunities=dict(opportunities),all_anchor_opportunities=sum(len(d['anchors']) for d in saved),all_retained_frames_and_links=dict(allmapped),summary=recalc,diagnostic_hold_counts=dict(diagnostic),verified_policy_evidence_counts=evidence_counts,definite_local_difference_still_compatible=escaped,
  original_physical_dev_summary_unchanged=previous['summary']['dev'],frozen_pins=len(frozen['pins']),protected_files=len(protected),cause_source_pins=len(cause['source_pins']),
  meaning=['All saved global links, feature group values and frame scopes remain exact. The mapping lists every query frame linked to each reference frame; its representative is display only.',
   'Disjoint, overlap_extends and contained are numeric interval relations. Contained and zero flags mean compatible window only, never exact stage, occurrence or sign correctness.',
   'Missing/ambiguous/local-order evidence retains unresolved flags. No educational correction instruction is generated.',
   'Person-pair baseline comes from original global DTW; numerical controls come from relational-feature full DTW. Their baseline accuracy is not pooled.',
   'Held physically same candidates are reported alongside held different-stage/ordinal candidates; prior provisional labels and all denominators are unchanged. Consumed DEV diagnosis, not new generalization.'],
  first_verifier_run_note='The prepared checker predated a pre-freeze policy addition: local_source_state_not_supported. The initial83 flag mismatches are preserved in SAVED_POLICY_FIRST_FAILURE.json; checker updated to independently validate the final frozen source-state rule. No source/result/policy changes or execution rerun.',
  activity=dict(new_searches=0,new_DP=0,new_model_inference=0,new_training=0,source_writes=0),hashes={n:sha(U/n) for n in ['FROZEN.json','flow_guard.py','run.py','POLICY.json','SUMMARY.json','RESULT_MANIFEST.json','POLICY_EVIDENCE_AUDIT.json','REPORT_DATA.json','review/BASELINE_CAUSE_AUDIT.json']})
 (U/'review/SAVED_POLICY_VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 print(out['status'],checks,'checks',len(failures),'failures',strata,opportunities)
 if failures:print('\n'.join(failures[:30]));raise SystemExit(1)
 print(json.dumps(dict(diagnostic),ensure_ascii=False,indent=2))
if __name__=='__main__':
 if sys.argv[1:]!=['--post-execution-notified']:raise SystemExit('Run only after explicit root execution-complete notification: --post-execution-notified')
 main()
