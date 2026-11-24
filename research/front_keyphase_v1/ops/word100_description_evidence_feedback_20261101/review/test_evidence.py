"""Independent synthetic evidence checks. Reads only this unit; no legacy arrays/models."""
from pathlib import Path
import copy, hashlib, importlib.util, json, math, ast
from collections import Counter
import numpy as np

U=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('reviewed_evidence',U/'evidence.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
checks=0; failures=[];categories=Counter()
def check(ok,name,detail=None):
 global checks
 checks+=1;categories[name.split('/')[0]]+=1
 if not ok:failures.append(dict(test=name,detail=detail))
def reject(fn,name):
 try:fn()
 except (AssertionError,ValueError,TypeError,KeyError,IndexError):check(True,name);return
 check(False,name,'Invalid input accepted')
def diag(n):return [[i,i] for i in range(n)]
def pair(x,mask=None):
 x=np.asarray(x,dtype=float)
 return x,np.ones(x.shape,bool) if mask is None else np.asarray(mask,bool)

def scalar_oracle(q,r,path,core):
 """Scalar loops, independently calculate expected/observed/path-missing cells."""
 x,qm=q;y,rm=r;out=[]
 for j in range(core[0],core[1]+1):
  ids=[int(a) for a,b in path if b==j];dd=[];expected=observed=missing=0
  for k in range(y.shape[1]):
   rv=bool(rm[j,k]) and math.isfinite(float(y[j,k]));expected+=int(rv)
   ds=[]
   for i in ids:
    qv=bool(qm[i,k]) and math.isfinite(float(x[i,k]))
    if rv and not qv:missing+=1
    if rv and qv:ds.append(min(abs(float(x[i,k])-float(y[j,k])),1.0))
   dd.append(max(ds) if ds else None);observed+=int(bool(ds))
  out.append(dict(reference_frame=j,query_frames=ids,difference_by_component=dd,
    expected_components=expected,observed_components=observed,path_missing_cells=missing,
    difference=sum(v for v in dd if v is not None)/observed if observed else None))
 return out

def compare(q,r,path,core,name):
 qbefore=(q[0].copy(),q[1].copy());rbefore=(r[0].copy(),r[1].copy());pbefore=copy.deepcopy(path)
 got=ev.trace(q,r,path,core);want=scalar_oracle(q,r,path,core)
 check(len(got)==core[1]-core[0]+1,name+'/reference_denominator')
 for g,w in zip(got,want):
  for k,v in w.items():
   if k=='difference':check(g[k] is None if v is None else abs(g[k]-v)<1e-12,name+'/'+k)
   elif k=='difference_by_component':
    check(all(a is None if b is None else a is not None and abs(a-b)<1e-12 for a,b in zip(g[k],v)) and len(g[k])==len(v),name+'/'+k)
   else:check(g[k]==v,name+'/'+k)
  check(g['possible_components']==r[0].shape[1],name+'/possible_denominator')
  for pos,i in enumerate(g['query_frames']):
   for k,val in enumerate(g['query_values'][pos]):
    valid=bool(q[1][i,k]) and math.isfinite(float(q[0][i,k]))
    check((val==float(q[0][i,k])) if valid else val is None,name+'/raw_query_value_and_mask')
 check(np.array_equal(q[0],qbefore[0],equal_nan=True) and np.array_equal(q[1],qbefore[1]) and np.array_equal(r[0],rbefore[0],equal_nan=True) and np.array_equal(r[1],rbefore[1]) and path==pbefore,name+'/inputs_unchanged')
 json.dumps(got,allow_nan=False)
 return got

# Identity, speed feature warp, crossing, short changes, component clipping.
r=pair([[i/10,1-i/10] for i in range(7)])
t=compare(r,r,diag(7),[0,6],'identity')
check(all(x['difference']==0 for x in t),'identity/all_zero')
ids=[0,0,1,1,2,2,3,3,4,4,5,5,6]
q=pair(r[0][ids]);p=[[i,j] for i,j in enumerate(ids)]
t=compare(q,r,p,[0,6],'speed_warp')
check(sum(len(x['query_frames']) for x in t)==len(p) and all(x['difference']==0 for x in t),'speed_warp/no_query_links_lost')
constant=pair(np.zeros((5,2)));p=[[0,0],[0,1],[1,2],[2,2],[3,3],[4,4]]
compare(constant,constant,p,[1,3],'reused_query_and_partial_core')
cross=pair([[-1,1],[-.5,.5],[0,0],[.5,-.5],[1,-1]])
check(all(x['difference']==0 for x in compare(cross,cross,diag(5),[0,4],'crossing')),'crossing/signed_zero_is_observed')
r=pair(np.zeros((9,3)));x=r[0].copy();x[3:6,1]=2.5
t=compare(pair(x),r,diag(9),[0,8],'short_change')
check([x['reference_frame'] for x in t if x['difference']>0]==[3,4,5],'short_change/exact_time_extent')
check(all(t[i]['difference_by_component']==[0,1,0] for i in [3,4,5]),'short_change/clip_one_per_component')
left=compare(r,r,diag(9),[0,8],'side_specific_left');right=compare(pair(x),r,diag(9),[0,8],'side_specific_right')
check(all(a['difference']==0 for a in left) and any(a['difference']>0 for a in right),'side_specific/no_opposite_side_propagation')

# Matching displacement does not prove matching positions; raw measurement has no motion verdict.
q=pair(np.arange(6,dtype=float)[:,None]/10+.2);r=pair(np.arange(6,dtype=float)[:,None]/10)
t=compare(q,r,diag(6),[0,5],'position_vs_change')
check(np.allclose(np.diff(q[0],axis=0),np.diff(r[0],axis=0)) and all(abs(x['difference']-.2)<1e-12 for x in t),'position_vs_change/offset_not_direction_error')
check(not any('error' in k or 'correct' in k or 'direction' in k for node in t for k in node),'position_vs_change/no_semantic_or_motion_label')

# Missing one link must remain even if another linked query supports the same cell.
q=pair([[.2,.8],[.4,.9],[.1,.7]],[[1,1],[0,1],[1,1]]);r=pair([[.2,.5],[.1,.7]])
t=compare(q,r,[[0,0],[1,0],[2,1]],[0,1],'partial_link_missing')
check(t[0]['path_missing_cells']==1 and t[0]['observed_components']==2 and t[0]['query_frames']==[0,1],'partial_link_missing/not_erased_by_any_observed')
q=pair([[np.nan,np.inf],[.2,.3]]);r=pair([[0,0],[np.nan,.2]])
t=compare(q,r,diag(2),[0,1],'nonfinite')
check(t[0]['difference'] is None and t[0]['path_missing_cells']==2 and t[1]['expected_components']==1,'nonfinite/null_not_zero')
q=pair([[1,0],[1,0]],[[1,1],[1,0]]);r=pair(np.zeros((2,2)))
t=compare(q,r,diag(2),[0,1],'weighted_components')
weighted=sum(sum(v for v in x['difference_by_component'] if v is not None) for x in t)/sum(x['observed_components'] for x in t)
check(abs(weighted-2/3)<1e-12 and abs(sum(x['difference'] for x in t)/2-.75)<1e-12,'weighted_components/do_not_average_curve_nodes_as_group_score')

# Random numeric/mask cases with independent scalar arithmetic; deterministic seed only.
rng=np.random.default_rng(20261101)
for trial in range(100):
 n=int(rng.integers(2,9));m=int(rng.integers(2,9));d=int(rng.integers(1,7));qpos=rpos=0;p=[[0,0]]
 while qpos<n-1 or rpos<m-1:
  steps=[(a,b) for a,b in [(1,0),(0,1),(1,1)] if qpos+a<n and rpos+b<m]
  a,b=steps[int(rng.integers(len(steps)))];qpos+=a;rpos+=b;p.append([qpos,rpos])
 q=pair(rng.normal(size=(n,d)),rng.random((n,d))>.2);r=pair(rng.normal(size=(m,d)),rng.random((m,d))>.2)
 compare(q,r,p,[0,m-1],'random_scalar')

metric=dict(compared=12,reference_unobserved=0,path_missing_cells=0,sufficient=True)
for range_state,status in [(None,'measurement_only_no_range'),('larger_than_fit_reference_range','observed_difference_to_review'),('within_small_reference_range','within_small_reference_sample')]:
 for missing in ['none','reference','query','sufficient','all']:
  met=copy.deepcopy(metric)
  if missing=='reference':met['reference_unobserved']=4
  if missing=='query':met['path_missing_cells']=1
  if missing=='sufficient':met['sufficient']=False
  if missing=='all':met['compared']=0
  before=copy.deepcopy(met);f=ev.feedback(met,{'state':range_state})
  check(f['status']==(status if missing=='none' else 'comparison_evidence_insufficient'),'feedback/missing_precedence_'+missing)
  check(all(f[k] is None for k in ['educational_correct','educational_error','correction_instruction']) and f['important_difference_proven'] is False and f['allowable_difference_proven'] is False,'feedback/no_educational_verdict')
  check(met==before,'feedback/legacy_metric_unchanged')

# Synthetic canonical source contract: no external/private/source files read.
c=dict(id='clause1',word_id='wordA',runtime_scope='active_reference_context',previous_visual_match_status='direct_fit_anchor',source_span='verified source excerpt',source_url='https://example.invalid/source',
 anchors=[dict(alias='referenceA',role='fit',anchors=[dict(frame_id=2),dict(frame_id=3)])],
 groups=['R_finger_angles14','R_hand_position2d','R_to_L_forearm2d','named_handshape','R_finger_angles14'],
 source_entry_id='entry1',source='synthetic source',unsupported_components=['contact pressure'],mapping_stage_restrictions=['anchor only'])
link=dict(clause_id='clause1',text=c['source_span'],url=c['source_url'],anchor_frames=[2,3],status='direct_fit_anchor')
segment=dict(segment_id='synthetic_s01',core=[1,4],source_links=[link]);clauses={'clause1':c}
before=copy.deepcopy((segment,clauses));cards=ev.source_cards(segment,'referenceA','wordA',clauses)
check((segment,clauses)==before,'source/input_unchanged')
check(cards[0]['applicable_numeric_groups']==['R_shape','R_location','R_to_L_forearm'],'source/side_specific_mapped_groups')
check(cards[0]['context_only_groups']==['named_handshape'],'source/unmapped_named_shape_context_only')
check(all(cards[0][k] is False for k in ['named_shape_verified','source_action_verified','training_target','full_span_semantic_verified']),'source/not_semantic_target')
check(cards[0]['reference_anchor_frames']==[2,3],'source/direct_anchor_only')
for name in ['wrong_word','wrong_reference','dev_reference','different_form','empty_anchors','duplicate_anchors','float_anchor','outside_direct_anchor','outside_core','wrong_text','wrong_url','wrong_status']:
 s=copy.deepcopy(segment);cc=copy.deepcopy(clauses);reference='referenceA';word='wordA'
 if name=='wrong_word':word='wordB'
 elif name=='wrong_reference':reference='another_person_same_word'
 elif name=='dev_reference':cc['clause1']['anchors'][0]['role']='development'
 elif name=='different_form':cc['clause1']['runtime_scope']='inactive_different_form'
 elif name=='empty_anchors':s['source_links'][0]['anchor_frames']=[]
 elif name=='duplicate_anchors':s['source_links'][0]['anchor_frames']=[2,2]
 elif name=='float_anchor':s['source_links'][0]['anchor_frames']=[2.0]
 elif name=='outside_direct_anchor':s['source_links'][0]['anchor_frames']=[4]
 elif name=='outside_core':s['core']=[3,4]
 elif name=='wrong_text':s['source_links'][0]['text']='unrelated meaning'
 elif name=='wrong_url':s['source_links'][0]['url']='https://example.invalid/other'
 elif name=='wrong_status':s['source_links'][0]['status']='invented_verified_status'
 reject(lambda s=s,cc=cc,reference=reference,word=word:ev.source_cards(s,reference,word,cc),'source/reject_'+name)
check(ev.source_cards(dict(core=[1,4]),'referenceA','wordA',clauses)==[],'source/uncovered_stays_uncovered')
check(ev.runs([4,2,3,2,8,10,9])==[[2,4],[8,10]],'runs/duplicate_unsorted_to_ranges')

# Range/path/mask shape exceptions; trace receives core-filtered global path, not a new full alignment.
q=r=pair(np.zeros((3,2)))
for bad in [[],[[0,0],[1.5,1],[2,2]],[[0,0],[-1,1],[2,2]],[[0,0],[3,1]],[[0,0],[0,0],[1,1]],[[0,0],[2,2]],[[1,0],[0,1],[2,2]],[[0,0],[np.nan,1]],[[0,0],[1,np.inf]]]:
 reject(lambda bad=bad:ev.trace(q,r,bad,[0,2]),'validation/invalid_path')
reject(lambda:ev.trace(q,r,[[0,0],[1,1]],[0,2]),'validation/reference_scope_cannot_shrink')
reject(lambda:ev.trace(q,r,diag(3),[2,1]),'validation/reversed_core')
reject(lambda:ev.trace((q[0],np.ones((3,1),bool)),r,diag(3),[0,2]),'validation/broadcast_mask_rejected')

# Static integration review without importing run.py or any legacy/protected data.
run_source=(U/'run.py').read_text(encoding='utf8');ast.parse(run_source)
check("assert g['path']==b['path']" in run_source,'integration/original_global_path_identity_assertion')
check("cnt==metric['compared']" in run_source and "metric['mean_difference']" in run_source and "metric['expected']" in run_source and "metric['path_missing_cells']" in run_source,'integration/reaggregate_against_saved_metrics')
check("if d['condition']=='B_same_path'" in run_source,'integration/reuse_saved_B_decisions')
check("len(results)==36" in run_source and "==117" in run_source,'integration/fixed_pair_stage_denominators')
check("'x'" in run_source and 'threshold_tuning=False' in run_source,'integration/no_output_overwrite_or_tuning')

result=dict(status='PASS' if not failures else 'FAIL',reviewer='same Codex B independent synthetic arithmetic/code review, not human KSL validation',
 checks=checks,failures=failures,categories=dict(categories),random_scalar_cases=100,
 reviewed_files={str(U/n):sha(U/n) for n in ['evidence.py','run.py','PLAN_KO.txt','review/test_evidence.py']},
 resolved_findings=[
  'Empty source anchor lists could previously pass subset checks; now nonempty unique integer direct-fit frames are required.',
  'Source prose/URL are checked against canonical clause source_span/source_url; mismatched or inactive different-form links fail.',
  'Path finite/integer/nonnegative/bounds/monotonicity and mask shapes now validated; reference frames cannot disappear.'
 ],
 implementation_findings=[
  'Trace preserves every original linked query per reference frame, max clipped absolute difference per component, reference expected count and path missing count.',
  'A missing query link remains counted even when another query link observes the same reference component; no missing-to-zero substitution in published values.',
  'Whole-group comparison is insufficient if reference channels are absent or query links missing, even if legacy sufficient is true. Legacy metrics/range states remain visible.',
  'Source mapping is specific to directly anchored fit reference frames. Numeric shape/position/axis/forearm features do not prove named handshape, action, contact, language meaning or required stage.',
  'Position differences are measurements, not derived movement errors; equal movement with offset positions remains nonzero without a direction verdict.',
  'Group score must aggregate observed component cells, not the unweighted mean of plotted reference-node means; current run does the component-weighted aggregation.',
  'Per-forearm source names describe the implemented 2D projection features only; this review does not validate contact or the broader historical 4D design.',
  'Alignment guard status remains separate from numeric/source evidence. Complete observation or small reference range does not prove correct stage or correctness.'
 ],
 scope=dict(synthetic_only=True,new_search=False,new_inference=False,new_training=False,real36_run=False,
  legacy_or_protected_inputs_read=False,user_data_read=False,original_code_modified=False),
 limitations=['Pass counts are implementation tests, not recognition accuracy or language validity.','Saved real-output numerical/source/hash verification is a later gate; this review did not execute run.py.','Synthetic speed warp is a numeric path test, not recaptured real signing speed.'])
dest=U/'review/CODE_REVIEW.json'
if dest.exists():
 preserved=U/'review'/('CODE_REVIEW_PREVIOUS_'+sha(dest)[:12]+'.json')
 if not preserved.exists():preserved.write_bytes(dest.read_bytes())
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps({k:result[k] for k in ['status','checks','failures','categories']},ensure_ascii=False))
