"""Synthetic function checks only: no model forward, training, media, or inference."""
from pathlib import Path
import sys, json, hashlib, copy, ast, contextlib, io, types
sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(U))
import numpy as np
import torch
import learning as L
import feedback as FB

results = []
def record(name, passed, detail):
    results.append(dict(name=name, passed=bool(passed), detail=detail))

# Loss function only, with prescribed logits, never Head.forward.
logits = torch.tensor([[[.2,-.3],[.9,.8]], [[.1,-.7],[.2,.3]]], requires_grad=True)
y = torch.tensor([[[1.,0.],[0.,1.]], [[0.,1.],[1.,0.]]])
m = torch.tensor([[[1,0],[0,1]], [[1,1],[0,0]]], dtype=torch.bool)
w = torch.tensor([[2.,3.],[4.,5.]])
loss = L.loss_fn(logits,y,m,w)
g = torch.autograd.grad(loss,logits,retain_graph=True)[0]
yn = y.clone(); yn[~m] = float('nan')
loss_n = L.loss_fn(logits,yn,m,w)
gn = torch.autograd.grad(loss_n,logits)[0]
record('masked_labels_loss_gradient_invariant', torch.equal(loss,loss_n) and torch.equal(g,gn) and bool((g[~m]==0).all()), {'loss':float(loss.detach())})

p = np.array([[.9,.8],[.1,.3],[.2,.7],[.7,.1]])
yy = np.array([[1,1],[0,0],[1,1],[0,0]])
mm = np.array([[1,0],[1,1],[1,1],[1,1]],bool)
met=L.metric(p,yy,mm)
record('score_confusion_matrix_mask', met['targets'][0]['tp']==1 and met['targets'][0]['tn']==1 and met['targets'][0]['fp']==1 and met['targets'][0]['fn']==1 and abs(met['macro_balanced_accuracy']-.75)<1e-12,met)

def case(support=True, present=True, probability=.1):
    n=5; k=2
    seq=dict(features=np.zeros((n,FB.F.FEATURE_WIDTH)),present=np.full((n,len(FB.F.GROUP_ORDER)),present,bool),frame_id=np.arange(n),timestamp_us=np.arange(n,dtype=np.int64)*33333)
    return dict(seq=seq,indices=np.arange(n),prob=np.full((n,k),probability),support=np.full((n,k),support,bool),full_support=np.full((n,k),support,bool),full_frame_ids=np.arange(n),full_times=seq['timestamp_us'],targets=['R_index_only','L_index_only'],role_evidence=None)

original=case()
base=FB.pair(original,original,'pose_plus_signrep')
missing=copy.deepcopy(original);missing['support'][:]=False;missing['full_support'][:]=False;missing['seq']['present'][:]=False
miss=FB.pair(missing,original,'pose_plus_signrep')
record('query_missing_keeps_reference_denominator', miss['reference_expected_cells']==base['reference_expected_cells'] and miss['missing_cells']==base['reference_expected_cells'] and miss['distance']>=base['distance'], {k:miss[k] for k in ['distance','reference_expected_cells','compared_cells','missing_cells','coverage','status']})

empty=case(False,False)
empty_pair=FB.pair(empty,empty,'pose_plus_signrep')
empty_pair['reference']='empty_1'
empty_pair2=copy.deepcopy(empty_pair);empty_pair2['reference']='empty_2'
different=FB.pair(case(probability=.9),original,'pose_plus_signrep');different['reference']='valid'
ag=FB.aggregate('synthetic',[different,empty_pair,empty_pair2],'pose_plus_signrep')
record('empty_evidence_references_excluded_and_partial_preserved', ag['distance']==different['distance'] and ag['status']=='PARTIAL_EVIDENCE' and ag['excluded_reference_count']==2, {'valid_distance':different['distance'],'empty_status':empty_pair['status'],'empty_distance':empty_pair['distance'],'aggregate_distance':ag['distance'],'aggregate_status':ag['status']})
all_empty=FB.aggregate('synthetic',[empty_pair,empty_pair2],'pose_plus_signrep')
record('all_empty_aggregate_withheld', all_empty['distance'] is None and all_empty['status']=='NO_COMPARISON_EVIDENCE' and all_empty['minimum_reference_coverage'] is None, all_empty)

tree=ast.parse((U/'feedback.py').read_text(encoding='utf-8-sig'))
control_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controls')
groups_node=next(n.value for n in ast.walk(control_fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='groups' for t in n.targets))
groups=eval(compile(ast.Expression(groups_node),'<sensor-groups-expression>','eval'),{'F':FB.F,'side':'R'})
record('sensor_missing_removes_bimanual_group', FB.F.GROUP_ORDER.index('bimanual') in groups, {'removed_groups':[FB.F.GROUP_ORDER[i] for i in groups]})

actual_seq,actual_uniform,actual_read=FB.F.sequence_features,FB.F.uniform_time_sample,FB.read
try:
    n=4; ob=types.SimpleNamespace(usable=np.ones((n,75),bool),xy_px=np.zeros((n,75,2)))
    ob.xy_px[1,0,0]=np.nan;ob.usable[2,2]=False
    fa=dict(frame_id=np.arange(n),timestamp_us=np.arange(n)*33333,mask=np.ones((n,2),bool))
    FB.F.sequence_features=lambda obs:dict(frame_id=fa['frame_id'],timestamp_us=fa['timestamp_us'],feedback_hand_roles=None)
    FB.F.uniform_time_sample=lambda seq,budget:seq
    FB.read=lambda path:dict(targets=['R_face_overlap'])
    fc=FB.prepare_case(dict(feature_names=['R_hand_wrist_x','R_hand_wrist_y']),fa,obs=ob)
    record('face_overlap_requires_usable_finite_face_anchors',fc['full_support'][:,0].tolist()==[True,False,False,True],{'support':fc['full_support'][:,0].tolist()})
finally:
    FB.F.sequence_features,FB.F.uniform_time_sample,FB.read=actual_seq,actual_uniform,actual_read

# Execute orchestration only with mocked loaders, synthetic probabilities, and in-memory saves.
# No checkpoint, media, feature cache, training, or model forward is accessed.
saved={}
real_load,real_save,real_read,real_sha=FB.load_cases,FB.save,FB.read,FB.sha
fake_rows=[dict(alias=f'fit{i}',role='fit',word_id='synthetic',label='synthetic',person=i,source='synthetic') for i in [1,2,3]]
fake_rows += [dict(alias='dev',role='development',word_id='synthetic',label='synthetic',person=4,source='synthetic'),dict(alias='held',role='held_person',word_id='synthetic',label='synthetic',person=5,source='synthetic')]
try:
    FB.save=lambda name,data:saved.__setitem__(name,copy.deepcopy(data))
    FB.read=lambda path:saved['FEEDBACK_FREEZE.json'] if Path(path).name=='FEEDBACK_FREEZE.json' else real_read(path)
    FB.sha=lambda path:'synthetic-protocol-pin' if Path(path).name=='PROTOCOL.json' else real_sha(path)
    for scenario,dev_case,held_case in [('all_empty',empty,empty),('calibration_missing',empty,original),('query_missing',original,empty),('fully_observed',original,original)]:
        saved.clear()
        cc={c:{r['alias']:(dev_case if r['alias']=='dev' else held_case if r['alias']=='held' else original) for r in fake_rows} for c in FB.CONDITIONS}
        FB.load_cases=lambda:({'rows':fake_rows},[],cc)
        FB.calibrate()
        with contextlib.redirect_stdout(io.StringIO()):FB.evaluate()
        frozen=saved['FEEDBACK_FREEZE.json']; output=saved['PAIRED_FEEDBACK.json']
        withheld=scenario!='fully_observed'
        passed=all((r['descriptive_distance_alert'] is None)==withheld for r in output['rows'])
        passed &= all(r['descriptive_distance_withheld_queries']==int(withheld) and r['descriptive_distance_evaluable_queries']==int(not withheld) for r in output['summary'])
        passed &= all((t is None)==(scenario in ['all_empty','calibration_missing']) for t in frozen['thresholds'].values())
        record('mocked_calibrate_evaluate_'+scenario,passed,{'thresholds':frozen['thresholds'],'alerts':[r['descriptive_distance_alert'] for r in output['rows']],'summary':output['summary']})
finally:
    FB.load_cases,FB.save,FB.read,FB.sha=real_load,real_save,real_read,real_sha

info=json.loads((U/'LEARNING_INPUTS.json').read_text(encoding='utf-8-sig'))
byrole={role:[r for r in info['rows'] if r['role']==role] for role in ['fit','development','held_person','held_expression_person']}
fit_ids={r['clip_id'] for r in byrole['fit']}
fit_people={r['person'] for r in byrole['fit']}
fit_words={r['word_id'] for r in byrole['fit']}
record('metadata_split_disjointness', all(fit_ids.isdisjoint({r['clip_id'] for r in byrole[g]}) for g in byrole if g!='fit') and fit_people=={1,2,3} and {r['person'] for r in byrole['development']}=={4} and all({r['person'] for r in byrole[g]}=={5,6} for g in ['held_person','held_expression_person']) and fit_words.isdisjoint({r['word_id'] for r in byrole['held_expression_person']}), {g:{'clips':len(rs),'people':sorted({r['person'] for r in rs}),'words':len({r['word_id'] for r in rs})} for g,rs in byrole.items()})

# No real assemble, Head, train or load_cases call occurs.
out=dict(scope='Pure function synthetic tests and split metadata; zero training/model forward/inference/media access',code_sha256={n:hashlib.sha256((U/n).read_bytes()).hexdigest() for n in ['learning.py','feedback.py']},checks=results)
outpath=U/'review'/'CODE_REVIEW_B_SELFTEST.json'
outpath.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
assert all(r['passed'] for r in results),[r['name'] for r in results if not r['passed']]
review=dict(
    status='PASS_WITH_RESEARCH_LIMITATIONS',
    reviewer='Codex subagent rgb_labels_b; implementation and synthetic verification by same agent, separate from root author',
    scope='learning.py and feedback.py review; no training, actual model forward, checkpoint load, video load, or model inference executed',
    authorization='Parent expanded write scope only for None-distance/partial-evidence fixes in feedback.py; all other writes under review/',
    code_sha256=out['code_sha256'],
    tests=dict(passed=len(results),failed=0,evidence=str(outpath),script=str(Path(__file__).resolve())),
    corrected_findings=[
        dict(id='B01',severity='P1',problem='Zero-comparison reference pairs were included in aggregate distance, allowing unsupported zero distances to dominate the median.',fix_owner='root',verification='A valid synthetic distance 0.2 plus two evidence-free pairs remains 0.2; evidence-free pairs are excluded.'),
        dict(id='B02',severity='P2',problem='Sensor-missing control removed only side-suffixed geometry groups and retained the bimanual group.',fix_owner='root',verification='Actual controls() group-expression AST now includes bimanual in the removed groups.'),
        dict(id='B03',severity='P1',problem='After withholding all-empty aggregates, calibration quantile, distance alert comparison, and summary expected numeric distances or missing coverage keys.',fix_owner='rgb_labels_b',verification='Mocked orchestration tests cover all-empty, calibration-missing, query-missing, and fully observed paths; missing alerts remain null and withheld/evaluable counts are explicit.'),
        dict(id='B04',severity='P2',problem='Dropping evidence-free references could promote the remaining reference to ESTIMATED_COMPARISON and hide excluded references.',fix_owner='rgb_labels_b',verification='Any excluded pair retains PARTIAL_EVIDENCE, excluded/compared reference counts, and minimum coverage across the original pairs.')
    ],
    learning_review=[
        'Pose/RGB normalization statistics and class counts are fit-only; training loss uses fit rows only. No evaluation-driven checkpoint or seed selection found in the inspected path.',
        'Both conditions instantiate the same trainable head, seed, epochs and optimizer. Baseline fixed pose projection adds no new information; equal parameter count does not imply equal independent input rank.',
        'Synthetic unknown-label NaNs leave masked loss and its gradients unchanged; masked logits have zero gradients. Padded targets remain masked by assembly.',
        'Synthetic confusion matrices and macro score agree with explicit arithmetic; metadata confirms 36 fit,12 development,24 held-person,8 held-expression-person with disjoint clip IDs and specified person/word splits.'
    ],
    dtw_review=[
        'The added-term denominator is reference-side support; synthetic query disappearance retains 10 expected cells and marks all10 missing, rather than reducing denominator.',
        'Reference-only resting exclusion is not triggered by query lowering or disappearance. DTW endpoint/path completeness preserves reference frame scope, but does not establish mandatory-stage detection.',
        'Full-query missing support is reported independently of warped alignment. Distance and feature events remain physical model comparison evidence, never an educational verdict.'
    ],
    limitations=[
        dict(topic='support_mask',detail='Support remains a detector-evidence proxy. Root added usable and finite nose/eye anchors (pose0/2/5) to face overlap; the new gate passed a synthetic missing/nonfinite-anchor test. Shape support uses a count of angle masks, not RGB confirmation. Do not describe these masks as direct physical ground truth.'),
        dict(topic='resting_hand',detail='A reference vertical wrist heuristic can suppress all side-specific added terms, including horizontal motion and shape. Low motion or low position does not establish linguistic irrelevance.'),
        dict(topic='metric_definition',detail='When a target or clip contains only one known class, metric() averages only the available recall. Clip macro balanced accuracy is therefore not always a two-class balanced accuracy and must remain distinct from pooled target scores.'),
        dict(topic='loss_definition',detail='Class weights use global fit-cell counts, then loss averages per known clip/target. This is a class-weighted clip/target average, not exact equality of each class contribution after clip normalization.'),
        dict(topic='evaluation_scope',detail='Same-Codex provisional RGB labels, correlated frames, previously used public TRAIN data, fixed normal-reference comparisons and synthetic controls do not measure real educational error precision or linguistic tolerance.')
    ],
    unverified=['Actual six training runs and saved predictions', 'Actual feature runtime and public replay outputs', 'Independent human KSL correctness and true error cases']
)
(U/'review'/'CODE_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'status':review['status'],'tests_passed':len(results),'code_sha256':out['code_sha256'],'review':str(U/'review'/'CODE_REVIEW.json')},ensure_ascii=False))
