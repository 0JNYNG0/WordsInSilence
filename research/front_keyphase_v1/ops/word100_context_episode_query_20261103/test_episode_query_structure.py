"""No model imports/loads/forwards. AST-isolated pure functions + synthetic inputs."""
from pathlib import Path
import ast, copy, hashlib, json
import numpy as np

U=Path(__file__).resolve().parent
O=U.parent
checks=[]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name, fn):
    try:
        assert fn()
        checks.append(dict(name=name,passed=True))
    except Exception as e:
        checks.append(dict(name=name,passed=False,error=type(e).__name__+': '+str(e)))
def rejects(fn):
    try: fn()
    except (ValueError,AssertionError,IndexError): return True
    return False
def functions(path,names,env):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env

ep=U/'episode_query.py'
motion=O/'word100_learned_motion_order_20261102/motion_model.py'
rgb=O/'word100_learned_rgb_motion_20261102/rgb_model.py'
support=O/'word100_process_contract_alignment_20261102/process_rgb_support.py'
env={'np':np}
functions(motion,['normalize'],env)
functions(rgb,['rgb_input'],env)
stree=ast.parse(support.read_text(encoding='utf8'))
assert all(isinstance(n,(ast.ImportFrom,ast.Assign,ast.FunctionDef,ast.Expr)) for n in stree.body)
exec(compile(stree,str(support),'exec'),env)
functions(ep,['integer_axis','prepare_input','select_episode','describe'],env)
prepare=env['prepare_input'];select=env['select_episode'];describe=env['describe']
n=20;ids=np.arange(100,100+n,dtype=np.int64);ts=np.arange(n,dtype=np.int64)*2;ts[8]=15
ws=np.array(env['window_starts'](n),dtype=np.int64)
centers=(ts[ws]+ts[ws+15])/2
wf=np.arange(len(ws)*768,dtype=np.float32).reshape(len(ws),768)/1000
rgb_values=np.stack([np.interp(ts,centers,wf[:,j]) for j in range(768)],axis=1).astype('float32')
a=dict(frame_id=ids,timestamp_us=ts,rgb=rgb_values,image_available=np.ones(n,bool),
       window_starts=ws,window_center_us=centers,window_features=wf)
z=np.arange(n*73,dtype=np.float32).reshape(n,73)/100
mask=np.ones((n,73),bool)
gs=dict(mean=np.linspace(-.2,.2,73).tolist(),std=np.ones(73).tolist())
rs=dict(mean=np.zeros(768).tolist(),std=np.ones(768).tolist())
p=prepare(a,z,mask,gs,rs)
expected=np.concatenate([env['normalize'](z,mask,gs),env['rgb_input'](a,rs)],axis=1)
check('identical_frozen_observer_preprocessing',lambda:np.array_equal(p['x'],expected) and p['x'].shape==(20,915))
check('native_axes_preserved_exact',lambda:np.array_equal(p['ids'],ids) and np.array_equal(p['ts'],ts))
check('exact_center_zero_weight_neighbor_excluded',lambda:p['support']['row_encoded_support'][8]==list(range(16)))
e=select(p,[105,110],0)
check('all_native_rows_selected_no_filter',lambda:e['indices'].tolist()==list(range(5,11)) and len(e['sequence'])==6)
check('RGB_auxiliary_support_not_nominal_truth',lambda:bool(e['rgb_support']['outside_nominal_consumed_rows']) and e['rgb_support']['structural_context_is_direct_nominal_evidence'] is False)
high=np.array([.9,.1,.1,.1,.1,.9])
out=describe(p,e,high)
check('raw_observed_separate_from_scope_attribution',lambda:out['raw_observer_status']=='DIRECTED_MOTION_OBSERVED' and out['scope_attribution']=='NEURAL_CONTEXT_EVIDENCE_NOT_EXACT_FRAME_TRUTH')
check('no_completion_or_correction_promotion',lambda:out['complete_action'] is None and out['whole_clause_correctness'] is None and out['educational_error'] is None and not out['correction_enabled'])
check('native_output_times_exact',lambda:out['timestamp_us']==ts[5:11].tolist() and out['frame_ids']==ids[5:11].tolist())
a_gap=copy.deepcopy(a);a_gap['image_available'][5:8]=False
pg=prepare(a_gap,z,mask,gs,rs);eg=select(pg,[105,110],0);og=describe(pg,eg,high)
check('missing_rows_retained_and_reported',lambda:len(eg['sequence'])==6 and og['unavailable_frame_ids']==[105,106,107])
check('high_probability_cannot_override_missing_evidence',lambda:og['raw_observer_status']=='MOTION_OBSERVATION_INSUFFICIENT')
check('unavailable_nominal_RGB_consumption_empty',lambda:all(pg['support']['row_consumed_support'][i]==[] for i in [5,6,7]))
check('neighbor_context_retains_encoded_blackout_support',lambda:5 in pg['support']['row_encoded_support'][8] and 105 in eg['rgb_support']['synthetic_context_source_coordinates'])
a_nan=copy.deepcopy(a);a_nan['rgb'][5]=np.nan
pn=prepare(a_nan,z,mask,gs,rs)
check('RGB_only_nonfinite_preserves_geometry_branch',lambda:pn['mask'][5].all() and np.array_equal(pn['x'][5,:146],p['x'][5,:146]) and not pn['available'][5])
check('RGB_only_nonfinite_has_no_RGB_consumed_support',lambda:pn['support']['row_consumed_support'][5]==[])
z_changed=z.copy();z_changed[:,63:]=999
p_changed=prepare(a,z_changed,mask,gs,rs)
check('tested_feature_values_do_not_select_scope',lambda:np.array_equal(select(p_changed,[105,110],0)['indices'],e['indices']))
for name,bounds in [('single',[105]),('reversed',[110,105]),('outside',[99,105]),('equal',[105,105])]:
    check('reject_'+name+'_scope',lambda b=bounds:rejects(lambda:select(p,b,0)))
for bad in [True,-1,5,0.0]:
    check('reject_target_'+str(bad),lambda b=bad:rejects(lambda:select(p,[105,110],b)))
ids_gap=copy.deepcopy(a);ids_gap['frame_id'][10:]+=1
pp=prepare(ids_gap,z,mask,gs,rs)
check('missing_frame_ID_not_silently_compressed',lambda:rejects(lambda:select(pp,[105,111],0)))
for name,values in [('float',[1.,2.]),('bool',[False,True]),('negative',[-1,2]),('duplicate',[1,1])]:
    check('reject_axis_'+name,lambda v=values:rejects(lambda:env['integer_axis'](v,'test')))
for name,field,value in [('bad_centers','window_center_us',centers+1),('bad_starts','window_starts',ws+1)]:
    aa=copy.deepcopy(a);aa[field]=value
    check(name,lambda aa=aa:rejects(lambda:prepare(aa,z,mask,gs,rs)))
aa=copy.deepcopy(a);aa['rgb'][5,0]+=1
check('reject_RGB_values_not_from_declared_windows',lambda:rejects(lambda:prepare(aa,z,mask,gs,rs)))
aa=copy.deepcopy(a);del aa['window_features']
check('reject_partial_window_metadata',lambda:rejects(lambda:prepare(aa,z,mask,gs,rs)))
recipe_a={k:v for k,v in a.items() if not k.startswith('window_')}
check('reject_missing_encoder_recipe',lambda:rejects(lambda:prepare(recipe_a,z,mask,gs,rs)))
recipe=dict(route='public_inputs_16_stride2_center_interp',frames=n,manifest_sha256='synthetic',code_pins={'synthetic':'synthetic'})
pr=prepare(recipe_a,z,mask,gs,rs,recipe)
check('recipe_only_lineage_explicit_not_exact_values',lambda:pr['rgb_lineage']=='VERIFIED_GENERATOR_RECIPE_WITHOUT_SAVED_WINDOW_VALUES' and p['rgb_lineage']=='SAVED_WINDOWS_AND_VALUES_EXACTLY_RECONSTRUCTED')
check('recipe_support_same_structural_algorithm',lambda:pr['support']==p['support'])
bad_recipe=dict(recipe,frames=n-1)
check('reject_recipe_frame_count_mismatch',lambda:rejects(lambda:prepare(recipe_a,z,mask,gs,rs,bad_recipe)))
a_short={k:v[:15] for k,v in recipe_a.items()}
check('short_full_cache_route_rejected_no_invented_padding',lambda:rejects(lambda:prepare(a_short,z[:15],mask[:15],gs,rs,dict(recipe,frames=15))))
check('two_native_frame_episode_preserved_but_insufficient',lambda:describe(p,select(p,[105,106],0),high)['raw_observer_status']=='MOTION_OBSERVATION_INSUFFICIENT')
check('three_frame_episode_old_gate_retained',lambda:describe(p,select(p,[105,107],0),high)['raw_observer_status']=='DIRECTED_MOTION_OBSERVED')
low=high.copy();low[0]=.2
check('low_probability_does_not_prove_error_or_absence',lambda:describe(p,e,low)['raw_observer_status']=='EXPECTED_MOTION_UNCONFIRMED')
pc=high.copy();pc[5]=.799
check('continuity_gate_preserved',lambda:describe(p,e,pc)['raw_observer_status']=='MOTION_CONTINUITY_UNCONFIRMED')
for bad in [np.ones(5),np.array([np.nan]*6),np.array([1.1]*6),np.array([-.1]*6)]:
    check('reject_invalid_prediction_'+str(len(checks)),lambda b=bad:rejects(lambda:describe(p,e,b)))

# Read/hash only the explicitly fixed public input/cache identities; no model data loaded.
contract=json.loads((U/'FROZEN.json').read_text(encoding='utf8'))
trusted_path=O/'word100_component_rgb_context_20261102/SCOPE_FROZEN.json'
trusted=json.loads(trusted_path.read_text(encoding='utf8'))
for row in contract['rows']:
    native=np.load(row['input'],allow_pickle=False)
    check('input_hash_'+str(row['person']),lambda r=row:sha(r['input'])==r['input_sha256']==trusted['pins'][r['input']])
    for kind,path in [
      ('shape',O/'word100_multipeople_live_feedback_20261101/bank_predictions'/f"{row['clip_id']}.npz"),
      ('relation',O/'word100_learned_body_relation_20261102/runs/rgb_geometry_42/predictions'/f"{row['clip_id']}.npz")]:
        observed=np.load(path,allow_pickle=False)
        check('trusted_'+kind+'_hash_'+str(row['person']),lambda path=path:sha(path)==trusted['pins'][str(path)])
        check(kind+'_native_axes_'+str(row['person']),lambda s=observed,a=native:np.array_equal(s['frame_id'],a['frame_id']) and np.array_equal(s['timestamp_us'],a['timestamp_us']))
# Assert no production model module was imported, constructed or invoked.
import sys
check('no_model_module_imports',lambda:not any(k in sys.modules for k in ['torch','rgb_model','motion_model','path_model']))
result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',
 checks=checks,check_count=len(checks),failures=sum(not c['passed'] for c in checks),
 model_imports=0,model_loads=0,model_forwards=0,media_views=0,
 method='AST-isolated pure production functions; synthetic arrays plus fixed public native/cache identity checks',
 code_pins={str(p):sha(p) for p in [ep,U/'run_diagnostic.py',motion,rgb,support,Path(__file__)]},
 limitations=['Recipe-only RGB route is structural provenance, not reconstruction of unsaved feature window values.','These checks establish no classifier accuracy or educational correctness.'])
dest=U/'REVIEW_STRUCTURAL_TESTS.json'
with dest.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(status=result['status'],checks=len(checks),failures=result['failures'],report_sha256=sha(dest))))
if result['failures']:raise SystemExit(1)

