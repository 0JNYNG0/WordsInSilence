"""Pre-execution static/pure-array review. No model creation/forward/training."""
from pathlib import Path
import sys, ast, json, hashlib, importlib.util
from collections import Counter
from datetime import datetime
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parents[1]; W=U/'local_motion_quality'; O=U.parent
sys.path.insert(0,str(W))
import labels, model, prepare
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
code={n:(W/n).read_text('utf-8') for n in ['model.py','labels.py','prepare.py','train.py']}
for s in code.values():ast.parse(s)
contract=json.loads((W/'CONTRACT.json').read_text('utf-8'))
anns, selected, caches=prepare.selected_metadata()
tests=[]
def check(name,actual,expected):
    tests.append(dict(name=name,passed=actual==expected,actual=actual,expected=expected))

example=dict(clip_id='public_fixture',axis=0,moving='R',events=[dict(onset=[2,3],offset=[7,8])],holds=[[8,10]],background=[[0,1]])
check('full_boundary_range_is_movement_extent',labels.assign_extent(2,8,0,'R',[example])[0],2)
check('within_event_is_fragment_not_wrong_sign',labels.assign_extent(3,6,0,'R',[example])[0],1)
check('explicit_hold_is_zero',labels.assign_extent(9,10,0,'R',[example])[0],0)
check('unannotated_left_is_unknown_not_zero',labels.assign_extent(3,6,0,'L',[example])[0],-1)
check('motion_plus_terminal_hold_unknown',labels.assign_extent(2,10,0,'R',[example])[0],-1)
check('ambiguous_hold_and_event_point_unknown',labels.assign_extent(8,8,0,'R',[example])[0],-1)
both=dict(example,events=example['events']+[dict(onset=[12,13],offset=[17,18])])
check('two_distinct_reviewed_events_is_three',labels.assign_extent(2,18,0,'R',[both])[0],3)
touch=dict(example,events=[dict(onset=[2,4],offset=[4,6])],holds=[],background=[])
check('one_frame_does_not_witness_motion_extent',labels.assign_extent(4,4,0,'R',[touch])[0],-1)

# Packing tests use invented arrays, not public/private recordings or models.
v=np.tile(np.arange(5,dtype=np.float32)[:,None],(1,4));good=np.array([1,1,0,1,1],bool);ts=np.array([0,33333,66667,120000,153333])
seq=model.make_sequence(v,good,ts)
check('missing_is_masked_not_observed_zero',seq[2,:4].tolist()+seq[2,8:10].tolist(),[0.,0.,0.,0.,0.,0.])
check('no_delta_bridged_across_missing',seq[3,4:10].tolist(),[0.,0.,0.,0.,1.,0.])
check('valid_next_pair_delta_preserved',seq[4,4:10].tolist(),[1.,1.,1.,1.,1.,1.])
crop=model.raw_sequence(v,good,ts,3,4)
check('crop_first_delta_and_dt_zero',crop[0,4:8].tolist()+crop[0,9:11].tolist(),[0.,0.,0.,0.,0.,0.])
check('irregular_actual_elapsed_time_preserved',bool(np.isclose(seq[3,10],.053333)),True)

unique={}
for cid in sorted({a['clip_id'] for a in anns}):
    az=[a for a in anns if a['clip_id']==cid]
    for c in labels.direct_candidates(az):
        key=(cid,c['a'],c['b'],c['axis'],c['moving'])
        y,eids,reason=labels.assign_extent(c['a'],c['b'],c['axis'],c['moving'],az)
        unique[key]=dict(y=y,reason=reason,episode_ids=eids)
singlefull=[list(k) for k,v in unique.items() if k[1]==k[2] and v['y']==2]
check('original_direct_candidates_have_no_single_frame_full_extent',len(singlefull),0)

root07=[r['clip_id'] for r in selected if r['person']==7]
check('REAL07_group_holdout_four_clips',len(root07),4)
check('all_FIT_metadata_30_clips',len(selected),30)
check('normal_only_context_caches_60',len(caches),60)
check('source_sha_groups_30',len({r['source_sha256'] for r in selected}),30)
auxfile=W/'AUX_ADMISSIONS.json'
auxrows=json.loads(auxfile.read_text('utf-8')).get('rows',[]) if auxfile.exists() else []
check('current_auxiliary_disabled',len(auxrows),0)

checks=dict(
    source_fit_filter_before_array_load=True,
    no_source_or_expected_order_neural_input=True,
    qnorm_reconstruction='Cached current z slices228:292(h),292:392(qnorm),392:456(fh); source q recomputed from unchanged physical full/upper; allclose1e-6 checks exact relation/view consistency.',
    boundary_reconstruction='Existing frozen J42 boundary head on145 features. This is one allowed new FIT head-only preparation inference; audit did not run it.',
    quality_summary='Frozen net.segment reconstructs304 features from saved source latents and q; old quality head not rerun.',
    latent_cache_integrity='R normal full/upper NPZ SHA checked; no reordering or common encoder inference for current no-aux run.',
    all_valid_exclusion='A candidate with any missing body-derived relation sample is excluded and listed in UNKNOWN_CANDIDATES. This strict observed subset cannot demonstrate missing-pose recovery.',
    denominator='SOURCE_INPUT_AUDIT has raw/unique pools per clip/view; result and unknown rows preserve disposition. Pre-training arithmetic must establish unique_pool = accepted + unknown.',
    train_only_weights='counts and inverse-square-root classweight use train list only. Probe never appears in optimizer batches.',
    fixed_budget='Frozen40epochs, matching conditions/seeds/batch order/last-epoch checkpoints. Additional local C parameters are the intended architecture difference.',
    no_fresh_test='Record07 is excluded only from this added fitting. Source common encoder, localizer AND copied quality initialization saw these records previously.',
    masks_and_padding='make_sequence never bridges missing values; packed GRU uses true length; current complete-observation admission yields observed1 within all training candidates.',
    auxiliary='No current admissions -> all masks0. Untrained auxiliary output ignored. Future edited branch and true sensor-error detection are not admitted by this review.'
)
findings=[
 dict(id='FIT_ADDITIONAL_ROLE',severity='blocking_before_prepare',description='Six original rows use fit_additional; permit both FIT annotation tags then enforce READY.fit/source SHA.',resolved="in ('fit','fit_additional')" in code['prepare.py']),
 dict(id='ZERO_DURATION_EXTENT',severity='blocking_before_prepare',description='Original uncertain boundary combinations admitted two single-frame class2 targets at WORD0003_REAL03 60/66. Preserve source, exclude these candidates as unknown.',resolved=not singlefull),
 dict(id='BOUNDARY_MODEL_PIN',severity='blocking_before_prepare',description='Pin the actual J/runs/42/model.pt used by role.frozen_model, not only quality R and localizer L weights.',resolved="J/'runs/42/model.pt'" in code['prepare.py'] or "J/'runs'/'42'/'model.pt'" in code['prepare.py'] or "word100_episode_joint_linkage_20261111/runs/42/model.pt" in code['prepare.py']),
 dict(id='INITIAL_QUALITY_EXPOSURE',severity='reporting',description='REAL07 was also used by copied quality initialization; incremental-fitting probe, not independently unseen-head generalization.',resolved='quality' in contract['scope']['prior_encoder_exposure'].lower()),
 dict(id='COPY_PREPROCESS_CLAMP',severity='consistency_recommendation',description='Original Localizer clamps normalized inputs to[-12,12]. Match that preprocessing when copying its temporal branch.',resolved='clamp(-12, 12)' in code['model.py'] or 'clamp(-12,12)' in code['model.py']),
 dict(id='TRAIN_ADMISSION_DIGESTS',severity='blocking_before_training',description='Bind admission to exact FROZEN and INPUTS_COMPLETE digests, not only approved boolean; verify both at train start.',resolved=('frozen_sha' in code['train.py'] and ('inputs_sha' in code['train.py'] or 'manifest_sha' in code['train.py'])))
]
out=dict(status='PREPARATION_REVIEW_PENDING_FIXES' if any(not f['resolved'] and f['severity']=='blocking_before_prepare' for f in findings) or any(not t['passed'] for t in tests) else 'PREPARATION_CODE_REVIEW_PASSED_TRAINING_REQUIRES_DATA_AUDIT',
 timestamp=datetime.now().astimezone().isoformat(),
 reviewer='Separate Codex code and deterministic metadata/array audit, not independent human ground truth',
 scope=dict(model_instantiation=0,model_inference=0,training=0,prepare_run=0,freeze_run=0,new_raw_video_read=0,private_data_read=0),
 findings=findings,pure_checks=tests,checked_contract=checks,
 direct_candidate_metadata=dict(unique_candidates=len(unique),class_counts=dict(Counter(v['y'] for v in unique.values())),
    unknown_reasons=dict(Counter(v['reason'] for v in unique.values() if v['y']<0)),single_frame_full_candidates=singlefull,
    actual_boundary_proposals_not_generated=True),
 split_metadata=dict(train_clips=26,head_probe_clips=4,probe_clips=root07,source_sha_duplicates=0,
    every_axis_view_fragment_edit_inherits_clip_split=True,prior_quality_initialization_exposed=True),
 pending_after_prepare=['Verify FROZEN code/model pins and exact selected public metadata.',
    'Independently reconstruct labels and mask disposition for every saved sample/unknown; include all-valid exclusion denominator.',
    'Check source original summary/qnorm/fh, raw sequence masks/deltas/timestamps and candidate source IDs.',
    'Confirm aux_mask0 everywhere, source SHA split disjoint, same B/C dataset and train-only class weights.',
    'Only then bind TRAIN_ADMISSION to FROZEN and INPUTS_COMPLETE hashes.'],
 code_sha256={str(W/n):sha(W/n) for n in [*code,'CONTRACT.json']})
(U/'reviews/local_quality_code_review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:out[k] for k in ['status','findings','direct_candidate_metadata']},ensure_ascii=False,indent=2))
