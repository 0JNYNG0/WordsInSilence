from pathlib import Path
import json,hashlib,datetime
V=Path(__file__).resolve().parent;U=V.parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
fit=read(O/'word100_autonomous_localization_20261111/FIT_LABELS.json')['rows']
pretrain=read(O/'word100_candidate_role_learning_20261111/PRETRAIN_CHECK.json')
policy=read(V/'candidate_quality_policy_arithmetic.json')
report=dict(
 status='MEASUREMENT_COMPONENT_IMPLEMENTED_TESTED_NO_SELECTION_OR_COMPLETION_CLAIM',
 reviewer='Separate Codex code/data-contract review; no human expert ground truth, model retraining or encoder inference',
 created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
 implemented=dict(file=str(U/'candidate_motion_evidence.py'),tests=str(U/'test_candidate_motion_evidence.py'),test_count=12,test_exit_code=0,
  test_command='python -B research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/test_candidate_motion_evidence.py',
  public_data_in_tests=False,web_changed=False,models_changed=False,
  api='candidate_motion_evidence(values[T,D], observed[bool T or T,D], timestamps_us[T], a,b); attach_candidate_motion_evidence(candidates,values,observed,timestamps_us)',
  measurements=['Every adjacent observed sample difference and its real elapsed time.','Observed/missing runs per relation component.','Endpoint difference separate from actually connected path variation.','Positive/negative variation; maximum single-step share; effective change-step count.','Actual time gaps retained; no fixed-FPS normalization.','Every candidate including single sample and ineligible candidates retained.'],
  never_claims=['A one-step change is an artifact.','A short change is wrong.','Interpolated positions were observed.','Multiple changing samples prove completion.','A model quality score proves educational correctness.'],
  unknown_outputs=['observation_discontinuity','physical_motion_complete','guide_requirement_satisfied']),
 fit_evidence=dict(source=str(O/'word100_autonomous_localization_20261111/FIT_LABELS.json'),relation_annotations=len(fit),public_clips=len({r['clip_id'] for r in fit}),words=sorted({r['clip_id'].split('_')[2] for r in fit}),record_ids=sorted({r['person'] for r in fit}),
  movement_events=sum(len(r['events']) for r in fit),holds=sum(len(r['holds']) for r in fit),background=sum(len(r['background']) for r in fit),moving_sides=sorted({r['moving'] for r in fit}),
  quality_training_counts=pretrain['quality_classes'],pair_training_counts=pretrain['pair_classes'],
  distinction='33 relation records cover30 public recordings of10 expressions;49 events,28holds,64background intervals.994 quality examples and204pair examples include derived/repeated views and are not independent facts. All current moving-side labels are R; bilateral code symmetry is not bilateral empirical validation.',
  direct_complete_occurrence_sources=[str(V/'obedience_occurrence_labels.json'),str(V/'egypt_occurrence_labels.json')]),
 observation_vs_quality=dict(
  observation='Were successive relation samples observed, with what time gaps and how is variation distributed? No semantic target is required.',
  quality=['0: hold/background according to reviewed event extent annotations.','1: partial fragment inside an observed movement.','2: whole observed movement with start/end inside reviewed uncertainty ranges.','3: multiple movements, or a movement mixed with reviewed nonmotion.'],
  pair=['0: fragments from the same observed physical event.','1: distinct same-axis movement/repetition.','2: different-axis movement.'],
  limitation='Neither target taxonomy explicitly supervises tracking discontinuity. A valid semantic action support interval may include a long terminal hold and be scored0; an edited jump span may be scored1/2. Head outputs remain evidence, not unconditional candidate validity.'),
 cached_head_reconstruction=dict(
  source_module=str(O/'word100_candidate_role_learning_20261111/role_core.py'),segment_module=str(O/'word100_episode_joint_linkage_20261111/joint_core.py'),
  z_layout='z[T,684] = before228 + current228 + after228. current228 = h64 + masked-normalized q100 + fh64.',
  slices=dict(h='z[:,228:292]',q_normalized='z[:,292:392].reshape(T,10,10)',fh='z[:,392:456]'),
  raw_q='role.old.relation_features(x): [T,10,10], relation order axis0..4, sideR thenL. This is deterministic coordinate computation, not encoder inference.',
  segment_call='ob={h:torch.tensor(z[:,228:292][None]), q:torch.tensor(qraw[None]), fh:torch.tensor(saved_fh[None])}; role.segment_features(role.frozen_model(),ob,[(0,a,b,axis*2+(moving==L)),...])',
  segment_dimension='304 = h(mean,first,last):192 + normalizedq(mean,first,last,last-first):40 + fhmean:64 + axis5 + side2 + duration1.',
  no_double_normalization='Never pass cached normalized q100 as raw q into JointNet.segment; it normalizes raw q again. Either recompute raw q from x or assemble304 directly from the already normalized cached q.',
  checks=['Assert saved fh equals z[:,392:456].','Assert normalization and masking of raw q equals z[:,292:392].','Use the identical frozen JointNet42 checkpoint that produced cached h; do not substitute a new encoder and treat its h as interchangeable.','Use matching seed quality/pair checkpoints for any claimed three-seed comparison; current quality probe is seed42 only.'],
  quality_load='RoleAdapter(seed).load_state_dict(R/runs/<seed>/model.pt); model.quality(segment_features).softmax(-1). Latest role continuation keeps these head weights unchanged.',
  pair_input='Concatenate the two304-vectors and existing frame-normalized gap:(second.a-first.b)/100 ->609. Existing pair training is not timestamp-invariant, so retain its contract for reuse and report this limitation.',
  latest_probe='Root has now run quality heads on124 phase candidates in82 exposed inputs. It checked cached h/q/fh reconstruction; this is new head inference, not an encoder replay. This reviewing agent only recalculated stored-score decisions.'),
 fixed_quality_policy_review=dict(
  arithmetic=str(V/'candidate_quality_policy_arithmetic.json'),seed=42,inputs=82,candidates=124,competitive_single_motion_groups=9,
  policy=policy['policy'],changes=policy['changes'],
  conclusion='NOT_ADOPT. It repairs REAL11 close candidate but replaces correctly mapped REAL13 C82..95 with C113..115, whose original frames89,90,115 cross a splice. Normal/speed unchanged selection merely lacks this kind of competing single-motion candidate; it does not establish quality reliability.',
  normal_speed_strong0or3_selected=len(policy['normal_speed_selected_strong_nonmotion_or_multiple']),
  no_posthoc_policy='Do not choose class3-only rejection, a new confidence margin, a minimum length or a special forearm exception to fix these DEV counterexamples.'),
 immediate_next_stage=[
  dict(step=1,title='Measure all fixed candidates',action='Attach the pure measurement output to every existing phase and motion proposal using prior.wrist_series(x,relation_kind,moving,anchor), its boolean observation mask and capture timestamps. Do not change spans, rank, filter or completion.',purpose='Expose measured intermediate variation and missingness without inventing a jump detector.'),
  dict(step=2,title='Check full-motion proposal recall using the existing boundary head',action='Reconstruct EpisodeNet.boundary input [h64,qnorm10,fh64,axis5,side2]=145 from the cached current features. Apply the identical frozen boundary head, then existing candidate_spans(bp,qraw,legacy). No encoder replay is needed.',
       safeguards=['Preserve the original phase pool and short arrival child observations; candidate_spans requires four frames and cannot replace all short evidence.','Keep complete=True as boundary-proposal metadata, not biological or sign completion.','Use source/frame mapping only in evaluation, never as input to proposal or quality.','Keep fragment and fragment-union proposals distinct; no automatic union across unseen gaps.']),
  dict(step=3,title='Separate missing proposals from wrong assignments',action='Audit proposals against direct FIT start/end uncertainty bands, intermediate witnesses and physical occurrence IDs. Report overlap-only, partial dynamic coverage and complete movement support separately. Terminal hold extent is not the whole required movement.',
       diagnostic='On the fixed dynamic DEV probes, source mapping can check if a proposal crosses a synthetic join and whether it contains B/C dynamic witnesses. Correct containment alone is not complete recall; retain that limitation. These scores cannot choose thresholds.'),
  dict(step=4,title='Only then use quality/pair on compatible input extents',action='Assess existing quality/pair on actual movement proposals rather than treating a broad phase interval plus long terminal hold as one motion. Assign role separately, retain contradictory/multiple hypotheses, and evaluate order only on real candidate identities. Do not prefer candidates because their direction matches the requested answer.',
       completion='Partial1 may establish direction or attempted action but cannot establish completion. Arrival child shares its parent and is not another repetition. Static descriptors may validly require a hold, so motion-quality gating never applies to all cards.')],
 independent_verification_plan=[
  dict(group='Pure function software contracts',status='12 tests passed',cases=['One step versus distributed path with same endpoints.','Three observed motion samples in8ms remain present.','Two-sample fast motion stays ambiguous.','Mask gap is never bridged, even with different endpoints.','Time dilation preserves path and rescales rates.','Uneven timestamps remain exposed.','Out-and-back motion retains variation despite zero endpoint difference.','Side reflection preserves support.','Different relation units and masks remain separate.','Nonfinite observations remain missing.','Every candidate kept, no input mutation.','Invalid timestamps/masks/spans rejected.']),
  dict(group='FIT evidence alignment audit',status='next, no new inference here',cases=['All49 annotated motion events,28 holds and64 background intervals; separate onset/end uncertainty from interiors.','Derived within-event partials versus unions of distinct events; keep episode IDs.','Six directly reviewed obedience/Egypt FIT occurrence sets, including movement-to-terminal-hold transitions.','Include other FIT expression structures from10-expression metadata; do not validate only one word.','Additional real left-moving and crossing examples require direct public labels before bilateral empirical claims.']),
  dict(group='Separate arithmetic reviewer',status='root or another agent to inspect',cases=['Mask and timestamp propagation.','No interpolation promoted to observation.','Every original candidate retained.','Cached h/q/fh identity and head checkpoint pins.','No source-index/word exception or expected-order selector.']),
  dict(group='Fresh independent public evaluation',status='reserved, not accessed',cases=['Use already reserved public final clips only after FIT contract and code are frozen.','Report normal/fast/slow/crossing preservation alongside changed direction/order/omission and unknowns.','Natural learner error accuracy still requires independently judged real errors; synthetic splice probes are only diagnostics.'])],
 failure_branches=[
  dict(if_case='No proposal contains directly observed start, intermediate change and end',then='Localization/extent problem. Audit missing onset/offset supervision and new boundary-head recall; improve that supervision if justified. Ranking alone cannot repair absent movement.'),
  dict(if_case='Correct proposal exists but role/quality picks another',then='Assignment problem. Compare actual proposal labels and feature distribution on FIT; preserve alternatives and abstain when conflicting. No expected-order shortcut.'),
  dict(if_case='A candidate contains real movement followed by hold',then='Keep parent interval and split movement evidence from terminal child; do not label whole candidate invalid because quality0 dominates its mean.'),
  dict(if_case='Observed frames have only one jump or a tracking gap',then='Keep observation continuity unknown. Do not claim either valid movement or learner error; a true fast motion can have the same sparse observations.'),
  dict(if_case='Normal fast/crossing evidence regresses',then='Do not deploy; preserve prior behavior and inspect interval/mask loss before changing weights.'),
  dict(if_case='Existing heads fail even on independently verified compatible FIT proposals',then='Only then consider a small supervised adaptation using reviewed extent/identity targets and observation-corruption negatives kept distinct from real motion mistakes. Freeze data/conditions first; do not add epochs because DEV is disappointing.')],
 no_new_learning_needed_for=['Candidate measurement extraction.','Existing cached-context boundary and quality feasibility checks.','Source mapping and direct episode coverage audit.'],
 training_required_only_if='Existing proposal/head failure is demonstrated on compatible direct FIT supervision; current continuity measurement is intentionally not a new learned classifier.',
 g2_status='NOT_PASSED; component and diagnosis are progress, not improved end-to-end feedback or web completion.',
 pins=[pin(p) for p in [U/'candidate_motion_evidence.py',U/'test_candidate_motion_evidence.py',O/'word100_autonomous_localization_20261111/FIT_LABELS.json',O/'word100_candidate_role_learning_20261111/PRETRAIN_CHECK.json',U/'candidate_quality_probe/RESULTS.json',V/'candidate_quality_policy_arithmetic.json']])
(V/'motion_continuity_design.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(report['status'])
