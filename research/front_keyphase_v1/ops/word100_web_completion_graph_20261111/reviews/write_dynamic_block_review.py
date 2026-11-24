"""Assemble review conclusions with the independently checked saved arithmetic."""
from pathlib import Path
import json,hashlib
V=Path(__file__).resolve().parent;U=V.parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=read(V/'dynamic_block_arithmetic.json')
def ref(p,line):return dict(path=str(p),line=line,sha256=sha(p))
r=dict(
 status='ARITHMETIC_PASS_LIMITED_DYNAMIC_ASSIGNMENT_GAIN_G2_NOT_PASSED',
 reviewer='Separate Codex source/code/array audit. Inherits source RGB review from another Codex agent; not independent human or KSL expert ground truth.',
 scope=dict(public_source_clips=4,limited_dynamic_order_clips=3,source_only_egypt_clips=1,conditions=3,seeds=[42,43,44],saved_records=36,limited_pair_records=27,
            new_inference=False,new_training=False,private_access=False,source_files_modified=False,web_changed=False,dev_threshold_tuning=False),
 arithmetic=dict(status='PASS',errors=a['errors'],report=str(V/'dynamic_block_arithmetic.json'),sha256=sha(V/'dynamic_block_arithmetic.json'),
  checks=['Target and driver frozen hashes match.', 'Four public video hashes match source records.', 'Every source frame appears exactly once; query/source maps are inverse.',
   'Each block preserves internal time direction, and original return is retained in suffix.', 'Saved x exactly equals old x indexed by frozen permutation; timestamps stay increasing.',
   'Saved fh and contextual z are not merely old features shuffled. Driver recomputed frozen encoders on the new input.',
   'All 36 saved result records, peak selections, 27 limited pair states, splice flags and containment flags agree with recomputed arithmetic.']),
 result=a['summary'],
 result_interpretation={
  'gain':'For the three previously exposed obedience DEV recordings, source and continued roles locate reversed B/C pair fragments in 1/3 at each seed; reviewed roles locate them in 2/3 for seeds42/43 and 1/3 for44. This is one additional recording at two initializations, not a three-seed gate pass.',
  'denominator':'27 pair records repeat the same three recordings under three model conditions and three seeds. They are not 27 independent human errors.',
  'mapped_is_not_complete':'spans_within_target_occurrences checks containment only. In reviewed42, REAL12 selects source61..69 and77..97; REAL13 selects80..88 and96..109. These do not include every direct start/end witness. Correctly locating action fragments and their order does not prove complete action coverage.',
  'synthetic_scope':'Only the expressly defined close-and-lower versus forearm-reorientation dynamic pair has a negative order target. Static open-front and Egypt source block order are not equivalent semantic negative targets. No natural learner error accuracy or whole-sign educational correctness is supplied.',
  'role_taxonomy':'Current role continuation retains64/67 under the same inherited8-family taxonomy at all six condition/seed runs. Raw11-output versus old8-gold differences are not regressions or improvements; preparation/return/transition collapse to old nonmain0 for that comparison.'},
 real11_failure={
  'candidate_generation':'The true B candidate exists. This is primarily unsupported candidate selection, not absence of a B proposal.',
  'selected_close':dict(query=[64,70],source=[64,65,66,80,81,82,83],score=.9999978542307346),
  'unselected_actual_B':dict(query=[94,101],source=list(range(70,78)),score=.9999582766804138),
  'score_gap':.9999978542307346-.9999582766804138,
  'explanation':'The selected close candidate skips all source B67..79, crosses source66→80, and contains early C frames. The selected C71..89 maps84..102 and contains late real C motion plus terminal hold. Their apparent B-before-C order therefore depends on assigning the artificial jump to B.',
  'geometry':'Selected close has72.96% of its absolute height variation in one synthetic adjacent step66→80, versus22.22% for the largest step in the true continuous B candidate. These are descriptive audit values, not proposed thresholds.',
  'second_spurious_candidate':'Other raise candidate102..107 maps78,79,104,105,106,107;87.46% of its forearm-upright variation is source79→104 splice. It must not count as performed upward reorientation.',
  'selection_cause':'occurrence_binding selects max peak sqrt(phase*role). Both scores are saturated; tiny peak preference is not credible evidence of physical motion. The independent order head cannot repair the wrong interval identity by itself.',
  'source_indices_use':'Source indices prove the editing artifact in this audit only. They are not available as normal inference evidence and must never become runtime filters.'},
 existing_components={
  'dynamic_driver':'Calls only phase/role interval selection and fixed guide order head. It does not call current event quality, event pair, bound movement or short-motion rescue on the selected dynamic spans. No saved quality output exists in this diagnostic.',
  'geometry_availability':'Current geometry requires coverage>=.8, four valid observations, both endpoints observed, then resamples to nine positions. Both REAL11 close candidates pass this availability contract. Many valid samples before and after a jump do not demonstrate observed intermediate movement.',
  'direction_hold':'Existing signed-direction gate checks endpoint sign agreement; hold always agrees. Short-motion rescue can replace a confident hold with signed motion. A jump can have the expected endpoint sign, so these checks do not guarantee continuity rejection. The current diagnostic does not save the neural direction output; whether a frozen direction head happens to predict hold here is unverified.',
  'quality_target':'quality classes are reviewed nonmotion/background(0), a fragment within an event(1), full event with reviewed endpoints(2), multiple movements/nonmotion contamination(3). They are not a classifier for reliable continuous motion versus tracking jump.',
  'quality_features':'304-dimensional feature combines context h mean/start/end, normalized body relation mean/start/end/delta, mean common100 features, axis/side and duration. Context can contain trajectory evidence, but no explicit contract requires intermediate samples or forbids one-step jumps.',
  'pair_target':'pair classes are fragments of the same observed event, separate events/repetitions on the same axis, and different-axis events. This can help avoid merging repetitions, but does not certify an across-splice movement as real or false.',
  'continuation_weights':'Latest continued_old_direct and reviewed_roles only update the role MLP; quality and pair parameters remain unchanged. Any new quality integration is a new, as-yet unmeasured candidate.',
  'training_artifact_question':'Old role preparation includes core reversal and retains source role labels without a seam-specific unknown mask. It therefore exposes the role classifier to contextual artificial joins. This is a plausible distribution weakness, not proof that the movement model was explicitly taught a jump as correct motion. Its complete quality targets require both endpoints inside one mapped reviewed event; transformed events straddling the reversal boundary are dropped. The saved manifests do not establish a seam-crossing complete positive target.'},
 proposed_common_implementation=[
  dict(step=1,title='Separate candidate identity from observed motion support',implementation='Retain every phase occurrence and its full span. Attach side, relation kind, observed-index set, motion-support intervals, boundary uncertainty, shape evidence and event quality separately. Never assign an action based on expected chronological position. Keep multiple incompatible candidates ambiguous.',
       reason='REAL11 already contains the correct B proposal; peak-only commitment discards it before evidence comparison.'),
  dict(step=2,title='Add a target-independent motion observation contract',implementation='Use actual timestamps, body-relative wrist/elbow relations, validity and adjacent observed increments. Record whether progress is distributed across observed intermediate states, concentrated in a single step, or separated by missing data. Preserve raw observation and interpolated estimates separately. Do not infer actual intermediate motion from nine resampled values alone.',
       decision='A suspected discontinuity should reduce observation confidence or keep correspondence unknown; it must not become a user wrong-direction warning. A fast real movement must not be rejected merely because it is short or has a large velocity.',
       fit_need='Freeze this contract from public FIT full actions, partial actions, terminal holds, fast/short changes, crossings and controlled observation drop/jump diagnostics. Never select a cutoff from these three DEV recordings.'),
  dict(step=3,title='Use existing quality/pair as additional, tested evidence',implementation='On frozen FIT candidate ledgers, check that full/partial/multiple/hold labels match the actual dynamic descriptor and preserve episode IDs. Test the existing heads before deciding to adapt. Evaluate quality on the same candidate type that production will use. A role score or complete-quality score alone cannot establish guide completion.',
       next_training_if_needed='If FIT evidence shows a discontinuity gap, train a small separate observation-support output or add explicitly reviewed observation-unreliable labels while retaining full/partial/multiple targets. Artificial seams are observation corruption examples, not performed sign mistakes. Recompute encoder features for every new sequence; mask seam-derived role/motion targets as appropriate. Continue all existing shape/direction normal regressions.'),
  dict(step=4,title='Resolve occurrences before reporting order/completion',implementation='Select by supported physical occurrence and descriptor association, not desired order. Connect arrival child to its actual parent occurrence and count it once. Preserve repeated/interleaved possibilities. Judge order only after both occurrences have sufficient observed dynamic evidence; distinguish partial attempt, observed endpoint, missing observation and omitted action.',
       falsifiable_checks='A jump between two holds is not supported motion; deleting observations never invents completion; actual fast continuous actions keep their evidence; true reverse direction remains visible; swapping complete FIT blocks changes occurrence order without reversing their internal action; no word-ID exceptions.'),
  dict(step=5,title='Freeze and evaluate without repeated DEV selection',implementation='After FIT contract review, pin code/data/weights and run the new candidate once. Reuse unchanged completed baseline caches. Compare normal/slow/fast/crossing, short changes, omission/masking, direction and complete-block order separately. Report errors, unknowns, valid recovery and three-seed behavior. Reserve fresh public final evaluation for after this gate.')],
 g2=dict(status='NOT_PASSED',already_supported=['Saved input/result arithmetic is consistent.', 'Review changes improve one dynamic pair correspondence at seeds42/43 while same8-family64/67 and normal24-card evidence are preserved in prior audits.'],
  missing=['REAL11 still supports an apparent normal pair using a splice as close action.', 'Seed44 has no improvement on these three dynamic probes.', 'Correctly mapped fragments are not complete action boundary evidence.', 'Multiple candidate/parent-arrival/repetition ambiguity is not resolved by max-peak selection.', 'Observation-quality compatibility and normal fast/short/crossing preservation after new integration are not measured.', 'No new held-out natural learner or expert educational-error validation exists.'],
  minimum='Before G2 pass, demonstrate supported correspondence recovery rather than only increased abstention, preserved normal evidence and no artificial-jump support under the frozen three-seed contract. A one-record gain on exposed synthetic inputs alone is insufficient.'),
 evidence=[
  ref(U/'occurrence_binding.py',29),ref(U/'interval_pipeline.py',14),ref(U/'run_dynamic_block_diagnostic.py',47),
  ref(O/'word100_broad_factorized_learning_20261110/core.py',47),ref(O/'word100_short_motion_completion_20261111/motion_core.py',26),
  ref(O/'word100_episode_joint_linkage_20261111/joint_core.py',23),ref(O/'word100_episode_joint_linkage_20261111/proposal_adapter.py',5),
  ref(O/'word100_candidate_role_learning_20261111/prepare_train.py',16),ref(U/'train_role_correction.py',24),
  ref(U/'occurrence_supervision/EVALUATION_TAXONOMY_CORRECTION.json',1)],
 limitations=['No new neural forward pass was performed; quality/direction behavior on these exact new spans is not experimentally established.', 'Coordinate-based step concentration is an audit description, not a human biomechanics limit.', 'Direct RGB source observations are inherited and are not independent expert annotations.', 'Known historical reversed-core9 targets were corrected; this review does not revive that invalid all-negative denominator.'])
(V/'dynamic_block_verification.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not r['arithmetic']['errors']
print(r['status'])
