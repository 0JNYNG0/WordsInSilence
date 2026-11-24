"""Write a descriptive diagnosis from saved results; never import model/evaluator."""
from pathlib import Path
from collections import Counter
import hashlib
import json

U=Path(__file__).resolve().parent
A=U.parent/'word100_action_completion_supervision_20261102'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    cp=U/'ALIGNMENT_CAUSE_COUNTS.json';ip=U/'ALIGNMENT_FALSE_INPUT_AUDIT.json'
    counts,availability=read(cp),read(ip)
    evaluation=read(U/'EVALUATION.json');old=read(A/'PROCESS_EVALUATION.json')
    old_by_id={r['view_id']:r for r in old['rows']}
    targets={r['view_id']:r for r in read(U/'DEV_VIEW_TARGETS_V2.json')['rows']}
    decision=read(U/'DECISION.json')
    assert availability['evaluation_sha256']==sha(U/'EVALUATION.json')
    changes=[]
    for r in evaluation['rows']:
        if r['role']!='development' or r['origin']!='reviewed_view' or r['truth'] is None:continue
        b=old_by_id[r['case_id']]
        before=(b['probability']>=.5)==bool(b['truth'])
        after=(r['probability']>=.5)==bool(r['truth'])
        if before!=after:
            changes.append(dict(case_id=r['case_id'],query_id=r['query_id'],truth=r['truth'],
                before_probability=b['probability'],after_probability=r['probability'],change='improved' if after else 'regressed'))
    points={}
    for role in ['fit','development']:
        rows=[r for r in evaluation['rows'] if r['role']==role and r['origin']=='reviewed_view' and r['kind']=='native']
        points[role]={}
        for side in ['setup','terminal']:
            pairs=[]
            for r in rows:
                now={p['source_frame']:p for p in r['sparse_point_scores']}
                for p in old_by_id[r['case_id']]['sparse_point_scores']:
                    truth=p[side+'_truth']
                    if truth is None:continue
                    n=now[p['source_frame']];assert n[side+'_truth']==truth
                    pairs.append((truth,p[side+'_probability'],n[side+'_probability']))
            points[role][side]={str(t):dict(known_count=sum(x==t for x,b,n in pairs),
                AB_ge_08=sum(x==t and b>=.8 for x,b,n in pairs),AC_ge_08=sum(x==t and n>=.8 for x,b,n in pairs)) for t in [0,1]}
    nominal=[p for r in evaluation['rows'] for p in r['nominal_process_scores']]
    nominal_totals=dict(total=len(nominal),known=sum(p['truth'] is not None for p in nominal),
       positive=sum(p['truth']==1 for p in nominal),positive_ge_08=sum(p['truth']==1 and p['probability']>=.8 for p in nominal),
       negative=sum(p['truth']==0 for p in nominal),negative_le_02=sum(p['truth']==0 and p['probability']<=.2 for p in nominal),
       unknown=sum(p['truth'] is None for p in nominal),unknown_ge_08=sum(p['truth'] is None and p['probability']>=.8 for p in nominal))
    unknowns=[]
    for r in evaluation['rows']:
        if r['truth'] is None and r['qualification']['qualified_display']==1:
            c=r['qualification']['witnessed_chain']
            unknowns.append(dict(case_id=r['case_id'],origin=r['origin'],probability=r['probability'],
                selected_source_frames=c['source_frame_ids'],local_probability=c['connected_process_probability']))
    false_cases=counts['development_false_confirmations']
    for pair in false_cases:
        pair['existing_review_reason']=targets[pair['after']['case_id']]['evidence_reason']
    groups={}
    for role in ['fit','development']:
        groups[role]={}
        for origin in ['reviewed_view','runtime_proposal']:
            groups[role][origin]={m:counts['summaries'][m][role][origin]['all'] for m in ['AB','AC']}
    arm_fit=[]
    for r in evaluation['rows']:
        if r['role']=='fit' and r['origin']=='runtime_proposal' and r['physical_query']=='right_hand_toward_left_wrist_along_arm':
            arm_fit.append(dict(case_id=r['case_id'],global_probability=r['probability'],
                candidates=[dict(source_frames=c['source_frame_ids'],probability=c['connected_process_probability']) for c in r['qualification']['interval_candidates']]))
    out={
      'status':'SAVED_CAUSE_DIAGNOSIS_COMPLETE_CANDIDATE_NOT_ADOPTED',
      'reviewer':'Codex retention_direction_review; separate arithmetic/contract analysis, same Codex not independent human or sign-language expert.',
      'scope':'Immutable AB/AC saved predictions, existing targets, frozen proposal code and two public DEV saved sensor-availability arrays. No new inference, training, threshold search or old-label/result change.',
      'source_pins':{str(p):sha(p) for p in [cp,ip,U/'EVALUATION.json',U/'EVALUATION_VERIFIED.json',U/'DECISION.json',U/'DEV_VIEW_TARGETS_V2.json',U/'NOMINAL_PROCESS_TARGETS.json',A/'PROCESS_EVALUATION.json',A/'PROCESS_AUTO_EVALUATION.json',A/'process_qualification.py',U/'process_runtime_contract.py',Path(__file__),U/'analyze_alignment_saved.py']},
      'fixed_thresholds':{'raw':.5,'high':.8,'low':.2},
      'decision':{'status':decision['status'],'passed_gates':decision['passed'],'failed_gates':decision['failed']},
      'summary':'Runtime-context/global evidence improved, but typed point proposals can truncate a process, while other DEV cases still have global/point/local failures. Two edited views are overconfirmed. One structural proposal repair is justified; another performance trial or generic training repeat is not yet justified.',
      'AB_AC_comparison':groups,
      'barrier_semantics':{
         'global_below_0.8':'Saved whole-context process score does not satisfy the unchanged high threshold.',
         'no_ordered_point_candidate':'The unchanged point-run proposal generated no ordered candidate; local outcome is not evaluated, not counted as a local failure.',
         'all_local_below_0.8':'At least one candidate exists, but every saved local score is below high. This is an operational bottleneck, not proof that its exact nominal span should be positive.',
         'combined':'Mutually exclusive combinations list all observed blockers; no hidden averaging of short events.',
         'target_change':'AB local score is net-motion; AC local score is linked-process. Numerical differences are not an isolated head/target causal ablation.'},
      'manual_auto_and_dependency_limits':[
         'Reviewed positive DEV30 consists of15 native queries,10 retimes and5 intact-context trims, all from12 clips/3 recorded IDs. These are not30 independent people.',
         'Each query has six positive reviewed views: three native plus two retimes and one trim of ID11. Native-only and automatic-query rows are separately available in the pinned counts artifact.',
         'FIT217 reviewed views have79 positive128 zero10 unknown, while45 automatic contexts have44 positive1 unknown. Automatic contexts were added to AC training, so FIT improvement there is not transfer evidence.',
         'DEV automatic15 global scores all exceed.8; automatic remaining barriers are seven local failures and one missing point proposal. Proposal witness containment was already complete and is not detection accuracy.'
      ],
      'arm_traversal_1133':{
         'observed':'DEV reviewed six positives are all raw-high but none confirmed: five local failures and one trim with no setup candidate. Three native and three automatic cases all have ordered candidates and low local connected-process scores.',
         'native_candidates':[dict(person=p['after']['person'],frames=p['after']['candidates'][0]['source_frame_range'],global_probability=p['after']['raw_probability'],local_probability=p['after']['candidates'][0]['local_probability'],first_direct_positive_terminal={11:65,12:77,13:73}[p['after']['person']]) for p in counts['development_arm_and_rise'] if p['after']['physical_query']=='right_hand_toward_left_wrist_along_arm' and p['after']['kind']=='native'],
         'automatic_candidates':[dict(person=p['after']['person'],frames=p['after']['candidates'][0]['source_frame_range'],local_probability=p['after']['candidates'][0]['local_probability']) for p in counts['development_arm_and_rise'] if p['after']['physical_query']=='right_hand_toward_left_wrist_along_arm' and p['after']['origin']=='runtime_proposal'],
         'unknown_point_scope':'Intervening direct terminal labels include null (for example person11 f56/59/61, person12 f68/70/72, person13 f60/64). Early high terminal values there are not known-negative classification errors and cannot be relabelled0 from this diagnosis.',
         'code_mechanism':{'path':str(A/'process_qualification.py'),'lines':[34,45],
           'finding':'For each setup-run first point, only after[0] of each terminal run is proposed. Later reviewed terminal witnesses within that same confident run never become alternative support. The cap is not responsible for these one-candidate cases.'},
         'fit_counterevidence_to_learning_inability':{'fit_auto_global_high':9,'fit_auto_confirmed':1,'fit_auto_local_only_failure':8,'fit_reviewed_positive_confirmed':'0/15','saved_fit_auto_cases':arm_fit,
           'interpretation':'The same local failure remains on FIT although the fixed direct nominal positive/negative training spans are learned. This supports a proposal-support mismatch more strongly than inability to optimize the existing process supervision. It does not certify any new candidate exact span.'},
         'direct_visual_next':'Parent assigned the six DEV native/auto spans to a separate pixel review. Preserve all existing labels and scores; examine nominal span only, with exterior context not completing missing action.'},
      'rise_1567':{
         'regression_unit':'ID11 native rise falls from.750579 to.480721; its two ordered-pixel retimes likewise fall from~.7505 to~.4804/.4812. This is three correlated view errors from one source query, not three independent people.',
         'same_case_bottlenecks':'An ordered candidate now exists at source56–64 but local=.065016. Known terminal positives start at67; candidate terminal64 is directly unknown. Global and local both block. Wider automatic context has global=.998879, but candidates57–65=.211299 and57–107=.417769 remain low.',
         'remaining_native_12':'ID12 native remains low(.126709; AB.050618), although a local candidate46–69=.895043 passes. The automatic version has global=.999493 and local57–68=.944313. This proves context-sensitive disagreement, not its unique sensor/recurrent cause.',
         'gain_13_and_new_false':'ID13 native rises from.022272 to.984839 and becomes confirmed, but its middle_splice changes from raw.011015 to.965527 and becomes a new false confirmation. Its suffix also becomes raw-high(.998952) without a qualified confirmation.',
         'matched_sparse_point_evidence':points,
         'point_interpretation':'Matched native direct labels show DEV setup-positive recovery49/66 vs42/66 but terminal-positive recovery42/67 vs43/67; negative high activations increase setup3→5 and terminal2→3. Thus global/context alignment did not uniformly improve direct point transfer.'},
      'two_splice_false_confirmations':{
         'cases':false_cases,
         'shared_structure':'Both preserve a setup relation, jump across three omitted source frames, and retain a later terminal configuration plus some real residual movement. All actually passing local candidates cross that seam; the saved known-seam diagnostic with identical scores holds both. That diagnostic is unavailable at real runtime.',
         '1567_detail':'REAL13 source52→56 omits crossing53–55. The first AC candidate45–59 is only.537052, but later candidates45–69 and45–71 are.983036/.986609. This demonstrates that simply supplying later/wider candidates can also amplify false confirmation.',
         '2523_detail':'REAL12 source55→59 omits56–58. AC candidate45–59 already scores.968139; wider RGB context sees retained later frames60–67. That support is from the same edited stream, not an accidental original-video intact chain.',
         'sensor_audit':availability,
         'interpretation_limit':'1567 has a local geometry mask drop to33/59 for two seam-adjacent frames;2523 remains59/59. Both have every image available and all RGB finite. A missing-image detector therefore has no common trigger. Neither finite masks nor a 3-frame source omission establishes educational error, and the compressed playback may resemble fast normal motion. Pixels/representation identifiability is not established by saved probabilities.',
         'future_semantics':'Keep the current frozen0 labels/results as the observation-continuity test they are. In a separately declared future task distinguish direct visible process, inference between sampled states, insufficient observation and independently justified learner error. Do not retroactively turn the two cases into correct predictions or train all synthetic splices as linguistic errors.'},
      'FIT_nominal_learning_and_unknowns':{
         'counts':nominal_totals,'by_binding':counts['nominal_fit_diagnostics'],
         'whole_unknown_confirmations':unknowns,
         'meaning':'All112 known nominal targets are confidently fitted (75 high positive,37 low negative). Four of six nominal unknowns are nevertheless high. WORD0739 ID08 is newly confirmed in native and auto despite preserved unknown truth. Zero loss on unknown labels is not a learned abstention target and cannot guarantee uncertainty protection.'},
      'raw_known_changes':{'AB_correct':68,'AC_correct':65,'known':75,'improved_views':5,'regressed_views':8,'cases':changes},
      'causal_separation':[
         {'hypothesis':'Insufficient epochs/capacity for existing known supervision','support':'Not supported as primary cause: FIT whole207/207 at.5 and nominal112/112 confident; repeat epochs/seed search is unjustified.'},
         {'hypothesis':'First-terminal proposal truncates the supported trace','support':'Strong structural support, including eight FIT arm auto failures and three DEV native/auto pairs. It does not prove a run endpoint or full-run span is physically sufficient.'},
         {'hypothesis':'Sensor failure causes both false confirmations','support':'Not sufficient: only1567 loses some geometry masks;2523 is fully masked-valid across the seam. RGB/image availability remains true in both.'},
         {'hypothesis':'Motion representation or temporal-context generalization is inadequate','support':'Plausible from native/auto disagreements and splice overconfirmation, but not isolated: labels, learning contexts and local target changed together; no feature ablation or counterfactual inference was performed.'},
         {'hypothesis':'Synthetic zero labels are educational-error truth','support':'Unsupported. Existing review reasons explicitly describe absent directly sampled connection, while retaining genuine tail movement. Sparse/fast motion ambiguity must be separated from actual incorrect performance.'}
      ],
      'one_next_action':{
         'decision':'Accept the root scope: implement and structurally verify a common uncertainty-preserving point-run proposal component; no performance evaluation or new training in that step.',
         'concrete_behavior':'Preserve the observed setup/terminal runs and their native times as support ranges rather than declaring the first high terminal point an exact end. Keep possible support and selected/verified evidence separate. No minimum dwell/length, required pause or repetition count. Real unavailable-image gaps split evidence; distinct repetitions/runs remain distinct. No source-frame seam oracle becomes runtime input.',
         'not_authorized_by_this_evidence':'Do not choose a later endpoint by maximum process score, promote a longer run to positive truth, lower thresholds, expand candidate search until success, or rescore AC as improved. The wider run may include holds, returns or another occurrence.',
         'structural_pass_conditions':['All original native indices/PTS preserved; no invented frames or event boundary','Single-frame predicates and passing terminal relations retained','Unavailable image gaps cannot be bridged; separate runs/repetitions are not collapsed','No word/person/source-seam/ground-truth feature or label-controlled selection','Bounded candidate/support representation with explicit uncertainty; no additional DTW/reference-video work','Frozen AB/AC outputs unchanged and new output cannot be mistaken for a validated connected-process result'],
         'performance_stop_condition':'A proposal-only score rerun cannot satisfy the existing reviewed-positive24/30 gate: unchanged raw-high20/30 is an absolute cap, and false splice2 may remain or worsen. Do not treat the structural repair as the next performance experiment.',
         'next_training_prerequisite':'Use the bounded pixel/meaning audits to decide how direct process, sampled-state inference and insufficient observation are supervised. Reuse existing FIT known chains/partial spans; do not broaden annotation or relabel DEV merely to fit scores. No new architecture or epoch choice is justified by this report alone.'},
      'fixed_pixel_review_cases':{
         'already_assigned_arm_six':[{'case_id':p['after']['case_id'],'candidates':[c['source_frame_range'] for c in p['after']['candidates']]} for p in counts['development_arm_and_rise'] if p['after']['physical_query']=='right_hand_toward_left_wrist_along_arm' and (p['after']['kind']=='native' or p['after']['origin']=='runtime_proposal')],
         'splice_two':[dict(case_id=p['after']['case_id'],source_gap=p['after']['candidates'][0]['source_seams'],purpose='Describe visible missing crossing vs residual motion and possible fast-motion ambiguity; do not change frozen labels.') for p in false_cases],
         'rise_one_source':{'query_id':'NIA_SL_WORD1567_REAL11_F_directed_57_67','nominal_native':[56,64],'nominal_auto_first':[57,65],'purpose':'If unresolved after existing direct-review evidence, examine whether candidate ending at the crossing contains a distinguishable post-crossing relation; native+two retimes are one dependent source. No new boundary/dwell rule.'}},
      'prohibitions_respected':{'new_model_calls':0,'training_steps':0,'threshold_or_seed_search':False,'old_outputs_or_labels_changed':False,'private_H14_H15_ID16_access':False,'root_documents_changed':False,'whole_goal_complete':False}
    }
    p=U/'RESULTS_ALIGNMENT_CAUSE.json';assert not p.exists()
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':out['status'],'path':str(p),'sha256':sha(p),'decision':out['decision'],'nominal':nominal_totals,'raw_changes':{'improved':5,'regressed':8}},ensure_ascii=False))

if __name__=='__main__':main()
