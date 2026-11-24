"""Saved KSL bridge audit. No model forward, detector, learner, or DTW rerun."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,math,sys
import numpy as np

U=Path(__file__).resolve().parents[1]
V=U/'verification'
F=('index_only','open_four','four_closed')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for block in iter(lambda:s.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12)
def same(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return close(a,b)
    return a==b

def main():
    checks=[]
    def check(name,ok,detail=None):
        checks.append({'name':name,'passed':bool(ok),'detail':detail});assert ok,name
    source=read(U/'data/KSL_TRANSFER.json');src=source['rows']
    pred=read(U/'KSL_TRANSFER_PREDICTIONS.json');rows=pred['rows']
    replay=read(U/'KSL_BRIDGE_REPLAY.json');paths=read(U/'KSL_REFERENCE_PATHS.json')
    protocol=read(U/'KSL_DIAGNOSTIC_PROTOCOL.json');bridgeprotocol=read(U/'BRIDGE_PROTOCOL.json')
    mapping=read(U/'research/DEEPHAND_PHYSICAL_PARTIAL_MAPPING.json')
    cfg=read(U/'EXPERIMENT.json');names=cfg['classes'];labels={r['class_name']:r for r in mapping['rows']}
    z=np.load(U/'data/KSL_OBSERVER_PROBABILITIES.npz',allow_pickle=False)
    probs=z['probabilities'];seeds=z['seed_probabilities']
    check('saved_probability_shapes_and_mean',probs.shape==(468,45) and seeds.shape==(3,468,45)
          and np.isfinite(seeds).all() and np.all(seeds>=0) and np.all(seeds<=1)
          and np.array_equal(seeds.mean(axis=0),probs)
          and np.allclose(seeds.sum(axis=-1),1.,rtol=0,atol=1e-6))
    check('fixed_mapping_and_selection_inputs',bridgeprotocol['mapping_sha256']==sha(U/'research/DEEPHAND_PHYSICAL_PARTIAL_MAPPING.json')
          and protocol['inputs_sha256']==sha(U/'data/KSL_TRANSFER.json')
          and protocol['selection_sha256']==sha(U/'OBSERVER_SELECTION.json')
          and protocol['paths_sha256']==replay['paths_sha256']==sha(U/'KSL_REFERENCE_PATHS.json')
          and all(sha(Path(p))==h for p,h in protocol['checkpoints'].items()))
    chosen=read(U/'OBSERVER_SELECTION.json')['selected']
    check('preselected_candidate_not_changed_for_transfer',chosen['kind']=='observer' and chosen['condition']=='RJ_F'
          and protocol['failed_external_gate_not_overridden'] is True
          and protocol['independent_KSL_truth_available'] is False)
    bycase=defaultdict(list);expectedhands={}; known=answered=agree=0;unknown=Counter();detected=0
    joint=np.load(U/'data/KSL_TRANSFER_JOINTS.npz',allow_pickle=False)
    raw=joint['raw_xyz'];rel=raw-raw[:,:1];scale=np.linalg.norm(rel[:,[5,9,13,17]],axis=-1).max(1)
    seen=joint['observed']&(scale>1e-6)
    for i,(s,r) in enumerate(zip(src,rows)):
        assert all(s[k]==r[k] for k in ['case_id','word_id','side','person','frame_id','pts_us','image','full_image','draft_only'])
        bycase[r['case_id']].append(i)
        available=bool(s['image']);identity=not s['identity_uncertain'];features={}
        for f in F:
            masses={v:math.fsum(float(probs[i,j]) for j,n in enumerate(names) if labels[n][f]==v) for v in (1,0,None)}
            reason='hand_not_observed' if not available else 'hand_identity_unresolved' if not identity else None
            state=None if reason else 1 if masses[1]>=.8 else 0 if masses[0]>=.8 else None
            if reason is None and state is None:reason='insufficient_source_mapped_model_mass'
            features[f]={'state':state,'kind':'model_estimate','reason':reason,
                         'p_pos':masses[1],'p_neg':masses[0],'p_unknown':masses[None],
                         'measurement':None,'calibrated_probability':False}
            assert same(features[f],r['features'][f])
            if state is None:unknown[reason]+=1
            if f in ['index_only','open_four'] and r['draft_only'][f] is not None:
                known+=1;answered+=state is not None;agree+=state is not None and state==r['draft_only'][f]
        hand={'hand_present':available,'identity_resolved':identity,'input_kind':'model_estimate',
              'features':features,'raw_class_names':names,'raw_class_probabilities':[float(x) for x in probs[i]],
              'source_mapping_sha256':bridgeprotocol['mapping_sha256'],'educational_verdict':None,
              'identity_basis':'inherited native assignment; not independent truth'}
        expectedhands[i]={'time_s':r['pts_us']/1e6,'hands':{r['side']:hand}}
        top=np.argsort(-probs[i])[:3]
        assert r['top3']==[{'class_name':names[j],'probability':float(probs[i,j])} for j in top]
        assert r['hand_crop_present']==available and r['crop_detector_present']==bool(seen[i]) and r['educational_verdict'] is None
        detected+=bool(seen[i])
    check('all_468_rows_mapping_and_visibility_recomputed',len(src)==len(rows)==468 and len(bycase)==36 and all(len(v)==13 for v in bycase.values()))
    check('independent_public_image_hashes',all(sha(Path(r['full_image']))==r['full_image_sha256']
          and (not r['image'] or sha(Path(r['image']))==r['image_sha256']) for r in src))
    check('source_native_inputs_unchanged',all(sha(Path(p))==h for p,h in source['source_pins'].items()))
    summary=pred['summary']
    check('AI_draft_agreement_not_accuracy',known==summary['draft_known_cells']==745
          and answered==summary['model_answered_known_draft_cells']==605
          and agree==summary['agreement_with_AI_draft']==547
          and summary['agreement_is_accuracy'] is False and summary['real_correctness_validation'] is False
          and summary['operating_candidate_enabled'] is False)
    check('crop_counts_are_input_availability',sum(bool(r['image']) for r in src)==summary['visible_crops']==456
          and detected==summary['crop_redetections']==422 and sum(r['identity_uncertain'] for r in src)==0)
    bypair={(p['query'],p['reference']):p for p in paths['pairs']}
    check('12_queries_and_24_unique_reference_paths',len(bypair)==24 and len(paths['pairs'])==24
          and len({p['query'] for p in paths['pairs']})==12
          and len(replay['comparisons'])==len(replay['all_missing_controls'])==12)
    for p in paths['pairs']:
        qi=p['query_indices'];ri=p['reference_indices'];q=src[qi[0]];r=src[ri[0]]
        assert qi==bycase[p['query']] and ri==bycase[p['reference']] and len(qi)==len(ri)==13
        assert q['person']==3 and r['person'] in [1,2] and q['word_id']==r['word_id'] and q['side']==r['side']==p['side']
        path=p['path'];assert path[0]==[0,0] and path[-1]==[12,12]
        assert all((b[0]-a[0],b[1]-a[1]) in [(1,0),(0,1),(1,1)] for a,b in zip(path,path[1:]))
        assert {v[0] for v in path}==set(range(13)) and {v[1] for v in path}==set(range(13))
    check('same_expression_separate_person_topology_and_paths',True)
    totals={'actual':Counter(),'all_missing':Counter()}
    all_pairs=set();all_missing_pairs=set()
    for mode,key in [('actual','comparisons'),('all_missing','all_missing_controls')]:
        for comparison in replay[key]:
            qid=comparison['query'];result=comparison['result'];qindices=bycase[qid]
            assert result['reference_splicing_or_best_selection'] is False and result['existing_alignment_modified'] is False
            assert result['educational_verdict'] is None and result['educational_feedback_enabled'] is False
            assert result['model_outputs_are_ground_truth'] is False and result['probabilities_are_calibrated'] is False
            assert result['bridge_protocol_sha256']==sha(U/'BRIDGE_PROTOCOL.json') and len(result['references'])==2
            queryframes=result['raw_query_frames']
            if mode=='actual':assert same(queryframes,[expectedhands[i] for i in qindices])
            else:
                assert len(queryframes)==13
                for q,i in zip(queryframes,qindices):
                    assert q['time_s']==src[i]['pts_us']/1e6
                    for h in q['hands'].values():
                        assert h['hand_present'] is False and h['identity_resolved'] is False
                        assert all(v['state'] is None and v['measurement'] is None for v in h['features'].values())
            for reference in result['references']:
                rid=reference['reference_id'];p=bypair[(qid,rid)];ri=p['reference_indices'];target=p['side']
                (all_pairs if mode=='actual' else all_missing_pairs).add((qid,rid))
                assert reference['path_original']==p['path']==reference['path_unique_pairs']
                assert reference['path_recomputed'] is False and reference['duplicate_pairs_ignored_for_count']==0
                assert reference['query_frames']==reference['reference_frames']==13
                assert same(reference['raw_reference_frames'],[expectedhands[i] for i in ri])
                required={h:{f:[h==target]*13 for f in F} for h in ['L','R']}
                assert reference['required_mask_preserved']==required
                links=defaultdict(list)
                for a,b in p['path']:links[b].append(a)
                for side in ['L','R']:
                    for feature in F:
                        group=reference['hands'][side][feature];counts=Counter();missing=[]
                        assert len(group['cells'])==13 and group['educational_verdict'] is None
                        for j,cell in enumerate(group['cells']):
                            req=side==target;refhand=reference['raw_reference_frames'][j]['hands'].get(side)
                            if refhand is None:rcell={'state':None,'kind':'unknown','reason':'hand_record_missing'}
                            elif not refhand['hand_present'] or not refhand['identity_resolved']:
                                rcell={'state':None,'kind':'unknown','reason':'hand_not_observed' if not refhand['hand_present'] else 'hand_identity_unresolved'}
                            else:rcell=refhand['features'][feature]
                            observed_pairs=[]
                            for qi in links[j]:
                                qhand=queryframes[qi]['hands'].get(side)
                                if qhand is None:qcell={'state':None,'kind':'unknown','reason':'hand_record_missing'}
                                elif not qhand['hand_present'] or not qhand['identity_resolved']:
                                    qcell={'state':None,'kind':'unknown','reason':'hand_not_observed' if not qhand['hand_present'] else 'hand_identity_unresolved'}
                                else:qcell=qhand['features'][feature]
                                both=qcell['state'] is not None and rcell['state'] is not None
                                pp={'query_index':qi,'query_time_s':queryframes[qi]['time_s'],'reference_index':j,
                                    'reference_time_s':reference['raw_reference_frames'][j]['time_s'],
                                    'query_state':qcell['state'],'reference_state':rcell['state'],
                                    'query_kind':qcell['kind'],'reference_kind':rcell['kind'],
                                    'query_reason':qcell.get('reason'),'reference_reason':rcell.get('reason'),
                                    'classification_state_difference':int(qcell['state']!=rcell['state']) if both else None,
                                    'observed_value_abs_difference':None,'observed_value_unit':None,
                                    'model_positive_mass_abs_difference':abs(qcell['p_pos']-rcell['p_pos']) if both else None}
                                observed_pairs.append(pp)
                            assert same(observed_pairs,cell['pairs'])
                            valid=[x for x in observed_pairs if x['classification_state_difference'] is not None]
                            complete=bool(observed_pairs) and len(valid)==len(observed_pairs)
                            state='complete' if complete else 'partial' if valid else 'unavailable'
                            relations={x['classification_state_difference'] for x in valid}
                            relation='same' if relations=={0} else 'different' if relations=={1} else 'mixed' if relations else None
                            assert cell['reference_index']==j and cell['required']==req and cell['reference_time_s']==src[ri[j]]['pts_us']/1e6
                            assert cell['aligned_pair_count']==len(observed_pairs) and cell['known_pair_count']==len(valid)
                            assert cell['classification_comparison']==state and cell['state_relation']==relation
                            assert cell['state_relation_scope']==('all_aligned_pairs' if complete else 'known_pairs_only_not_whole_required_cell')
                            assert cell['complete_observed_only'] is False and cell['measured_pair_count']==0
                            assert cell['complete_observed_value_comparison'] is False
                            assert cell['complete_involving_estimate']==complete and cell['educational_verdict'] is None
                            if req:
                                counts['required']+=1;counts[state]+=1
                                if not complete:missing.append(j)
                        assert group['required_sample_count']==counts['required']
                        assert group['complete_comparison_count']==counts['complete']
                        assert group['partial_comparison_count']==counts['partial']
                        assert group['insufficient_count']==counts['required']-counts['complete']
                        assert group['complete_observed_only_count']==group['complete_observed_value_comparison_count']==0
                        assert group['complete_involving_estimate_count']==counts['complete']
                        assert group['complete_comparison_fraction']==(counts['complete']/counts['required'] if counts['required'] else None)
                        assert group['sample_count_not_duration_fraction'] is True
                        runs=[]
                        for j in missing:
                            if not runs or j!=runs[-1][-1]+1:runs.append([j])
                            else:runs[-1].append(j)
                        expectedruns=[{'reference_indices':indices,'sample_times_s':[src[ri[j]]['pts_us']/1e6 for j in indices],
                            'start_sample_s':src[ri[indices[0]]]['pts_us']/1e6,'end_sample_s':src[ri[indices[-1]]]['pts_us']/1e6,
                            'continuous_interval_claim':False} for indices in runs]
                        assert same(group['insufficient_sample_runs'],expectedruns)
                        totals[mode].update(counts)
    check('all_reference_pairs_kept_separate',all_pairs==all_missing_pairs==set(bypair))
    check('every_saved_pair_cell_and_insufficient_run_recomputed',True)
    check('all_missing_controls_keep_936_required_cells',totals['actual']['required']==totals['all_missing']['required']==24*13*3
          and totals['all_missing']['complete']==totals['all_missing']['partial']==0
          and totals['all_missing']['unavailable']==936)
    check('all_outputs_are_estimates_not_true_observed_values',True)
    check('no_educational_or_crossing_accuracy_claim',replay['normal_crossing_accuracy']=='not established'
          and replay['educational_verdict_enabled'] is False and replay['production_changed'] is False)
    protected=read(U/'FROZEN_PREPARATION.json')['protected']
    check('protected32_byte_hashes_unchanged',len(protected)==32 and all(sha(Path(p))==h for p,h in protected.items()))
    recovery=read(U/'logs/KSL_BRIDGE_RECOVERY.json')
    check('first_caller_failure_preserved',recovery['exit_code']==1 and recovery['final_external_test_recomputed'] is False
          and recovery['KSL_partial_predictions_saved_before_failure'] is False)
    report={'status':'PASS','at':datetime.now(timezone.utc).isoformat(),'checks':checks,'check_groups':len(checks),
        'verifier':'Separate saved-data arithmetic; no producer imports, model inference, training, or alignment recomputation',
        'frames':468,'windows':36,'queries':12,'fixed_reference_paths':24,
        'required_cells':936,'actual_comparison_counts':dict(totals['actual']),
        'all_missing_comparison_counts':dict(totals['all_missing']),
        'feature_unknown_reasons':dict(unknown),'draft_known':known,'answered_AI_draft_cells':answered,'AI_draft_agreements':agree,
        'protected_files_verified':32,'educational_accuracy_validated':False,'normal_crossing_accuracy_validated':False,
        'real_missing_hand_recovery_accuracy_validated':False,
        'limits':['456 visible_crops is availability of landmark-derived input crops, not independent RGB hand-visibility ground truth.',
                  'No crop-candidate identity-uncertain cases occurred; the unknown-identity branch is covered by earlier synthetic contracts, not by this natural replay.',
                  'Left/right comes from earlier native assignment and is not freshly certified anatomical identity.',
                  'All three feature states here are model estimates. Same states can share the same model error.',
                  'All13 samples are technically required for this diagnostic; this does not establish linguistic necessity or valid hand-rest boundaries.',
                  'Each window has only13 frames. No alignment quality improvement, full-expression equivalence, normal crossing correctness, or education correctness was tested.',
                  '547/605 is agreement conditional on answering previously known AI-draft cells, not independently measured accuracy.'],
        'first_failure':'Caller omitted the unused-hand false required-mask. First failure and same-condition repair retained; no completed final-test output recomputed.',
        'files':{name:sha(U/name) for name in ['KSL_TRANSFER_PREDICTIONS.json','KSL_BRIDGE_REPLAY.json','KSL_REFERENCE_PATHS.json','KSL_DIAGNOSTIC_PROTOCOL.json','BRIDGE_PROTOCOL.json','data/KSL_OBSERVER_PROBABILITIES.npz']},
        'script_sha256':sha(Path(__file__))}
    out=V/'KSL_BRIDGE_INDEPENDENT.json';assert not out.exists(),'Preserve prior review'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','check_groups':len(checks),'actual':dict(totals['actual']),'all_missing':dict(totals['all_missing']),'protected':32},ensure_ascii=False))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');main()
