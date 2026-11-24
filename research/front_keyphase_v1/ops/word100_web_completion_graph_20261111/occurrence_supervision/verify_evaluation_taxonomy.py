"""Saved prediction/result arithmetic only. Never imports or executes a model."""
from pathlib import Path
from collections import Counter
import json, hashlib
import numpy as np

W=Path(__file__).resolve().parent;G=W.parent;O=G.parent;R=O/'word100_candidate_role_learning_20261111'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    target=W/'EVALUATION_TAXONOMY_CORRECTION.json'
    assert not target.exists()
    paths=[W/'FRAME_ROLE_SCORES.json',W/'EVALUATION_COMPLETE.json',W/'RESULTS.json',W/'EVALUATION_FROZEN.json',G/'evaluate_role_correction.py',R/'FRAME_ROLE_SCORES.json',R/'evaluate.py',O/'word100_event_guide_linkage_20261110/LINK_LABELS.json',O/'word100_action_episode_supervision_20261110/episode_core.py',G/'INTERVAL_PRESERVATION_RESULTS.json',G/'reviews/interval_preservation_verification.json',G/'reviews/sequence_truth_audit.json']
    beforepins={str(p):sha(p) for p in paths}
    frames=read(W/'FRAME_ROLE_SCORES.json');summary=read(W/'EVALUATION_COMPLETE.json');rows=read(W/'RESULTS.json')
    old={(r['seed'],r['clip_id'],r['frame']):r for r in read(R/'FRAME_ROLE_SCORES.json')}
    prior={(r['seed'],r['clip_id'],r['kind']):r for r in read(G/'INTERVAL_PRESERVATION_RESULTS.json')}
    links={r['clip_id']:r for r in read(O/'word100_event_guide_linkage_20261110/LINK_LABELS.json')['rows']}
    mapping={0:0,1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:6,9:7}
    labels={(cid,a['frame']):mapping[a['label']] for cid,r in links.items() for a in r['anchors'] if a['label'] in mapping}
    result_by={(r['condition'],r['seed'],r['clip_id'],r['kind']):r for r in rows}
    assert len(rows)==len(result_by)==468
    configs=[(c,s) for c in ['continued_old_direct','reviewed_roles'] for s in [42,43,44]]
    metric={};converted=[];prediction_pins=[]
    for condition,seed in configs:
        ff=[r for r in frames if r['condition']==condition and r['seed']==seed]
        assert len(ff)==67
        raw=sum(r['prediction']==r['target'] for r in ff)
        corrected=sum((r['prediction'] if r['prediction']<8 else 0)==r['target'] for r in ff)
        baseline=sum(old[seed,r['clip_id'],r['frame']]['after']==r['target'] for r in ff)
        assert raw==summary['anchors'][condition][str(seed)]['correct']
        assert summary['anchors'][condition][str(seed)]['total']==67
        for r in ff:
            assert r['target']==labels[r['clip_id'],r['frame']]==old[seed,r['clip_id'],r['frame']]['target']
            assert r['target']<8 and r['prediction']!=10
            if r['prediction']>=8:converted.append(dict(**r,collapsed_prediction=0,raw_correct=False,collapsed_correct=r['target']==0))
        metric[f'{condition}/{seed}']=dict(raw_11_output_vs_8_gold_correct=raw,like_for_like_8family_correct=corrected,baseline_8family_correct=baseline,total=67,nonmain_predictions_collapsed=sum(r['prediction']>=8 for r in ff),changed_accuracy_points=(corrected-baseline)/67*100)
        assert corrected==baseline==64
    # Check cached role outputs match saved labels; no forward pass or feature load.
    for r in rows:
        assert sha(r['role_prediction'])==r['role_sha256']
        with np.load(r['role_prediction'],allow_pickle=False) as z:
            assert z.files==['role'];rp=z['role']
            assert rp.shape[1]==11 and np.isfinite(rp).all()
            assert (rp[:,10]==0).all(), 'Untrained transition class was not masked'
            if r['kind']=='full_normal':
                ff=[x for x in frames if x['condition']==r['condition'] and x['seed']==r['seed'] and x['clip_id']==r['clip_id']]
                for f in ff:assert int(rp[f['frame']].argmax())==f['prediction']
        prediction_pins.append(dict(path=r['role_prediction'],sha256=r['role_sha256']))

    selection=[];pairchanges=[];all_null={};normal={}
    for condition,seed in configs:
        for kind in sorted({r['kind'] for r in rows}):
            rr=[r for r in rows if r['condition']==condition and r['seed']==seed and r['kind']==kind]
            changes=[];nullbefore=nullafter=0
            for r in rr:
                p=prior[seed,r['clip_id'],kind];pc={c['id']:c for c in p['cards']}
                for c in r['cards']:
                    oc=pc[c['id']];nullbefore+=oc['span'] is None;nullafter+=c['span'] is None
                    different=c['span']!=oc['span'] or c['evidence_intervals']!=oc['evidence_intervals'] or c['observed_indices']!=oc['observed_indices']
                    if different:changes.append(dict(clip_id=r['clip_id'],card=c['id'],before_span=oc['span'],after_span=c['span'],before_support=oc['evidence_intervals'],after_support=c['evidence_intervals']))
                    assert c['semantic_completion'] is None
                pp={tuple(x['cards']):x for x in p['pairs']}
                for q in r['pairs']:
                    op=pp[tuple(q['cards'])]
                    if q['state']!=op['state']:pairchanges.append(dict(condition=condition,seed=seed,clip_id=r['clip_id'],kind=kind,cards=q['cards'],before=op['state'],after=q['state'],before_spans=op['spans'],after_spans=q['spans']))
            selection.append(dict(condition=condition,seed=seed,kind=kind,clips=len(rr),cards=sum(len(r['cards']) for r in rr),changed_support_cards=len(changes),changes=changes))
            all_null[f'{condition}/{seed}/{kind}']=dict(before=nullbefore,after=nullafter)
            if kind=='full_normal':
                assert sum(len(r['cards']) for r in rr)==24
                assert not changes
                normal[f'{condition}/{seed}']=dict(cards=24,source_support_spans_unchanged=24,all_observed_indices_unchanged=True,normal_direct_anchor_coverage_inherited=True,meaning='Same selected support preserves previously audited anchor coverage; not fresh semantic feedback validation.')

    truth={r['clip_id']:r for r in read(G/'reviews/sequence_truth_audit.json')['clips']}
    conflicts=[]
    inside=lambda i,span:span is not None and span[0]<=i<=span[1]
    for condition,seed in configs:
        for cid in ['NIA_SL_WORD0739_REAL12_F','NIA_SL_WORD0739_REAL13_F']:
            r=result_by[condition,seed,cid,'reversed_core_auto_order'];p=prior[seed,cid,'reversed_core_auto_order']
            detail=[]
            for card,family in [('close_chest',2),('raise_forearm',3)]:
                current=next(c for c in r['cards'] if c['id']==card);previous=next(c for c in p['cards'] if c['id']==card)
                selected=[a for a in truth[cid]['reviewed_source_anchors'] if inside(a['query_index'],current['span'])]
                detail.append(dict(card=card,span=current['span'],prior_span=previous['span'],matched=[a for a in selected if a['source_reviewed_family']==family],conflicting=[a for a in selected if a['source_reviewed_family']!=family]))
                assert current['span']==previous['span']
            pair=next(q for q in r['pairs'] if q['cards']==['close_chest','raise_forearm'])
            assert pair['state']=='association_unknown'
            assert next(x for x in detail if x['card']=='raise_forearm')['span'] is None
            conflicts.append(dict(condition=condition,seed=seed,clip_id=cid,details=detail,state=pair['state'],false_source_role_raise_assignment_stays_rejected=True,recovered_complete_action=False))
    # Recompute saved aggregate pair counts without classifying all reversals as errors.
    recomputed={c:{str(s):{k:dict(Counter(p['state'] for r in rows if r['condition']==c and r['seed']==s and r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in rows})} for s in [42,43,44]} for c in ['continued_old_direct','reviewed_roles']}
    assert recomputed==summary['summary']
    assert beforepins=={str(p):sha(p) for p in paths}
    out=dict(status='PASS_SAVED_TAXONOMY_AND_SELECTION_AUDIT_NO_IMPROVEMENT_CLAIM',issue='New evaluator compared raw11-class role output directly to legacy8-class gold. Existing evaluator folds preparation8/return9/transition10 into legacy nonmain0.',correction='Use prediction if prediction<8 else0 for the SAME 67 original targets. This is inherited label-taxonomy compatibility, not a tuned threshold or a model change.',metric=metric,converted_nonmain_frames=converted,original_files_preserved=True,untrained_transition_class10_zero_in_all_468_saved_outputs=True,normal_selected_support=normal,source_conflict_checks=conflicts,all_condition_selection_changes=selection,pair_state_changes=pairchanges,null_card_counts=all_null,recomputed_pair_summary=recomputed,limitations=['The 64/67 measure distinguishes legacy event families only. It does not validate entry/return subclass accuracy, boundary timing, handshape correctness, direction, semantic completion, or natural learner feedback.','Normal24 counts cards from the existing12 public DEV clips, not24 independent videos. Each condition/seed reuses those sources.','The two source-conflict cases stay withheld. That is preserved rejection of wrong assignment, not successful recovered action understanding.','Do not label all partial-core reversal pairs semantic errors: unedited fragments and changed source action meanings remain.','Root/delegated RGB audits are not independent human/KSL truth. Existing DEV remains exposed, not a fresh held-out result.'],no_model_inference=True,no_training=True,no_threshold_tuning=True,no_private_access=True,pins=[dict(path=p,sha256=h) for p,h in beforepins.items()],saved_prediction_pins=prediction_pins,reviewer='Separate Codex saved-result arithmetic audit')
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=out['status'],metric=metric,normal=normal,source_conflict_checks=len(conflicts),support_changes={f"{x['condition']}/{x['seed']}/{x['kind']}":x['changed_support_cards'] for x in selection if x['changed_support_cards']},pair_state_changes=pairchanges,output=str(target)),ensure_ascii=False))

if __name__=='__main__':main()
