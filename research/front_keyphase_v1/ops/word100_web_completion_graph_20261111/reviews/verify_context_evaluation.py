"""Saved outputs only: compare actual support spans, not just aggregate counts."""
from pathlib import Path
from collections import Counter
import json,hashlib,datetime
import numpy as np

HERE=Path(__file__).resolve().parent;U=HERE.parent;O=U.parent;W=U/'context_role_training';OLD=U/'occurrence_supervision'
R=O/'word100_candidate_role_learning_20261111'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def card_view(c):
    return {k:c[k] for k in ['id','family','span','evidence_intervals','observed_indices','required','parent','semantic_completion']}

def pair_view(p):
    return {k:p.get(k) for k in ['cards','spans','state','limited_dynamic_target','spans_within_target_occurrences','selected_span_crosses_splice']}

def main():
    target=HERE/'context_evaluation_verification.json';assert not target.exists()
    pins=[W/'RESULTS.json',W/'FRAME_SCORES.json',W/'EVALUATION_COMPLETE.json',W/'EVALUATION_FROZEN.json',OLD/'RESULTS.json',U/'dynamic_blocks/RESULTS.json',U/'dynamic_blocks/INPUTS_COMPLETE.json',R/'EVALUATION_INPUTS.json',O/'word100_event_guide_linkage_20261110/LINK_LABELS.json',HERE/'context_training_verification.json',Path(__file__)]
    hashes={str(p):sha(p) for p in pins}
    frozen=read(W/'EVALUATION_FROZEN.json')
    for p,h in frozen['source_pins'].items():assert sha(p)==h
    rows=read(W/'RESULTS.json');frames=read(W/'FRAME_SCORES.json');complete=read(W/'EVALUATION_COMPLETE.json')
    normal_sources=read(R/'EVALUATION_INPUTS.json')['rows'];dynamic_sources=read(U/'dynamic_blocks/INPUTS_COMPLETE.json')['rows']
    sources={(r['clip_id'],r['kind']):r for r in normal_sources}
    sources.update({(r['clip_id'],'dynamic_complete_blocks'):r for r in dynamic_sources})
    assert len(normal_sources)==78 and len(dynamic_sources)==4 and len(sources)==82
    for s in sources.values():
        assert sha(s['path'])==s['sha256'];assert sha(s['baseline'])==s['baseline_sha256']
    by={(r['condition'],r['seed'],r['clip_id'],r['kind']):r for r in rows}
    assert len(rows)==len(by)==492
    prior={(r['seed'],r['clip_id'],r['kind']):r for r in read(OLD/'RESULTS.json') if r['condition']=='reviewed_roles'}
    prior.update({(r['seed'],r['clip_id'],'dynamic_complete_blocks'):r for r in read(U/'dynamic_blocks/RESULTS.json') if r['condition']=='reviewed_roles'})
    assert len(prior)==246
    configs=[(c,s) for c in ['source_context_control','permuted_context'] for s in [42,43,44]]
    gold={r['clip_id']:r for r in read(O/'word100_event_guide_linkage_20261110/LINK_LABELS.json')['rows']}
    family_map={0:0,1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:6,9:7}
    labels={(cid,a['frame']):family_map[a['label']] for cid,r in gold.items() for a in r['anchors'] if a['label'] in family_map}
    score_lookup={(r['condition'],r['seed'],r['clip_id'],r['frame']):r for r in frames};assert len(frames)==len(score_lookup)==402
    predictions={};predpin=[];pairchecks=0;order_probability_matches=0;dynamic_details=[]
    baseline_changed=[];baseline_pair_changed=[]
    for r in rows:
        c,s,cid,kind=r['condition'],r['seed'],r['clip_id'],r['kind'];assert (cid,kind) in sources
        assert sha(r['role_prediction'])==r['role_sha256'];predpin.append(dict(path=r['role_prediction'],sha256=r['role_sha256']))
        with np.load(r['role_prediction'],allow_pickle=False) as z:
            assert z.files==['role'];rp=z['role'].copy();assert rp.shape[1]==11 and np.isfinite(rp).all() and (rp[:,10]==0).all()
        predictions[c,s,cid,kind]=rp
        if kind=='full_normal':
            for f in [x for x in frames if (x['condition'],x['seed'],x['clip_id'])==(c,s,cid)]:
                raw=int(rp[f['frame']].argmax());assert raw==f['prediction_raw'] and (raw if raw<8 else 0)==f['prediction_legacy8']
                assert f['target']==labels[cid,f['frame']]
        p=prior[s,cid,kind];pc={x['id']:x for x in p['cards']}
        for a in r['cards']:
            assert a['semantic_completion'] is None
            if card_view(a)!=card_view(pc[a['id']]):baseline_changed.append(dict(condition=c,seed=s,clip_id=cid,kind=kind,card=a['id'],before=card_view(pc[a['id']]),after=card_view(a)))
        pp={tuple(x['cards']):x for x in p['pairs']}
        for q in r['pairs']:
            a,b=q['spans'];prob=q['order_probability'];state='association_unknown'
            if a is not None and b is not None:state='supported' if a[1]<b[0] and prob>=.65 else 'reversed' if b[1]<a[0] and prob<=.35 else 'ambiguous'
            assert state==q['state'];pairchecks+=1
            old=pp[tuple(q['cards'])]
            if old['spans']==q['spans'] and prob is not None:
                oldp=old.get('order_probability',old.get('probability'));assert prob==oldp;order_probability_matches+=1
            if old['spans']!=q['spans'] or old['state']!=q['state']:baseline_pair_changed.append(dict(condition=c,seed=s,clip_id=cid,kind=kind,cards=q['cards'],before_spans=old['spans'],after_spans=q['spans'],before_state=old['state'],after_state=q['state']))
            if kind=='dynamic_complete_blocks':
                st=sources[cid,kind]['target'];seams=[j['query_right'] for j in st['splice_joins'] if j['synthetic_discontinuity']]
                crosses=any(l<j<=h for sp in q['spans'] if sp for l,h in [sp] for j in seams)
                assert crosses==q['selected_span_crosses_splice']
                limited=cid.split('_')[2]=='WORD0739' and q['cards']==['close_chest','raise_forearm'];assert limited==q['limited_dynamic_target']
                mapped=None
                if limited and a and b:
                    ba,ca=[st['blocks'][k]['query_frame_span'] for k in ['B','C']]
                    mapped=ba[0]<=a[0]<=a[1]<=ba[1] and ca[0]<=b[0]<=b[1]<=ca[1]
                assert mapped==q['spans_within_target_occurrences']
                if limited:dynamic_details.append(dict(condition=c,seed=s,clip_id=cid,cards=q['cards'],spans=q['spans'],state=state,inside_correct_source_occurrences=mapped,crosses_artificial_splice=crosses,correct_limited_reversed=state=='reversed' and mapped is True and not crosses,whole_sign_correctness=None))

    summaries=[];normal=[]
    for c,s in configs:
        rr=[r for r in rows if r['condition']==c and r['seed']==s];ff=[f for f in frames if f['condition']==c and f['seed']==s]
        dd=[d for d in dynamic_details if d['condition']==c and d['seed']==s]
        sm=dict(condition=c,seed=s,legacy8=dict(correct=sum(f['target']==f['prediction_legacy8'] for f in ff),total=len(ff)),states={k:dict(Counter(q['state'] for r in rr if r['kind']==k for q in r['pairs'])) for k in sorted({r['kind'] for r in rr})},limited_dynamic=dict(correct_reversed_mapped=sum(d['correct_limited_reversed'] for d in dd),total=len(dd)))
        summaries.append(sm);assert sm==next(x for x in complete['summary'] if x['condition']==c and x['seed']==s)
        for kind in ['full_normal','speed075','speed125']:
            x=[r for r in rr if r['kind']==kind];ncards=sum(len(r['cards']) for r in x);changed=[v for v in baseline_changed if v['condition']==c and v['seed']==s and v['kind']==kind]
            normal.append(dict(condition=c,seed=s,kind=kind,clips=len(x),cards=ncards,changed_support_cards=len(changed),unknown_support_cards=sum(a['span'] is None for r in x for a in r['cards']),supported_order_pairs=sum(q['state']=='supported' for r in x for q in r['pairs'])))
            assert ncards==24 and not changed

    matched=[];matched_card_changes=[];matched_pair_changes=[];matched_anchor_changes=[]
    for seed in [42,43,44]:
        for cid,kind in sources:
            control=by['source_context_control',seed,cid,kind];candidate=by['permuted_context',seed,cid,kind]
            cc={c['id']:c for c in control['cards']};pp={tuple(p['cards']):p for p in control['pairs']}
            changed=[];anchor_changed=[]
            for c in candidate['cards']:
                o=cc[c['id']]
                if card_view(c)!=card_view(o):changed.append(c['id']);matched_card_changes.append(dict(seed=seed,clip_id=cid,kind=kind,card=c['id'],control=card_view(o),candidate=card_view(c)))
                if [g['anchors'] for g in c['groups']]!=[g['anchors'] for g in o['groups']]:anchor_changed.append(c['id']);matched_anchor_changes.append(dict(seed=seed,clip_id=cid,kind=kind,card=c['id'],control=[g['anchors'] for g in o['groups']],candidate=[g['anchors'] for g in c['groups']]))
            pairchanged=[]
            for p in candidate['pairs']:
                if pair_view(p)!=pair_view(pp[tuple(p['cards'])]):pairchanged.append(p['cards']);matched_pair_changes.append(dict(seed=seed,clip_id=cid,kind=kind,control=pair_view(pp[tuple(p['cards'])]),candidate=pair_view(p)))
            a=predictions['source_context_control',seed,cid,kind];b=predictions['permuted_context',seed,cid,kind]
            assert a.shape==b.shape
            matched.append(dict(seed=seed,clip_id=cid,kind=kind,changed_support_cards=changed,changed_order_pairs=pairchanged,changed_internal_anchor_cards=anchor_changed,probability_max_abs_difference=float(np.abs(a-b).max()),raw_top1_changed_frames=int((a.argmax(1)!=b.argmax(1)).sum()),frames=len(a)))
    assert len(matched)==246
    assert not matched_card_changes and not matched_pair_changes
    assert order_probability_matches==complete['order_scores_reused']==282 and complete['new_order_observations']==0
    assert hashes=={str(p):sha(p) for p in pins}
    out=dict(status='PASS_SAVED_EVALUATION_AUDIT_NO_OBSERVED_AUGMENTATION_GAIN',at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),reviewer='Separate Codex saved-result arithmetic only; no inference or independent human truth',counts=dict(scenarios=82,conditions=2,seeds=3,result_rows=len(rows),frame_scores=len(frames),pair_state_checks=pairchecks,prediction_hashes_checked=len(predpin),matched_case_pairs=len(matched),matched_selected_support_changes=len(matched_card_changes),matched_order_state_or_span_changes=len(matched_pair_changes),matched_internal_anchor_changes=len(matched_anchor_changes),vs_previous_reviewed_support_changes=len(baseline_changed),vs_previous_reviewed_order_changes=len(baseline_pair_changed),old_order_scores_reused=order_probability_matches,new_order_observations=0),recomputed_summaries=summaries,normal_speed_regression=normal,dynamic_limited_targets=dynamic_details,matched_case_differences=matched,matched_card_changes=matched_card_changes,matched_pair_changes=matched_pair_changes,matched_internal_anchor_changes=matched_anchor_changes,previous_reviewed_card_changes=baseline_changed,previous_reviewed_pair_changes=baseline_pair_changed,conclusion='Both matched conditions choose exactly the same actual support intervals/observed-index sets and order states in all246case comparisons. Saved role probabilities may differ, but no selected-feedback improvement is established. Normal and two speed conditions preserve all24card supports each, across all6runs.',limitations=['64/67 is legacy8family anchor accuracy, not full feedback or completion accuracy.','Limited dynamic2/3,2/3,1/3 is the same under control and augmentation; it covers3reviewed obedience source pairs only. Egypt synthetic input is not a whole-sign semantic truth test.','Changing internal joint-anchor indices without changing selected support is not by itself a demonstrated feedback improvement.','Existing exposed DEV78 plus4synthetic scenarios are not an independent final test. No threshold or candidate selection was performed by this verifier.','Only saved role/selection/order outputs were audited. Full handshape/direction feedback and browser behavior were not rerun. Frozen parameters alone do not prove those runtime regressions.'],pins=[dict(path=p,sha256=h) for p,h in hashes.items()],prediction_pins=predpin,model_inference=False,training=False,private_access=False,original_files_modified=False)
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=out['status'],counts=out['counts'],normal_speed_preserved=True,output=str(target),previous_card_changes=baseline_changed,previous_pair_changes=baseline_pair_changed),ensure_ascii=False))

if __name__=='__main__':main()
