"""Keep physical predicates, source clauses and empirical event IDs distinct."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parent
D=O/'word100_action_episode_supervision_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    assert not (U/'EVIDENCE_CONTRACT.json').exists()
    m=read(U/'REVIEW_MANIFEST.json');parent=read(U/'PARENT_TRAVERSAL_REVIEW.json');control=read(U/'AGENT_CONTROL_REVIEW.json')
    assert read(U/'MEDIA_VERIFIED.json')['status']=='PUBLIC_NATIVE_REVIEW_MEDIA_VERIFIED'
    by={r['alias']:r for r in m['rows']};covered=[]
    for r in parent['rows']:
        assert r['reviewed_native_frames']==by[r['alias']]['frames'];covered.append(r['alias'])
    for r in control['rows']:
        assert r['reviewed_frames']==by[r['alias']]['frames'] and r['all_sheet_cells_visually_inspected'];covered.append(r['alias'])
        for s in r['reviewed_sheets']:assert sha(s['path'])==s['sha256']
    assert len(covered)==len(set(covered))==18 and set(covered)==set(by)
    targets=read(D/'ACTION_TARGETS.json');families=[];clauses={};pins={}
    for fam in targets['families']:
        links=[]
        for desc in fam.get('source_descriptions',[]):
            for c in desc['source_cards']:
                src=c['source'];assert sha(src['path'])==src['sha256'];pins[src['path']]=src['sha256']
                key=c['url']+'#'+c['clause_id'];links.append(key)
                row=clauses.setdefault(key,dict(clause_id=c['clause_id'],text=c['text'],url=c['url'],source=src,event_families=[],guide_ids=[],context_only=c.get('context_only',False)))
                assert row['text']==c['text']
                row['event_families']=sorted(set(row['event_families']+[fam['id']]))
                row['guide_ids']=sorted(set(row['guide_ids']+fam['guide_ids']))
        families.append(dict(id=fam['id'],name=fam['name'],guide_ids=fam['guide_ids'],source_clause_links=sorted(set(links)),
            role='empirical video event-content label; not automatically a required separate linguistic action'))
    # The same clause can support multiple local observed phases. This relation
    # is retained, never used to merge phases or fabricate mandatory counts.
    shared=[r for r in clauses.values() if len(r['event_families'])>1]
    assert any(r['event_families']==[3,4] for r in shared) and any(r['event_families']==[1,2] for r in shared)
    for p in [D/'ACTION_TARGETS.json',U/'FROZEN.json',U/'REVIEW_MANIFEST.json',U/'PARENT_TRAVERSAL_REVIEW.json',U/'AGENT_CONTROL_REVIEW.json',U/'MEDIA_VERIFIED.json',U/'build_contract.py']:pins[str(p)]=sha(p)
    contract=dict(schema_version=1,
        physical_evidence=dict(nonexclusive=True,fields=['handshape','body_relative_relation','directed_movement','visibility','native_frames','support_provenance'],
            values=['observed','estimated_from_temporal_context','unknown'],missing_is_not_negative=True,scope='A shape or body relation may persist during movement. Multiple properties need not compete for a single class.'),
        event_binding=dict(fields=['guide_clause_id','event_instance','native_evidence_frames','ownership_support','uncertain_boundary'],
            physical_coexistence_does_not_prove_event_overlap=True,shared_clause_does_not_merge_events=True,empirical_order_is_not_automatic_language_rule=True,
            no_forced_expected_order=True,no_frame_removal=True),
        feedback=dict(source_clause_text_required_for_claim=True,observed_predicate_difference_required=True,event_binding_support_required=True,
            an_internal_family_disagreement_alone_is_not_a_linguistic_error=True,uncertain_boundary_is_not_wrong_handshape=True,
            current_image_absent_cannot_become_direct_observation=True,no_unstated_mandatory_pause=True,no_unstated_repeat_count=True,
            preserve_short_contrary_motion=True,normal_speed_not_error=True,whole_sign_correctness_not_established=True),
        families=families,clauses=list(clauses.values()),shared_clauses=shared,pins=pins,
        evidence=dict(public_fit_windows=18,source_frames=248,word_count=4,recorded_ids=list(range(1,10)),
            direct_visual_review='root9arm-traversal windows plus separate Codex agent9other-word controls. No independent human or KSL expert claim.',
            exact_event_boundaries='not certified',existing_label_revisions=0,new_dense_targets=0),
        decision='Do not tune the same role head again. First compile predicate-specific masked supervision and event-instance support, retaining old exclusive-family scores as separate diagnostics. Compare whether the learned scene evidence supports the source action, not whether a sparse phase boundary is matched exactly.',
        whole_goal_complete=False)
    write(U/'EVIDENCE_CONTRACT.json',contract)
    write(U/'REVIEW_VERIFIED.json',dict(status='SOURCE_EVIDENCE_CONTRACT_REVIEW_COMPLETE',windows=18,source_frames=248,all_selected_sheets_reviewed=True,
        source_links_checked=True,shared_source_clauses=len(shared),no_model_or_label_change=True,
        qualification='Structural/source correspondence and recorded visual observations, not model performance gain or proof that existing event IDs are false.',whole_goal_complete=False))
    print('18windows248frames; shared clauses',len(shared),'old models and labels unchanged')
if __name__=='__main__':main()
