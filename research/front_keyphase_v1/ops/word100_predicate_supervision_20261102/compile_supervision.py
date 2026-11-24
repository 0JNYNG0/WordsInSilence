"""Compile existing direct observations without turning context into new truth."""
from pathlib import Path
from collections import defaultdict
import hashlib, json, re, traceback
import numpy as np

U=Path(__file__).resolve().parent; O=U.parent
P=O/'word100_learned_guide_feedback_20261102'
V=O/'word100_learned_body_relation_20261102'
M=O/'word100_learned_motion_order_20261102'
D=O/'word100_action_episode_supervision_20261102'
Q=O/'word100_event_evidence_contract_20261102'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
    with (U/name).open('x',encoding='utf8') as f: json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)

def counts(rows,names):
    out={}
    for role in ['fit','development']:
        out[role]={}
        for j,name in enumerate(names):
            groups={s:[] for s in ['positive','negative','unknown']}
            for r in rows:
                if r['role']!=role: continue
                state='unknown' if not r['mask'][j] else 'positive' if r['y'][j] else 'negative'
                groups[state].append(r)
            out[role][name]={s:dict(count=len(rr),recorded_ids=sorted({r['person'] for r in rr}),
                words=sorted({r['word_id'] for r in rr}),clips=sorted({r['clip_id'] for r in rr})) for s,rr in groups.items()}
    return out

def main():
    assert not (U/'SUPERVISION.json').exists()
    files=[P/'INPUT_MANIFEST.json',V/'RELATION_TARGETS.json',M/'MOTION_TARGETS.json',D/'ACTION_TARGETS.json',Q/'EVIDENCE_CONTRACT.json']
    pins={str(p):sha(p) for p in files}
    manifest,relations,motions,events,contract=map(read,files)
    mr={r['clip_id']:r for r in manifest['rows']}
    rel={(r['clip_id'],r['center_frame']):(i,r) for i,r in enumerate(relations['rows'])}
    em={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in events['rows']}
    point_names=manifest['targets']+relations['targets']; motion_names=motions['targets']
    assert len(point_names)==10 and len(motion_names)==6 and len(mr)==48
    points=[]; windows=[]; arrays={}; sources=[]
    for i,r in enumerate(manifest['rows']):
        cid=r['clip_id']; assert re.fullmatch('NIA_SL_WORD[0-9]{4}_REAL[0-9]{2}_F',cid)
        assert r['person'] in list(range(1,10))+[11,12,13]
        assert r['role']==('fit' if r['person']<10 else 'development')
        assert sha(r['input'])==r['input_sha256']
        assert sha(r['source'])==r['source_sha256']
        pins[r['input']]=r['input_sha256'];pins[r['source']]=r['source_sha256']
        a=dict(np.load(r['input'],allow_pickle=False));arrays[cid]=a
        assert np.array_equal(a['frame_id'],np.arange(len(a['frame_id'])))
        assert np.all(np.diff(a['timestamp_us'])>0)
        source={k:r[k] for k in ['clip_id','word_id','person','role','source','source_sha256','input','input_sha256']}
        source['native_frames']=len(a['frame_id']);sources.append(source)
        for k,l in enumerate(r['labels']):
            f=l['center_frame'];ri,rr=rel[(cid,f)]
            assert rr['timestamp_us']==l['center_timestamp_us']==int(a['timestamp_us'][f])
            assert rr['image']==l['center_image'] and sha(rr['image'])==rr['image_sha256']
            pins[rr['image']]=rr['image_sha256']
            y=l['y']+rr['y'];mask=l['target_mask']+rr['mask']
            assert all(v in [0,1] for v in y) and all(type(v) is bool for v in mask)
            points.append(dict(id=cid+':frame'+str(f),clip_id=cid,word_id=r['word_id'],person=r['person'],role=r['role'],
                frame=f,timestamp_us=int(a['timestamp_us'][f]),y=y,mask=mask,
                scope='Exact native center only; neighbouring frames provide context, not copied labels',
                observation_provenance=['direct_image' if known else 'unknown' for known in mask],
                reasons=[l['reasons'][n] for n in manifest['targets']]+[rr['provenance']]*3,
                image=rr['image'],image_sha256=rr['image_sha256'],
                source_rows=[dict(path=str(files[0]),pointer=f'/rows/{i}/labels/{k}'),dict(path=str(files[1]),pointer=f'/rows/{ri}')],
                local_event_witness=em.get(cid,{}).get(f),
                event_binding_scope='Saved exact frame witness if present; not a boundary or source-clause entailment',
                boundary_uncertain=True,guide_clause_gold=None))
    for i,r in enumerate(motions['rows']):
        source=mr[r['clip_id']];a=arrays[r['clip_id']]
        assert r['person']==source['person'] and r['role']==source['role']
        assert r['source_sha256']==source['source_sha256'] and r['input_sha256']==source['input_sha256']
        lo,hi=r['frames'];assert 0<=lo<hi<len(a['frame_id'])
        assert r['timestamps_us']==[int(a['timestamp_us'][lo]),int(a['timestamp_us'][hi])]
        windows.append(dict(id=r['id'],clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],role=r['role'],
            frames=[lo,hi],native_frame_ids=list(range(lo,hi+1)),timestamps_us=r['timestamps_us'],
            y=r['y'],mask=r['mask'],kind=r['kind'],scope=r['claim'],provenance=r['review'],
            observation_provenance=['direct_chronological_images' if known else 'unknown' for known in r['mask']],
            source_row=dict(path=str(files[2]),pointer=f'/rows/{i}'),framewise_target_propagation=False,
            boundary_uncertain=True,guide_clause_gold=None))
    assert len(points)==144 and len(windows)==108 and len({r['id'] for r in points})==144
    assert len({r['source_sha256'] for r in sources})==48
    # Relations are semantic scope mappings, not inferred target labels.
    links=[
        ('obedience_closed_hand',['R_closed_four'],['thumb configuration','certain fist']),
        ('obedience_lateral_arm_then_lower',['right_hand_descends_face_to_torso'],['whole lateral-then-lower sequence','required amplitude']),
        ('well_palm_left_arm',['R_open_four','R_hand_at_L_arm'],['palm orientation','actual touch','left-arm surface']),
        ('well_brush',['right_hand_toward_left_wrist_along_arm','continuous_path_evidence'],['actual brushing contact','precise endpoint','mandatory pause']),
        ('ancestor_thumbs',['R_closed_four','L_closed_four'],['thumb extension and orientation']),
        ('ancestor_alternation',['right_hand_rises_relative_left','right_hand_falls_relative_left'],['alternation from one interval alone','required cycle count']),
        ('egypt_tip_meeting',['bilateral_upper_ends_converged'],['actual contact','all fingertip identities']),
        ('egypt_oblique_pose',[],['hand surface orientation','angle','full temporal relation']),
        ('egypt_lower_shared',[],['source variant match','downward sequence evidence']),
        ('egypt_gather_suffix',[],['all fingertips gathered'])]
    clause_lookup={c['clause_id']:c for c in contract['clauses']};clause_links=[]
    for cid,names,unsupported in links:
        c=clause_lookup[cid];assert all(n in point_names+motion_names for n in names)
        clause_links.append(dict(clause_id=cid,text=c['text'],url=c['url'],source=c['source'],
            related_observable_components=names,unsupported_components=unsupported,
            link_type='partial_physical_component_relevance_not_entailment',
            training_label=False,linguistic_correctness_supported=False,source_context_only=c['context_only']))
    audit=dict(point=counts(points,point_names),window=counts(windows,motion_names))
    no_both=[]
    for gran in ['point','window']:
        for name,c in audit[gran]['fit'].items():
            if not c['positive']['count'] or not c['negative']['count']:
                no_both.append(dict(granularity=gran,predicate=name,positive=c['positive']['count'],negative=c['negative']['count']))
    frozen=dict(goal='Source-traceable supervised physical evidence, distinct from event-instance and educational correctness',
        point_targets=point_names,window_targets=motion_names,fit_recorded_ids=list(range(1,10)),development_recorded_ids=[11,12,13],
        development_previously_exposed=True,point_labels_dense_propagation=False,motion_labels_framewise_propagation=False,
        missing_is_negative=False,event_family_to_predicate_label_conversion=False,context_to_source_gold_conversion=False,
        existing_models_and_labels_unchanged=True,locked14_15_and_private_access=False,pins=pins)
    save('FROZEN.json',frozen)
    save('SUPERVISION.json',dict(point_targets=point_names,window_targets=motion_names,sources=sources,points=points,windows=windows))
    save('CLAUSE_COMPONENT_LINKS.json',dict(rows=clause_links,scope='Input for source-action supervision design; not a trained rule or a correctness claim'))
    save('COVERAGE.json',dict(**audit,without_both_fit_classes=no_both,
        evidence_grade='Prior Codex visual labels, no independent human/KSL expert ground truth',
        training_decision='Physical heads already exist. Do not train another duplicate head solely because this inventory is compiled. Event-binding requires actual support/conflict/unknown labels, not copied DTW or source clauses.',
        whole_goal_complete=False))
    print(json.dumps(dict(points=len(points),windows=len(windows),sources=len(sources),without_both_fit_classes=no_both),ensure_ascii=False))

if __name__=='__main__':
    try: main()
    except BaseException:
        p=U/'COMPILE_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
