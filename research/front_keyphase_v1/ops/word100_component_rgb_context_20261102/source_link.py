"""Attach a same-definition learned observation to a sourced partial clause.

This is a transparent connection/formatting layer, not a new language learner
or a new educational-correctness classifier. No word-specific routing.
"""
from pathlib import Path
import json,hashlib,copy
T=Path(__file__).resolve().parent;O=T.parent
CATALOG=O/'word100_predicate_supervision_20261102/CLAUSE_COMPONENT_LINKS.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
    with (T/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

# A shared physical predicate, never a particular word or performer exception.
ROUTES={
 'right_hand_traverses_left_arm_toward_wrist':dict(
    model='rgb_residual_42',target_index=0,
    source_component='right_hand_toward_left_wrist_along_arm',
    name='오른손이 왼팔을 따라 왼손목 쪽으로 이동하는 흐름')
}

def link(row,catalog):
    record=row['record'];route=ROUTES.get(record['target']);clauses=[]
    if route is None:return dict(component_id=record['target'],status='NO_EQUIVALENT_OBSERVER',whole_sign_correctness=None)
    for clause_id in record['source_clause_ids']:
        c=catalog[clause_id]
        if route['source_component'] not in c['related_observable_components']:continue
        clauses.append(dict(id=clause_id,text=c['text'],url=c['url'],source=c['source'],
                            unchecked=list(c['unsupported_components'])))
    pp=row['probabilities'];observed=row['usable'] and row['image_coverage']>=.8 and row['geometry_coverage']>=.5
    if not observed:status='INSUFFICIENT_OBSERVATION';message='이 시간대는 영상·관절 근거가 부족하여 움직임을 확인할 수 없습니다.'
    elif pp[5]<.8:status='CONTINUITY_UNCONFIRMED';message='움직임의 연결 과정을 확인하지 못했습니다. 동작 오류라는 뜻은 아닙니다.'
    elif pp[0]>=.8:status='RELATED_MOVEMENT_OBSERVED';message=route['name']+'을 확인했습니다.'
    else:status='RELATED_MOVEMENT_UNCONFIRMED';message=route['name']+'을 확실하게 확인하지 못했습니다. 잘못했다는 판정은 아닙니다.'
    return dict(clip_id=record['clip_id'],role=record['role'],component_id=record['target'],learned_observer=route,source_clauses=clauses,
      reviewed_sparse_witness_frames=record['native_frames'],dense_context_is_not_dense_truth=True,
      native_frames=list(range(row['extent'][0],row['extent'][1]+1)),
      start_seconds=row['source_times_us'][0]/1e6,end_seconds=row['source_times_us'][1]/1e6,
      status=status,message=message,probability=pp[0],continuity_probability=pp[5],
      proposed_scope='reviewed_public_annotation_window_not_automatic_user_localization',
      whole_sign_correctness=None,whole_clause_correctness=None,
      correction_enabled=False,diagnostic_geometry=row['comparators'],
      old_models_modified=False,reference_video_comparisons=0)

def main():
    assert read(T/'SCOPE_SAVED_VERIFIED.json')['status']=='PASS_SAVED_NATIVE_SCOPE_DIAGNOSTIC'
    rows=read(T/'SCOPE_RESULT.json')['rows'];catalog={r['clause_id']:r for r in read(CATALOG)['rows']}
    linked=[]
    for i,r in enumerate(rows):
        item=link(r,catalog)
        item['input_evidence']=dict(path=str(T/'SCOPE_RESULT.json'),sha256=sha(T/'SCOPE_RESULT.json'),pointer='/rows/'+str(i))
        linked.append(item)
    assert all(r['source_clauses'] and r['whole_clause_correctness'] is None and not r['correction_enabled'] for r in linked)
    # A real missing image must defeat a high cached posterior. A low posterior
    # alone must not become a claim that a learner used the wrong sign.
    missing=copy.deepcopy(rows[0]);missing['image_coverage']=0
    assert link(missing,catalog)['status']=='INSUFFICIENT_OBSERVATION'
    low=copy.deepcopy(rows[0]);low['probabilities'][0]=.001
    assert link(low,catalog)['status']=='RELATED_MOVEMENT_UNCONFIRMED' and not link(low,catalog)['correction_enabled']
    different=copy.deepcopy(rows[0]);different['record']['target']='right_elbow_lowers_hand_remains_raised'
    assert link(different,catalog)['status']=='NO_EQUIVALENT_OBSERVER'
    for a,b in zip(rows,linked):
        assert [b['start_seconds'],b['end_seconds']]==[t/1e6 for t in a['source_times_us']]
        assert len(b['native_frames'])==a['extent'][1]-a['extent'][0]+1
    save('SOURCE_LINKED_EVIDENCE_v2.json',dict(rows=linked,records=len(linked),checks=dict(
       all_native_extents_preserved=True,source_unsupported_requirements_preserved=True,
       high_probability_missing_image_held=True,low_probability_not_correction=True,
       different_physical_predicate_not_substituted=True),
       scope='Offline partial source-evidence connection verified; automatic user localization and demo deployment not performed',
       pins={str(p):sha(p) for p in [T/'SCOPE_RESULT.json',T/'source_link.py',CATALOG]},whole_goal_complete=False))
    print('Linked12source scopes; uncertainty, native time and predicate identity preserved')
if __name__=='__main__':main()
