"""Add common body-relative measurements without modifying frozen first results."""
from run import U,OLD,read,write,sha,load_data,observation_digest
import engine_v2 as engine
from relational_check import forearm_relation,local_relation,selftest
from pathlib import Path
import numpy as np,copy,time,json

def main():
    f=read(U/'FROZEN.json');initial=read(U/'RESULT.json');spec=read(U/'source_audit/SOURCE_CONTRACTS.json')
    contracts={c['word_id']:c for c in spec['expressions']}
    # Explicit semantic relation from source phrase. No word ID branch exists.
    link=[]
    for expression in spec['expressions']:
        for c in expression['clauses']:
            if '왼 팔등' in c['source_span'] and '스쳐' in c['source_span'] and c['link_status']=='proxy_only':
                link.append(dict(clause_id=c['id'],relation='hand_along_opposite_forearm',actor='R',other='L',source_span=c['source_span'],source_url=c['source_url'],contact_supported=False,required_complete_stroke_fraction=None))
    protocol=dict(status='FROZEN_BEFORE_BODY_RELATIVE_MEASUREMENTS',source_links=link,window='five consecutive original frames centered on existing mapped anchor; no missing fill',
        decision='If two references have same-sign nonzero local progress and query has that sign on every step while original binary motion says opposite, mark evidence conflict. Do not call normal, wrong, or prove contact.',
        coordinate_basis='Projection of acting palm center on current opposite elbow→pose-wrist axis. Measure both anatomical hand/arm pairs for ALL48 fit/dev clips.',
        residual='Projection may agree despite no actual contact; planar estimate and detector noise remain. No linguistic tolerance or partial-stroke threshold is invented.',
        pins={str(p):sha(p) for p in [Path(__file__),U/'engine_v2.py',U/'relational_check.py',U/'FROZEN.json',U/'RESULT.json',U/'source_audit/SOURCE_CONTRACTS.json']})
    write(U/'RELATIONAL_PROTOCOL.json',protocol)
    controls=selftest();info,arrays,cases,obs=load_data(f)
    digest={a:observation_digest(o) for a,o in obs.items()};tracks={};trackrows=[]
    names=f['rows'][0]['feature_names'];dest=U/'relations';dest.mkdir(exist_ok=False)
    for row in f['rows']:
        alias=row['alias'];parts={}
        for actor,other in [('R','L'),('L','R')]:
            rel=forearm_relation(arrays[alias],names,actor,other);tracks[(alias,actor)]=rel
            for field in ['progress','perpendicular_distance','valid']:parts[actor+'_'+field]=rel[field]
        np.savez_compressed(dest/(alias+'.npz'),frame_id=obs[alias]['frame_ids'],timestamp_us=obs[alias]['times_us'],**parts)
        trackrows.append(dict(alias=alias,frames=len(obs[alias]['frame_ids']),valid_R=int(parts['R_valid'].sum()),valid_L=int(parts['L_valid'].sum()),sha256=sha(dest/(alias+'.npz'))))
    final=copy.deepcopy(initial);supplements=[];v2checks=[]
    for out in final['rows']:
        alias=out['query'];q=obs[alias];old=next(r for r in read(OLD/'DEVELOPMENT_COMPARISON.json')['rows'] if r['query']==alias and r['condition']=='pose_capacity_matched')
        refs=[]
        for pair in old['pairs']:
            ref=copy.copy(obs[pair['reference']]);ref['reference_to_query']=pair['reference_to_query'];refs.append(ref)
        revised=engine.run_expression(contracts[out['word_id']],refs,q,f['targets'])
        assert revised['clauses']==out['clauses'] and revised['orders']==out['orders']
        v2checks.append(dict(alias=alias,boundary_fix_does_not_change_saved_development_output=True))
        for clause in out['clauses']:
            links=[l for l in link if l['clause_id']==clause['id']]
            if not links:continue
            rule=links[0];actor=rule['actor'];relation=tracks[(alias,actor)];pairs=[]
            for ref in clause['references']:
                ro=obs[ref['reference']];rlookup={int(fr):i for i,fr in enumerate(ro['frame_ids'])}
                for cell in ref['cells']:
                    if cell.get('query_index') is None:continue
                    qi=cell['query_index'];ri=rlookup[cell['reference_frame']]
                    qr=local_relation(relation,qi,q['frame_ids'],q['times_us'])
                    rr=local_relation(tracks[(ref['reference'],actor)],ri,ro['frame_ids'],ro['times_us'])
                    same=False
                    if qr['status']==rr['status']=='OBSERVED_2D_RELATION':
                        same=(qr['every_step_toward_wrist'] and rr['displacement']>0) or (qr['every_step_toward_elbow'] and rr['displacement']<0)
                    pairs.append(dict(reference=ref['reference'],query_frame=cell['query_frame'],reference_frame=cell['reference_frame'],original_state=cell['state'],query_relation=qr,reference_relation=rr,same_local_along_arm_direction=bool(same)))
            conflict={}
            for pair in pairs:
                if pair['same_local_along_arm_direction'] and pair['original_state']=='OPPOSITE_STATE':conflict.setdefault(pair['query_frame'],set()).add(pair['reference'])
            agreed=[dict(query_frame=k,references=sorted(v)) for k,v in conflict.items() if len(v)>=2]
            supplement=dict(clause_id=clause['id'],alias=alias,rule=rule,pairs=pairs,conflict_consensus=agreed,
                educational_verdict=None,contact_verdict=None,linguistic_tolerance=None)
            supplements.append(supplement);clause['body_relation_evidence']=supplement
            if agreed:
                clause['initial_status']=clause['status'];clause['status']='OBSERVER_RELATION_CONFLICT'
                clause['message_ko']='기존 이동 판정은 참고와 다르지만, 손이 반대팔을 따라 이동하는 방향은 같은 근거가 있습니다. 관측 판단이 충돌하므로 동작 오류로 안내하지 않습니다.'
    assert digest=={a:observation_digest(o) for a,o in obs.items()}
    final['status']='COMPLETE_SOURCE_LINKED_WITH_BODY_RELATION_CONFLICT_GUARD'
    final['initial_summary']=initial['summary']
    final['summary']=dict(initial['summary'])
    final['summary']['clause_statuses']={s:sum(c['status']==s for r in final['rows'] for c in r['clauses']) for s in ['GUIDE_RELEVANT_DIFFERENCE_CANDIDATE','OBSERVER_RELATION_CONFLICT','DESCRIBED_FEATURE_OBSERVED','INSUFFICIENT_LINKED_EVIDENCE','SOURCE_LINK_WITHHELD']}
    final['summary'].update(body_relation_tracks=len(trackrows)*2,body_relation_clips=len(trackrows),claimed_contact_accuracy=None,claimed_linguistic_error_accuracy=None)
    write(U/'RELATIONAL_RESULT.json',dict(status='COMPLETE',controls=controls,tracks=trackrows,supplements=supplements,v2_checks=v2checks,observer_outputs_unchanged=True))
    write(U/'FINAL_RESULT.json',final)
    print(json.dumps(final['summary'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
