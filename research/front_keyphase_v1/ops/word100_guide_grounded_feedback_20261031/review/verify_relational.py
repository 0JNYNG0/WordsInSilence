"""Independent stored 2D relation audit, with no project module imports."""
from pathlib import Path
import json,hashlib,math
import numpy as np
U=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    count=0;fail=[]
    def check(ok,name,detail=None):
        nonlocal count
        count+=1
        if not bool(ok):fail.append(dict(check=name,detail=detail))
    def equivalent(a,b):
        if isinstance(a,dict):return set(a)==set(b) and all(equivalent(v,b[k]) for k,v in a.items())
        if isinstance(a,list):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
        if isinstance(a,float):return isinstance(b,(int,float)) and math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12)
        return a==b
    frozen=read(U/'FROZEN.json');proto=read(U/'RELATIONAL_PROTOCOL.json');rr=read(U/'RELATIONAL_RESULT.json');initial=read(U/'RESULT.json');final=read(U/'FINAL_RESULT.json')
    for p,h in proto['pins'].items():check(sha(p)==h,'relational_pin/'+Path(p).name)
    rows={r['alias']:r for r in frozen['rows']};tracks={};frames_total=0
    check(len(rr['tracks'])==48 and {r['alias'] for r in rr['tracks']}==set(rows),'all48_clips_both_sides')
    for record in rr['tracks']:
        alias=record['alias'];row=rows[alias];path=U/'relations'/f'{alias}.npz';check(sha(path)==record['sha256'],'relation_file_pin/'+alias)
        with np.load(row['input'],allow_pickle=False) as z:x=z['x'];mask=z['mask'];frames=z['frame_id'];ts=z['timestamp_us']
        with np.load(path,allow_pickle=False) as z:saved={k:z[k] for k in z.files}
        check(np.array_equal(saved['frame_id'],frames) and np.array_equal(saved['timestamp_us'],ts),'relation_preserves_all_original_frames/'+alias)
        check(row['feature_names']==frozen['rows'][0]['feature_names'],'consistent_coordinate_schema/'+alias)
        names=row['feature_names'];frames_total+=len(frames)
        for actor,other in [('R','L'),('L','R')]:
            columns=[[names.index(prefix+'_'+axis) for axis in ['x','y']] for prefix in [actor+'_palm_center',other+'_elbow',other+'_pose_wrist']]
            u=np.full(len(frames),np.nan);dist=np.full(len(frames),np.nan);valid=np.zeros(len(frames),bool)
            for i in range(len(frames)):
                palm,elbow,wrist=[[float(x[i,j]) for j in pair] for pair in columns]
                vx=wrist[0]-elbow[0];vy=wrist[1]-elbow[1];den=vx*vx+vy*vy
                if not all(mask[i,j] for pair in columns for j in pair) or not all(math.isfinite(v) for v in palm+[vx,vy]) or den<=1e-12:continue
                px=palm[0]-elbow[0];py=palm[1]-elbow[1]
                u[i]=(px*vx+py*vy)/den
                # Independent cross-product distance to the infinite arm axis,
                # divided by arm length, rather than projection/norm code.
                dist[i]=abs(px*vy-py*vx)/den;valid[i]=True
            check(np.array_equal(valid,saved[actor+'_valid']),'independent_valid_mask/'+alias+'/'+actor)
            check(np.allclose(u,saved[actor+'_progress'],rtol=1e-12,atol=1e-12,equal_nan=True),'independent_axis_progress/'+alias+'/'+actor)
            check(np.allclose(dist,saved[actor+'_perpendicular_distance'],rtol=1e-12,atol=1e-12,equal_nan=True),'independent_cross_product_distance/'+alias+'/'+actor)
            check(record['valid_'+actor]==int(valid.sum()),'valid_frame_count/'+alias+'/'+actor)
            tracks[(alias,actor)]=dict(progress=u,distance=dist,valid=valid,frame_ids=frames,times_us=ts)
    def local(track,frame):
        i=int(np.flatnonzero(track['frame_ids']==frame)[0]);low=i-2;high=i+3
        if low<0 or high>len(track['frame_ids']):return dict(status='INCOMPLETE_WINDOW')
        ids=track['frame_ids'][low:high];ts=track['times_us'][low:high]
        if not track['valid'][low:high].all() or not np.all(np.diff(ids)==1) or not np.all(np.diff(ts)>0):return dict(status='UNOBSERVED_WINDOW')
        progress=track['progress'][low:high];steps=np.diff(progress)
        return dict(status='OBSERVED_2D_RELATION',first_frame=int(ids[0]),last_frame=int(ids[-1]),seconds=(ts/1e6).tolist(),progress=progress.tolist(),perpendicular_distance=track['distance'][low:high].tolist(),displacement=float(progress[-1]-progress[0]),every_step_toward_wrist=bool(np.all(steps>0)),every_step_toward_elbow=bool(np.all(steps<0)),actual_contact=None,linguistic_acceptability=None)
    conflict_frames=0
    for supplement in rr['supplements']:
        actor=supplement['rule']['actor'];alias=supplement['alias'];consensus={}
        for pair in supplement['pairs']:
            query=local(tracks[(alias,actor)],pair['query_frame']);reference=local(tracks[(pair['reference'],actor)],pair['reference_frame'])
            check(equivalent(query,pair['query_relation']),'independent_query_five_frame_relation')
            check(equivalent(reference,pair['reference_relation']),'independent_reference_five_frame_relation')
            same=False
            if query['status']==reference['status']=='OBSERVED_2D_RELATION':same=(query['every_step_toward_wrist'] and reference['displacement']>0) or (query['every_step_toward_elbow'] and reference['displacement']<0)
            check(pair['same_local_along_arm_direction']==same,'independent_direction_match')
            if same and pair['original_state']=='OPPOSITE_STATE':consensus.setdefault(pair['query_frame'],set()).add(pair['reference'])
        expected=[dict(query_frame=frame,references=sorted(refs)) for frame,refs in consensus.items() if len(refs)>=2]
        check(supplement['conflict_consensus']==expected,'independent_conflict_consensus')
        check(supplement['educational_verdict'] is None and supplement['contact_verdict'] is None and supplement['linguistic_tolerance'] is None,'no_contact_or_tolerance_claim')
        conflict_frames+=len(expected)
    check(final['initial_summary']==initial['summary'],'initial_summary_retained')
    changes=0
    for before,after in zip(initial['rows'],final['rows']):
        check(before['query']==after['query'],'query_order_retained')
        for key,value in before.items():
            if key!='clauses':check(after[key]==value,'unchanged_nonclause/'+before['query']+'/'+key)
        for old,new in zip(before['clauses'],after['clauses']):
            check(old['id']==new['id'],'clause_order_retained')
            for key,value in old.items():
                if key not in ['status','message_ko']:check(new[key]==value,'original_anchor_evidence_retained/'+old['id']+'/'+key)
            if old['status']!=new['status']:
                changes+=1;check(new['status']=='OBSERVER_RELATION_CONFLICT','only_abstention_status_change')
                conflicts={x['query_frame'] for x in new['body_relation_evidence']['conflict_consensus']}
                remaining={x['query_frame'] for x in old['coherent_query_frame_consensus'] if x['state']=='OPPOSITE_STATE'}-conflicts
                check(not remaining,'whole_clause_override_does_not_hide_other_candidates',dict(clause=old['id'],remaining=sorted(remaining)))
    counts={key:sum(c['status']==key for r in final['rows'] for c in r['clauses']) for key in final['summary']['clause_statuses']}
    check(final['summary']['clause_statuses']==counts,'final_status_summary')
    check(final['summary']['body_relation_tracks']==96 and final['summary']['body_relation_clips']==48,'final_relation_counts')
    check(final['summary']['claimed_contact_accuracy'] is None and final['summary']['claimed_linguistic_error_accuracy'] is None,'no_accuracy_claim')
    check(len(rr['v2_checks'])==12 and all(x['boundary_fix_does_not_change_saved_development_output'] for x in rr['v2_checks']),'recorded_v2_replay_identity')
    out=dict(status='PASS_WITH_LIMITATIONS' if not fail else 'FAIL',checks=count,failures=fail,model_inference=0,training=0,project_imports=False,clips=48,tracks=96,original_frames_per_side=frames_total,conflict_frames=conflict_frames,changed_clause_statuses=changes,final_clause_statuses=counts,pins={str(U/name):sha(U/name) for name in ['RELATIONAL_PROTOCOL.json','RELATIONAL_RESULT.json','FINAL_RESULT.json','engine_v2.py','add_relational_evidence.py','relational_check.py']},verifier_sha256=sha(__file__),limitations=['Axis projection is not hand-on-forearm contact. Perpendicular distance is reported but no distance tolerance is inferred.','The current dataset produces no relation-conflict consensus and no status change; this extension has not demonstrated a reduction of the remaining difference candidate.','A future clause with several opposite consensus frames must retain unresolved frames if only a subset becomes relationally conflicting.','The reference test uses net local displacement sign while the query requires every step to share that sign; no linguistic necessity or full-stroke coverage is asserted.'])
    with (U/'review/RELATIONAL_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=out['status'],checks=count,failures=len(fail),conflict_frames=conflict_frames,changed_clause_statuses=changes),ensure_ascii=False))
    return int(bool(fail))

if __name__=='__main__':raise SystemExit(main())
