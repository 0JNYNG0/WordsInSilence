"""Saved-file arithmetic only; no implementation/model imports or inference."""
import ast, collections, datetime, json, sys
from pathlib import Path
import numpy as np
from verify_phase_projection import UNIT, OLD, KNOWN, read, digest, independent_projection, synthetic_controls

checks=[]
def ck(name,actual,expected=True):
    checks.append(dict(name=name,passed=actual==expected,detail=None if actual==expected else dict(actual=actual,expected=expected)))
def normalized(v):
    return None if v in (None,'unknown','uncertain') else v

def run(output_name):
    auth=read(UNIT/'AUTHORIZATION.json'); rows={r['alias']:r for r in auth['rows']}
    fit=read(UNIT/'PHASE_FIT.json'); dev=read(UNIT/'PHASE_DEV.json')
    raw={c['alias']:c for c in fit['clips']+dev['clips']}
    data=read(UNIT/'PHASE_DATA.json'); saved=read(UNIT/'STAGE_PROJECTION.json')
    freeze=read(UNIT/'FROZEN_DIAGNOSTIC.json'); sources=read(UNIT/'SOURCE_PHASE_CONTRACTS.json')
    empirical=read(OLD/'EMPIRICAL_RANGE.json')
    oracle=independent_projection(raw,auth['rows'],empirical['development'])
    for pin in read(UNIT/'review/PHASE_INDEPENDENT_ORACLE.json')['inputs']:
        ck('oracle_input_unchanged/'+Path(pin['path']).name,digest(pin['path']),pin['sha256'])
    for path,expected in freeze['pins'].items(): ck('freeze/'+Path(path).name,digest(path),expected)
    ck('saved_contract',saved['contract'],freeze)
    for key in ['new_training','new_model_inference','new_DTW_search']: ck(key,freeze[key],False)
    ck('diagnostic_status',saved['status'],'DIAGNOSTIC_ONLY_NOT_ADOPTED')
    ck('semantic_identity',saved['semantic_stage_identity_proven'],False)
    ck('fit12',len(fit['clips']),12); ck('dev4',len(dev['clips']),4)
    ck('disjoint',not ({c['alias'] for c in fit['clips']}&{c['alias'] for c in dev['clips']}))
    ck('all_aliases',sorted(c['alias'] for c in data['clips']),sorted(rows))
    ck('dev_aliases',sorted(c['alias'] for c in saved['clips']),sorted(oracle))
    expressions={e['word_id']:e for e in sources['expressions']}
    role_counts=collections.Counter(); support_totals=collections.Counter()
    linked_segments=associations=anchor_occurrences=0
    for c in data['clips']:
        a=c['alias']; r=rows[a]; original=raw[a]
        ck(a+'/input_pin',digest(r['input']),r['input_sha256'])
        with np.load(r['input'],allow_pickle=False) as z:
            x=z['x']; mask=z['mask']; ids=z['frame_id']; times=z['timestamp_us']
        ck(a+'/all_frame_ids',ids.tolist(),list(range(r['frames'])))
        ck(a+'/frame_count',c['frames'],r['frames'])
        ck(a+'/PTS',c['seconds'],[int(t)/1000000 for t in times])
        ck(a+'/PTS_monotonic',all(int(p)<int(q) for p,q in zip(times,times[1:])))
        names=r['feature_names']; ck(a+'/width',x.shape[1],len(names));ck(a+'/mask_shape',list(mask.shape),list(x.shape))
        original_segments={s['segment_id']:s for s in original['segments']}
        ck(a+'/segment_ids',[s['segment_id'] for s in c['segments']],list(original_segments))
        occupied=set()
        for s in c['segments']:
            sid=s['segment_id']; origin=original_segments[sid]
            for key,value in origin.items(): ck(sid+'/preserved/'+key,s.get(key),value)
            lo,hi=s['core'];n=hi-lo+1
            ck(sid+'/bounds',0<=lo<=hi<r['frames']);ck(sid+'/role',s['role'] in KNOWN|{'uncertain'})
            ck(sid+'/nonoverlap',not(occupied&set(range(lo,hi+1))));occupied.update(range(lo,hi+1))
            role_counts[s['role']]+=1
            for side in ['L','R']:
                groups={'position_frames':['hand_wrist_x','hand_wrist_y'],'palm_frames':['palm_center_x','palm_center_y'],'arm_frames':['pose_wrist_x','pose_wrist_y','elbow_x','elbow_y']}
                computed={'frames':n}
                for metric,cols in groups.items():
                    indices=[names.index(side+'_'+name) for name in cols]
                    computed[metric]=sum(all(bool(mask[f,j]) and bool(np.isfinite(x[f,j])) for j in indices) for f in range(lo,hi+1))
                angle_names=['thumb_mcp_angle','thumb_ip_angle']+[finger+'_'+joint+'_angle' for finger in ['index','middle','ring','little'] for joint in ['mcp','pip','dip']]
                indices=[names.index(side+'_'+name) for name in angle_names]
                computed['angle_cells']=sum(bool(mask[f,j]) and bool(np.isfinite(x[f,j])) for f in range(lo,hi+1) for j in indices)
                computed['angle_expected']=n*len(indices)
                ck(sid+'/support/'+side,s['coordinate_support'][side],computed)
                for key,value in computed.items(): support_totals[side+'_'+key]+=value
            e=expressions[r['word_id']]; expected=[]
            for clause in e['clauses']:
                anchors=sorted({q['frame_id'] for ev in clause.get('evidence',[]) if ev['alias']==a for q in ev['anchors'] if lo<=q['frame_id']<=hi})
                if anchors: expected.append(dict(clause_id=clause['id'],text=clause['source_span'],url=clause['source_url'],anchor_frames=anchors,status=clause['visible_match_status'],training_target=False,scope='listed anchors only; whole span not proven'))
            ck(sid+'/source_links',s['source_links'],expected)
            uncovered=sorted({q['frame_id'] for ev in e.get('visible_source_uncovered_actions',[]) if ev['alias']==a for q in ev['anchors'] if lo<=q['frame_id']<=hi})
            ck(sid+'/uncovered',s['source_uncovered_anchors'],uncovered)
            linked_segments+=bool(s['source_links']);associations+=len(s['source_links'])
            anchor_occurrences+=sum(len(link['anchor_frames']) for link in s['source_links'])
        text=json.dumps(c,ensure_ascii=False,sort_keys=True)
        for key in ['unknown','notes','reviewer_note']:
            if key in original:
                values=original[key] if isinstance(original[key],list) else [original[key]]
                for j,item in enumerate(values): ck(a+'/preserved_caveat/'+key+'/'+str(j),json.dumps(item,ensure_ascii=False,sort_keys=True) in text)
    summary=data['summary']
    for key,value in dict(clips=16,frames=sum(r['frames'] for r in rows.values()),segments=sum(role_counts.values()),role_counts=dict(role_counts),source_linked_segments=linked_segments).items():ck('summary/'+key,summary[key],value)
    association_key='source_clause_links' if 'source_clause_links' in summary else 'source_anchor_links'
    ck('summary/clause_associations',summary[association_key],associations)
    for key in ['source_anchor_occurrences','source_anchor_frame_occurrences']:
        if key in summary:ck('summary/'+key,summary[key],anchor_occurrences)
    diagnostics=[]
    for c in saved['clips']:
        a=c['alias']; e=oracle[a]; n=e['all_native_frames'];refs=c['reference_projections']
        ck(a+'/expected_refs',sorted(p['reference'] for p in refs),sorted(e['references']))
        ck(a+'/3distinct',len(refs)==len({p['reference'] for p in refs})==3)
        for p in refs:
            ref=p['reference']
            ck(a+'/'+ref+'/roles',[normalized(v) for v in p['projected_roles']],e['per_reference'][ref])
            ck(a+'/'+ref+'/all_mapped',p['reference_frames_per_query'],e['mapped_reference_frames'][ref])
        ck(a+'/consensus',[normalized(v) for v in c['consensus_roles']],e['consensus'])
        compared=[i for i,(t,p) in enumerate(zip(e['dev_draft_roles'],e['consensus'])) if t is not None and p is not None]
        mismatch=[dict(frame=i,projected=e['consensus'][i],rgb_draft=e['dev_draft_roles'][i]) for i in compared if e['consensus'][i]!=e['dev_draft_roles'][i]]
        values=dict(frames=n,native_frames_retained=n,consensus_frames=e['known_projection'],unknown_frames=e['unknown_projection'],draft_comparable_frames=len(compared),draft_agreement_frames=e['known_draft_match'],draft_disagreement=mismatch,not_accuracy=True)
        for key,value in values.items():ck(a+'/'+key,c[key],value)
        diag={k:v for k,v in e.items() if k not in ['per_reference','mapped_reference_frames','consensus','dev_draft_roles']}
        diag.update(alias=a,both_known=len(compared),projected_known_on_draft_unknown=sum(t is None and p is not None for t,p in zip(e['dev_draft_roles'],e['consensus'])))
        diagnostics.append(diag)
        ck(a+'/data_projection_copy',next(x for x in data['clips'] if x['alias']==a)['projection'],c)
    for key in ['frames','consensus_frames','draft_comparable_frames','draft_agreement_frames','unknown_frames','native_frames_retained']:ck('projection_summary/'+key,saved['summary'][key],sum(c[key] for c in saved['clips']))
    for control in synthetic_controls():ck('oracle_control/'+control['name'],control['pass'])
    tree=ast.parse((UNIT/'integrate_evidence.py').read_text(encoding='utf-8'))
    imports=[name.name for n in ast.walk(tree) if isinstance(n,ast.Import) for name in n.names]
    ck('no_model_imports',not any(s.startswith(('torch','tensorflow','mediapipe','onnx')) for s in imports))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='roles')
    namespace={}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'reviewed_roles_function_only','exec'),namespace)
    fixture=dict(alias='synthetic',frames=6,segments=[dict(segment_id='a',role='rest',core=[0,1]),dict(segment_id='u',role='uncertain',core=[3,3]),dict(segment_id='b',role='return',core=[5,5])])
    ck('implementation_roles_unknown_gap_endpoints',namespace['roles'](fixture),[('a','rest'),('a','rest'),None,None,None,('b','return')])
    per_role={role:{key:sum(d['per_draft_role'][role][key] for d in diagnostics) for key in ['draft_frames','match','projected_unknown','wrong_known_role']} for role in sorted(KNOWN)}
    failures=[c for c in checks if not c['passed']]
    report=dict(status='PASS_STORED_ARITHMETIC_AND_BOUNDED_CONTRACT' if not failures else 'FAIL_REVIEW_FINDINGS_REMAIN',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer='Codex /root/rgb_labels_b; separate code/arithmetic reviewer and DEV RGB author, not independent human linguistic truth',checks=dict(total=len(checks),passed=len(checks)-len(failures),failed=len(failures)),failures=failures,check_details=checks,per_clip_diagnostics=diagnostics,per_draft_role_diagnostics=per_role,coordinate_support_totals_within_draft_cores_only=dict(support_totals),source_link_units=dict(linked_segments=linked_segments,segment_clause_associations=associations,anchor_frame_occurrences=anchor_occurrences),scope_limits=['No new inference/training/DTW search or prior model rescoring.','Sampled phase-draft agreement is diagnostic, not recognition accuracy.','Rest and active denominators are separated; unannotated gaps remain unknown.','A projected role does not certify handshape visibility/contact/depth/semantic identity/required order/repetition completion.','Source contract preservation/linkage checked; linguistic truth not certified.','Synthetic control count is not accuracy.'])
    paths=[UNIT/n for n in ['AUTHORIZATION.json','PHASE_FIT.json','PHASE_DEV.json','SOURCE_PHASE_CONTRACTS.json','integrate_evidence.py','FROZEN_DIAGNOSTIC.json','PHASE_DATA.json','STAGE_PROJECTION.json']]+[OLD/'EMPIRICAL_RANGE.json',Path(__file__),UNIT/'review/verify_phase_projection.py']
    report['files']=[dict(path=str(p),sha256=digest(p)) for p in paths]
    out=UNIT/'review'/output_name
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(output=str(out),status=report['status'],checks=report['checks'],per_role=per_role),ensure_ascii=False))
    if failures:print(json.dumps(failures[:12],ensure_ascii=False))
    return int(bool(failures))

if __name__=='__main__':
    raise SystemExit(run(sys.argv[1] if len(sys.argv)>1 else 'PHASE_VERIFICATION.json'))
