"""One frozen public experiment. No detector, model training, or word-aware path input."""
from pathlib import Path
import json, hashlib, sys, time, traceback, numpy as np
U=Path(__file__).resolve().parent; OLD=U.parent/'word100_relational_sequence_compare_20261031'
sys.path.insert(0,str(OLD));import relational
from prepare import read,sha,write,load
def path_checks(p,n,m):
    p=np.asarray(p,int);steps=np.diff(p,axis=0)
    assert p.ndim==2 and p.shape[1]==2
    assert np.array_equal(p[0],[0,0]) and np.array_equal(p[-1],[n-1,m-1])
    assert all(tuple(s) in [(1,0),(0,1),(1,1)] for s in steps)
    assert set(p[:,0])==set(range(n)) and set(p[:,1])==set(range(m))
    return p
def evidence(q,r,path):
    # Both original axes stay visible, even when no comparison can be made.
    x,y=q['x'],r['x'];qm,rm=q['mask'],r['mask'];p=path_checks(path,len(x),len(y))
    qs=np.zeros_like(qm);rs=np.zeros_like(rm);qd=np.zeros_like(x);rd=np.zeros_like(y)
    conflicts=[];supported_relations=0
    for i,j in p:
        shared=qm[i]&rm[j]&np.isfinite(x[i])&np.isfinite(y[j]);d=np.where(shared,np.minimum(np.abs(x[i]-y[j]),1),0)
        qs[i]|=shared;rs[j]|=shared;qd[i]=np.maximum(qd[i],d);rd[j]=np.maximum(rd[j],d)
        if qm[i,[1,29]].all() and rm[j,[1,29]].all():
            supported_relations+=1
            a=x[i,1]-x[i,29];b=y[j,1]-y[j,29]
            if a*b<0:conflicts.append([int(i),int(j)])
    def axis(mask,supported,d):
        return {'total_slots':int(mask.size),'observed_slots':int(mask.sum()),'unobserved_slots':int((~mask).sum()),'compared_slots':int(supported.sum()),'observed_without_comparison':int((mask&~supported).sum()),'not_compared_slots':int((~supported).sum()),'shape':{h:{'observed':int(mask[:,cols].sum()),'compared':int(supported[:,cols].sum()),'peak_difference':float(d[:,cols].max(initial=0)) if supported[:,cols].any() else None} for h,cols in [('L',slice(11,25)),('R',slice(39,53))]}}
    return {'query':axis(qm,qs,qd),'reference':axis(rm,rs,rd),'height_conflicts':conflicts,'height_supported_path_cells':supported_relations,'height_relation_is_not_linguistic_truth':True,'no_correctness_verdict':True},qd,rd,qs,rs
def truth_metrics(path,meta,kind,n,m):
    if 'query_unrolled_index' in meta:
        a=np.asarray(meta['query_unrolled_index']);b=np.asarray(meta['reference_unrolled_index'])
        qo=np.asarray(meta['query_occurrence']);ro=np.asarray(meta['reference_occurrence'])
    elif 'original_indices' in meta:
        a=np.asarray(meta['original_indices']);b=np.arange(m);qo=np.zeros(n,int);ro=np.zeros(m,int)
    elif n==m:
        a=np.arange(n);b=np.arange(m);qo=np.zeros(n,int);ro=np.zeros(m,int)
    else:return None
    p=np.asarray(path,int);e=np.abs(a[p[:,0]]-b[p[:,1]]).astype(float)
    qe=np.zeros(n);re=np.zeros(m);np.maximum.at(qe,p[:,0],e);np.maximum.at(re,p[:,1],e)
    mismatch=qo[p[:,0]]!=ro[p[:,1]]
    def seam_mask(occ):
        out=np.zeros(len(occ),bool);edges=np.flatnonzero(np.diff(occ)!=0);out[edges]=True;out[edges+1]=True;return out
    qseam,rseam=seam_mask(qo),seam_mask(ro)
    interior_mean=float((qe[~qseam].mean()+re[~rseam].mean())/2)
    return {'query_mean_max_error_frames':float(qe.mean()),'reference_mean_max_error_frames':float(re.mean()),'mean_two_axes':float((qe.mean()+re.mean())/2),'peak_error_frames':float(e.max(initial=0)),'occurrence_mismatch_cells':int(mismatch.sum()),'query_wrong_occurrence_frames':int(len(set(p[mismatch,0]))),'reference_wrong_occurrence_frames':int(len(set(p[mismatch,1]))),'interior_mean_two_axes':interior_mean,'synthetic_seam':{'query_frames':np.flatnonzero(qseam).tolist(),'reference_frames':np.flatnonzero(rseam).tolist(),'query_mean_max_error':float(qe[qseam].mean()) if qseam.any() else None,'reference_mean_max_error':float(re[rseam].mean()) if rseam.any() else None,'mismatched_path_cells_touching_seam':int((mismatch&(qseam[p[:,0]]|rseam[p[:,1]])).sum())},'source_truth_not_passed_to_candidate':True,'counterfactual_feature_edit':kind not in ['identity','speed_075','speed_125','hold15','repeat2_same','repeat3_same']}
def baseline(q,r):return relational.dtw(relational.alignment_cost(relational.features(q),relational.features(r))).tolist()
def input_digest(q,r):
    return {side:{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in a.items()} for side,a in [('query',q),('reference',r)]}
def guarded_align(align,q,r,bp):
    before=input_digest(q,r)
    for a in [q,r]:
        for v in a.values():v.setflags(write=False)
    st=time.perf_counter();out=align(q,r,bp);elapsed=time.perf_counter()-st
    assert before==input_digest(q,r),'Matcher modified input'
    return out,elapsed,before
def seal(n,v):
    f=U/'results'/f'{n}.json';write(f,v);return {'file':str(f),'sha256':sha(f)}
def check_frozen():
    f=read(U/'FROZEN.json');inp=read(U/'INPUTS.json')
    for p,h in {**f['pins'],**inp['pins']}.items():assert sha(p)==h,('pin changed',p)
    return f,inp
def main():
    from candidate import align
    frozen,inp=check_frozen();assert not (U/'RESULT_MANIFEST.json').exists()
    lock=U/'RUNNING.json';write(lock,{'started':time.time(),'purpose':'one frozen candidate run','no_pid_only_unlock':True})
    state={'status':'RUNNING','active_session':'root exec recorded in external log','frozen_sha256':sha(U/'FROZEN.json')}
    (U/'CURRENT_EXECUTION.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    manifest={'frozen_sha256':sha(U/'FROZEN.json'),'controls':[],'natural':[]}
    try:
        for c in inp['controls']:
            q,r=load(c['input']),load(c['reference']);bp=read(c['baseline'])['path'] if c['baseline'] else baseline(q,r)
            out,elapsed,digests=guarded_align(align,q,r,bp)
            path=out['path'];be,bqd,brd,bqs,brs=evidence(q,r,bp);ce,cqd,crd,cqs,crs=evidence(q,r,path)
            changed=c['metadata'].get('changed_frames',[]);cols=slice(11,25) if c['kind']=='left_short_shape_change_5f' else slice(39,53)
            changed_values=None
            if changed:
                bm=bqd[changed,cols].max(axis=1);cm=cqd[changed,cols].max(axis=1)
                known=q['mask'][changed,cols]
                changed_values={'frames':changed,'before_max':bm.tolist(),'after_max':cm.tolist(),'nonzero_before':int((bm>1e-6).sum()),'nonzero_after':int((cm>1e-6).sum()),'actual_observed_changed_cells':c['metadata'].get('actual_observed_changed_cells',0),'known_edited_cells':int(known.sum()),'baseline_compared_edited_cells':int((known&bqs[changed,cols]).sum()),'candidate_compared_edited_cells':int((known&cqs[changed,cols]).sum()),'lost_compared_edited_cells':int((known&bqs[changed,cols]&~cqs[changed,cols]).sum())}
            d={'id':c['id'],'kind':c['kind'],'input':c['input'],'reference':c['reference'],'baseline_source':c['baseline'],'baseline_path':bp,'candidate':out,'seconds':elapsed,'baseline_evidence':be,'candidate_evidence':ce,'baseline_truth':truth_metrics(bp,c['metadata'],c['kind'],len(q['x']),len(r['x'])),'candidate_truth':truth_metrics(path,c['metadata'],c['kind'],len(q['x']),len(r['x'])),'changed_shape':changed_values,'input_digests_checked_before_and_after':digests,'existing_input_values_unchanged':True}
            manifest['controls'].append({'id':c['id'],**seal(c['id'],d)})
            if len(manifest['controls'])%16==0:print('controls',len(manifest['controls']),flush=True)
        for pair in inp['natural']:
            a,b=pair['query'],pair['reference']
            for panel in ['original','merged']:
                q,r=load(inp['inputs'][a][panel]),load(inp['inputs'][b][panel])
                bp=read(pair['same_cost_path'])['path'] if panel=='original' else baseline(q,r)
                out,elapsed,digests=guarded_align(align,q,r,bp)
                be,*_=evidence(q,r,bp);ce,*_=evidence(q,r,out['path'])
                d={'query':a,'reference':b,'panel':panel,'input':inp['inputs'][a][panel],'reference_input':inp['inputs'][b][panel],'baseline_source':pair['same_cost_path'] if panel=='original' else 'new matched-input unconstrained control','global_saved_source':pair['global_path'],'baseline_path':bp,'candidate':out,'seconds':elapsed,'baseline_evidence':be,'candidate_evidence':ce,'true_semantic_time_correspondence':None,'parent_ids':[x['id'] for x in inp['parents'] if x['query']==a and x['reference']==b],'input_digests_checked_before_and_after':digests}
                manifest['natural'].append({'id':f'{panel}_{a}_{b}',**seal(f'{panel}_{a}_{b}',d)})
            print('natural',a,b,flush=True)
        write(U/'RESULT_MANIFEST.json',manifest);state.update(status='COMPLETE',active_session=None,control_results=len(manifest['controls']),natural_results=len(manifest['natural']),no_new_training=True,no_new_observation=True)
        (U/'CURRENT_EXECUTION.json').write_text(json.dumps(state,indent=2),encoding='utf-8');lock.unlink()
        print('COMPLETE',len(manifest['controls']),len(manifest['natural']),flush=True)
    except Exception:
        f=U/'FIRST_EXECUTION_ERROR.txt'
        if not f.exists():f.write_text(traceback.format_exc(),encoding='utf-8')
        state.update(status='FAILED_PRESERVED',active_session=None);(U/'CURRENT_EXECUTION.json').write_text(json.dumps(state,indent=2),encoding='utf-8');raise
if __name__=='__main__':main()
