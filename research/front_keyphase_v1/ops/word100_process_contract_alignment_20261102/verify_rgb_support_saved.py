"""Independent np.interp basis-vector support audit; no models/media/extraction."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import ast
import hashlib
import json
import numpy as np

U=Path(__file__).resolve().parent
COMMON=U.parent/'word100_signrep_feasibility_20261029/common.py'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    dest=U/'RGB_SUPPORT_VERIFIED.json';assert not dest.exists()
    # Execute only the inspected starts function, not common module side effects.
    node=next(n for n in ast.parse(COMMON.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='starts')
    env={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(COMMON),'exec'),env)
    starts=env['starts'];checks=Counter()
    def ok(value,name):
        assert value,name
        checks[name]+=1
    def independent(ts,av,finite):
        n=len(ts)
        if n<2:return [],np.empty((n,0)),[[] for _ in ts],[[] for _ in ts]
        if n<16:
            windows=[list(range(n))+[n-1]*(16-n)];basis=np.ones((n,1))
        else:
            ss=starts(n);windows=[list(range(s,s+16)) for s in ss]
            centers=(np.asarray(ts,np.int64)[ss]+np.asarray(ts,np.int64)[np.asarray(ss)+15])/2
            basis=np.stack([np.interp(ts,centers,np.eye(len(ss))[j]) for j in range(len(ss))],axis=1)
        encoded=[];consumed=[]
        for i in range(n):
            union=sorted({r for w in np.flatnonzero(basis[i]>0) for r in windows[w]})
            encoded.append(union);consumed.append(union if av[i] and finite[i] else [])
        return windows,basis,encoded,consumed
    # Synthetic unit tests call only the new pure helper and the original pure
    # starts definition. These are interpolation arithmetic, not model inference.
    from process_rgb_support import build_rgb_support,span_support
    synthetic=[]
    for n in [0,1,2,3,7,15,16,17,18,19,20,31,32,33,80,120]:
        for axis in ('regular','irregular'):
            ts=[round(i*1e6/30) for i in range(n)] if axis=='regular' else np.cumsum([0]+[10000+(i*7919)%17000 for i in range(1,n)]).tolist() if n else []
            av=[True]*n;s=build_rgb_support(ts,av)
            windows,basis,encoded,consumed=independent(ts,av,[True]*n)
            ok(s['row_encoded_support']==encoded,'synthetic_structural_support')
            ok(s['row_consumed_support']==consumed,'synthetic_consumed_support')
            ok([w['slot_rows'] for w in s['windows']]==windows,'synthetic_window_layout')
            synthetic.append(dict(n=n,axis=axis,status=s['status']))
    ts=list(range(0,140,10))+[135,140,150,160,170,180]
    s=build_rgb_support(ts,[True]*20)
    for i in (7,9,11):
        ok(len(s['row_window_terms'][i])==1,'exact_center_excludes_zero_neighbors')
    ok(s['row_window_terms'][0]==[[0,1,1]],'left_extrapolation_constant')
    ok(s['row_window_terms'][-1]==[[len(s['windows'])-1,1,1]],'right_extrapolation_constant')
    s=build_rgb_support([0,1000],[True,False],source_frame_ids=[10,30],source_timestamp_us=[2000,4000])
    ok(s['padding_slots']==14 and s['row_encoded_support']==[[0,1],[0,1]],'short_padding_no_extra_observation')
    ok(s['row_consumed_support']==[[0,1],[]],'blackout_row_removed_neighbor_keeps_synthetic_context')
    sp=span_support(s,[0,1])
    ok(sp['observed_context_source_frames']==[10] and sp['synthetic_context_source_coordinates']==[30],'blackout_not_original_observation')
    ok(20 not in sp['observed_context_source_frames'],'source_gap_not_filled')
    s=build_rgb_support(list(range(20)),[True]*20,rgb_finite=[False]+[True]*19)
    ok(s['row_consumed_support'][0]==[] and s['row_encoded_support'][0]!=[],'nonfinite_row_removed')

    manifest_path=U/'RGB_SUPPORT_MANIFEST.json';m=read(manifest_path)
    for p,h in m['pins'].items():ok(sha(p)==h,'pin_unchanged')
    independent_by={}
    boundary=Counter()
    for key,item in m['inputs'].items():
        s=item['support'];ts=s['timestamp_us'];av=s['image_available'];finite=s['rgb_finite'];n=len(ts)
        windows,basis,encoded,consumed=independent(ts,av,finite)
        independent_by[key]=(encoded,consumed)
        ok([w['slot_rows'] for w in s['windows']]==windows,'actual_window_layout')
        for i,terms in enumerate(s['row_window_terms']):
            expected=set(np.flatnonzero(basis[i]>0).tolist())
            ok({w for w,num,den in terms}==expected,'actual_nonzero_window_set')
            ok(all(num>0 and den>0 for w,num,den in terms),'no_zero_weight_support')
            ok(sum(Fraction(num,den) for w,num,den in terms)==1,'rational_weights_sum_one')
            for w,num,den in terms:ok(abs(num/den-basis[i,w])<=2e-15,'np_interp_coefficient_match')
            ok(s['row_encoded_support'][i]==encoded[i],'actual_encoded_union')
            ok(s['row_consumed_support'][i]==consumed[i],'actual_consumed_union')
            boundary['one_window_rows' if len(terms)==1 else 'two_window_rows']+=1
            boundary['masked_rgb_rows']+=not (av[i] and finite[i])
        ok(s['padding_slots']==sum(16-n for _ in windows) if n<16 else s['padding_slots']==0,'actual_padding_count')
    summary=Counter()
    for r in m['contexts']+m['candidates']:
        item=m['inputs'][r['input_sha256']];s=item['support'];sp=r['span'];ii=sp['nominal_cache_rows']
        encoded,consumed=independent_by[r['input_sha256']]
        raw=sorted({j for i in ii for j in encoded[i]});used=sorted({j for i in ii for j in consumed[i]})
        outside=[j for j in used if j not in set(ii)]
        good=[j for j in used if s['image_available'][j]];black=[j for j in used if not s['image_available'][j]]
        ok(ii==list(range(ii[0],ii[-1]+1)),'span_contiguous_cache_axis')
        ok(sp['encoded_context_cache_rows']==raw and sp['consumed_context_cache_rows']==used,'span_union_independent')
        ok(sp['outside_nominal_consumed_rows']==outside,'span_outside_nominal')
        ok(sp['nominal_source_frames']==[s['source_frame_ids'][i] for i in ii],'nominal_source_axis')
        ok(sp['nominal_source_timestamps_us']==[s['source_timestamp_us'][i] for i in ii],'nominal_source_time')
        ok(sp['observed_context_source_frames']==[s['source_frame_ids'][i] for i in good],'observed_source_axis')
        ok(sp['synthetic_context_cache_rows']==black,'synthetic_context_separate')
        ok(sp['local_process_truth'] is None and sp['exact_completion_boundary'] is None,'support_never_target')
        if 'candidate_id' in r:
            summary['candidates']+=1
            summary['with_outside_context']+=bool(outside)
            summary['with_synthetic_context']+=bool(black)
    ok(len(m['inputs'])==344 and len(m['contexts'])==417,'expected_input_context_counts')
    ok(summary['candidates']==m['summary']['candidate_count'],'candidate_population_preserved')
    obj=dict(status='PASS_EXACT_RGB_STRUCTURAL_SUPPORT_NO_MODEL_EXECUTION',
        manifest_sha256=sha(manifest_path),helper_sha256=sha(U/'process_rgb_support.py'),verifier_sha256=sha(Path(__file__)),
        checks=dict(checks),checks_total=sum(checks.values()),synthetic_cases=synthetic,
        actual_summary=m['summary'],interpolation_boundary_counts=dict(boundary),candidate_summary=dict(summary),
        algorithm='Independent dense np.interp one-hot window basis; source window starts function extracted by AST without module/model import.',
        new_model_inference=0,new_training=0,new_extraction=0,new_scores=0,source_media_decoded=0,
        model_weights_loaded=False,targets_changed=False,old_results_changed=False,
        limitation='Exact structural input dependence is not effective neural influence, physical observation correctness or exact event boundaries.')
    dest.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=obj['status'],checks_total=obj['checks_total'],summary=m['summary'],path=str(dest)),ensure_ascii=False))

if __name__=='__main__':main()
