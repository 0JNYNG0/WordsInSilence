"""Independent arithmetic for new fixed-reference thirds and fixed control ranges."""
import json, math, datetime
from collections import Counter
from pathlib import Path
import numpy as np
import verify_saved_relational as v

U=v.U
def mean(values):return math.fsum(values)/len(values) if values else 0.0
def run():
    auth=v.read(U/'AUTHORIZATION.json');rows={r['alias']:r for r in auth['rows']}
    feature={}
    for a,r in rows.items():
        v.ck(a+'/input_pin',v.sha(r['input']),r['input_sha256'],'provenance')
        feature[a]=v.independent_features(v.load_array(r['input']),r['feature_names'])
    previous=v.read(U/'review/SAVED_RELATIONAL_VERIFICATION.json')
    for pin in previous['files']:
        v.ck('previous_verify_pin/'+Path(pin['path']).name,v.sha(pin['path']),pin['sha256'],'provenance')
    doc=v.read(U/'CHANGE_EVIDENCE.json')
    v.ck('72new_change_pairs',len(doc['pairs']),72,'scope')
    v.ck('thirds_not_semantic_stages',doc['thirds_are_not_semantic_stages'],category='scope')
    status_counts=Counter();unavailable_dimensions=0
    for item in doc['pairs']:
        p=U/'pairs'/item['pair_file'];d=v.read(p)
        v.ck(item['pair_file']+'/hash',item['pair_sha256'],v.sha(p),'provenance')
        path=d['path'];q=feature[d['query']];r=feature[d['reference']]
        v.ck(item['pair_file']+'/stage_ids',[s['segment_id'] for s in item['stages']],[s['segment_id'] for s in d['stages']],'scope')
        for stage,base in zip(item['stages'],d['stages']):
            lo,hi=base['reference_core'];ids=list(range(lo,hi+1))
            quotient,remainder=divmod(len(ids),3)
            sizes=[quotient+(j<remainder) for j in range(3)]
            chunks=[];offset=0
            for size in sizes:chunks.append(ids[offset:offset+size]);offset+=size
            for g in v.GROUPS:
                x,xok=q[g];y,yok=r[g];saved=stage['groups'][g];ref_means=[];query_means=[];valids=[]
                for ti,chunk in enumerate(chunks):
                    prefix=item['pair_file']+'/'+stage['segment_id']+'/'+g+'/third'+str(ti)
                    if not chunk:
                        v.check_fields(prefix,saved['thirds'][ti],dict(expected=0,compared=0,sufficient=False),'thirds')
                        ref_means.append([0.0]*y.shape[1]);query_means.append([0.0]*y.shape[1]);valids.append([False]*y.shape[1]);continue
                    mapping={j:[int(i) for i,k in path if k==j] for j in chunk}
                    expected=compared=missing=0;unique=set();reference=[];query=[];valid=[]
                    for k in range(y.shape[1]):
                        rvalues=[float(y[j,k]) for j in chunk if yok[j,k]]
                        qvalues=[]
                        for j in chunk:
                            if not yok[j,k]:continue
                            expected+=1;selected=[]
                            for i in mapping[j]:
                                if xok[i,k]:selected.append(float(x[i,k]));unique.add(i)
                                else:missing+=1
                            if selected:qvalues.append(mean(selected));compared+=1
                        reference.append(mean(rvalues));query.append(mean(qvalues))
                        valid.append(len(rvalues)>=2 and len(qvalues)==len(rvalues))
                    if missing or len(unique)<2:valid=[False]*y.shape[1]
                    actual=saved['thirds'][ti]
                    v.check_fields(prefix,actual,dict(reference_frames=chunk,expected=expected,compared=compared,path_missing_cells=missing,distinct_observed_query_frames=len(unique),sufficient=any(valid),valid_dimensions=valid),'thirds')
                    for k in range(y.shape[1]):
                        v.ck(prefix+'/reference_mean/'+str(k),actual['reference_mean'][k],reference[k],'thirds')
                        v.ck(prefix+'/query_mean/'+str(k),actual['query_observed_mean'][k],query[k],'thirds')
                    ref_means.append(reference);query_means.append(query);valids.append(valid)
                common=[all(valid[k] for valid in valids) for k in range(y.shape[1])]
                differences=[]
                for t in [0,1]:
                    for k,good in enumerate(common):
                        if good:differences.append(min(abs((query_means[t+1][k]-query_means[t][k])-(ref_means[t+1][k]-ref_means[t][k])),1.0))
                value=mean(differences) if differences else None
                expected_status='observed_change_candidate' if value is not None else 'insufficient'
                v.check_fields(item['pair_file']+'/'+stage['segment_id']+'/'+g,saved,dict(valid_dimensions=sum(common),change_difference=value,status=expected_status,semantic_order_proven=False),'change')
                status_counts[expected_status]+=1
                unavailable_dimensions+=len(common)-sum(common)
    ranges=v.read(U/'REFERENCE_RANGES.json')['ranges']['C_relational_path']
    limits={(r['reference'],r['segment_id'],r['group']):r['max_difference'] for r in ranges}
    diagnosis=v.read(U/'CONTROL_RANGE_DIAGNOSTIC.json')
    v.check_fields('range_diagnostic_scope',diagnosis,dict(fit_only=True,development_excluded=4,new_inference=False),'control_range')
    bycontrol={}
    for row in diagnosis['rows']:
        a=row['alias'];control=row['control'];d=v.read(U/'controls'/row['original_control'])
        v.ck(a+'/'+control+'/fit_only',rows[a]['role'],'fit','split')
        target='L_shape' if control=='left_short_shape_change_5f' else 'R_shape'
        v.ck(a+'/'+control+'/target',row['target_group'],target,'control_range')
        decisions=[]
        for s in d['stages']:
            group=s['groups'][target];threshold=limits[a,s['segment_id'],target]
            state='reference_range_unavailable' if threshold is None else 'observation_insufficient' if not group['sufficient'] else 'larger_than_fit_reference_range' if group['mean_difference']>threshold+1e-6 else 'within_small_reference_range'
            decisions.append(dict(segment_id=s['segment_id'],reference_core=s['reference_core'],query_frames=s['query_frames'],group=target,state=state,mean_difference=group['mean_difference'],peak_difference=group['peak_difference'],range_max=threshold,coverage=group['coverage'],path_missing_cells=group['path_missing_cells'],not_educational_verdict=True))
        v.ck(a+'/'+control+'/not_empty',bool(decisions),category='control_range')
        expected=dict(decisions=decisions,counts=dict(Counter(d['state'] for d in decisions)),larger_any=any(d['state']=='larger_than_fit_reference_range' for d in decisions),all_target_unavailable=all(d['state'] in ['reference_range_unavailable','observation_insufficient'] for d in decisions),observed_edit_cells=d['metadata'].get('actual_observed_changed_cells'))
        v.check_fields(a+'/'+control,row,expected,'control_range')
        bycontrol.setdefault(control,[]).append(row)
    v.ck('96fit_control_rows',len(diagnosis['rows']),96,'control_range')
    for s in diagnosis['summary']:
        ds=bycontrol[s['control']]
        expected=dict(control=s['control'],fit_cases=len(ds),larger_than_range_cases=sum(d['larger_any'] for d in ds),all_target_unavailable=sum(d['all_target_unavailable'] for d in ds),observed_edit_cases=sum(d['observed_edit_cells'] is not None and d['observed_edit_cells']>0 for d in ds))
        v.ck('control_range_summary/'+s['control'],s,expected,'control_range')
    report=dict(status='PASS_FIXED_REFERENCE_CHANGES_AND_CONTROL_RANGE_ARITHMETIC' if not v.failures else 'FAIL',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=dict(total=sum(v.counts.values()),failed=len(v.failures),by_category=dict(v.counts)),max_absolute_float_difference=v.max_float_error,failures=v.failures,change_status_counts=dict(status_counts),unavailable_dimension_instances=unavailable_dimensions,control_range_summary=diagnosis['summary'],limitations=['Thirds are index partitions inside sampled physical core drafts, not linguistic stages.','New reference means depend only on reference values/masks; query means average all observed mapped query values per reference frame. Missing path cells or fewer than2unique query frames withhold change values.','Raw mean vectors may contain zero placeholders when unsupported; valid_dimensions/status/null change_difference must be honored.','Control-range diagnostic is fit12self-reference perturbation sensitivity under fixed C ranges, not independent errors or recognition accuracy.','Omitting five source samples produced no range exceedance in fit12; those samples are not labeled linguistically mandatory.','Source/model/pair/range/DEV outputs were read only. No model inference, DTW rerun, tuning or prior-result overwrite.'])
    files=[U/n for n in ['CHANGE_EVIDENCE.json','CONTROL_RANGE_DIAGNOSTIC.json','change_evidence.py','control_ranges.py','REFERENCE_RANGES.json']]+[Path(__file__)]
    report['files']=[dict(path=str(p),sha256=v.sha(p)) for p in files]
    out=U/'review/CHANGE_CONTROL_RANGE_VERIFICATION.json'
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(out),status=report['status'],checks=report['checks'],change_counts=dict(status_counts),control_range_summary=diagnosis['summary'],failures=v.failures[:10]),ensure_ascii=False))
    return int(bool(v.failures))
if __name__=='__main__':raise SystemExit(run())
