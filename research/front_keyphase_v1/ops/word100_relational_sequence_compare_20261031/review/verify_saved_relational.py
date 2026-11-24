"""Independent arithmetic from saved inputs/paths, without importing evaluator.
Never searches a DTW path, calls a model or rewrites experiment results.
"""
from pathlib import Path
from collections import Counter, defaultdict
import datetime, hashlib, json, math
import numpy as np

U=Path(__file__).resolve().parents[1]
OPS=U.parent
OLD=OPS/'word100_core_stage_evidence_20261031'
EP=OPS/'word100_semantic_episode_pilot_20261031'
GROUPS=['L_shape','R_shape','L_location','R_location','L_axis','R_axis','L_arm','R_arm','hands_relation','L_to_R_forearm','R_to_L_forearm','L_direction','R_direction']
failures=[];counts=Counter();max_float_error=0.0
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(name,actual,expected=True,category='arithmetic'):
    global max_float_error
    counts[category]+=1
    if isinstance(expected,(float,np.floating)) and not isinstance(expected,bool):
        ok=isinstance(actual,(float,int)) and math.isfinite(actual) and math.isclose(actual,float(expected),rel_tol=2e-9,abs_tol=2e-11)
        if isinstance(actual,(float,int)) and math.isfinite(actual):max_float_error=max(max_float_error,abs(actual-float(expected)))
    else:ok=actual==expected
    if not ok:failures.append(dict(name=name,actual=actual,expected=expected,category=category))
def check_fields(name,actual,expected,category='arithmetic'):
    for k,v in expected.items():ck(name+'/'+k,actual.get(k),v,category)
def load_array(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}

def independent_features(a,names):
    values=np.asarray(a['x'],dtype=np.float64)
    valid=np.asarray(a['mask'],dtype=bool)&np.isfinite(values)
    idx={name:i for i,name in enumerate(names)}
    def columns(side,suffixes,scale=1.0):
        js=[idx[side+'_'+s] for s in suffixes]
        return np.where(valid[:,js],values[:,js]/scale,0.0),valid[:,js].copy()
    result={}
    for side,other in [('L','R'),('R','L')]:
        fingers=['thumb_mcp_angle','thumb_ip_angle']+[f+'_'+j+'_angle' for f in ['index','middle','ring','little'] for j in ['mcp','pip','dip']]
        result[side+'_shape']=columns(side,fingers)
        result[side+'_location']=columns(side,['hand_wrist_x','hand_wrist_y','palm_center_x','palm_center_y'])
        result[side+'_axis']=columns(side,['palm_axis_x','palm_axis_y'],2.0)
        result[side+'_direction']=columns(side,['wrist_motion_direction_x','wrist_motion_direction_y'],2.0)
        elbow,em=columns(side,['elbow_x','elbow_y'])
        wrist,wm=columns(side,['pose_wrist_x','pose_wrist_y'])
        raw_elbow=values[:,[idx[side+'_elbow_x'],idx[side+'_elbow_y']]]
        raw_wrist=values[:,[idx[side+'_pose_wrist_x'],idx[side+'_pose_wrist_y']]]
        delta=raw_wrist-raw_elbow
        lengths=np.sqrt(np.sum(delta*delta,axis=1))
        supported=em.all(axis=1)&wm.all(axis=1)&(lengths>1e-8)
        angle,am=columns(side,['elbow_angle'])
        axes=np.zeros_like(delta)
        axes[supported]=delta[supported]/lengths[supported,None]/2.0
        result[side+'_arm']=(np.column_stack([elbow,angle,axes,np.where(supported,lengths,0.0)]),np.column_stack([em,am,supported,supported,supported]))
        origin=values[:,[idx[other+'_elbow_x'],idx[other+'_elbow_y']]]
        distal=values[:,[idx[other+'_pose_wrist_x'],idx[other+'_pose_wrist_y']]]
        hand=values[:,[idx[side+'_hand_wrist_x'],idx[side+'_hand_wrist_y']]]
        vector=distal-origin;offset=hand-origin;squared=np.sum(vector*vector,axis=1)
        needed=[side+'_hand_wrist_x',side+'_hand_wrist_y',other+'_elbow_x',other+'_elbow_y',other+'_pose_wrist_x',other+'_pose_wrist_y']
        supported=valid[:,[idx[s] for s in needed]].all(axis=1)&(squared>1e-8)
        relation=np.zeros((len(values),2),float)
        relation[supported,0]=np.sum(offset[supported]*vector[supported],axis=1)/squared[supported]
        relation[supported,1]=(vector[supported,0]*offset[supported,1]-vector[supported,1]*offset[supported,0])/squared[supported]
        result[side+'_to_'+other+'_forearm']=(relation,np.column_stack([supported,supported]))
    js=[idx[n] for n in ['hands_wrist_delta_x','hands_wrist_delta_y','hands_wrist_distance']]
    result['hands_relation']=(np.where(valid[:,js],values[:,js],0.0),valid[:,js].copy())
    return result

def summary(q,r,path,reference_ids=None):
    selection=list(range(len(next(iter(r.values()))[0]))) if reference_ids is None else list(reference_ids)
    chosen=set(selection);pairs=[(int(i),int(j)) for i,j in path if int(j) in chosen]
    query_ids=sorted({i for i,j in pairs})
    out={}
    for g in GROUPS:
        x,xok=q[g];y,yok=r[g]
        # A dictionary gives each expected reference cell one aggregate slot;
        # no query-specific denominator or best mapped sample is selected.
        maxima={};missing=0;observed_query=set()
        for i,j in pairs:
            for k in range(y.shape[1]):
                if yok[j,k]:
                    if xok[i,k]:
                        observed_query.add(i)
                        error=min(abs(float(x[i,k])-float(y[j,k])),1.0)
                        maxima[j,k]=max(maxima.get((j,k),0.0),error)
                    else:missing+=1
        expected=sum(bool(yok[j,k]) for j in selection for k in range(y.shape[1]))
        compared=len(maxima);available=sum(any(yok[j]) for j in selection)
        out[g]=dict(expected=expected,compared=compared,reference_possible=len(selection)*y.shape[1],
            reference_unobserved=len(selection)*y.shape[1]-expected,path_missing_cells=missing,
            coverage=compared/expected if expected else None,
            mean_difference=math.fsum(maxima.values())/compared if compared else None,
            peak_difference=max(maxima.values()) if compared else None,
            sufficient=bool(expected and compared/expected>=.75 and missing==0 and available>=2 and len(observed_query)>=2),
            distinct_shared_query_frames=len(observed_query),
            query_unobserved_frames=[i for i in query_ids if not any(xok[i])])
    return out

def verify_order(name,q,r,core_segments,stored):
    n=len(next(iter(q.values()))[0]);ck(name+'/anchor_count',len(stored['anchors']),len(core_segments),'order')
    width=np.convolve(np.ones(n),np.ones(5),mode='same')
    for segment,anchor in zip(core_segments,stored['anchors']):
        ids=list(range(segment['core'][0],segment['core'][1]+1))
        group_costs=[];observed=np.zeros(n);total_expected=0
        for g in GROUPS:
            x,xok=q[g];y,yok=r[g]
            need=np.sum(yok[ids],axis=0)>=math.ceil(.75*len(ids))
            if not any(need):continue
            js=np.flatnonzero(need)
            template=np.array([math.fsum(float(y[f,j]) for f in ids if yok[f,j])/sum(bool(yok[f,j]) for f in ids) for j in js])
            difference=np.minimum(np.abs(x[:,js]-template),1.0)
            present=xok[:,js]
            per_frame_cost=np.sum(np.where(present,difference,1.0),axis=1)
            group_costs.append(np.convolve(per_frame_cost,np.ones(5),mode='same')/(width*len(js)))
            observed+=np.convolve(np.sum(present,axis=1),np.ones(5),mode='same')
            total_expected+=len(js)
        costs=np.mean(group_costs,axis=0) if group_costs else np.ones(n)
        coverage=observed/(width*total_expected) if total_expected else np.zeros(n)
        best=anchor['query_center'];minimum=float(np.min(costs))
        ck(name+'/'+segment['segment_id']+'/best_is_numerical_min',abs(float(costs[best])-minimum)<=2e-11,category='order')
        ck(name+'/'+segment['segment_id']+'/cost',anchor['cost'],float(costs[best]),'order')
        ck(name+'/'+segment['segment_id']+'/coverage',anchor['coverage'],float(coverage[best]),'order')
        ck(name+'/'+segment['segment_id']+'/window',anchor['query_window'],[max(0,best-2),min(n-1,best+2)],'order')
        ties=np.flatnonzero(np.abs(costs-costs[best])<=1e-9).tolist()
        ck(name+'/'+segment['segment_id']+'/ties',anchor['ties'],ties,'order')
        ck(name+'/'+segment['segment_id']+'/unique',anchor['unique'],len(ties)==1,'order')
        ck(name+'/'+segment['segment_id']+'/supported',anchor['supported'],bool(coverage[best]>=.75),'order')
        second=anchor['second_candidate']['center']
        ck(name+'/'+segment['segment_id']+'/second_cost',anchor['second_candidate']['cost'],float(costs[second]),'order')
        ck(name+'/'+segment['segment_id']+'/second_gap',anchor['second_candidate']['gap'],float(costs[second]-costs[best]),'order')
        second_min=min(float(costs[i]) for i in range(n) if i!=best)
        ck(name+'/'+segment['segment_id']+'/second_rank',abs(float(costs[second])-second_min)<=2e-11,category='order')
        ceiling=float(np.sort(costs)[min(4,n-1)])
        for k,candidate in enumerate(anchor['five_nearest_candidates']):
            j=candidate['center']
            ck(name+'/'+segment['segment_id']+'/top5cost/'+str(k),candidate['cost'],float(costs[j]),'order')
            ck(name+'/'+segment['segment_id']+'/top5coverage/'+str(k),candidate['coverage'],float(coverage[j]),'order')
            ck(name+'/'+segment['segment_id']+'/top5rank/'+str(k),float(costs[j])<=ceiling+2e-11,category='order')
    relations=[]
    for a,b in zip(stored['anchors'],stored['anchors'][1:]):
        overlap=max(a['query_window'][0],b['query_window'][0])<=min(a['query_window'][1],b['query_window'][1])
        state='insufficient' if not a['supported'] or not b['supported'] else 'ambiguous' if not a['unique'] or not b['unique'] or overlap else 'forward' if a['query_center']<b['query_center'] else 'reverse'
        relations.append(dict(first=a['segment_id'],second=b['segment_id'],state=state))
    ck(name+'/relations',stored['relations'],relations,'order')
    ck(name+'/semantic_order_false',stored['semantic_order_verified'],False,'order')

def verify_result(name,q,r,segments,d,band):
    path=np.asarray(d['path'],dtype=int);n=len(next(iter(q.values()))[0]);m=len(next(iter(r.values()))[0])
    ck(name+'/path_start',path[0].tolist(),[0,0],'scope')
    ck(name+'/path_end',path[-1].tolist(),[n-1,m-1],'scope')
    ck(name+'/query_frame_scope',sorted(set(path[:,0].tolist())),list(range(n)),'scope')
    ck(name+'/reference_frame_scope',sorted(set(path[:,1].tolist())),list(range(m)),'scope')
    ck(name+'/monotonic_steps',all(tuple(x) in {(1,0),(0,1),(1,1)} for x in np.diff(path,axis=0)),category='scope')
    if band:ck(name+'/normalized_band',bool(np.all(np.abs(path[:,0]/max(n-1,1)-path[:,1]/max(m-1,1))<=.25+1e-14)),category='scope')
    for key,value in dict(query_frames=n,reference_frames=m,query_scope_retained=n,reference_scope_retained=m,max_query_reuse=max(Counter(path[:,0]).values()),max_reference_reuse=max(Counter(path[:,1]).values())).items():
        ck(name+'/'+key,d[key],int(value),'scope')
    calculated=summary(q,r,path)
    for g in GROUPS:check_fields(name+'/whole/'+g,d['groups'][g],calculated[g])
    cores=[s for s in segments if s['role']=='core_candidate']
    ck(name+'/stage_ids',[s['segment_id'] for s in d['stages']],[s['segment_id'] for s in cores],'scope')
    for s,stored in zip(cores,d['stages']):
        lo,hi=s['core'];expected=summary(q,r,path,range(lo,hi+1))
        ck(name+'/'+s['segment_id']+'/reference_core',stored['reference_core'],s['core'],'scope')
        ck(name+'/'+s['segment_id']+'/query_frames',stored['query_frames'],sorted({int(i) for i,j in path if lo<=j<=hi}),'scope')
        for g in GROUPS:check_fields(name+'/'+s['segment_id']+'/'+g,stored['groups'][g],expected[g])
    verify_order(name,q,r,cores,d['free_order'])
    return calculated

def main():
    auth=read(U/'AUTHORIZATION.json');rows={r['alias']:r for r in auth['rows']}
    clips={c['alias']:c for c in read(OLD/'PHASE_DATA.json')['clips']}
    freeze=read(U/'FROZEN.json');v2=read(U/'FROZEN_V2.json')
    ck('v2_parent',v2['parent_sha256'],sha(U/'FROZEN.json'),'provenance')
    for p,h in v2['pins'].items():ck('v2_pin/'+Path(p).name,sha(p),h,'provenance')
    for p,h in freeze['pins'].items():
        if Path(p).name=='run_compare.py':
            preserved=[f for f in U.rglob('run_compare.py') if f!=Path(p) and sha(f)==h]
            ck('preserved_original_driver',bool(preserved),category='provenance')
        else:ck('original_pin/'+Path(p).name,sha(p),h,'provenance')
    for item in read(U/'review/PAIR_VERIFICATION_INPUT_SNAPSHOT.json')['files']:
        ck('initial_result_unchanged/'+Path(item['path']).name,sha(item['path']),item['sha256'],'provenance')
    arrays={};features={}
    for a,r in rows.items():
        ck(a+'/input_sha',sha(r['input']),r['input_sha256'],'provenance')
        arrays[a]=load_array(r['input'])
        ck(a+'/native_frame_ids',arrays[a]['frame_id'].tolist(),list(range(r['frames'])),'scope')
        ck(a+'/native_PTS_monotonic',bool(np.all(np.diff(arrays[a]['timestamp_us'])>0)),category='scope')
        features[a]=independent_features(arrays[a],r['feature_names'])
    emp=read(EP/'EMPIRICAL_RANGE.json');saved_paths={}
    for w in emp['calibration']:
        for q in w['lopo']:
            for pair in q['pairs']:saved_paths[q['query'],pair['reference']]=pair
    for q in emp['development']:
        for pair in q['pairs']:saved_paths[q['alias'],pair['reference']]=pair
    provenance=read(U/'PAIR_PROVENANCE.json')
    pairs=[];pair_by={}
    expected_keys={(a,b,kind) for a,ra in rows.items() for b,rb in rows.items() if a!=b and rb['role']=='fit' and ra['word_id']==rb['word_id'] for kind in ['B_same_path','C_relational_path']}
    for p in sorted((U/'pairs').glob('*.json')):
        d=read(p);a,b,kind=d['query'],d['reference'],d['condition']
        key=(a,b,kind);ck(p.name+'/unique',key not in pair_by,category='scope')
        pair_by[key]=d;pairs.append(d)
        seal=provenance[p.name]
        check_fields(p.name+'/manifest',seal,dict(result_sha256=sha(p),frozen_sha256=sha(U/'FROZEN.json'),evaluator_sha256=sha(U/'relational.py'),query_input_sha256=rows[a]['input_sha256'],reference_input_sha256=rows[b]['input_sha256']),'provenance')
        ck(p.name+'/reference_fit',rows[b]['role'],'fit','split')
        ck(p.name+'/same_expression',rows[a]['word_id'],rows[b]['word_id'],'split')
        ck(p.name+'/no_self',a!=b,category='split')
        if kind=='B_same_path':ck(p.name+'/saved_path_exact',d['path'],saved_paths[a,b]['path_original_frames'],'scope')
        verify_result(p.name,features[a],features[b],clips[b]['segments'],d,kind=='C_relational_path')
    ck('all72pair_keys',sorted(pair_by),sorted(expected_keys),'scope')
    ck('all72pair_manifest_keys',sorted(provenance),sorted(p.name for p in (U/'pairs').glob('*.json')),'provenance')
    print('72 pair summaries and unconstrained window arithmetic checked',flush=True)
    calibration=read(U/'REFERENCE_RANGES.json');range_lookup={}
    for condition in ['B_same_path','C_relational_path']:
        grouped=defaultdict(list)
        relevant=[d for d in pairs if d['condition']==condition and rows[d['query']]['role']=='fit']
        ck(condition+'/24calibration_pairs',len(relevant),24,'split')
        for d in relevant:
            for s in d['stages']:
                for group,val in s['groups'].items():grouped[d['reference'],s['segment_id'],group].append((d['query'],val))
        actual={(r['reference'],r['segment_id'],r['group']):r for r in calibration['ranges'][condition]}
        ck(condition+'/range_keys',sorted(actual),sorted(grouped),'range')
        for key,items in grouped.items():
            ref,seg,g=key
            expected_people=sorted(a for a,r in rows.items() if r['role']=='fit' and r['word_id']==rows[ref]['word_id'] and a!=ref)
            ck(condition+'/'+str(key)+'/two_other_people',sorted(a for a,v in items),expected_people,'split')
            supported=len(items)==2 and all(v['sufficient'] and v['mean_difference'] is not None for a,v in items)
            threshold=max(v['mean_difference'] for a,v in items) if supported else None
            check_fields(condition+'/'+str(key),actual[key],dict(fit_pairs=2,supported=supported,max_difference=threshold),'range')
            range_lookup[condition,key]=threshold
    development=read(U/'DEVELOPMENT.json');dev_counts=defaultdict(Counter)
    for d in development['results']:
        a,b,kind=d['query'],d['reference'],d['condition'];pair=pair_by[a,b,kind]
        ck(a+'/'+b+'/'+kind+'/dev_role',rows[a]['role'],'development','split')
        generated=[]
        for s in pair['stages']:
            for g,v in s['groups'].items():
                threshold=range_lookup[kind,(b,s['segment_id'],g)]
                state='reference_range_unavailable' if threshold is None else 'observation_insufficient' if not v['sufficient'] else 'larger_than_fit_reference_range' if v['mean_difference']>threshold+1e-6 else 'within_small_reference_range'
                generated.append(dict(segment_id=s['segment_id'],reference_core=s['reference_core'],query_frames=s['query_frames'],group=g,state=state,mean_difference=v['mean_difference'],peak_difference=v['peak_difference'],range_max=threshold,coverage=v['coverage'],path_missing_cells=v['path_missing_cells'],not_educational_verdict=True))
        ck(a+'/'+b+'/'+kind+'/decisions',d['decisions'],generated,'decision')
        ck(a+'/'+b+'/'+kind+'/counts',d['counts'],dict(Counter(x['state'] for x in generated)),'decision')
        ck(a+'/'+b+'/'+kind+'/old_A_value',d['old_saved_mean'],saved_paths[a,b]['distance'],'provenance')
        ck(a+'/'+b+'/'+kind+'/free_order_copy',d['free_order'],pair['free_order'],'order')
        dev_counts[kind].update(x['state'] for x in generated)
    for kind in ['B_same_path','C_relational_path']:
        for a,ra in rows.items():
            if ra['role']=='development':
                found=[d['reference'] for d in development['results'] if d['query']==a and d['condition']==kind]
                expected=[b for b,rb in rows.items() if rb['role']=='fit' and rb['word_id']==ra['word_id']]
                ck(a+'/'+kind+'/three_references',sorted(found),sorted(expected),'scope')
    controls=[];control_by={}
    for p in sorted((U/'controls').glob('*.json')):
        d=read(p);a=d['alias'];name=d['control'];key=(a,name);base=arrays[a];r=rows[a];n=len(base['x'])
        npz=p.with_suffix('.npz');stored=load_array(npz)
        check_fields(p.name+'/pins',d,dict(array_sha256=sha(npz),source_input_sha256=r['input_sha256'],frozen_sha256=sha(U/'FROZEN_V2.json')),'provenance')
        cores=[s for s in clips[a]['segments'] if s['role']=='core_candidate']
        midpoint=sum(cores[0]['core'])//2;changed=list(range(max(0,midpoint-2),min(n,midpoint+3)))
        expected={k:v.copy() for k,v in base.items()}
        if name in ['speed_075','speed_125','omit_short_core_5f','swap_adjacent_core_blocks']:
            if name.startswith('speed'):
                speed=.75 if name=='speed_075' else 1.25
                ids=np.round(np.linspace(0,n-1,round(n/speed))).astype(int)
            elif name=='omit_short_core_5f':
                ids=np.array([i for i in range(n) if i not in changed])
                ck(p.name+'/omitted_original_frames',d['metadata']['omitted_original_frames'],changed,'control_transform')
            else:
                s0,e0=cores[0]['core'];s1,e1=cores[1]['core']
                ids=np.array(list(range(s0))+list(range(s1,e1+1))+list(range(e0+1,s1))+list(range(s0,e0+1))+list(range(e1+1,n)))
            expected={k:v[ids].copy() for k,v in base.items()}
            expected['frame_id']=base['frame_id'].copy() if name=='swap_adjacent_core_blocks' else np.arange(len(ids))
            expected['timestamp_us']=base['timestamp_us'].copy() if name=='swap_adjacent_core_blocks' else np.arange(len(ids))*33333
            ck(p.name+'/source_index_map',d['metadata']['original_indices'],ids.tolist(),'control_transform')
        elif name in ['short_shape_change_5f','left_short_shape_change_5f']:
            side='R' if name=='short_shape_change_5f' else 'L'
            names=r['feature_names'];suffix=['thumb_mcp_angle','thumb_ip_angle']+[f+'_'+j+'_angle' for f in ['index','middle','ring','little'] for j in ['mcp','pip','dip']]
            cols=[names.index(side+'_'+s) for s in suffix]
            for f in changed:
                for j in cols:expected['x'][f,j]=1.0 if expected['x'][f,j]<.5 else 0.0
            ck(p.name+'/changed_frames',d['metadata']['changed_frames'],changed,'control_transform')
            ck(p.name+'/observed_changed_count',d['metadata']['actual_observed_changed_cells'],int(np.sum(base['mask'][np.ix_(changed,cols)])),'control_transform')
        elif name=='right_hand_unobserved':
            names=r['feature_names'];suffix=['hand_wrist_x','hand_wrist_y','index_tip_x','index_tip_y','palm_center_x','palm_center_y','thumb_mcp_angle','thumb_ip_angle']+[f+'_'+j+'_angle' for f in ['index','middle','ring','little'] for j in ['mcp','pip','dip']]+['palm_axis_x','palm_axis_y','palm_width','wrist_estimator_distance','wrist_velocity_x','wrist_velocity_y','wrist_motion_direction_x','wrist_motion_direction_y']
            cols=[names.index('R_'+s) for s in suffix]+[names.index(s) for s in ['hands_wrist_delta_x','hands_wrist_delta_y','hands_wrist_distance']]
            expected['mask'][:,cols]=False
        else:ck(p.name+'/identity_name',name,'identity','control_transform')
        for field in expected:
            ck(p.name+'/array/'+field,bool(np.array_equal(stored[field],expected[field],equal_nan=True)),category='control_transform')
            if name=='identity':ck(p.name+'/identity_dtype/'+field,str(stored[field].dtype),str(base[field].dtype),'control_transform')
        qfeatures=independent_features(stored,r['feature_names'])
        verify_result(p.name,qfeatures,features[a],clips[a]['segments'],d,True)
        if name=='right_hand_unobserved':
            for group in ['R_shape','R_location','R_axis','R_direction','R_to_L_forearm','hands_relation']:
                g=d['groups'][group]
                ck(p.name+'/'+group+'/zero_compared',g['compared'],0,'missing_contract')
                ck(p.name+'/'+group+'/null_mean',g['mean_difference'],None,'missing_contract')
                ck(p.name+'/'+group+'/not_sufficient',g['sufficient'],False,'missing_contract')
        controls.append(d);control_by[key]=d
    ck('126controls',len(controls),126,'scope')
    summary_doc=read(U/'CONTROL_SUMMARY.json');control_summary=[]
    for item in summary_doc['summary']:
        name=item['control'];ds=[d for d in controls if d['control']==name]
        expected=dict(control=name,cases=len(ds),positive_shape_peak=sum((d['groups']['R_shape']['peak_difference'] or 0)>1e-6 for d in ds),shape_insufficient=sum(not d['groups']['R_shape']['sufficient'] for d in ds),order_reverse=sum(any(v['state']=='reverse' for v in d['free_order']['relations']) for d in ds),order_ambiguous=sum(any(v['state'] in ['ambiguous','insufficient'] for v in d['free_order']['relations']) for d in ds),scope_retained=all(d['query_scope_retained']==d['query_frames'] and d['reference_scope_retained']==d['reference_frames'] for d in ds))
        ck('control_summary/'+name,item,expected,'control_summary')
        independent=dict(control=name,cases=len(ds),R_shape_nonzero=sum((d['groups']['R_shape']['peak_difference'] or 0)>1e-6 for d in ds),L_shape_nonzero=sum((d['groups']['L_shape']['peak_difference'] or 0)>1e-6 for d in ds),R_shape_insufficient=sum(not d['groups']['R_shape']['sufficient'] for d in ds),L_shape_insufficient=sum(not d['groups']['L_shape']['sufficient'] for d in ds),reverse_candidates=expected['order_reverse'])
        control_summary.append(independent)
    ck('control_rows_match',sorted((r['alias'],r['control']) for r in summary_doc['rows']),sorted(control_by),'scope')
    # The already-frozen evaluator's dynamic-programming implementation was
    # read. We validate the saved full paths and all arithmetic, but do not
    # rerun DTW or claim independent optimal-path reproduction.
    report=dict(status='PASS_SAVED_ARITHMETIC_WITH_EXPLICIT_SCOPE_LIMITS' if not failures else 'FAIL',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=dict(total=sum(counts.values()),failed=len(failures),by_category=dict(counts)),max_absolute_float_difference=max_float_error,failures=failures,development_counts={k:dict(v) for k,v in dev_counts.items()},control_summary_by_hand=control_summary,coverage=dict(original_clips=16,comparison_pairs=72,fit_pairs_per_condition=24,dev_pairs_per_condition=12,control_npz_results=126),limitations=['No model inference, neural training, new DTW path search or rewriting of stored experiment results.','C saved path admissibility and arithmetic verified; optimal path not independently searched again.','free_order is nearest-window retrieval, not event detection; second candidate may overlap; edge windows contain3or4rather than5frames.','Identity already yields a reverse candidate in one case, so reverse cannot certify a real language-order error.','pair.changes remains deprecated shared-only/query-conditioned historical diagnostic; separate fixed-reference audit required.','Range from only2other fit people per reference/core/group is an empirical span, not a linguistic tolerance.','positive_shape_peak in root summary refers toR_shape even forLcontrols; separateLvalues reported here.','Speed/omission controls deliberately change synthetic query sampling/timeline; retained scope refers to each generated query, not preservation of all original source samples.','Synthetic numerical perturbations and nonzero distances are not real KSL error detection accuracy.'])
    files=[U/n for n in ['FROZEN.json','FROZEN_V2.json','PAIR_PROVENANCE.json','REFERENCE_RANGES.json','DEVELOPMENT.json','CONTROL_SUMMARY.json','relational.py','run_compare.py']]+[Path(__file__)]
    report['files']=[dict(path=str(p),sha256=sha(p)) for p in files]
    out=U/'review/SAVED_RELATIONAL_VERIFICATION.json'
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(out),status=report['status'],checks=report['checks'],development_counts=report['development_counts'],control_summary=control_summary,failures=failures[:10]),ensure_ascii=False),flush=True)
    return int(bool(failures))

if __name__=='__main__':raise SystemExit(main())
