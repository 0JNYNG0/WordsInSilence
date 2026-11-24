"""Synthetic-only checks and an explicitly unresolved missing-island risk.

run() returns JSON-safe evidence. CLI prints status/check count. This file does
not read public arrays, run models, create result files or update frozen docs.
"""
import copy
import json
import numpy as np
import split_alignment as S


def names():
    out=[]
    for h in ('L','R'):
        out += [h+'_'+s for s in ('hand_wrist_x','hand_wrist_y','pose_wrist_x','pose_wrist_y',
            'elbow_x','elbow_y','index_tip_x','index_tip_y','palm_center_x','palm_center_y','elbow_angle')]
        out += [h+'_thumb_'+j+'_angle' for j in ('mcp','ip')]
        out += [h+'_'+f+'_'+j+'_angle' for f in ('index','middle','ring','little') for j in ('mcp','pip','dip')]
        out += [h+'_palm_axis_x',h+'_palm_axis_y',h+'_palm_width']
    out += ['hands_wrist_delta_x','hands_wrist_delta_y','hands_wrist_distance',
            'L_wrist_estimator_distance','R_wrist_estimator_distance']
    for h in ('L','R'):
        out += [h+'_wrist_velocity_x',h+'_wrist_velocity_y',h+'_wrist_motion_direction_x',h+'_wrist_motion_direction_y']
    return out


def sample(values, observed=None, side='R'):
    values=np.asarray(values,dtype=float);n=len(values);cols=names()
    x=np.zeros((n,len(cols)));mask=np.zeros_like(x,dtype=bool)
    column=cols.index(side+'_index_mcp_angle');x[:,column]=values
    mask[:,column]=True if observed is None else np.asarray(observed,dtype=bool)
    return {'x':x,'mask':mask,'names':cols,'frame_id':np.arange(n,dtype=np.int64),
        'timestamp_us':np.arange(n,dtype=np.int64)*33333,
        'probs':np.zeros((n,3)),'support':np.zeros((n,3),dtype=bool)}


def brute(matrix):
    n,m=matrix.shape;found=[]
    def visit(i,j,axis,run,total,warps,path):
        if i==n-1 and j==m-1:
            found.append((total,warps,path));return
        for di,dj,next_axis,w in ((1,1,0,2),(1,0,1,1),(0,1,2,1)):
            ni,nj=i+di,j+dj
            nr=0 if next_axis==0 else run+1 if next_axis==axis else 1
            if ni<n and nj<m and nr<=2:
                visit(ni,nj,next_axis,nr,total+w*matrix[ni,nj],warps+(next_axis!=0),path+[[ni,nj]])
    visit(0,0,0,0,2*matrix[0,0],0,[[0,0]])
    if not found:return None,0,None
    best=min(v for v,w,p in found);opt=[(w,p) for v,w,p in found if v==best]
    return best,min(2,len(opt)),min(w for w,p in opt)


def run():
    checks=[]
    def check(name,condition):
        checks.append({'name':name,'pass':bool(condition)})
        if not condition:raise AssertionError(name)
    q=sample([0,.3,.8,.2]);r=copy.deepcopy(q)
    ans=S.align(q,r);dim=2
    qr=ans['evidence']['groups']['R_shape']['query'];rr=ans['evidence']['groups']['R_shape']['reference']
    check('identity_full_axes',ans['path']==[[i,i] for i in range(4)])
    check('identity_shared_residual_zero',all(row[dim]==0 for row in qr['mean_residual']))
    check('unknown_dimensions_null',all(row[0] is None for row in qr['mean_residual']))
    check('all_opportunities_fixed',qr['counts']['possible']==56 and rr['counts']['possible']==56)
    check('independent_masks',qr['counts']['own_observed']==4 and ans['evidence']['groups']['L_shape']['status']=='UNASSESSABLE')
    check('no_scalar_sign_grade',ans['semantic_verdict'] is None and ans['scalar_agreement'] is None)
    constant=S.align(sample([0,0,0,0]),sample([0,0,0,0]))
    check('constant_minimal_warp',constant['non_diagonal_moves']==0)
    check('ambiguity_before_secondary_retained',constant['optimal_path_count_capped']==2 and constant['optimal_path_ambiguous'])
    unknown=sample([0,0,0,0],[False]*4);no=S.align(unknown,unknown)
    check('no_support_not_certainty',no['non_diagonal_moves']==0 and no['optimal_path_ambiguous'] and no['status']=='UNASSESSABLE')
    check('no_support_zero_objective_not_agreement',no['normalized_cost']==0 and no['alignment_objective_is_agreement'] is False)
    check('no_support_null_readout',all(v is None for row in no['evidence']['groups']['R_shape']['query']['max_residual'] for v in row))
    # Small positive cost may not be exchanged for fewer warp steps, even < EPS.
    tiny=np.zeros((3,3));tiny[1,1]=S.EPS/10
    tiny_result=S.solve(tiny)
    check('positive_primary_cannot_trade_for_leastwarp',tiny_result['weighted_cost_sum']==0 and tiny_result['non_diagonal_moves']==2)
    check('numeric_near_tie_only_marks_ambiguity',tiny_result['numeric_tie_ambiguous'])
    check('fixed_dimension_denominator',np.isclose(S.cost(S.features(sample([1])),S.features(sample([0])))[0,0],1/14/5/3))
    check('query_missing_no_objective_penalty',S.cost(S.features(sample([1],[False])),S.features(sample([0])))[0,0]==0)
    check('reference_unknown_no_objective_penalty',S.cost(S.features(sample([1])),S.features(sample([0],[False])))[0,0]==0)
    # Speed change uses known synthetic correspondence, not a model or RGB claim.
    speed=S.align(sample([0,0,1,1,0,0]),sample([0,1,0]))
    check('speed_full_scope',len(speed['query_frame_ids'])==6 and len(speed['reference_frame_ids'])==3
          and {i for i,j in speed['path']}==set(range(6)) and {j for i,j in speed['path']}==set(range(3)))
    check('speed_known_values_preserved',all(row[dim]==0 for row in speed['evidence']['groups']['R_shape']['query']['max_residual']))
    change=S.align(sample([0,0,1,0,0]),sample([0,0,0,0,0]))
    qc=change['evidence']['groups']['R_shape']['query'];rc=change['evidence']['groups']['R_shape']['reference']
    check('short_change_query_peak_preserved',qc['max_residual'][2][dim]==1 and qc['mean_residual'][2][dim]==1)
    check('short_change_reference_peak_preserved',max(row[dim] for row in rc['max_residual'] if row[dim] is not None)==1)
    outside=S.align(sample([1,0,0,0,1]),sample([0,0,0,0,0]))
    oq=outside['evidence']['groups']['R_shape']['query']
    check('outside_boundaries_not_dropped',oq['max_residual'][0][dim]==1 and oq['max_residual'][-1][dim]==1)
    # Missing and shared partners at one original sample remain distinguishable.
    mixed=S.readout(S.features(sample([1,0])),S.features(sample([0,0,0],[False,True,True])),[[0,0],[0,1],[1,2]])
    mx=mixed['groups']['R_shape']['query']
    check('mixed_support_not_all_observed',mx['matched_support'][0][dim] and mx['mixed_counterpart_missing'][0][dim]
          and mx['missing_counterpart_link_count'][0][dim]==1 and mx['cell_status'][0][dim]=='PARTIAL_SHARED_SUPPORT')
    check('conditional_mean_does_not_impute_missing',mx['mean_residual'][0][dim]==1)
    # Required unresolved counterexample: shared-only cost rewards an unknown island.
    risk_q=sample([0,1,0,0]);risk_r=sample([0,0,0,0],[True,True,False,True])
    risk=S.align(risk_q,risk_r);risk_axis=risk['evidence']['groups']['R_shape']['query']
    check('hidden_change_risk_reproduced',risk['normalized_cost']==0 and [1,2] in risk['path'])
    check('hidden_change_not_comparable',risk_axis['missing_counterpart'][1][dim]
          and not risk_axis['matched_support'][1][dim] and risk_axis['cell_status'][1][dim]=='NOT_COMPARABLE')
    check('hidden_change_null_not_zero',risk_axis['mean_residual'][1][dim] is None and risk_axis['max_residual'][1][dim] is None)
    check('hidden_change_missing_time_preserved',1 in risk_axis['missing_indices_by_dimension']['observed_without_shared_counterpart'][dim]
          and risk_axis['frame_id'][1]==1 and risk_axis['timestamp_us'][1]==33333)
    check('hidden_change_no_normal_verdict',risk['whole_sign_verdict'] is None and risk['scalar_agreement'] is None)
    # Crossing hands do not change anatomical channel assignment or split time.
    cross=sample([.1,.1,.1,.1],side='L')
    for h,positions in [('L',[-.3,-.1,.1,.3]),('R',[.3,.1,-.1,-.3])]:
        for axis in ('x','y'):
            col=cross['names'].index(h+'_hand_wrist_'+axis)
            cross['x'][:,col]=positions if axis=='x' else 0.;cross['mask'][:,col]=True
    cf=S.features(cross)
    check('hand_cross_keeps_anatomical_channels',np.allclose(cf['groups']['L_location'][0][:,0],[-.3,-.1,.1,.3])
          and np.allclose(cf['groups']['R_location'][0][:,0],[.3,.1,-.1,-.3]))
    cr=S.align(cross,cross)
    check('hand_cross_whole_temporal_path',cr['path']==[[i,i] for i in range(4)] and len(cr['evidence']['groups'])==16)
    # Word/phase/prior correspondence metadata cannot direct a new alignment.
    meta=copy.deepcopy(q);meta.update(word_id='fake',phase_labels=[99]*4,source_indices=[3,2,1,0],old_alignment=[[3,0]])
    before=S.align(q,r);after=S.align(meta,r)
    check('word_phase_old_indices_ignored',before['path']==after['path'] and before['evidence']==after['evidence'])
    check('no_input_mutation',np.array_equal(meta['x'],q['x']) and meta['source_indices']==[3,2,1,0])
    for name,path in [('dropped_query',[[0,0],[1,1],[3,2],[3,3]]),('duplicate',[[0,0],[0,0]]),('out_of_range',[[0,0],[4,3]])]:
        try:S.readout(S.features(q),S.features(r),path)
        except ValueError:check('reject_'+name,True)
        else:check('reject_'+name,False)
    rng=np.random.default_rng(42)
    for n in range(1,5):
        for m in range(1,5):
            for kind in ('constant','random'):
                matrix=np.zeros((n,m)) if kind=='constant' else rng.integers(0,5,size=(n,m))/4
                expected,count,warps=brute(matrix);actual=S.solve(matrix);prefix=f'{n}x{m}_{kind}'
                if expected is None:
                    check(prefix+'_infeasible',actual['path']==[] and actual['normalized_cost'] is None);continue
                check(prefix+'_primary',actual['weighted_cost_sum']==expected)
                check(prefix+'_exact_ambiguity',actual['optimal_path_count_capped']==count)
                check(prefix+'_leastwarp_only_tie',actual['non_diagonal_moves']==warps)
                check(prefix+'_normalization',np.isclose(actual['normalized_cost'],expected/(n+m)) and actual['path_weight_sum']==n+m)
    risk_record={'status':'KNOWN_UNRESOLVED_ALIGNMENT_RISK','query_values':[0,1,0,0],
        'reference_values':[0,0,None,0],'path':risk['path'],'normalized_cost':risk['normalized_cost'],
        'observed_query_change_frame':1,'mapped_reference_unknown_frame':2,
        'query_change_readout':{'mean':risk_axis['mean_residual'][1][dim],
            'max':risk_axis['max_residual'][1][dim],'matched_support':risk_axis['matched_support'][1][dim],
            'missing_counterpart':risk_axis['missing_counterpart'][1][dim],'status':risk_axis['cell_status'][1][dim]},
        'risk_solved':False,'normal_sign_verdict':None,
        'explanation':'Cost-zero masks can hide an observed change. Readout exposes no comparable reference; no whole-sign success.'}
    return {'status':'PASS_SYNTHETIC_CONTRACT_WITH_KNOWN_UNRESOLVED_RISK','checks':len(checks),'results':checks,
        'known_risk_counterexample':risk_record,'actual_data_alignment_runs':0,'new_inference':0,'semantic_accuracy':None}


if __name__=='__main__':
    result=run();print(json.dumps({'status':result['status'],'checks':result['checks'],
        'known_risk_counterexample':result['known_risk_counterexample']}))
