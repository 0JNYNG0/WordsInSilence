"""Synthetic-only module tests; no source data, model run, scores or file writes."""
import copy
import json
import numpy as np
import joint_alignment as J


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


def sample(n=5):
    cols=names();x=np.zeros((n,len(cols)),dtype=float)
    for h in ('L','R'):
        x[:,cols.index(h+'_pose_wrist_x')]=.5
        x[:,cols.index(h+'_hand_wrist_x')]=.5
        x[:,cols.index(h+'_palm_center_x')]=.5
    return {'x':x,'mask':np.ones_like(x,dtype=bool),'names':cols,
        'frame_id':np.arange(n,dtype=np.int64),'timestamp_us':np.arange(n,dtype=np.int64)*33333,
        'probs':np.tile([.1,.9,.9],(n,1)),'support':np.ones((n,3),dtype=bool)}


def brute(matrix):
    n,m=matrix.shape;paths=[]
    def visit(i,j,axis,run,value,path):
        if i==n-1 and j==m-1:
            paths.append((value,path));return
        for di,dj,next_axis,weight in ((1,1,0,2),(1,0,1,1),(0,1,2,1)):
            ni,nj=i+di,j+dj
            nr=0 if next_axis==0 else run+1 if axis==next_axis else 1
            if ni<n and nj<m and nr<=2:
                visit(ni,nj,next_axis,nr,value+weight*matrix[ni,nj],path+[[ni,nj]])
    visit(0,0,0,0,2*matrix[0,0],[[0,0]])
    if not paths:return None,0
    best=min(v for v,p in paths)
    return best,min(2,sum(abs(v-best)<=J.EPS for v,p in paths))


def run():
    checks=[]
    def check(name,condition):
        checks.append({'name':name,'pass':bool(condition)})
        if not condition:raise AssertionError(name)
    a=sample();f=J.features(a)
    check('shape_14_per_hand',f['groups']['L_shape'][0].shape==(5,14) and f['groups']['R_shape'][0].shape==(5,14))
    check('fixed_subgroups',tuple(map(len,J.FAMILIES.values()))==(5,7,4))
    check('observed_zero_direction_not_unknown',f['groups']['L_direction'][1].all()
          and np.all(f['groups']['L_direction'][0]==0) and f['observed_zero_is_semantic_hold'] is False)
    b=copy.deepcopy(a);b['mask'][2,b['names'].index('L_hand_wrist_x')]=False
    g=J.features(b)
    check('movement_gap_no_bridge',not g['groups']['L_displacement'][1][1:4].any())
    check('other_hand_mask_independent',g['groups']['R_displacement'][1].all())
    check('shape_not_blocked_by_wrist_gap',g['groups']['L_shape'][1].all())
    b=copy.deepcopy(a);b['frame_id'][3:]+=2
    check('skipped_native_frame_no_motion',not J.features(b)['groups']['L_direction'][1][2:4].any())
    b=copy.deepcopy(a);b['x'][:,b['names'].index('L_hand_wrist_x')]=np.arange(5)*.1
    g=J.features(b)
    check('signed_native_displacement',np.allclose(g['groups']['L_displacement'][0][:,0],[.1,.2,.2,.2,.1]))
    check('unit_direction_divide2',np.allclose(g['groups']['L_direction'][0][:,0],.5))
    b['x'][:,b['names'].index('L_hand_wrist_x')]*=-1
    check('opposite_direction_preserved',np.allclose(J.features(b)['groups']['L_direction'][0][:,0],-.5))
    b=copy.deepcopy(a);b['mask'][:]=False;b['support'][:]=False
    missing=J.features(b)
    check('unknown_reference_neutral_not_match',np.allclose(J.cost(f,missing),.5))
    check('known_reference_missing_query_penalty',np.allclose(J.cost(missing,f),1.))
    check('both_unknown_not_zero_match',np.allclose(J.cost(missing,missing),.5))
    check('unknown_not_imputed',all(not v[1].any() for v in missing['groups'].values()))
    c,families=J.cost(f,f,True)
    check('fixed_family_equal_mean',np.allclose(c,sum(families.values())/3))
    altered=copy.deepcopy(f);altered['groups']['learned3'][0][:]=1-f['groups']['learned3'][0]
    qcost,qfamilies=J.cost(altered,f,True)
    check('learned_shape_signal_active',np.all(qfamilies['shape']>families['shape']))
    check('learned_change_does_not_modify_other_families',np.array_equal(qfamilies['relations'],families['relations'])
          and np.array_equal(qfamilies['movement'],families['movement']))
    b=copy.deepcopy(a);b['word_id']='different';b['phase_labels']=['fake']*5;b['correct_class']='wrong'
    check('labels_ignored',np.array_equal(J.cost(J.features(b),f),J.cost(f,f)))
    b=copy.deepcopy(a);b['probs'][2,0]=np.nan
    check('nonfinite_probability_unknown',not J.features(b)['groups']['learned3'][1][2,0])
    for name,mutate in [('missing_mask',lambda d:d.pop('mask')),
        ('float_mask',lambda d:d.update(mask=d['mask'].astype(float))),
        ('bad_probability_shape',lambda d:d.update(probs=np.zeros((5,4)))),
        ('bad_supported_probability',lambda d:d['probs'].__setitem__((0,0),2)),
        ('unordered_time',lambda d:d['timestamp_us'].__setitem__(2,0))]:
        b=copy.deepcopy(a);mutate(b)
        try:J.features(b)
        except ValueError:check(name,True)
        else:check(name,False)
    # Exact independent tiny enumeration validates objective, constraints, and tie count.
    rng=np.random.default_rng(42)
    for n in range(1,5):
        for m in range(1,5):
            for style in ('zeros','grid'):
                mat=np.zeros((n,m)) if style=='zeros' else rng.integers(0,5,size=(n,m))/4
                answer=J.solve(mat);best,count=brute(mat)
                prefix=f'{n}x{m}_{style}'
                if best is None:
                    check(prefix+'_infeasible',answer['path']==[] and answer['normalized_cost'] is None)
                    continue
                check(prefix+'_optimal_weighted_cost',np.isclose(answer['weighted_cost_sum'],best))
                check(prefix+'_tie_count',answer['optimal_path_count_capped']==count)
                check(prefix+'_n_plus_m',answer['path_weight_sum']==n+m
                      and np.isclose(answer['normalized_cost'],best/(n+m)))
                p=answer['path']
                check(prefix+'_all_frames',{i for i,j in p}==set(range(n)) and {j for i,j in p}==set(range(m)))
                axis=0;run=0;ok=True
                for (i,j),(ni,nj) in zip(p,p[1:]):
                    di,dj=ni-i,nj-j;next_axis=0 if (di,dj)==(1,1) else 1 if (di,dj)==(1,0) else 2
                    ok&=(di,dj) in ((1,1),(1,0),(0,1))
                    run=0 if next_axis==0 else run+1 if next_axis==axis else 1
                    axis=next_axis;ok&=run<=2
                check(prefix+'_same_axis_run2',ok)
    answer=J.align(a,a)
    check('align_scope_metadata',answer['query_frame_ids']==a['frame_id'].tolist()
          and answer['reference_timestamp_us']==a['timestamp_us'].tolist())
    check('forced_order_not_semantics',answer['global_order_forced'] and not answer['semantic_order_verified']
          and answer['whole_sign_verdict'] is None)
    check('family_cost_sum',np.isclose(answer['normalized_cost'],sum(answer['family_costs'].values())/3))
    unknown=sample(3);unknown['mask'][:]=False;unknown['support'][:]=False
    all_unknown=J.align(unknown,unknown)
    check('all_unknown_full_alignment_neutral',np.isclose(all_unknown['normalized_cost'],.5))
    check('all_unknown_remains_ambiguous',all_unknown['optimal_path_ambiguous'])
    check('timing_reported',answer['seconds']>=0)
    return {'status':'PASS','checks':len(checks),'results':checks,'scope':'synthetic only; no actual source/output scores read',
            'new_training':0,'new_inference':0,'semantic_accuracy':None}


if __name__=='__main__':
    result=run();print(json.dumps({'status':result['status'],'checks':result['checks']}))
