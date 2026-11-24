"""Small synthetic contract checks; not a public-data performance experiment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback
import numpy as np

U=Path(__file__).resolve().parent


def run():
    assert not (U/'FIXTURE_RESULT.json').exists(), 'Do not rerun saved fixtures'
    spec=importlib.util.spec_from_file_location('coherent_candidate_fixture', U/'candidate.py')
    C=importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
    assert Path(C.__file__).resolve()==(U/'candidate.py').resolve()
    n=9
    value=np.broadcast_to(np.arange(n,dtype=float)[:,None,None],(n,2,21)).copy()
    value[:,1]=8-value[:,1]
    c=dict(lo=value.copy(),hi=value.copy(),valid=np.ones(value.shape,bool))
    rest=np.zeros((n,2),bool);domain=np.ones(value.shape,bool)
    path=np.column_stack([np.arange(n),np.arange(n)])
    m=C.mapped(c,path,n,rest);d=C.descriptor(m,domain)
    assert d['valid'][2:7].all() and not d['valid'][[0,1,7,8]].any()
    np.testing.assert_array_equal(d['lo'][4,0,0], [2.,4.,6.,2.,2.])
    np.testing.assert_array_equal(d['lo'][4,1,0], [6.,4.,2.,-2.,-2.])
    checks={'ordered_three_levels_two_increments_hand_computed':True,
            'crossing_hand_identity_preserved':True}
    doubled={k:np.repeat(v,2,axis=0) for k,v in c.items()}
    slowpath=np.array([(i,j) for j in range(n) for i in (2*j,2*j+1)])
    slow=C.descriptor(C.mapped(doubled,slowpath,n,np.repeat(rest,2,0)),domain)
    for key in ('lo','hi','valid'):
        np.testing.assert_allclose(slow[key],d[key],rtol=0,atol=0,equal_nan=True)
    checks['slower_duplicate_sampling_same_order_same_descriptor']=True
    missing={k:v.copy() for k,v in c.items()}
    missing['valid'][4,0,7]=False;missing['lo'][4,0,7]=missing['hi'][4,0,7]=np.nan
    md=C.descriptor(C.mapped(missing,path,n,rest),domain)
    assert md['span_bad'][4,0,7] and not md['valid'][4,0,7]
    assert md['valid'][4,0,6] and md['valid'][4,0,8]
    checks['missing_middle_unknown_neighbors_preserved']=True
    rs=rest.copy();rs[4,0]=True
    rd=C.descriptor(C.mapped(c,path,n,rs),domain)
    assert rd['rest_bad'][4,0].all() and not rd['valid'][4,0].any() and rd['valid'][4,1].all()
    checks['positive_rest_context_not_silently_compared']=True
    lower=domain.copy();lower[4,0]=False
    ld=C.descriptor(C.mapped(c,path,n,rest),lower)
    assert not ld['eligible'][4,0].any() and ld['eligible'][4,1].all()
    checks['reference_domain_context_kept_separate']=True
    cal=C.calibrate(d,[d]*7)
    np.testing.assert_array_equal(cal['count'][4],np.full((2,21),7))
    assert (cal['levels_threshold'][4]==0).all() and (cal['sequence_threshold'][4]==0).all()
    result=C.compare(md,d,cal,domain)
    assert result['unknown'][4,0,7] and not result['levels_within'][4,0,7] and not result['sequence_within'][4,0,7]
    assert np.array_equal(result['expected'],domain)
    lowcal=C.calibrate(d,[d]*4)
    assert not lowcal['support'].any() and np.isnan(lowcal['levels_threshold']).all()
    checks['insufficient_peers_or_middle_never_within_range']=True
    offset={k:v.copy() for k,v in d.items()}
    offset['lo'][...,0:3]+=1;offset['hi'][...,0:3]+=1
    oc=C.calibrate(d,[offset]*7)
    assert oc['levels_threshold'][4,0,0]==1 and oc['sequence_threshold'][4,0,0]==1
    swapped={k:v.copy() for k,v in d.items()}
    swapped['lo'][4,0,0]=swapped['hi'][4,0,0]=[2,6,4,4,-2]
    assert C.distance(swapped,d,'levels')[4,0,0]==2
    assert C.distance(swapped,d,'sequence')[4,0,0]==4
    changes=C.compare(swapped,d,oc,domain)
    assert changes['levels_bad'][4,0,0] and changes['sequence_bad'][4,0,0]
    checks['order_change_has_joint_increment_evidence']=True
    flat={k:v[:1].copy() for k,v in c.items()}
    cp=np.column_stack([np.zeros(n,int),np.arange(n)])
    cd=C.descriptor(C.mapped(flat,cp,n,np.zeros((1,2),bool)),domain)
    assert cd['collapsed'][4] and cd['valid'][4].all()
    assert np.all(cd['phase_index'][4]==0)
    checks['collapsed_sampling_recorded_not_automatic_error']=True
    out={'status':'PASS','checks':checks,'scope':'Nine synthetic arithmetic and preservation checks; not empirical normal-speed, sign-language or crossing accuracy.',
         'sha256':{name:hashlib.sha256((U/name).read_bytes()).hexdigest() for name in ('candidate.py','fixtures.py')}}
    with (U/'FIXTURE_RESULT.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print('FIXTURES PASS',len(checks),flush=True)


if __name__=='__main__':
    try:run()
    except BaseException:
        p=U/'FIXTURE_FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
