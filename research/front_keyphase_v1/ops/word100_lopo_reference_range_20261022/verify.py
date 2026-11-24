import traceback,copy
import numpy as np
import study as d
U=d.U
V=d.module('lopo_dp_verifier',d.prev.K/'verify.py')
def eq(a,b):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True)

def project(c,path,n,neighbors):
    shape=(n,2,21);low=np.full(shape,np.nan);high=low.copy();complete=np.zeros(shape,bool);partial=complete.copy()
    for j in range(n):
        ids=sorted(set(int(i) for i,jj in path if jj==j));valid=c['valid'][ids]
        complete[j]=np.sum(valid,axis=0)==len(ids);partial[j]=(np.sum(valid,axis=0)>0)&~complete[j]
        for h,a in np.argwhere(valid.any(0)):
            use=[]
            for i in ids:
                if c['valid'][i,h,a]:
                    use.append(i)
                    if neighbors:
                        if i and c['valid'][i-1,h,a]:use.append(i-1)
                        if i+1<len(c['valid']) and c['valid'][i+1,h,a]:use.append(i+1)
            low[j,h,a]=min(float(c['lo'][i,h,a]) for i in use);high[j,h,a]=max(float(c['hi'][i,h,a]) for i in use)
    return dict(lo=low,hi=high,complete=complete,partial=partial)

def runmask(bad):
    out=np.zeros_like(bad)
    for h in range(2):
        for a in range(21):
            for j in range(len(bad)-2):
                if bad[j:j+3,h,a].all():out[j:j+3,h,a]=True
    return out

def check_case(z,c,anchor,t):
    p=project(c,z['path'],len(z['expected']),True);domain=z['expected'];comp=domain&t['support']&p['complete']
    bad=comp&((p['hi']<t['low']-1e-6)|(p['lo']>t['high']+1e-6))
    # Use same max arithmetic without invoking candidate.
    gap=np.fmax(np.fmax(t['low']-p['hi'],p['lo']-t['high']),0.);gap[~comp]=np.nan
    eq(z['comparable'],comp);eq(z['unknown'],domain&~comp);eq(z['mismatch'],bad);eq(z['persistent'],runmask(bad));eq(z['gap'],gap)
    eq(z['within_range'],comp&~bad);eq(z['reference_insufficient'],domain&~t['support']);eq(z['query_incomplete'],domain&~p['complete']);eq(z['query_partial'],domain&p['partial'])
    eq(z['query_low'],p['lo']);eq(z['query_high'],p['hi']);eq(z['query_complete'],p['complete'])
    bc=domain&anchor['valid']&p['complete'];bg=np.fmax(np.fmax(anchor['lo']-p['hi'],p['lo']-anchor['hi']),0.);bg[~bc]=np.nan;bb=bc&(bg>1e-6)
    eq(z['base_comparable'],bc);eq(z['base_gap'],bg);eq(z['base_mismatch'],bb);eq(z['base_persistent'],runmask(bb))
    assert not np.any(z['unknown']&z['within_range']) and np.isnan(z['gap'][z['unknown']]).all()
    common=comp&bc;assert np.all(z['gap'][common]<=z['base_gap'][common]+1e-10)

def fixtures():
    n=12;shape=(n,2,21);c=dict(lo=np.zeros(shape),hi=np.zeros(shape),valid=np.ones(shape,bool));path=np.column_stack([np.arange(n),np.arange(n)]);dom=np.ones(shape,bool)
    refs=[]
    for _ in range(8):
        r=copy.deepcopy(c);r['lo'][5:8,1,0]=40;r['hi'][5:8,1,0]=40;refs.append(d.C.project(r,path,n,False))
    t=d.C.pool(refs);out=d.C.compare(c,r,t,path,dom);assert out['persistent'][5:8,1,0].all()
    result={'common_three_frame_excursion_missing':True}
    r=copy.deepcopy(c);r['lo'][6,1,0]=40;r['hi'][6,1,0]=40;t=d.C.pool([d.C.project(r,path,n,False) for _ in range(8)])
    out=d.C.compare(c,r,t,path,dom);assert out['mismatch'][6,1,0] and not out['persistent'].any();result['one_frame_difference_not_deleted']=True
    q=copy.deepcopy(c);q['valid'][:]=False;q['lo'][:]=np.nan;q['hi'][:]=np.nan;out=d.C.compare(q,r,t,path,dom)
    assert np.array_equal(out['unknown'],dom) and not out['within_range'].any();result['all_missing_unknown']=True
    refs=[d.C.project(c,path,n,False) for _ in range(8)]
    for ref in refs[:4]:ref['complete'][:]=False
    low=d.C.pool(refs);out=d.C.compare(c,c,low,path,dom);assert out['unknown'].all();result['four_people_insufficient']=True
    q=copy.deepcopy(c);q['valid'][5,1,0]=False;path2=np.array([(i,i) for i in range(n)]+[(5,6)])
    full=d.C.pool([d.C.project(c,path,n,False) for _ in range(8)]);out=d.C.compare(q,c,full,path2,dom)
    assert out['unknown'][6,1,0] and out['query_partial'][6,1,0];result['partial_plateau_not_full_observation']=True
    # Observed inter-person range is not a temporal/semantic rule.
    refs=[]
    for level in [0,0,10,10,20,20,40,40]:
        r=copy.deepcopy(c);r['lo'][:,:,0]=level;r['hi'][:,:,0]=level;refs.append(d.C.project(r,path,n,False))
    broad=d.C.pool(refs);out=d.C.compare(c,c,broad,path,dom);assert not out['mismatch'].any();result['broad_range_can_hide_excursion_limitation']=True
    return result

def run():
    f=d.read(U/'FROZEN.json');d.check(f['pins']);d.check(f['protected']);assert not (U/'RUNNING.json').exists()
    result=d.read(U/'RESULT.json');lookup={r['clip_id']:r for r in f['rows']};samples={cid:d.prev.data(r) for cid,r in lookup.items()};channels={cid:d.C.channels(e) for cid,(q,e) in samples.items()}
    cache={};templates={};domains={};count=0
    for meta in result['templates']:
        cid,rid=meta['query'],meta['anchor'];people=[lookup[c]['signer_id'] for c in meta['contributors']]
        assert len(set(people))==8 and lookup[cid]['signer_id'] not in people and cid not in meta['contributors'] and rid in meta['contributors']
        projected=[];re=samples[rid][1]
        for contributor in meta['contributors']:
            key=(contributor,rid)
            if key not in cache:
                p=np.column_stack([np.arange(len(re['core'])),np.arange(len(re['core']))]) if contributor==rid else d.load(d.P/'cases'/('identity__'+contributor+'__'+rid+'.npz'))['path']
                cache[key]=project(channels[contributor],p,len(re['core']),False)
            projected.append(cache[key])
        t=d.load(U/'templates'/(meta['name']+'.npz'));vv=np.array([p['complete'] for p in projected]);lo=np.array([p['lo'] for p in projected]);hi=np.array([p['hi'] for p in projected])
        nums=np.count_nonzero(vv,axis=0);valid=nums>=5
        expectedlow=np.where(valid,np.where(vv,lo,np.inf).min(0),np.nan);expectedhigh=np.where(valid,np.where(vv,hi,-np.inf).max(0),np.nan)
        eq(t['count'],nums);eq(t['support'],valid);eq(t['low'],expectedlow);eq(t['high'],expectedhigh);eq(t['member_low'],lo);eq(t['member_high'],hi);eq(t['member_complete'],vv)
        req=f['roles'].get(lookup[cid]['word_code'],{}).get('required_sides',['L','R'])
        dom=np.zeros_like(t['expected'])
        for h,s in enumerate(('L','R')):
            if s in req:dom[:,h]=np.repeat((re['core']&~re['source_rest'][:,h])[:,None],21,axis=1)
        eq(t['expected'],dom);templates[cid,rid]=t;domains[cid,rid]=dom
    assert len(templates)==216
    seen=set();newdp=0;reused=0
    for row in result['rows']:
        cid,rid,kind=row['query'],row['anchor'],row['kind'];q,e=samples[cid];qq,qe,ix=d.prev.transform(q,e,kind);qc=d.C.channels(qe)
        z=d.load(U/'cases'/(row['name']+'.npz'));t=templates[cid,rid];eq(z['expected'],domains[cid,rid]);eq(z['source_index'],ix)
        _,deleted=d.prev.prior.mapping(len(e['core']),e['core'],kind);eq(z['deleted'],deleted)
        if row['reused_path']:
            old=d.load(d.P/'cases'/(row['name']+'.npz'));eq(z['path'],old['path']);eq(z['cost'],old['cost']);reused+=1
        else:
            p=z['path'];eq(p[0],[0,0]);eq(p[-1],[len(ix)-1,len(t['expected'])-1]);delta=np.diff(p,axis=0)
            assert (delta>=0).all() and (delta<=1).all() and (delta.sum(1)>0).all()
            eq(z['cost'][tuple(p.T)].sum(),V.optimum(z['cost']));newdp+=1
        check_case(z,qc,channels[rid],t)
        for key,value in d.stats(z).items():assert row[key]==value
        assert row['whole_action_correct'] is None
        seen.add((cid,rid,kind));count+=1
        if count%264==0:print('VERIFIED',count,flush=True)
    assert len(seen)==2376 and reused==1080 and newdp==1296
    cc={}
    byname={r['name']:r for r in result['rows']}
    for row in result['controls']:
        parent=byname[row['parent']];cid,rid=parent['query'],parent['anchor'];q=copy.deepcopy(channels[cid]);z=d.load(U/'controls'/(row['name']+'.npz'));before=d.load(U/'cases'/(row['parent']+'.npz'))
        h=('L','R').index(row['side']);ids=z['removed'];sl=slice(17,21) if row['mode']=='wholearm' else slice(0,17)
        q['valid'][ids,h,sl]=False;q['lo'][ids,h,sl]=np.nan;q['hi'][ids,h,sl]=np.nan
        check_case(z,q,channels[rid],templates[cid,rid]);eq(z['expected'],before['expected']);eq(z['path'],before['path'])
        assert not z['within_range'][:,h,sl].any() if row['mode']!='shortgap' else True
        if row['mode']=='shortgap':
            for j in range(len(z['expected'])):
                if any(i in ids for i in z['path'][z['path'][:,1]==j,0]):assert z['unknown'][j,h,sl][z['expected'][j,h,sl]].all()
        cc[row['mode']]=cc.get(row['mode'],0)+1
    fs=fixtures();d.check(f['pins']);d.check(f['protected'])
    d.write(U/'VERIFY.json',dict(status='PASS',at=d.now(),templates=216,cases=2376,reused_paths=1080,new_dp=1296,controls=cc,fixtures=fs,
      protected=len(f['protected']),person_exclusion=True,expected_fixed=True,unknown_not_good=True,
      verifier='Codex independent scalar interval projection/pooling and DP; previous cost arithmetic verification reused, not external human semantic review.',
      limits=['Reference-excluded people were used in past public model/development; not untouched person generalization.', 'Per-axis empirical range cannot prove joint pose or semantic order.', 'Deletion core labels are not linguistic essentiality labels.']))
    print('VERIFIED ALL',flush=True)

if __name__=='__main__':
    try:run()
    except BaseException:
        p=U/'VERIFY_FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
