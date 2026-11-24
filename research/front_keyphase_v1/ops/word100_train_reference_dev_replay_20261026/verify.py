"""Separate saved-result arithmetic; no re-alignment, inference or threshold fitting."""
import study as S
from collections import defaultdict, Counter
import numpy as np, time, traceback
V=S.R.mod('replay_independent_arithmetic',S.B/'verify_alignment.py')
E=S.R.mod('replay_independent_eligibility',S.B/'verify_range.py')

def main():
    assert not(S.U/'RUNNING.json').exists()
    f=S.read(S.U/'FROZEN.json'); done=S.read(S.U/'DONE.json'); bd=S.read(S.U/'BOUNDS_DONE.json')
    S.check(f['protected']); S.check(bd['pins']); S.check(done['pins'])
    rows=f['train']+f['dev']; samples={r['clip_id']:S.load(r['output']) for r in rows}; metadata={r['clip_id']:r for r in rows}
    for r in rows:assert S.sha(r['output'])==f['pins'][r['output']]
    groups=defaultdict(list)
    for p in f['calibration_pairs']:groups[p['reference']].append(p)
    bounds={}; n=Counter(); start=time.perf_counter()
    for row in bd['rows']:
        rid=row['reference']; pp=groups[rid]; actual=S.load(row['output']); vals=[]; evidence=[]
        assert sorted(p['query'] for p in pp)==row['calibration_ids'] and len(pp) in (7,8)
        people={metadata[p['query']]['signer_id'] for p in pp}; assert len(people)==len(pp) and max(people)<=9 and metadata[rid]['signer_id'] not in people
        for p in pp:
            assert S.sha(p['pair'])==f['pins'][p['pair']]
            z=S.load(p['pair']); vals.append(z); evidence.append(E.eligibility(z,samples[p['query']],samples[rid]))
        for mode,k in [('point',2),('sequence',3)]:
            valid=np.array([e[k] for e in evidence]); count=valid.sum(0); delta=np.array([z[mode+'_delta'] for z in vals]); mask=valid[...,None] if mode=='sequence' else valid
            maximum=np.max(np.where(mask,delta,-np.inf),0); expected=np.where((count>=5)[...,None] if mode=='sequence' else count>=5,maximum,np.nan)
            V.same(actual[mode+'_peer_count'],count); V.same(actual[mode+'_bound'],expected)
        bounds[rid]=actual; n['train_only_bounds']+=1
        if n['train_only_bounds']%200==0:print('VERIFY_TRAIN_BOUNDS',n['train_only_bounds'],899,flush=True)
    assert {p['query'] for p in done['rows']}=={r['clip_id'] for r in f['dev']}
    counts=Counter(); seen=set()
    for i,row in enumerate(done['rows']):
        qid=row['query'];rid=row['reference'];assert (qid,rid) not in seen;seen.add((qid,rid));q=samples[qid];r=samples[rid];z=S.load(row['output']);b=bounds[rid]
        assert metadata[qid]['signer_id'] in (11,12,13) and metadata[rid]['signer_id']<=9 and metadata[qid]['word_code']==metadata[rid]['word_code']
        assert not row['repeat'], 'Nontrivial compressed paths need explicit independent verification'
        nr=len(r['core']); domain=np.broadcast_to(r['core'][:,None,None],(nr,2,23)); V.same(domain,z['domain'])
        path=z['path']; assert (path[0]==[0,0]).all() and (path[-1]==[len(q['core'])-1,nr-1]).all()
        step=np.diff(path,axis=0); assert ((step>=0)&(step<=1)).all() and (step.sum(1)>0).all()
        ids=[path[path[:,1]==j,0] for j in range(nr)]; phase=np.array([np.median(x) for x in ids]); first=np.array([x.min() for x in ids]);last=np.array([x.max() for x in ids])
        V.same(phase,z['phase_old']);V.same(first,z['span_first']);V.same(last,z['span_last'])
        new=phase.copy()
        if nr>=5:
            for j in range(2,nr-2):new[j]=np.clip(np.dot(phase[j-2:j+3],[1,4,6,4,1])/16,phase[1],phase[-2])
            new=np.clip(new,phase-1,phase+1)
        V.close(new,z['phase_new']); assert (np.diff(new)>=0).all()
        value,av=V.mapped(q,phase); po=domain&av&r['available'];V.same(po,z['point_ok']);V.close(np.where(po,abs(value-r['values']),np.nan),z['point_delta'])
        od,ov=V.descriptor(q,phase,first,last,domain); nd,nv=V.descriptor(q,new,np.minimum(first,np.floor(new).astype(int)),np.maximum(last,np.ceil(new).astype(int)),domain); rd,rv=V.descriptor(r,np.arange(nr,dtype=float),np.arange(nr),np.arange(nr),domain)
        src=np.where(nv,1,np.where(ov,2,0)).astype(np.uint8);V.same(src,z['source']);ok=(src>0)&rv;V.same(ok,z['sequence_ok']);V.close(np.where(ok[...,None],abs(np.where((src==1)[...,None],nd,od)-rd),np.nan),z['sequence_delta'])
        evidence=E.eligibility(z,q,r); c=Counter(pairs=1,exact_repeat_realignment=0)
        for mode,k in [('point',0),('sequence',1)]:
            strong=evidence[k];V.same(strong,z[mode+'_evidence']);raw=z[mode+'_ok'];cut=b[mode+'_peer_count']>=5;ex=z[mode+'_delta']>b[mode+'_bound']+1e-6
            if mode=='sequence':ex=ex.any(-1)
            state=np.select([~domain,~raw,~strong,~cut,ex],[0,1,2,3,5],default=4).astype(np.uint8);V.same(state,z[mode+'_status'])
            assert not ((state>=4)&(~domain|~raw|~strong|~cut)).any()
            assert not np.isfinite(z[mode+'_delta'][~raw]).any()
            for name,a,bb in S.GROUPS:
                c[f'{mode}_{name}_domain']=int(domain[:,:,a:bb].sum());c[f'{mode}_{name}_raw']=int(raw[:,:,a:bb].sum())
                for code in range(1,6):c[f'{mode}_{name}_state{code}']=int((state[:,:,a:bb]==code).sum())
                assert sum(c[f'{mode}_{name}_state{code}'] for code in range(1,6))==c[f'{mode}_{name}_domain']
        assert dict(c)==row['counts'];counts.update(c);n['verified_pairs']+=1
        if (i+1)%300==0:print('VERIFY_SAVED_PAIRS',i+1,2697,flush=True)
    assert len(seen)==2697 and counts==Counter(done['counts']);S.check(f['protected'])
    S.write(S.U/'VERIFIED.json',dict(at=S.now(),status='PASS2697_FULL_SAVED_ARITHMETIC_899_TRAIN_ONLY_BOUNDS',tests=dict(n),counts=dict(counts),seconds=time.perf_counter()-start,pins={str(p):S.sha(p) for p in [S.U/'FROZEN.json',S.U/'DONE.json',S.U/'BOUNDS_DONE.json',__file__,V.__file__,E.__file__]},independence='Same Codex separate source-window, bound, path and status reconstruction; no new alignment or classifier scoring. Not human anatomical truth, sign correctness, or fresh heldout model accuracy.'))
    print('VERIFIED_ALL',dict(n),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=S.U/'FIRST_ERROR_verify.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
