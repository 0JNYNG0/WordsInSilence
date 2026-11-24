"""Exact-condition resume after exclusive PROGRESS write failure. Preserve 80 outputs."""
import os,traceback
import numpy as np
import study as d
U=d.U

def main():
    f=d.read(U/'FROZEN.json');d.check(f['pins']);d.check(f['protected'])
    assert (U/'FIRST_ERROR.txt').is_file() and not (U/'RESULT.json').exists()
    assert 'FileExistsError' in (U/'FIRST_ERROR.txt').read_text('utf8')
    assert len(list((U/'cases').glob('*.npz')))==80
    # Prior exec d08cb7 returned exit1. Its lock is retained as failure evidence.
    (U/'RUNNING.json').rename(U/'FAILED_EXECUTION_LOCK.json')
    d.write(U/'RECOVERY_AUTHORIZATION.json',dict(at=d.now(),code=d.sha(__file__),prior_exec='d08cb7 exit1',
      cause='Exclusive write helper cannot overwrite PROGRESS.json. Calculations unchanged; 80partial cases preserved and exactly checked before reuse.',
      preserved={str(p):d.sha(p) for sub in ('cases','inputs','controls') for p in (U/sub).glob('*.npz')}))
    d.write(U/'RECOVERY_RUNNING.json',dict(at=d.now(),pid=os.getpid()));(U/'progress').mkdir()
    samples={r['clip_id']:d.data(r) for r in f['rows']};allrows=[];controls=[];reused=0
    for qr in f['rows']:
        cid=qr['clip_id'];q,e=samples[cid];req=f['roles'].get(qr['word_code'],{}).get('required_sides',['L','R'])
        refs=[r for r in f['rows'] if r['word_code']==qr['word_code'] and r['signer_id']!=qr['signer_id']]
        weights=d.A.C.weights_for(req,(False,False),d.A.F.GROUPS)
        for rr in refs:
            rid=rr['clip_id'];r,re=samples[rid]
            for kind in d.KINDS:
                qq,qe,ix=d.transform(q,e,kind)
                cost,ev=d.A.legacy(qq,r,group_weights=weights,hand_participation=False,one_sided_mode='flat')
                path=np.asarray(d.A.dp(cost,.2),int);out=d.evaluate(qe,re,path,req)
                name=kind+'__'+cid+'__'+rid;p=U/'cases'/(name+'.npz')
                arrays=dict(out,path=path,source_index=ix,cost=cost)
                if p.exists():
                    saved=d.load(p);assert saved.keys()==arrays.keys()
                    for k in arrays:assert np.array_equal(saved[k],arrays[k],equal_nan=True),(name,k)
                    reused+=1
                else:np.savez_compressed(p,**arrays)
                allrows.append(dict(name=name,query=cid,reference=rid,label=qr['label'],kind=kind,**d.stats(out),alignment_evidence=float(ev[tuple(path.T)].mean())))
                if kind=='identity' and rr==refs[0]:
                    for h,side in enumerate(('L','R')):
                        if side not in req:continue
                        for mode in ('wholehand','shortgap','wholearm'):
                            zz={k:v.copy() for k,v in qe.items()};ids=np.arange(len(qe['core']))
                            if mode=='shortgap':
                                act=np.flatnonzero(qe['core']);mid=act[len(act)//2];ids=np.arange(mid-1,mid+2)
                            if mode=='wholearm':zz['arm_valid'][ids,h]=False;zz['arm_xy'][ids,h]=np.nan
                            else:
                                zz['angle_valid'][ids,h]=False;zz['angle_low'][ids,h]=np.nan;zz['angle_high'][ids,h]=np.nan
                                zz['palm_valid'][ids,h]=False;zz['centre'][ids,h]=np.nan;zz['radius'][ids,h]=np.nan
                            co=d.evaluate(zz,re,path,req);cn=mode+'_'+side+'__'+cid;cp=U/'controls'/(cn+'.npz')
                            ca=dict(co,path=path,removed=ids)
                            if cp.exists():
                                saved=d.load(cp)
                                for k in ca:assert np.array_equal(saved[k],ca[k],equal_nan=True),(cn,k)
                            else:np.savez_compressed(cp,**ca)
                            controls.append(dict(name=cn,parent=name,mode=mode,side=side,**d.stats(co)))
        d.write(U/'progress'/(cid+'.json'),dict(at=d.now(),query=cid,completed=len(allrows),total=1080))
        print('DONE',cid,len(allrows),flush=True)
    assert len(allrows)==1080 and len(controls)==135 and reused==80
    d.write(U/'RESULT.json',dict(at=d.now(),rows=allrows,controls=controls,training=0,user_data=0,reused=reused))
    d.check(f['pins']);d.check(f['protected']);d.check(d.read(U/'RECOVERY_AUTHORIZATION.json')['preserved'])
    (U/'RECOVERY_RUNNING.json').unlink()
    print('COMPLETE1080 controls135 reused80',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'RECOVERY_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
