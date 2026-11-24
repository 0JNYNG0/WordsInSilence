"""Separate saved-input replay and channel preservation checks; no inference."""
from common import *
from collections import Counter

def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x

def verify(stage):
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);assert not (U/'RUNNING.json').exists()
    rows=[r for r in f['rows'] if stage=='all' or r['uid'] in f['smoke_uids']]
    selective=module('verify_wrist_selective',Y/'candidate.py');counts=Counter();review=[]
    for row in rows:
        uid=row['uid'];rp=U/'records'/(uid+'.json');rec=read(rp);check(rec['outputs']);check(row['input_pins'])
        assert rec['context_sha256']==sha(U/'FROZEN.json')
        info=row['info'];a=info['source_arrays'];base=load(a['observation']);native=load(a['capture']);raw=load(a['raw']);ass=load(Path(a['capture']).parent/'assignment.npz')['sides']
        old=load(a['channels']);out=load(U/'outputs'/(uid+'.npz'));new=load(U/'features'/(uid+'.npz'))
        assert set(old)==set(new)
        for k in ('xy_px','observed','usable','conf','hand_score'):assert same(out[k][0],base[k]),(uid,k)
        for k in ('frame_id','timestamp_us'):assert same(out[k],base[k]) and equal(out[k],native[k])
        assert same(out['full_xy_px'][0],native['xy_px']) and same(out['full_usable'][0],native['usable'])
        target={q['index']:q for q in row['targets']};assert {fr['i'] for fr in rec['frames']}==set(target)
        assert len(rec['frames'])==len(target)
        for k in (1,2):
            keep=base['usable'];assert out['usable'][k][keep].all() and same(out['xy_px'][k][keep],base['xy_px'][keep])
            assert same(out['xy_px'][k,:,:33],base['xy_px'][:,:33]) and same(out['usable'][k,:,:33],base['usable'][:,:33])
            counts['original_point_frames_preserved']+=int(keep.sum())
        untouched=np.array([i not in target for i in range(info['frames'])])
        for k in (1,2):
            assert same(out['xy_px'][k][untouched],base['xy_px'][untouched]);assert same(out['full_xy_px'][k][untouched],native['xy_px'][untouched])
        cache={}
        if row['clip_id'] in f['cache']:
            cr=f['cache'][row['clip_id']];assert sha(cr['file'])==cr['sha256'];cc=read(cr['file'])
            if cc['canvas']==info['canvas']:cache=cc['frames']
        for fr in rec['frames']:
            i=fr['i'];fid=int(native['frame_id'][i]);assert fid==fr['frame_id']==target[i]['frame_id']
            assert fr['canvas']==info['canvas'];w,h=info['canvas'];box=info['original_roi']
            p=native['xy_px'][i,33:].reshape(2,21,2);v=native['usable'][i,33:].reshape(2,21)
            pose=raw['pose_xy_px'][i];pu=raw['pose_usable'][i];nn=int(raw['raw_count'][i])
            assert clean(selective.trigger(p,v,pose,pu,w,h,box))==fr['boxes']
            oldp=list(raw['hand_xy_px'][i,:nn]);sides=ass[i,:nn].tolist()
            for s in (0,1):
                if v[s].any() and s not in sides:oldp.append(p[s]);sides.append(s)
            ds=rule.merge(restore(fr['mp_full']),restore(fr['mp_crop']),oldp,sides,pose,pu,w,h)
            pp=p.copy();pv=v.copy()
            for d in ds:
                if d['accepted']:
                    s=d['side']
                    if str(s) not in fr['boxes'] or v[s].any():d.update(accepted=False,reason='not_missing_target');continue
                    good=np.asarray(d['joint_supported'],bool);pp[s]=np.where(good[:,None],points(fr['mp_full'][d['index']]),np.nan);pv[s]=good
            assert clean(ds)==fr['mp_decisions']
            assert same(pp.reshape(42,2),out['full_xy_px'][1,i,33:]) and same(pv.reshape(42),out['full_usable'][1,i,33:])
            yt=selective.trigger(pp,pv,pose,pu,w,h,box);assert clean(yt)==fr['yolo_boxes']
            fp=pp.copy();fv=pv.copy();yd=[]
            if yt:_,_,fp,fv,yd=selective.supplement(restore(fr['yolo_full']),restore(fr['yolo_crop']),pp,pv,raw['hand_xy_px'][i,:nn],ass[i,:nn].tolist(),pose,pu,w,h,yt)
            assert clean(yd)==fr['yolo_decisions'];assert same(fp.reshape(42,2),out['full_xy_px'][2,i,33:])
            for k,(xy,valid) in enumerate(((pp,pv),(fp,fv)),1):
                cropped,ok=roi(xy,valid,box)
                assert same(cropped.reshape(42,2),out['xy_px'][k,i,33:]) and same(ok.reshape(42),out['usable'][k,i,33:])
            for call in fr['calls']:
                b=call['box']
                if b:assert 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h
                if call['cached']:
                    assert str(fid) in cache and call['detector'] in cache[str(fid)]
                    entry=cache[str(fid)][call['detector']];assert entry['source']==call['source']
                    if b:assert ','.join(map(str,b)) in entry['crops']
                    counts['cache_calls_verified']+=1
                else:counts['new_calls_recorded']+=1
            for s in (0,1):
                j=33+21*s;expected=not v[s].any() and bool(out['usable'][2,i,j])
                assert out['wrist_usable'][i,s]==expected
                if expected:
                    assert same(out['wrist_xy'][i,s],out['xy_px'][2,i,j])
                    review.append(dict(uid=uid,index=i,side=s,frame_id=fid,wrist=out['wrist_xy'][i,s].tolist(),origin=int(out['point_origin'][2,i,j]),image=str(U/'frames'/f'{uid}_{i:04d}.jpg')))
            counts['geometry_frames']+=1
        assert not out['wrist_usable'][untouched].any()
        for temporal in info['original_spans']:
            key='x_'+temporal;lo,hi=info['original_spans'][temporal];x=old[key];y=new[key];assert x.shape==y.shape and x.dtype==y.dtype
            allowed=np.zeros(x.shape,bool);expected=x.copy()
            scale=info['original_scales'][temporal]['body']
            for i in range(lo,hi+1):
                for s in (0,1):
                    if not out['wrist_usable'][i,s] or not base['usable'][i,[11,12]].all():continue
                    assert not base['usable'][i,33+21*s:54+21*s].any()
                    j=9+21*s;cols=[j*2,j*2+1,186+j];allowed[i-lo,cols]=True
                    midpoint=base['xy_px'][i,[11,12]].sum(axis=0)/np.float32(2.0)
                    expected[i-lo,2*j:2*j+2]=(out['wrist_xy'][i,s]-midpoint)/scale;expected[i-lo,186+j]=1
                    counts['wrist_feature_insertions']+=1
            assert same(x[~allowed],y[~allowed]),uid
            assert same(expected,y),(uid,temporal)
            assert np.isfinite(y).all();counts['protected_feature_cells']+=int((~allowed).sum())
            counts['feature_cells']+=x.size
        for k in old:
            if not k.startswith('x_'):assert same(old[k],new[k])
        counts['views']+=1
    check(f['pins']);check(f['protected'])
    write(U/(stage.upper()+'_VERIFIED.json'),dict(at=now(),status='PASS',counts=dict(counts),frozen_sha256=sha(U/'FROZEN.json'),verifier_sha256=sha(__file__),
        scope='Separate arithmetic/replay by same Codex, not independent human anatomical truth. Normal original hands/arms/crossings exactly preserved. Wrist-only additions need visual review.',review=review))
    print(json.dumps(dict(counts)),flush=True)
if __name__=='__main__':verify(sys.argv[1])
