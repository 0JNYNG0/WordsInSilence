"""Independent vectorized raw-observation reconstruction of prepared inputs."""
import argparse
from collections import Counter
import json
import numpy as np
from common import U,OUT,read,write,sha,pin_check

BODY=[0,11,12,13,14,15,16,23,24]
POINTS=BODY+list(range(33,75))

def rebuild(z,lo,hi):
    xy=z['xy_px'][lo:hi+1][:,POINTS]; pr=z['usable'][lo:hi+1][:,POINTS]
    ts=z['timestamp_us'][lo:hi+1]
    shoulders=pr[:,1]&pr[:,2]
    size=np.median(np.linalg.norm(xy[shoulders,1]-xy[shoulders,2],axis=1))
    center=(xy[:,1]+xy[:,2])/2
    valid=pr&shoulders[:,None]
    body=(xy-center[:,None,:])/size;body[~valid]=0
    hands=[]; masks=[]
    for base in [9,30]:
        pts=xy[:,base:base+21]; good=pr[:,base:base+21]
        wrist=pts[:,:1]; mcp=pts[:,[5,9,13,17]]
        lengths=np.linalg.norm(mcp-wrist,axis=-1)
        mask=good[:,[5,9,13,17]]&good[:,:1]
        s=np.median(lengths[mask]) if mask.any() else float('nan')
        hm=good&good[:,:1]
        if not np.isfinite(s) or s<=1e-6:
            hands.append(np.zeros_like(pts));masks.append(np.zeros_like(hm));continue
        rel=(pts-wrist)/s;rel[~hm]=0
        hands.append(rel);masks.append(hm)
    dt=np.r_[np.float32(0),np.diff(ts).astype(np.float32)/1e6].astype(np.float32)
    return np.concatenate([body.reshape(len(ts),-1),np.concatenate(hands,axis=1).reshape(len(ts),-1),valid,np.concatenate(masks,axis=1),dt[:,None]],axis=1).astype(np.float32)

def main(partial):
    frozen=pin_check(); rows=read(U/'MANIFEST.json'); counts=Counter(); frames=Counter(); fail=[]
    records=[]; record_hashes={}; manifest_hash=sha(U/'MANIFEST.json')
    for r in rows:
        p=OUT/'records'/f"{r['clip_id']}.json"
        if not p.exists():
            assert partial, 'Preparation incomplete'
            continue
        a=read(p);assert a['clip_id']==r['clip_id'] and a['role']==r['role'] and a['label_index']==r['label_index']
        assert a['signer_id']==r['signer_id']
        assert a['manifest_sha256']==manifest_hash
        record_hashes[r['clip_id']]=sha(p)
        assert sha(a['observation'])==a['observation_sha256']
        with np.load(a['observation'],allow_pickle=False) as z:
            t=z['timestamp_us']; n=len(t)
            assert np.array_equal(z['frame_id'],np.arange(n)) and np.all(np.diff(t)>0)
            assert np.all(~z['usable']|z['observed']) and not np.isfinite(z['xy_px'][~z['observed']]).any()
            anyhand=z['usable'][:,33:75].any(1); ids=np.flatnonzero(anyhand)
            if a['channel_file'] is None:
                assert len(ids)==0 or len(ids)<4 or ids[-1]-ids[0]+1<4
                fail.append(dict(clip_id=r['clip_id'],role=r['role'],reason=a['feature_check']['reason']))
                counts[r['role']+'_failed']+=1
                continue
            assert len(ids)>=4
            lo,hi=int(ids[0]),int(ids[-1]); assert [lo,hi]==a['feature_check']['whole_span']
            assert sha(a['channel_file'])==a['channel_sha256']
            with np.load(a['channel_file'],allow_pickle=False) as c:
                ref=rebuild(z,lo,hi)
                np.testing.assert_allclose(ref,c['x_whole'],rtol=1e-6,atol=1e-6)
                np.testing.assert_array_equal(t[lo:hi+1],c['whole_ts'])
                if r['role']=='train_candidate':
                    s,e=r['intervals'][0]; ctx=np.flatnonzero((t>=round((s-.3)*1e6))&(t<=round((e+.3)*1e6)))
                    cl=max(lo,int(ctx[0]));ch=min(hi,int(ctx[-1]))
                    np.testing.assert_array_equal(t[cl:ch+1],c['context_ts'])
                    np.testing.assert_array_equal(z['frame_id'][cl:ch+1],c['context_frame_id'])
                    np.testing.assert_allclose(rebuild(z,cl,ch),c['x_context'],rtol=1e-6,atol=1e-6)
                else: assert 'x_context' not in c.files, 'Do not prepare oracle development core input'
            counts[r['role']]+=1;frames[r['role']]+=n;records.append(a)
    if not partial:
        assert len(records)+len(fail)==1199
        per=Counter(r['label_index'] for r in records if r['role']=='train_candidate')
        assert set(per)==set(range(100)) and counts['train_candidate_failed']/899<=.05
    result=dict(status='PASS',scope='partial first inputs' if partial else 'complete1199 manifest inputs',
        counts=dict(counts),frames=dict(frames),failures=fail,
        checks=['frozen sources','source manifest separation','raw observation hashes/missing/PTS','independent vectorized body and hand normalization','exact dt and context row preservation','no oracle core channels for development'],
        verifier='Codex separate reconstruction, not a second human/agent',
        record_hashes=record_hashes)
    path=U/('INPUT_CHECK_FIRST6.json' if partial else 'INPUT_CHECK.json')
    assert not path.exists();write(path,result)
    print(json.dumps({k:v for k,v in result.items() if k!='record_hashes'},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
