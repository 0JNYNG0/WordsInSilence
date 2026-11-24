"""Untrained generic input primitive. Not connected to any operational model.
Preserve missingness separately instead of averaging absent coordinates as zeros.
Expose projection numerator and denominator instead of a poorly conditioned ratio.
"""
from pathlib import Path
import numpy as np
import json,hashlib
U=Path(__file__).resolve().parent;P=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def valid_context(coords,available,timestamps,halo_us=133333):
    coords=np.asarray(coords,dtype=np.float64);available=np.asarray(available,dtype=bool);ts=np.asarray(timestamps)
    assert coords.shape[:-1]==available.shape and len(coords)==len(ts)
    lo=np.searchsorted(ts,ts-halo_us);hi=np.searchsorted(ts,ts+halo_us,side='right');out=[];coverage=[]
    for i,(a,b)in enumerate(zip(lo,hi)):
        slots=[];covers=[]
        for u,v in [(a,i+1),(i,i+1),(i,b)]:
            m=available[u:v];count=m.sum(0);s=(coords[u:v]*m[...,None]).sum(0)
            slots.append(np.divide(s,count[...,None],out=np.zeros_like(s),where=count[...,None]>0));covers.append(m.mean(0))
        out.append(slots);coverage.append(covers)
    return np.asarray(out),np.asarray(coverage)
def relative_valid_context(coords,available,timestamps):
    c,m=valid_context(coords,available,timestamps);out=c[:,[1,0,2]].copy();mm=m[:,[1,0,2]].copy()
    for slot in [1,2]:
        good=(mm[:,0]>0)&(mm[:,slot]>0);out[:,slot]=np.where(good[...,None],out[:,slot]-out[:,0],0)
    return out,mm
def projection_terms(ratio,anchor_length):
    # body-normalized dot product and squared length are finite as length shrinks.
    ratio=np.asarray(ratio);length=np.asarray(anchor_length)
    return np.stack([ratio*length**2,length**2],axis=-1)
def main():
    checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    ts=np.arange(9)*33333;q=np.full((9,1,4),2.);m=np.ones((9,1),bool);m[5:7]=False
    rel,cov=relative_valid_context(q,m,ts)
    ck('constant_observed_pose_with_gap_has_zero_change',np.max(abs(rel[:,1:]))==0)
    ck('missingness_retained',np.any(cov<1))
    a=np.arange(9)[:,None,None]*np.ones((1,1,4));forward,cm=valid_context(a,np.ones_like(m),ts)
    reverse,rm=valid_context(a[::-1],np.ones_like(m),ts)
    ck('time_reversal_exchanges_context',np.allclose(reverse,forward[::-1,::-1]))
    length=np.array([.2,.1,.05,.001]);ratio=.3/length;terms=projection_terms(ratio,length)
    ck('projection_roundtrip',np.allclose(terms[:,0]/terms[:,1],ratio))
    ck('finite_small_denominator_features',np.isfinite(terms).all())
    z=np.load(P/'INPUTS.npz');x=z['absolute_context'].reshape(-1,3,10,5);rows=read(P/'INDEX.json')['rows'];summaries=[]
    for r in rows:
        if r['reverse']:continue
        sl=slice(r['start'],r['stop']);cur=x[sl,1];xx,mask=valid_context(cur[...,:4],cur[...,4]>.5,z['timestamp_us'][sl])
        ck('availability_unchanged_'+str(r['input_index']),np.allclose(mask,x[sl,...,4],atol=2e-7))
        if np.all(cur[...,4]>.5):ck('allobserved_context_equal',np.allclose(xx,x[sl,...,:4],atol=2e-5))
        projected=projection_terms(cur[:,4,0],cur[:,3,2]);good=(cur[:,4,4]>.5)&(cur[:,3,4]>.5)
        ck('public_ratio_reconstruction_'+str(r['input_index']),np.allclose(projected[good,0]/projected[good,1],cur[good,4,0],rtol=1e-5,atol=1e-6))
        known=(z['mask'][sl]>0);nmissing=int(((mask[:,0,:,None]<1)|(mask[:,2,:,None]<1)).sum())
        summaries.append(dict(clip_id=r['clip_id'],view=r['view'],max_projection_ratio=float(np.max(abs(cur[:,4,0]))),
            primary_projection_min_valid_length=float(np.min(cur[good,3,2]))if good.any()else None,
            context_entries_with_gap=nmissing))
    report=dict(status='INPUT_PRIMITIVE_PASS_NOT_TRAINED_OR_DEPLOYED',checks=len(checks),check_names=checks,rows=summaries,
        limits='This validates arithmetic and mask behavior only, not feedback accuracy. No labels changed; no extra model selected.',
        next='Freeze a new comparison before any training; retain needed body context and expose availability/conditioning; do not use this primitive to declare G2 solved.')
    (U/'CONDITIONING_VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    print(report['status'],report['checks'])
if __name__=='__main__':main()
