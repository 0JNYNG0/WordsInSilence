"""Separate scalar verification of saved paths, units and source immutability."""
import math
import inspect_alignment as S
import numpy as np

def optimal_sum(cost,ratio):
    n,m=cost.shape; band=max(1,round(ratio*max(n,m))); prev=[math.inf]*(m+1);prev[0]=0.
    for i in range(1,n+1):
        cur=[math.inf]*(m+1);center=(i-1)*(m-1)/(n-1)
        for j in range(max(1,math.floor(center-band)+1),min(m,math.ceil(center+band)+1)+1):
            cur[j]=float(cost[i-1,j-1])+min(prev[j-1],prev[j],cur[j-1])
        prev=cur
    return prev[-1]

def main():
    f=S.read(S.U/'FROZEN.json');result=S.read(S.U/'RESULT.json');S.check(f['sources']);S.check(f['protected']);count=0
    for row in result['rows']:
        p=S.U/'arrays'/f"{row['word_id']}.npz";assert S.sha(p)==row['array_sha256']
        with np.load(p) as z:
            path=z['path'];n,m=z['cost'].shape
            assert tuple(path[0])==(0,0) and tuple(path[-1])==(n-1,m-1)
            assert all(tuple(d) in [(1,0),(0,1),(1,1)] for d in np.diff(path,axis=0))
            assert set(path[:,0])==set(range(n)) and set(path[:,1])==set(range(m))
            total=sum(float(z['cost'][i,j]) for i,j in path)
            assert abs(total-row['dtw_sum'])<1e-8
            assert abs(total-optimal_sum(z['cost'],f['band']))<1e-8
            assert abs(sum(float(z['evidence'][i,j]) for i,j in path)/len(path)-row['evidence'])<1e-8
            for key,points in [('before',z['linear']),('after',path)]:
                for r in row[key]:
                    g=r['group']+'_'+r['side'];gi=S.A.F.GROUP_ORDER.index(g);sl=S.A.OFFSETS[g];errs=[]
                    for i,j in points:
                        a=z['q_features'][i,sl];b=z['r_features'][j,sl]
                        if not (z['q_present'][i,gi] and z['r_present'][j,gi] and np.isfinite(a).all() and np.isfinite(b).all()):continue
                        if r['group']=='handpos':err=math.hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))*100
                        else:err=sum(abs(float(x)-float(y)) for x,y in zip(a[:15],b[:15]))/15*180
                        errs.append(err)
                    assert len(errs)==r['paired']
                    assert (not errs and r['difference'] is None) or abs(sum(errs)/len(errs)-r['difference'])<3e-5
                    count+=1
    S.write('VERIFIED.json',dict(status='PASS',clips=12,metric_recounts=count,independent_scalar_dp=12,saved_matches=12,source_hashes=len(f['sources']),protected=len(f['protected']),
        meaning='Same Codex separate arithmetic implementation; no human motion-alignment ground truth. Missing is not zero. C/T3F/web/source unchanged.'))
    print('VERIFY_PASS',count,flush=True)
if __name__=='__main__':main()
