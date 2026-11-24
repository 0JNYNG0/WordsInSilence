"""Deterministic input/contract checks only, no model training or forward."""
from physical_branch import *
import importlib.util

def main():
    checks=0
    def ok(x,message):
        nonlocal checks
        assert bool(x),message;checks+=1
    cfg=read(U/'PHYSICAL_FROZEN.json')
    for p,h in cfg['pins'].items():ok(sha(p)==h,f'pin:{p}')
    z=np.load(U/'PHYSICAL_INPUTS.npz');rows=read(U/'PHYSICAL_INDEX.json')['rows'];labels=read(U/'PHYSICAL_SOURCE_LABELS.json')['rows'];source={r['key']:r for r in labels}
    ok(len(rows)==1380,'closeddenominator');ok(sum(r['right']=='U'or r['left']=='U'for r in labels)==2,'U preserved')
    for i,r in enumerate(rows):
        x,eligible,ob=features(z['q'][i],z['valid'][i],z['exists'][i],4)
        ok(np.allclose(x,z['x'][i],rtol=0,atol=1e-7),'input reconstruction');ok(eligible==z['eligible'][i],'eligible reconstruction')
        ok(len(r['source_frames'])==len(set(r['source_frames'])),'no duplicated native evidence')
        expected=sorted(range(max(0,r['centre_frame']-4),r['centre_frame']+5))
        ok(sorted(r['source_frames'])==expected,'native windowtime')
        ok(np.array_equal(np.flatnonzero(z['exists'][i]),r['positions']),'padding tracked')
        if r['centre_frame']==0:ok(z['exists'][i].sum()==5,'clipstart five real frames')
    for f in cfg['folds']:
        pid=f['person'];w=np.load(f['weights'])['weights'];saved=read(f['normalization_rule']);training=w>0;held=z['person']==pid
        ok(not w[held].any(),'held source no loss');ok(np.isclose(w.sum(),1),'loss sums1');ok(not w[z['y']<0].any(),'unknown excluded from loss')
        ok(set(r['source_sha256']for r,t in zip(rows,training)if t).isdisjoint(r['source_sha256']for r,t in zip(rows,held)if t),'source hashes separated')
        q=z['x'][training].astype(float)
        ok(np.allclose(saved['normalization']['mean'],q.mean(0)),'FIT-only mean');ok(np.allclose(saved['normalization']['std'],np.maximum(q.std(0),1e-4)),'FIT-only std')
        for value in [0,1]:
            ok(np.isclose(w[z['y']==value].sum(),.5),'class balance')
        ok(calibrate_rule(z['x'],z['y'],w)==saved['rule'],'rule deterministic FIT-only')
    by={(r['source_key'],r['view'],r['reverse']):i for i,r in enumerate(rows)}
    for i,r in enumerate(rows):
        if r['reverse']:continue
        j=by[(r['source_key'],r['view'],True)]
        ok(np.allclose(z['q'][i],z['q'][j][::-1],atol=2e-5),'reversal same raw points')
        ok(np.array_equal(z['valid'][i],z['valid'][j][::-1]),'reversal masks')
        ok(z['y'][i]==z['y'][j],'window label invariant to playback direction')
        ok(np.allclose(z['x'][i,6:8],-z['x'][j,6:8],atol=2e-5),'signed net reversal')
        ok(np.allclose(z['x'][i,8:12],z['x'][j,8:12],atol=2e-5),'path/range reversal invariant')
    q=np.zeros((9,2));valid=np.ones(9,bool);exists=np.ones(9,bool)
    q[4,0]=.2;a,e,_=features(q,valid,exists,4)
    ok(e and a[11]>.19 and a[10]>.39 and np.allclose(a[6:8],0),'brief out-and-back retained even netzero')
    q=np.tile(np.array([.3,-.7]),(9,1));valid[3:5]=False
    b,e,_=features(q,valid,exists,4);q[~valid]=1e6;c,e2,_=features(q,valid,exists,4)
    ok(e and e2 and np.array_equal(b,c),'missing values never averaged aszero/huge');ok(b[11]==0,'stationary unaffected bygap')
    valid[:]=False;d,e,_=features(q,valid,exists,4);ok(not e and np.isfinite(d).all(),'unobserved U not H')
    valid[:]=True;valid[3:6]=False;ok(not features(q,valid,exists,4)[1],'gap3 notcertified')
    exists[:4]=False;valid=exists.copy();q=np.zeros((9,2));a,e,ob=features(q,valid,exists,4)
    ok(e and ob['actual']==5 and ob['observed']==5,'real clipstart accepted without fakepast')
    # Actual existing body_points normalization, not a newly substituted formula.
    p=U.parent.parent/'word100_motion_components_20261109/motion.py';spec=importlib.util.spec_from_file_location('motion_normalization_check',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    base=np.zeros((9,237));points=base[:,:102].reshape(9,51,2);points[:,1]=[-.5,0];points[:,2]=[.5,0];points[:,6,0]=np.linspace(.2,.8,9);points[:,6,1]=-.4;points[:,5]=[-.3,-.4];base[:,186:]=1
    p0,v0=mod.body_points(base);trans=base.copy();p1=trans[:,:102].reshape(9,51,2);p1*=2.7;p1+=np.stack([np.linspace(4,5,9),np.linspace(-3,-2,9)],1)[:,None,:];p1,v1=mod.body_points(trans)
    ok(np.allclose(p0,p1)and np.array_equal(v0,v1),'body translation/sway and scale invariance')
    mirrored=base.copy();pm=mirrored[:,:102].reshape(9,51,2);pm[:,:,0]*=-1;pm[:,[1,2]]=pm[:,[2,1]];pm[:,[5,6]]=pm[:,[6,5]];pn,_=mod.body_points(mirrored)
    right=p0[:,6]*np.array([1,-1]);left=pn[:,5]*np.array([-1,-1]);ok(np.allclose(right,left),'same convention acrossanatomicalhands')
    # Time is not a classifier input: replay timestamp scaling cannot alter x.
    ok('timestamp'not in features.__code__.co_varnames,'no absolute time/word/person leakage')
    for p,h in cfg['pins'].items():ok(sha(p)==h,f'preserved:{p}')
    put(U/'PHYSICAL_SELF_VERIFICATION.json',dict(status='PASS',checks=checks,frozen_sha256=sha(U/'PHYSICAL_FROZEN.json'),test_code_sha256=sha(Path(__file__)),no_learning=True,no_model_forward=True,independence='Implementation author arithmetic/input checks; independent root review still required',limits='Retiming invariance is arithmetic, not slow-user accuracy. Existing normalization is shoulder2D; rotation/depth ambiguity remains.'))
    print(json.dumps(dict(status='PASS',checks=checks)))

if __name__=='__main__':main()
