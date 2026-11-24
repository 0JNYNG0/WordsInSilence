"""Independent oracle descriptor reconstruction. Does not import diagnose.py."""
from pathlib import Path
import sys,json,hashlib,datetime,importlib.util
from collections import Counter,defaultdict
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_point_sequence_split_20261022';B=O/'word100_coherent_reference_sequence_20261022';I=O/'word100_cross_person_motion_support_20261022/inputs'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
spec=importlib.util.spec_from_file_location('verified_channel_representation',B/'candidate.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
def interp(c,t):
    a=t.astype(int);b=np.ceil(t).astype(int);w=(t-a).reshape(-1,1,1);ok=c['valid'][a]&c['valid'][b]
    return {k:np.where(ok,c[k][a]*(1-w)+c[k][b]*w,np.nan) for k in ('lo','hi')}|{'valid':ok}
def inv(ix,t):
    low=np.zeros(len(t));high=low.copy()
    for j,target in enumerate(t):
        matches=np.flatnonzero(ix==target)
        if len(matches):low[j]=matches[0];high[j]=matches[-1]
        else:
            a=np.flatnonzero(ix<target)[-1];b=np.flatnonzero(ix>target)[0]
            low[j]=high[j]=a+(target-ix[a])/(ix[b]-ix[a])*(b-a)
    return low,high
def descriptor(c,t,left,right,rest,domain):
    z=interp(c,t);n=len(t);l=np.full(domain.shape+(5,),np.nan);h=l.copy();ok=np.zeros(domain.shape,bool)
    for j in range(2,n-2):
        a=int(left[j-2]);b=int(right[j+2]);ii=[j-2,j,j+2]
        good=domain[j-2:j+3].all(0)&c['valid'][a:b+1].all(0)&~rest[a:b+1].any(0)[:,None]&z['valid'][ii].all(0)
        before,mid,after=(z['lo'][k] for k in ii);bhi,mhi,ahi=(z['hi'][k] for k in ii)
        l[j]=np.stack([before,mid,after,mid-bhi,after-mhi],-1);h[j]=np.stack([bhi,mhi,ahi,mhi-before,ahi-mid],-1);ok[j]=good
    return np.where(ok[...,None],l,np.nan),np.where(ok[...,None],h,np.nan),ok
def main():
    f=read(U/'FROZEN.json');r=read(U/'RESULT.json');assert r['frozen_sha256']==sha(U/'FROZEN.json') and not (U/'RUNNING.json').exists()
    for p,h in {**f['pins'],**f['protected'],**r['output_pins']}.items():assert sha(p)==h,p
    es={row['clip_id']:load(I/(row['clip_id']+'.npz')) for row in f['rows']};cs={k:C.channels(e) for k,e in es.items()}
    groups={'angle':slice(0,15),'palm':slice(15,17),'elbow':slice(17,19),'wrist':slice(19,21)};tot=defaultdict(Counter);cells=0;examples=[]
    for row in r['rows']:
        cid=row['query'];rid=row['reference'];folder=row['folder'];z=load(U/'cases'/(row['name']+'.npz'))
        original_name=row['name'][len(folder)+2:];saved=load(P/folder/(original_name+'.npz'));ident='identity__'+cid+('__'+rid if folder=='cross' else '')
        old=load(P/folder/(ident+'.npz'));cal=load(P/'calibration'/(cid+'__'+rid+'.npz'));ix=saved['source_index'];n=len(z['expected'])
        path=old['path'];indices=[path[path[:,1]==j,0] for j in range(n)];t=np.array([np.median(ids) for ids in indices]);left=np.array([min(ids) for ids in indices]);right=np.array([max(ids) for ids in indices])
        low,high=inv(ix,t);ot=(low+high)/2;ol,_=inv(ix,left);_,orr=inv(ix,right)
        cc=interp(cs[cid],ix);rest=es[cid]['source_rest'][ix.astype(int)]&es[cid]['source_rest'][np.ceil(ix).astype(int)]
        qlo,qhi,qok=descriptor(cc,ot,np.floor(ol).astype(int),np.ceil(orr).astype(int),rest,z['expected'])
        ii=np.arange(n);alo,ahi,aok=descriptor(cs[rid],ii,ii,ii,es[rid]['source_rest'],z['expected'])
        bok=z['expected']&qok&aok&cal['support'];gap=np.maximum(0,np.maximum(qlo-ahi,alo-qhi));bad=bok[...,None]&(gap-cal['component_bound']>1e-6)
        assert np.array_equal(bok,z['oracle_ok']) and np.array_equal(bad,z['oracle_bad'])
        for key in ('expected','point_comparable','point_bad','native_expected','native_unknown'):assert np.array_equal(z[key],saved[key])
        assert np.array_equal(z['actual_ok'],saved['sequence_comparable']) and np.array_equal(z['actual_bad'],saved['component_bad'])
        assert np.array_equal(z['base_ok'],old['sequence_comparable']) and np.array_equal(z['base_bad'],old['component_bad'])
        common=z['base_ok']&z['actual_ok']&bok;assert np.array_equal(common,z['common'])
        a=common[...,None]&z['actual_bad']&~z['base_bad'];b=common[...,None]&bad&~z['base_bad']
        for name,v in [('actual_new',a),('oracle_new',b),('removed_by_phase',a&~b),('new_after_phase',b&~a),('remaining',a&b)]:assert np.array_equal(v,z[name])
        expected_phase=np.interp(ot,np.arange(len(ix)),ix);assert np.allclose(expected_phase,t,atol=1e-10,rtol=0) and np.allclose(expected_phase,z['oracle_phase'],atol=1e-10,rtol=0)
        cells+=z['oracle_bad'].size
        for g,sl in groups.items():
            d=tot[folder+'__'+g];d['common']+=int(common[:,:,sl].sum());d['expected']+=int(z['expected'][:,:,sl].sum())
            for name in ('actual_new','oracle_new','removed_by_phase','new_after_phase','remaining'):
                for j in range(5):d[name+'_'+str(j)]+=int(z[name][:,:,sl,j].sum())
            d['lost_oracle']+=int((z['actual_ok'][:,:,sl]&~bok[:,:,sl]).sum());d['gained_oracle']+=int((~z['actual_ok'][:,:,sl]&bok[:,:,sl]).sum())
        # Representative largest alignment perturbations, no classification results involved.
        examples.append(dict(name=row['name'],phase_error_max=float(z['phase_error'].max()),actual_new_changes=int(a[...,3:].sum()),oracle_new_changes=int(b[...,3:].sum())))
    for key,values in tot.items():
        for k,v in values.items():assert r['aggregate'][key][k]==v,(key,k)
    out=dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS_SEPARATE_TIME_INVERSE_DESCRIPTOR_AND_RECOUNT',cases=len(r['rows']),component_cells=cells,result_sha256=sha(U/'RESULT.json'),verifier_sha256=sha(__file__),aggregate={k:dict(v) for k,v in tot.items()},examples=sorted(examples,key=lambda e:e['phase_error_max'],reverse=True)[:12],
        existing_pointwise_native_missingness_unchanged=True,known_phase_not_available_at_runtime=True,no_new_DTW=True,independence='Same Codex independent inverse/span/interval arithmetic; existing frozen channel representation shared. Not human semantic phase truth.')
    with (U/'VERIFIED.json').open('x',encoding='utf8') as h:json.dump(out,h,ensure_ascii=False,indent=2)
    print(json.dumps({k:out[k] for k in ('status','cases','component_cells')}))
if __name__=='__main__':main()
