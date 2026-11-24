"""Isolated public reference-bank comparison. No classifier, capture, or user I/O."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_k,'1')
from pathlib import Path
import sys,json,hashlib,time,traceback,copy
from datetime import datetime,timezone,timedelta
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
U=Path(__file__).resolve().parent;B=U.parents[1];ROOT=B.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(B/'src'))
from wis_lab.core.observation import Observation
from wis_lab.service import word100_analysis as A,word100_multiref as M,word100_role_dtw as R
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
VIEWS=('legacy','masked','sensor')
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf-8');t.replace(p)
def frozen():
    d=read(U/'FROZEN.json')
    for p,h in d['protected'].items():assert sha(p)==h,p
    return d
def freeze():
    assert not (U/'FROZEN.json').exists()
    previous=read(B/'ops/word100_multiref_dtw_plan_20261019/AUDIT.json')
    assert all(sha(p)==h for p,h in previous['protected'].items())
    manifestpath=B/'ops/word100_training_20261013/MANIFEST.json'
    rows=read(manifestpath);oldroot=Path(read(manifestpath.parent/'FROZEN_EXPERIMENT.json')['output'])
    roles=read(B/'ops/word100_role_dtw_20261017/HAND_ROLES.json')['words']
    inputs=[]
    for row in rows:
        cid=row['clip_id'];oldp=oldroot/'records'/(cid+'.json');old=read(oldp)
        cp=B/'ops/word100_body_boundary_20261017/data/records'/(cid+'.json');c=read(cp)
        assert row['role']==old['role']==c['role'] and row['signer_id']==c['signer_id']
        assert old['source_sha256']==c['source']['sha256']
        assert c['context_sha256']=='9cc72d7ecbcb97732fe9dcb6412e71dc7809023df5c780fa81517b85984f100d'
        paths={'legacy':old['observation'],**{v:c['views'][v]['observation'] for v in VIEWS[1:]}}
        hashes={'legacy':old['observation_sha256'],**{v:c['outputs'][paths[v]] for v in VIEWS[1:]}}
        inputs.append({**row,'inputs':paths,'input_hashes':hashes,'record_hashes':{str(oldp):sha(oldp),str(cp):sha(cp)},'calibration_method':c['calibration']['method']})
    assert len(rows)==1199 and sum(x['role']=='train_candidate' for x in rows)==899
    protocol={'at':now(),'authorization':'2026-10-19 user: 공개100개 다중참고 확장·기존DTW/C관측경로 비교 승인. 인식보조/추가학습은 결과검토후.',
      'scope':'same-word comparison only; no ranking, model inference, training, user/mentor records, heldout14-18, detector runs, production changes',
      'views':list(VIEWS),'sampling_budget':96,'band_ratio':.2,'min_refs':5,'min_frames':8,'min_support':.5,'path_evidence':.35,
      'roles':roles,'unknown_role_policy':'joint alignment retains both hands; per-hand descriptive metrics; combined semantic verdict unavailable for88 unreviewed words',
      'aggregation':'median of valid distinct-reference distances per hand; TRAIN query person excluded; no nearest-reference selection',
      'range':'TRAIN leave-one-person-out per-side min/max/median; DEV range exceedance descriptive, not sign-error rate',
      'no_tuning':'same frozen feature scales, weights, DTW and support rules for all views; train-only baseline per view',
      'time_quality':'midpoint Voronoi cell durations on real source timestamps; missing internal run between first/last usable; no gap filled',
      'local_difference':'per-pair metric residual p90/max supplementary only, no new acceptance threshold',
      'success':'report paired coverage gains/losses, measured differences on common support, source-time support and gaps, new regression. Lower mean alone never promotion.',
      'splits':{'train':list(range(1,10)),'dev':[11,12,13]},'expected_pairs_per_view':{'train':7184,'dev':2697},
      'manifest':str(manifestpath),'manifest_sha256':sha(manifestpath),'protected':previous['protected'],
      'implementation_sha256':sha(__file__),'inputs':inputs,'new_training':0}
    write(U/'FROZEN.json',protocol)
    write(U/'STATUS.json',{'status':'FROZEN_INPUT_PREPARATION_PENDING','at':now()})
    print('frozen1199 public sources',flush=True)

def cells(ts):
    ts=np.asarray(ts,dtype=np.float64)/1e6
    if len(ts)<2:return np.ones(len(ts))
    edges=np.r_[ts[0],(ts[:-1]+ts[1:])/2,ts[-1]]
    return np.diff(edges)
def missing_run(mask,weights):
    ix=np.flatnonzero(mask)
    if not len(ix):return None
    best=run=0.
    for i in range(ix[0],ix[-1]+1):
        run=0 if mask[i] else run+float(weights[i]);best=max(best,run)
    return best
def prepare_one(row):
    cid=row['clip_id'];records={}
    for v in VIEWS:
        p=row['inputs'][v];assert sha(p)==row['input_hashes'][v]
        obs=Observation.load(p)
        rec={'view':v,'clip_id':cid,'role':row['role'],'signer_id':row['signer_id'],'word_code':row['word_code'],'source_path':p,'source_sha256':row['input_hashes'][v],'full_frames':obs.n_frames}
        try:
            seq=A.F.sequence_features(obs);s=A.F.uniform_time_sample(seq,96)
            # Numeric arrays only. No object pickle or user metadata is serialized.
            target=U/'samples'/v/(cid+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,**{k:s[k] for k in ('features','present','timestamp_us','frame_id','picked_rows')})
            rec.update(status='OK',sample_path=str(target),sample_sha256=sha(target),span=list(map(int,seq['span'])),span_start_us=int(seq['timestamp_us'][0]),span_end_us=int(seq['timestamp_us'][-1]),sampled_frames=len(s['features']),shoulder_width_px=seq['shoulder_width_px'],sides={})
            weights=cells(seq['timestamp_us']);fullweights=cells(obs.timestamp_us)
            for side,sl in [('L',slice(33,54)),('R',slice(54,75))]:
                vals={}
                for group in ('bend','handpos'):
                    _,valid=A._group(seq,f'{group}_{side}')
                    vals[group]={'observed_frames':int(valid.sum()),'source_time_fraction':float(np.dot(weights,valid)/max(weights.sum(),1e-12)),'internal_missing_seconds':missing_run(valid,weights)}
                valid=obs.usable[:,sl].all(axis=1)
                vals['full_clip_complete_hand_time_fraction']=float(np.dot(fullweights,valid)/max(fullweights.sum(),1e-12))
                rec['sides'][side]=vals
        except (A.F.FeatureError,ValueError) as exc:
            rec.update(status='INSUFFICIENT',reason=str(exc))
        write(U/'inputs'/v/(cid+'.json'),rec);records[v]=rec
    return {v:records[v]['status'] for v in VIEWS}
def sample(info):
    assert info['status']=='OK'
    with np.load(info['sample_path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}

def diag(q,r,required):
    weights=None if len(required)==2 else {g:(w if g.endswith('_'+required[0]) else 0) for g,(_,w) in A.F.GROUPS.items()}
    cost,e=A.frame_cost_matrix(q,r,group_weights=weights,hand_participation=False,one_sided_mode='flat')
    path=A._optimal_path(cost,.2)
    if not path:return {'path':[]}
    qi,ri=np.asarray(path).T;delta=np.diff(np.asarray(path),axis=0)
    maximum=current=0;previous=None
    for step in delta:
        st=tuple(step)
        current=current+1 if st==previous and st in ((0,1),(1,0)) else (1 if st in ((0,1),(1,0)) else 0)
        maximum=max(maximum,current);previous=st
    out={'path':path,'max_repeated_alignment_steps':maximum,'diagonal_fraction':float(np.mean((delta==[1,1]).all(axis=1))) if len(delta) else 1.,'sides':{}}
    for side in required:
        out['sides'][side]={}
        for key,g in [('handshape','bend'),('location','handpos'),('movement','handpos')]:
            a,av=A._group(q,g+'_'+side);b,bv=A._group(r,g+'_'+side);valid=av[qi]&bv[ri]
            qm=np.zeros(len(a),bool);rm=np.zeros(len(b),bool);qm[qi[valid]]=True;rm[ri[valid]]=True
            wq,wr=cells(q['timestamp_us']),cells(r['timestamp_us'])
            value={'query_sample_time_support':float(wq@qm/max(wq.sum(),1e-12)),'reference_sample_time_support':float(wr@rm/max(wr.sum(),1e-12)),'residual_p90':None,'residual_max':None}
            if valid.any():
                aa=a[qi[valid]].copy();bb=b[ri[valid]].copy()
                if key=='handshape':d=np.abs(aa[:,:15]-bb[:,:15]).mean(axis=1)*180
                else:
                    if key=='movement':aa-=a[np.flatnonzero(av)[0]];bb-=b[np.flatnonzero(bv)[0]]
                    d=np.linalg.norm(aa-bb,axis=1)*100
                value.update(residual_p90=float(np.quantile(d,.9)),residual_max=float(d.max()))
            out['sides'][side][key]=value
    return out

_reuse=None
def reuse():
    global _reuse
    if _reuse is None:
        old=read(B/'ops/word100_multiref_20261014/TRAIN_PAIRS.json')
        _reuse={(x['query_clip'],x['reference_clip']):x['result'] for x in old}
        for d in read(B/'ops/word100_multiref_20261014/DEVELOPMENT.json'):
            for p in d['pairs']:_reuse[(d['query_clip'],p['reference_clip'])]=p['result']
        singles=set(k for k,v in read(B/'ops/word100_role_dtw_20261017/HAND_ROLES.json')['words'].items() if len(v['required_sides'])==1)
        _reuse={k:v for k,v in _reuse.items() if not any(c in k[0] for c in singles)}
        for p in read(B/'ops/word100_role_dtw_20261017/TRAIN_PAIRS.json'):_reuse[(p['query_clip'],p['reference_clip'])]=p['result']
    return _reuse
def compare_word(task):
    v,code,rows,required,role_known=task
    target=U/'words'/v/(code+'.json')
    assert not target.exists(),'Preserve completed result'
    infos={x['clip_id']:read(U/'inputs'/v/(x['clip_id']+'.json')) for x in rows}
    samples={k:sample(i) for k,i in infos.items() if i['status']=='OK'}
    tr=[x for x in rows if x['role']=='train_candidate'];dev=[x for x in rows if x['role']=='development_only']
    pairs=[];queries=[];reused=0
    for q in rows:
        cid=q['clip_id'];subset=[]
        for ref in tr:
            rid=ref['clip_id']
            if q['signer_id']==ref['signer_id']:continue
            if cid not in samples or rid not in samples:
                measured={'status':'INSUFFICIENT','reason':'FEATURE_UNAVAILABLE','metrics':[]};extra={};source='unavailable'
            else:
                stored=reuse().get((cid,rid)) if v=='legacy' else None
                measured=stored if stored is not None else R.compare_samples(samples[cid],samples[rid],required)
                source='reused_frozen' if stored is not None else 'new';reused+=int(stored is not None)
                # Path diagnostics are new; reused numeric metric results remain untouched.
                extra=diag(samples[cid],samples[rid],required)
            item={'query_clip':cid,'reference_clip':rid,'query_signer':q['signer_id'],'reference_signer':ref['signer_id'],'role':q['role'],'result':measured,'diagnostics':extra,'result_source':source}
            pairs.append(item);subset.append(item)
        agg=R.aggregate(subset,required)
        queries.append({'clip_id':cid,'signer_id':q['signer_id'],'role':q['role'],'metrics':agg})
    baseline={}
    for key in M.AXES:
        mm=[next(m for m in q['metrics'] if m['key']==key) for q in queries if q['role']=='train_candidate']
        baseline[key]=M.distribution([m['value'] for m in mm]);baseline[key]['sides']={s:M.distribution([next(x for x in m['sides'] if x['side']==s)['value'] for m in mm]) for s in ('L','R')}
    result={'word_code':code,'label':rows[0]['canonical_label'],'view':v,'role_known':role_known,'alignment_sides':list(required),'combined_semantic_verdict':None,'baseline':baseline,'references':[infos[x['clip_id']] for x in tr],'queries':queries,'pairs':pairs,'reused_metric_pairs':reused}
    write(target,result)
    return {'view':v,'word':code,'pairs':len(pairs),'reused':reused,'sha256':sha(target)}

def preflight():
    f=frozen();cid='NIA_SL_WORD0003_REAL01_F';q=sample(read(U/'inputs/legacy'/(cid+'.json')));tests=[]
    ident=R.compare_samples(q,q,('L','R'));assert ident['status']=='OK'
    assert max(m['value'] for m in ident['metrics'])<1e-5;tests.append('identical observed features zero difference')
    empty=copy.deepcopy(q);empty['features'][:]=np.nan;empty['present'][:]=False
    assert R.compare_samples(empty,q,('L','R'))['status']=='INSUFFICIENT';tests.append('missing observation not zero match')
    shifted=copy.deepcopy(q)
    for side in ('L','R'):shifted['features'][:,A.OFFSETS['handpos_'+side]]+=np.array([.1,0],dtype=np.float32)
    r=R.compare_samples(shifted,q,('L','R'));vals={m['key']:m['value'] for m in r['metrics']}
    assert abs(vals['location']-10)<1e-3 and vals['movement']<1e-3 and vals['handshape']<1e-4;tests.append('10percent shoulder wrist shift: location10 trajectory0 bend0')
    doubled=copy.deepcopy(q);doubled['timestamp_us']*=2
    assert R.compare_samples(q,q,('L','R'))==R.compare_samples(doubled,q,('L','R'));tests.append('timestamp-only change does not change fixed feature comparison')
    for code in ['WORD0949','WORD2177','WORD2551']:
        cid=f'NIA_SL_{code}_REAL01_F';a=sample(read(U/'inputs/legacy'/(cid+'.json')));b=copy.deepcopy(a)
        for i,g in enumerate(A.F.GROUP_ORDER):
            if g.endswith('_L') or g=='bimanual':b['present'][:,i]=False;b['features'][:,A.OFFSETS[g]]=np.nan
        assert R.compare_samples(a,a,('R',))==R.compare_samples(b,a,('R',));tests.append(code+' frozen unrequired hand invariance')
    # Baseline cached144 numeric transforms must reproduce exactly; metrics reused.
    cached=read(B/'ops/word100_multiref_20261014/INPUTS.json')
    for x in cached:
        a=sample(read(U/'inputs/legacy'/(x['clip_id']+'.json')));b=M.sample(x['sample_path'],x['sample_sha256'])
        assert np.array_equal(a['present'],b['present']) and np.array_equal(a['features'],b['features'],equal_nan=True)
    tests.append('144 existing legacy numeric samples exactly match frozen cache')
    write(U/'PREFLIGHT.json',{'status':'PASS','tests':tests,'frozen_sha256':sha(U/'FROZEN.json'),'at':now()})

def main():
    mode=sys.argv[1]
    if mode=='freeze':return freeze()
    f=frozen();assert sha(__file__)==f['implementation_sha256']
    if mode=='preflight':return preflight()
    lock=U/'RUNNING.json';assert not lock.exists(),'Check recorded active execution first'
    write(lock,{'pid':os.getpid(),'mode':mode,'at':now()});start=time.monotonic()
    try:
        if mode=='prepare':tasks=f['inputs'];func=prepare_one
        elif mode=='compare':
            assert read(U/'PREFLIGHT.json')['status']=='PASS'
            tasks=[]
            for code in sorted({x['word_code'] for x in f['inputs']}):
                rows=[x for x in f['inputs'] if x['word_code']==code];role=f['roles'].get(code)
                for v in VIEWS:tasks.append((v,code,rows,tuple(role['required_sides']) if role else ('L','R'),role is not None))
            func=compare_word
        else:raise ValueError(mode)
        done=[]
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(func,t) for t in tasks]
            for future in as_completed(futures):
                done.append(future.result())
                if len(done)%10==0 or len(done)==len(tasks):
                    status={'status':mode.upper()+'_RUNNING','complete':len(done),'total':len(tasks),'seconds':time.monotonic()-start,'at':now()};write(U/'STATUS.json',status);print(json.dumps(status),flush=True)
        write(U/(mode.upper()+'_RESULT.json'),{'status':'COMPLETE','items':done,'seconds':time.monotonic()-start,'frozen_sha256':sha(U/'FROZEN.json')})
        write(U/'STATUS.json',{'status':mode.upper()+'_COMPLETE_PENDING_VERIFICATION','complete':len(done),'total':len(tasks),'at':now()})
        write(U/(mode.upper()+'_EXIT.json'),{'exit':0,'at':now()})
    except BaseException:
        p=U/('FIRST_'+mode.upper()+'_ERROR.txt')
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        write(U/(mode.upper()+'_EXIT.json'),{'exit':1,'at':now()});raise
    finally:lock.unlink()
if __name__=='__main__':main()
