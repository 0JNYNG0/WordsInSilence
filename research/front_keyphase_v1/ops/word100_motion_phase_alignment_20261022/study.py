import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import sys,shutil,traceback,importlib.util
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];P=O/'word100_lopo_reference_range_20261022'
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
L=module('phase_lopo',P/'study.py');C=module('phase_candidate',U/'candidate.py')
read,load,write,sha,now,check=L.read,L.load,L.write,L.sha,L.now,L.check
K=L.prev.prior;A=L.prev.A;KINDS=K.KINDS

def freeze():
    f=read(P/'FROZEN.json');check(f['pins']);check(f['protected']);assert read(P/'SYNC_CHECK.json')['status']=='PASS'
    assert not (P/'RUNNING.json').exists() and not (U/'FROZEN.json').exists()
    pins=dict(f['pins'])
    for p in [U/'candidate.py',U/'study.py',P/'candidate.py',P/'study.py',P/'FINAL_AUDIT.json',P/'VERIFY.json',P/'FROZEN.json']+list((P/'templates').glob('*.npz'))+list((P/'cases').glob('*.npz'))+list((K.U/'cases').glob('*.npz')):pins[str(p)]=sha(p)
    (U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
    write(U/'FROZEN.json',dict(at=now(),rows=f['rows'],roles=f['roles'],folds=f['folds'],pins=pins,protected=f['protected'],kinds=list(KINDS),
      authorization='2026-10-22 user: palm position,arm trajectory,direction aided same-phase DTW; public verification normal speed/brief omission/missing; preserve C/T3-F and exclude user; no training before joint review.',
      scope='Same3publicTRAIN words9people27clips.297known-time self-source comparisons,2376crossperson comparisons,135newpath missing controls. No100word or unseen classifier-person claim.',
      candidate='Existing base DTW cost +.25 mean position gap/(1+gap) +.25 mean (1-cos)/2 directional mismatch. Palm centre,elbow,wrist coordinates in existing body frame. Palm radius subtracted; arm radius0. Incoming/outgoing adjacent-frame direction only when both endpoints observed and displacement>radii sum+.01 shoulderwidth. Undefined stationary/noisy/missing directions ignored as evidence, never zero-direction agreement. No speed magnitude penalty. No gap bridging. Heuristic constants fixed before outcomes,not calibrated.',
      unchanged='Base DTW weight/one-sided penalty/band.2. Source points,identity,crossing,scales/ROI. Positive rest suppresses new palm terms only; arm anchors retained. No query core/deleted/source index in cost. Existing8person range templates strictly unchanged and query excluded; no rebuilding/widening.',
      evaluation='Known-time median path error on core and originally measured changing frames; true source indices evaluator-only. Crossperson warning/coverage/common evidence and core/rest contamination are proxies,not true phase labels. Same query conditions compare against OWN identity; deletion region fixed from prior baseline identity path. Record new raw/persistent differences inside and outside region. No whole-action pass.',
      missing='135controls wholehand/short3framehand/wholearm per27query requiredhand tolowestotherperson. Remove e evidence AND legacy q feature/presence groups (hand6groups+bimanual; arm only its group). Handmotion immediately after removed frame also invalidated. Recompute oldcostpath and newcostpath. Reference expected fixed; keep nativequery missing timeline independent of chosen path. No fake fill.',
      acceptance='Report all cases and regressions; no threshold search. Numerical correctness does not imply improvement. Fewer warnings with lower evidence are not benefit. Require known-time evidence plus normal-speed/omission/missing checks before any adoption. Runtime/wholeaction unchanged.',
      limitations=['Fixed frame neighborhood direction may change under resampling or noise.','Unsupported newterms contribute0 and can attract paths; measure support loss explicitly.','Existing angle interval hull may hide omissions even with better paths.','Public reference labels give core not linguistically essential stage truth.','No newRGB detection; exact same inputs preserve normal/crossing observations not certified semantics.'],
      forbidden=['training','classifier inference','user/mentor data','DEV/heldout','camera','downloads','external transfer','Git','C/T3-F/runtime/web edit']))
    print('FROZEN 297known+2376cross+135missing',flush=True)

def damage(q,e,side,mode):
    qq={k:v.copy() for k,v in q.items()};z={k:v.copy() for k,v in e.items()};h=('L','R').index(side)
    ids=np.arange(len(e['core']))
    if mode=='shortgap':
        core=np.flatnonzero(e['core']);mid=core[len(core)//2];ids=np.arange(mid-1,mid+2)
    names=['arm_'+side] if mode=='wholearm' else [g+'_'+side for g in ('shape','orient','handpos','facerel','motion','bend')]+['bimanual']
    offset=0
    for gi,(name,(width,_)) in enumerate(A.F.GROUPS.items()):
        if name in names:
            use=np.unique(np.r_[ids,ids[ids+1<len(e['core'])]+1]) if name=='motion_'+side else ids
            qq['features'][use,offset:offset+width]=np.nan;qq['present'][use,gi]=False
        offset+=width
    if mode=='wholearm':z['arm_valid'][ids,h]=False;z['arm_xy'][ids,h]=np.nan
    else:
        z['angle_valid'][ids,h]=False;z['angle_low'][ids,h]=np.nan;z['angle_high'][ids,h]=np.nan
        z['palm_valid'][ids,h]=False;z['centre'][ids,h]=np.nan;z['radius'][ids,h]=np.nan
        z['source_rest'][ids,h]=False;z['observed'][ids,h]=False;z['missing'][ids,h]=True;z['unconfirmed'][ids,h]=True
    return qq,z,ids

def evaluate(qe,re,path,template,req):
    return L.C.compare(L.C.channels(qe),L.C.channels(re),template,path,L.expected(re,req))

def timing(e,qe,path,ix):
    pred=np.array([np.median(path[path[:,0]==i,1]) for i in range(len(ix))]);error=np.abs(pred-ix)
    lo=np.floor(ix).astype(int);hi=np.ceil(ix).astype(int)
    moving=qe['core']&(e['measured_change'][lo].any(1)|e['measured_change'][hi].any(1))
    out={}
    for name,mask in [('core',qe['core']),('moving',moving)]:
        x=error[mask];out[name]=dict(n=len(x),mae=float(x.mean()) if len(x) else None,p95=float(np.percentile(x,95)) if len(x) else None,within2=float((x<=2).mean()) if len(x) else None)
    return out,error,moving

def compare_stats(b,n):
    out={}
    for name,sl in [('angle',slice(0,15)),('palm',slice(15,17)),('elbow',slice(17,19)),('wrist',slice(19,21))]:
        common=b['comparable'][:,:,sl]&n['comparable'][:,:,sl]
        out[name]=dict(common=int(common.sum()),resolved=int((common&b['mismatch'][:,:,sl]&~n['mismatch'][:,:,sl]).sum()),
            added=int((common&~b['mismatch'][:,:,sl]&n['mismatch'][:,:,sl]).sum()),old_difference_now_unknown=int((b['mismatch'][:,:,sl]&n['unknown'][:,:,sl]).sum()),
            old_comparable_now_unknown=int((b['comparable'][:,:,sl]&n['unknown'][:,:,sl]).sum()))
    return out

def proxy(qe,re,path,req):
    wanted=re['core'][path[:,1]]
    coreoutside=wanted&~qe['core'][path[:,0]]
    target=L.expected(re,req)[path[:,1],:,0]
    rest=target&qe['source_rest'][path[:,0]]
    return dict(core_path_steps=int(wanted.sum()),query_outside_core_steps=int(coreoutside.sum()),query_rest_to_active_handsteps=int(rest.sum()))

def calculate(folder,name,q,qe,r,re,template,req,base,meta,ix=None):
    if base is None:
        weights=A.C.weights_for(req,(False,False),A.F.GROUPS)
        bc,_=A.legacy(q,r,group_weights=weights,hand_participation=False,one_sided_mode='flat');bp=np.asarray(A.dp(bc,.2),int)
        b=evaluate(qe,re,bp,template,req)
    else:bc,bp,b=base
    motion=C.costs(qe,re,req);cost=bc+motion['addition'];path=np.asarray(A.dp(cost,.2),int)
    n=evaluate(qe,re,path,template,req);native=C.query_unknown(qe,req)
    assert np.array_equal(b['expected'],n['expected'])
    arr=dict(n,path=path,base_path=bp,cost=cost,**motion,**native,
             old_comparable=b['comparable'],old_mismatch=b['mismatch'],old_persistent=b['persistent'])
    row=dict(meta,name=name,old=L.stats(b)['groups'],new=L.stats(n)['groups'],comparison=compare_stats(b,n),
        old_proxy=proxy(qe,re,bp,req),new_proxy=proxy(qe,re,path,req),path_changed=not np.array_equal(bp,path),
        old_path_length=len(bp),new_path_length=len(path),
        position_support_fraction=float(motion['position_count'][tuple(path.T)].mean()/(3*len(req))),
        direction_support_fraction=float(motion['direction_count'][tuple(path.T)].mean()/(6*len(req))),
        native_unknown_angles=int(native['query_angle_unknown'].sum()),native_unknown_palms=int(native['query_palm_unknown'].sum()),native_unknown_arms=int(native['query_arm_unknown'].sum()),whole_action_correct=None)
    if ix is not None:arr['source_index']=ix
    if folder=='known':
        row['old_timing'],arr['old_error'],arr['moving']=timing(re,qe,bp,ix)
        row['new_timing'],arr['new_error'],_=timing(re,qe,path,ix)
    np.savez_compressed(U/folder/(name+'.npz'),**arr)
    return row

def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);assert not (U/'RUNNING.json').exists()
    write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()))
    for sub in ('known','cross','missing','progress'):(U/sub).mkdir()
    samples={r['clip_id']:L.prev.data(r) for r in f['rows']};out={k:[] for k in ('known','cross','missing')}
    for row in f['rows']:
        cid=row['clip_id'];q,e=samples[cid];req=f['roles'].get(row['word_code'],{}).get('required_sides',['L','R'])
        refs=[r for r in f['rows'] if r['word_code']==row['word_code'] and r['signer_id']!=row['signer_id']]
        ch=L.C.channels(e);single=dict(low=ch['lo'],high=ch['hi'],support=ch['valid'])
        for kind in KINDS:
            qq,qe,ix=L.prev.transform(q,e,kind);saved=load(K.U/'cases'/(kind+'__'+cid+'.npz'));bp=saved['path']
            b=evaluate(qe,e,bp,single,req)
            meta=dict(query=cid,reference=cid,kind=kind,label=row['label'])
            out['known'].append(calculate('known',kind+'__'+cid,qq,qe,q,e,single,req,(saved['cost'],bp,b),meta,ix))
        for ref in refs:
            rid=ref['clip_id'];r,re=samples[rid];template=load(P/'templates'/(cid+'__'+rid+'.npz'))
            for kind in KINDS:
                qq,qe,ix=L.prev.transform(q,e,kind);name=kind+'__'+cid+'__'+rid;saved=load(P/'cases'/(name+'.npz'))
                meta=dict(query=cid,reference=rid,kind=kind,label=row['label'])
                out['cross'].append(calculate('cross',name,qq,qe,r,re,template,req,(saved['cost'],saved['path'],saved),meta,ix))
            if ref==refs[0]:
                for side in req:
                    for mode in ('wholehand','shortgap','wholearm'):
                        qq,qe,ids=damage(q,e,side,mode);name=mode+'_'+side+'__'+cid
                        meta=dict(query=cid,reference=rid,kind=mode,side=side,removed=ids.tolist(),label=row['label'])
                        out['missing'].append(calculate('missing',name,qq,qe,r,re,template,req,None,meta))
        write(U/'progress'/(cid+'.json'),dict(at=now(),completed={k:len(v) for k,v in out.items()}))
        print('DONE',cid,{k:len(v) for k,v in out.items()},flush=True)
    assert [len(out[k]) for k in ('known','cross','missing')]==[297,2376,135]
    write(U/'RESULT.json',dict(at=now(),**out,training=0,user_data=0))
    check(f['pins']);check(f['protected']);(U/'RUNNING.json').unlink()

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if not (U/'FIRST_ERROR.txt').exists():(U/'FIRST_ERROR.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
