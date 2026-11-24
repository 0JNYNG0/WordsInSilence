from pathlib import Path
import sys,json,hashlib,datetime,importlib.util,traceback,os,time
import numpy as np
import candidate as N
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf8')
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_point_sequence_split_20261022';B=O/'word100_coherent_reference_sequence_20261022';I=O/'word100_cross_person_motion_support_20261022/inputs'
NORMAL={'fast075','slow15','pace','pause'}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=module('phase_channels',B/'candidate.py');S=module('phase_components',P/'split_support.py')
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def damaged(e,side,mode,ids):
    z={k:v.copy() for k,v in e.items()};h=('L','R').index(side)
    if mode=='wholearm':z['arm_valid'][ids,h]=False;z['arm_xy'][ids,h]=np.nan
    else:
        z['angle_valid'][ids,h]=False;z['angle_low'][ids,h]=np.nan;z['angle_high'][ids,h]=np.nan
        z['palm_valid'][ids,h]=False;z['centre'][ids,h]=np.nan;z['radius'][ids,h]=np.nan
        z['source_rest'][ids,h]=False;z['observed'][ids,h]=False;z['missing'][ids,h]=True;z['unconfirmed'][ids,h]=True
    return z
def fixtures():
    tests=N.selftest();n=13;c=dict(lo=np.zeros((n,2,21)),hi=np.zeros((n,2,21)),valid=np.ones((n,2,21),bool));rest=np.zeros((n,2),bool);domain=np.ones((n,2,21),bool)
    path=np.column_stack([np.arange(n),np.arange(n)]);base=C.mapped(c,path,n,rest)
    c['valid'][6,0,0]=False;c['lo'][6,0,0]=np.nan;c['hi'][6,0,0]=np.nan
    m=N.map_continuous(c,base,rest,C.resample);d=C.descriptor(m,domain)
    assert not d['valid'][4:9,0,0].any() and d['valid'][4:9,0,1].all()
    rest[6,0]=True;m=N.map_continuous(c,base,rest,C.resample);d=C.descriptor(m,domain);assert not d['valid'][4:9,0].any()
    tests.update(missing_angle_not_filled=True,neighbor_angle_preserved=True,rest_not_bridged=True)
    return tests
def freeze():
    f=read(P/'FROZEN.json');r=read(P/'RESULT.json');assert read(P/'VERIFY.json')['status']=='PASS';check(f['protected'])
    paths=[U/'candidate.py',U/'study.py',U/'verify.py',P/'FROZEN.json',P/'RESULT.json',P/'VERIFY.json',B/'candidate.py',P/'split_support.py']+list(I.glob('*.npz'))+list((P/'calibration').glob('*.npz'))
    for folder in ('known','cross','missing'):paths+=list((P/folder).glob('*.npz'))
    write(U/'SELFTEST.json',dict(at=now(),status='PASS',checks=fixtures()))
    write(U/'FROZEN.json',dict(at=now(),rows=f['rows'],roles=f['roles'],cases={k:r[k] for k in ('known','cross','missing')},protected=f['protected'],pins={str(p):sha(p) for p in paths},
      authorization='Existing autonomous goal: reduce real feedback errors using public TRAIN; no operating/model change.',
      method='One fixed binomial1/4/6/4/1 filter on median savedDTW source indices, interior only; max shift1 inputframe; monotonic endpoints preserved. Source channels unchanged. Fractional interpolation only adjacent observed endpoint values. Required/reference domain and original pointwise result retained. Source evidence span union(original match,new bracketing frames), never narrowed. No labels/known time map to candidate.',
      comparison='Baseline old continuous descriptor vs new phase, frozen old leave-query-out peer bounds unchanged to isolate mapping. Same condition own identity baseline used for new warnings. Original pointwise discrepancies retained separately. Changed sequence evidence cannot establish correct action. No new DTW search or neuraltraining.',
      evaluation='All297known+2376cross+135missing. Known normal speed target-time MAE on reference core; cross speed consistency relative to own identity, not true cross-person correspondence. Common-supported new warnings and evidence losses separately. Omission local region from old identity path,removed source indices evaluator-only. Count >=5 of8 anchors per omitted variant. Missing masks and fixed expected domain exact.',
      gate='Exploratory candidate only if known normal timing mean improves, cross normal common-supported new warnings decrease, no sequence coverage loss, no reduction in >=5-of8 short3frame omission detection, all invariants pass. No tuning/retries based on outcomes. Even a pass is not semantic/cross-person human groundtruth and cannot replace operating.',
      limitations='Three repeatedly developed words27TRAINclips9recordIDs, not independent all100 or user evaluation. Temporal smoothing can conceal short events; test and reject if so. Old peer bounds held fixed are not candidate recalibrated tolerance. Missing controls reuse saved missing-input paths; not detector recovery. Interpolated values are estimates, never new measurements.',
      training=0,user_data=0,production_changed=False))
    print('FROZEN', {k:len(r[k]) for k in ('known','cross','missing')},flush=True)
def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);assert not (U/'RUNNING.json').exists();write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()))
    (U/'cases').mkdir();rows={r['clip_id']:r for r in f['rows']};es={cid:load(I/(cid+'.npz')) for cid in rows};cs={cid:C.channels(e) for cid,e in es.items()}
    domains={cid:C.expected(e,f['roles'].get(rows[cid]['word_code'],{}).get('required_sides',['L','R'])) for cid,e in es.items()};anchors={}
    for cid,e in es.items():
        ix=np.arange(len(e['core']));anchors[cid]=C.descriptor(C.mapped(cs[cid],np.column_stack([ix,ix]),len(ix),e['source_rest']),domains[cid])
    output=[];pins={};started=time.perf_counter()
    for folder,metas in f['cases'].items():
        for count,meta in enumerate(metas):
            cid=meta['query'];rid=meta['reference'];old=load(P/folder/(meta['name']+'.npz'));ix=old['source_index'];e=es[cid]
            if folder=='missing':
                qe=damaged(e,meta['side'],meta['kind'],np.array(meta['removed'],int));ch=C.channels(qe);rest=qe['source_rest']
            else:ch=C.resample(cs[cid],ix);rest=e['source_rest'][np.floor(ix).astype(int)]&e['source_rest'][np.ceil(ix).astype(int)]
            base=C.mapped(ch,old['path'],len(domains[rid]),rest);mapped=N.map_continuous(ch,base,rest,C.resample);desc=C.descriptor(mapped,domains[rid]);cal=load(P/'calibration'/(cid+'__'+rid+'.npz'))
            ok=domains[rid]&desc['valid']&anchors[rid]['valid']&cal['support'];bad=ok[...,None]&(S.gaps(desc,anchors[rid])-cal['component_bound']>1e-6)
            assert not (ok&~old['sequence_comparable']).any(),'Cannot manufacture original missing evidence'
            name=folder+'__'+meta['name'];out=U/'cases'/(name+'.npz')
            np.savez_compressed(out,index=mapped['index'],first=mapped['first'],last=mapped['last'],sequence_comparable=ok,component_bad=bad,
              expected=old['expected'],point_comparable=old['point_comparable'],point_bad=old['point_bad'],native_expected=old['native_expected'],native_unknown=old['native_unknown'])
            pins[str(out)]=sha(out);output.append(dict(folder=folder,**{k:meta[k] for k in ('name','kind','query','reference','side','removed') if k in meta},output=str(out),old=str(P/folder/(meta['name']+'.npz'))))
            if (count+1)%216==0:print('PROGRESS',folder,count+1,len(metas),flush=True)
        print('DONE',folder,len(metas),flush=True)
    check(f['pins']);check(f['protected']);write(U/'RESULT.json',dict(at=now(),rows=output,pins=pins,frozen_sha256=sha(U/'FROZEN.json'),seconds=time.perf_counter()-started,status='COMPLETE_PENDING_INDEPENDENT_VERIFICATION',model_changed=False))
    (U/'RUNNING.json').unlink();print('COMPLETE',len(output),flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        with (U/('FAILURE_'+now().replace(':','_')+'.txt')).open('x',encoding='utf8') as f:f.write(traceback.format_exc())
        raise
