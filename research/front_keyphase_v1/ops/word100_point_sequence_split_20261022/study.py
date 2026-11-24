import os,sys,json,hashlib,shutil,traceback,importlib.util
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import split_support as S
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
B=O/'word100_coherent_reference_sequence_20261022'
sys.path.insert(0,str(B))
spec=importlib.util.spec_from_file_location('frozen_sequence_study',B/'study.py');BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE)
C=BASE.C;I=BASE.I
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
read=BASE.read;load=BASE.load;sha=BASE.sha;check=BASE.check
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def freeze():
    assert not (U/'FROZEN.json').exists() and not (B/'RUNNING.json').exists()
    f=read(B/'FROZEN.json');assert read(B/'VERIFY.json')['status']=='PASS';check(f['protected'])
    paths=[B/'FROZEN.json',B/'RESULT.json',B/'VERIFY.json',B/'ANALYSIS.json',B/'candidate.py',B/'study.py']
    paths+=list(I.glob('*.npz'))+list((B/'calibration').glob('*.npz'))
    for k in ('known','cross','missing'):paths+=list((B/k).glob('*.npz'))
    paths +=[U/'split_support.py',U/'study.py',U/'verify.py']
    (U/'before').mkdir()
    for p in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json'):shutil.copyfile(p,U/'before'/p.name)
    write(U/'SELFTEST.json',{'at':now(),'status':'PASS','checks':S.selftest()})
    write(U/'FROZEN.json',{'at':now(),'rows':f['rows'],'roles':f['roles'],'protected':f['protected'],
        'pins':{str(p):sha(p) for p in paths},'authorization':'User approved independent point and sequence evidence plus separation of levels and changes, public validation only; no training or user data.',
        'protocol':'Reuse frozen named-reference 3-point/2-increment descriptors, old DTW path and same strict full-span evidence. Preserve all original pointwise flags/native timeline/expected domain exactly. Five components keep separate distances and per-component max peer bounds, no max across components. Each coordinate/angle independent, same excluded q/r and minimum5 of7 or8 peers. No new thresholds or word-specific rule. No whole-action correctness or fused warning; level/change flags diagnostics only.',
        'evaluation':'2808 saved cases and243 calibrations. Primary preservation of pointwise/support/unknown/rest/crossing. Secondary component sensitivity: normal speed original-vs-variant common support; short deleted81 variants inside fixed original path region with common-original-variant evidence; all8 refs kept and1/5 summaries descriptive, not acceptance rules. Compare old combined vs level-only/change-only/split-any. No new inference/DP/RGB/extraction. Same27 developed TRAIN videos, not independent final test.',
        'limits':'Still strict continuous evidence; preserved point result is no new recovered measurement. Component separation may increase normal warnings; report any regression without threshold tuning. No semantic omission labels; no repeat-generalization claim.',
        'forbidden':['training','classifier','new DTW','new MediaPipe','user/mentor data','runtime or C/T3F edit','Git','download','external']})
    print('FROZEN',flush=True)

def prepare(f):
    rows={r['clip_id']:r for r in f['rows']};es={cid:load(I/(cid+'.npz')) for cid in rows};cs={cid:C.channels(e) for cid,e in es.items()}
    req={cid:f['roles'].get(r['word_code'],{}).get('required_sides',['L','R']) for cid,r in rows.items()}
    dom={cid:C.expected(e,req[cid]) for cid,e in es.items()};anchors={};peers={}
    for rid,e in es.items():
        ix=np.arange(len(e['core']));anchors[rid]=C.descriptor(C.mapped(cs[rid],np.column_stack((ix,ix)),len(ix),e['source_rest']),dom[rid])
        for cid in rows:
            if cid==rid or rows[cid]['word_code']!=rows[rid]['word_code']:continue
            p=load(B/'cross'/('identity__'+cid+'__'+rid+'.npz'))['path']
            peers[cid,rid]=C.descriptor(C.mapped(cs[cid],p,len(ix),es[cid]['source_rest']),dom[rid])
    return rows,es,cs,dom,anchors,peers

def query(meta,z,es,cs,dom):
    cid=meta['query'];rid=meta['reference'];e=es[cid]
    if 'side' in meta:
        qe=BASE.damaged(e,meta['side'],meta['kind'],np.array(meta['removed'],int));cc=C.channels(qe);rest=qe['source_rest']
    else:
        cc=C.resample(cs[cid],z['source_index']);rest=BASE.rest_resample(e,z['source_index'])
    return C.descriptor(C.mapped(cc,z['path'],len(dom[rid]),rest),dom[rid])

def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);assert not (U/'RUNNING.json').exists()
    write(U/'RUNNING.json',{'at':now(),'pid':os.getpid()})
    for k in ('calibration','known','cross','missing'):(U/k).mkdir()
    rows,es,cs,dom,anchors,peers=prepare(f);prior=read(B/'RESULT.json');cals={}
    for cid in rows:
        refs=[rid for rid in rows if rows[cid]['word_code']==rows[rid]['word_code']]
        for rid in refs:
            ids=[s for s in refs if s not in (cid,rid)];cal=S.calibration(anchors[rid],[peers[s,rid] for s in ids]);old=load(B/'calibration'/(cid+'__'+rid+'.npz'))
            assert np.array_equal(cal['count'],old['count']) and np.array_equal(cal['support'],old['support'])
            assert np.allclose(cal['component_bound'].max(-1),old['sequence_threshold'],equal_nan=True)
            np.savez_compressed(U/'calibration'/(cid+'__'+rid+'.npz'),**cal,peer_ids=np.array(ids));cals[cid,rid]=cal
    results={}
    for folder in ('known','cross','missing'):
        results[folder]=[]
        for meta in prior[folder]:
            cid,rid=meta['query'],meta['reference'];z=load(B/folder/(meta['name']+'.npz'))
            q=query(meta,z,es,cs,dom);out=S.evaluate(z,q,anchors[rid],cals[cid,rid])
            assert np.array_equal(out['sequence_comparable'],z['comparable'])
            assert not (out['prior_combined_bad'] & ~out['split_any_bad']).any()
            np.savez_compressed(U/folder/(meta['name']+'.npz'),**out)
            groups={}
            for g,sl in C.GROUPS.items():
                keys=('expected','point_comparable','point_unknown','point_bad','point_within','sequence_comparable','sequence_unknown','level_bad','change_bad','split_any_bad','prior_combined_bad')
                groups[g]={k:int(out[k][:,:,sl].sum()) for k in keys}
                groups[g]['states']={str(i):int((out['evidence_state'][:,:,sl]==i).sum()) for i in (1,2,3,4)}
                groups[g]['components_bad']=[int(out['component_bad'][:,:,sl,j].sum()) for j in range(5)]
            results[folder].append({k:meta[k] for k in ('name','query','reference','kind','label','side','removed') if k in meta}|{'groups':groups,'whole_action_correct':None})
        print(folder,len(results[folder]),flush=True)
    assert [len(results[k]) for k in ('known','cross','missing')]==[297,2376,135]
    check(f['pins']);check(f['protected']);write(U/'RESULT.json',{'at':now(),**results,'training':0,'user_data_access':0})
    (U/'RUNNING.json').unlink();print('COMPLETE',flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if not (U/'FIRST_ERROR.txt').exists():(U/'FIRST_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
