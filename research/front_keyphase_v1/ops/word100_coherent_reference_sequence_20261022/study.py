import os,sys,json,hashlib,shutil,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import candidate as C
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
P=O/'word100_motion_phase_alignment_20261022';D=O/'word100_short_omission_cause_20261022';I=O/'word100_cross_person_motion_support_20261022'/'inputs'
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p

def freeze():
    assert not (U/'FROZEN.json').exists() and not (D/'RUNNING.json').exists()
    f=read(D/'FROZEN.json');assert read(D/'SYNC_CHECK.json')['status']=='PASS';check(f['pins']);check(f['protected'])
    paths=[U/'candidate.py',U/'study.py',D/'ANALYSIS.json',D/'FINAL_AUDIT.json',D/'VERIFY.json',P/'RESULT.json',P/'ANALYSIS.json',P/'VERIFY.json',O/'word100_image_arm_connectivity_20261018'/'CROSSING.json']
    paths+=list(I.glob('*.npz'))+list((P/'cross').glob('*.npz'))+list((P/'known').glob('*.npz'))+list((P/'missing').glob('*.npz'))
    (U/'before').mkdir()
    for p in (R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json'):shutil.copyfile(p,U/'before'/p.name)
    write(U/'FROZEN.json',dict(at=now(),rows=f['rows'],roles=f['roles'],folds=f['folds'],protected=f['protected'],pins={str(p):sha(p) for p in paths},
      authorization='User20261022 coherent multiple-reference before/middle/after comparison; public normal speed/rest/crossing and unknown preservation; maintain C/T3F; no users/training before review.',
      scope='Same publicTRAIN3words9people27clips,297known+2376cross+135missing saved oldpaths; no fresh independent validation set or all100claim.',
      candidate='At every reference center j, fixed5frame neighborhood j-2..j+2; samples j-2,j,j+2. Each source mapped at median oldDTW index; only adjacent observed endpoint interpolation; no frame extrema pooling. Descriptor three levels plus two chronological increments interval arithmetic, max intervalgap across components separately per21channel, no angle/position unit mixing. LEVEL ablation same3levels/max/support/calibration. Candidate neither newDTW nor classifier.',
      evidence='Reference existing core/required/not-confirmed-rest domain unchanged. Local context must stay inside domain; all reference and mapped source span channel points observed, no positive source-rest in span. Missing/partial/transition boundary=>sidecarunknown, never shrinks expected or original pointwise sidecar. Before/mid/after are fixed temporal samples not certified semantic stages. Each angle independent to avoid neighbor loss.',
      calibration='For each excluded query q and fixed anchor r, every other person s contributes a whole coherent descriptor to r, not pooled coordinates. Cross7 people(q,r excluded), self8(q=r excluded). Perchannel bound=max complete peer distances, >=5people; levels/sequence bounds separate. Keep all8anchors scores and 1/5reference summaries; no bestreference-only choice. Query and synthetic variants never calibrate. EPS1e-6 inherited numerical tolerance only, no threshold search. Bounds descriptive not error probability.',
      evaluation='Compare each variant to own identity in same condition. Main incrementalnewbad requires both ownidentity and variant comparable; rawnew including priorunknown separate only for prior reconciliation. Cross-condition benefit/regression requires common-all4. Fixed local deleted region inherited prior oldidentity. Report newlybad and lostcomparability separately, original whole data insufficient staysunknown. Known-time and missing native timelines retained. Normal original+4speed deltas vsidentity. Existing reviewed crossing WORD0185_REAL03 frame_id101..111,113: preserve angularunknown (0/360 supported), palmunknown(0/24), arm48/48; not semantic correct-shape truth. Keep pertriplet sourceindices/spans and collapsedphase diagnostics without automatic error. No automatic whole-action pass.',
      limitations=['Window scale is inputframe not semanticstage, samples can miss betweenphase excursions.','Max peer bound may be too permissive; no outcome tuning.','Median oldpath sampling cannot prove correct crosspersonphase.','Strict fullspan evidence can reduce coverage; report notimprovement.','Observed interval difference not linguisticerror.','Reference counts self8/cross7 differ, results separate.'],
      forbidden=['training','classifier inference','new detector','new DTW','user/mentor access','DEVheldout','camera','download','external','Git','C/T3F/runtime/web edit']))
    print('FROZEN',flush=True)

def damaged(e,side,mode,ids):
    z={k:v.copy() for k,v in e.items()};h=('L','R').index(side)
    if mode=='wholearm':z['arm_valid'][ids,h]=False;z['arm_xy'][ids,h]=np.nan
    else:
        z['angle_valid'][ids,h]=False;z['angle_low'][ids,h]=np.nan;z['angle_high'][ids,h]=np.nan
        z['palm_valid'][ids,h]=False;z['centre'][ids,h]=np.nan;z['radius'][ids,h]=np.nan
        z['source_rest'][ids,h]=False;z['observed'][ids,h]=False;z['missing'][ids,h]=True;z['unconfirmed'][ids,h]=True
    return z

def rest_resample(e,ix):return e['source_rest'][np.floor(ix).astype(int)]&e['source_rest'][np.ceil(ix).astype(int)]

def native(c,e,req):
    d=C.expected(e,req)
    return {'native_expected':d,'native_unknown':d&~c['valid']}

def counts(out):
    result={}
    for g,sl in C.GROUPS.items():
        result[g]={k:int(out[k][:,:,sl].sum()) for k in ('expected','comparable','unknown','levels_bad','sequence_bad','context_unavailable','anchor_incomplete','query_incomplete','peer_insufficient')}
    return result

def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);assert not (U/'RUNNING.json').exists()
    write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()))
    for d in ('calibration','known','cross','missing','progress'):(U/d).mkdir()
    rows={r['clip_id']:r for r in f['rows']};es={cid:load(I/(cid+'.npz')) for cid in rows};cs={cid:C.channels(e) for cid,e in es.items()}
    reqs={cid:f['roles'].get(r['word_code'],{}).get('required_sides',['L','R']) for cid,r in rows.items()}
    dom={cid:C.expected(e,reqs[cid]) for cid,e in es.items()};descriptors={};anchors={}
    for rid,e in es.items():
        ids=np.arange(len(e['core']));identity=np.column_stack([ids,ids]);anchors[rid]=C.descriptor(C.mapped(cs[rid],identity,len(ids),e['source_rest']),dom[rid])
        peers=[cid for cid in rows if cid!=rid and rows[cid]['word_code']==rows[rid]['word_code']]
        for cid in peers:
            p=load(P/'cross'/('identity__'+cid+'__'+rid+'.npz'))['base_path']
            descriptors[cid,rid]=C.descriptor(C.mapped(cs[cid],p,len(ids),es[cid]['source_rest']),dom[rid])
    results={k:[] for k in ('known','cross','missing')};prior=read(P/'RESULT.json')
    for cid in rows:
        refs=[cid]+[rid for rid in rows if rid!=cid and rows[rid]['word_code']==rows[cid]['word_code']]
        for rid in refs:
            peers=[s for s in rows if s not in (cid,rid) and rows[s]['word_code']==rows[cid]['word_code']]
            assert len(peers)==(8 if rid==cid else 7)
            cal=C.calibrate(anchors[rid],[descriptors[s,rid] for s in peers]);calname=cid+'__'+rid
            np.savez_compressed(U/'calibration'/(calname+'.npz'),**cal,peer_ids=np.array(peers),query=np.array(cid),anchor=np.array(rid))
            folder='known' if rid==cid else 'cross'
            cases=[x for x in prior[folder] if x['query']==cid and x['reference']==rid]
            if rid!=cid:cases +=[x for x in prior['missing'] if x['query']==cid and x['reference']==rid]
            for meta in cases:
                where='missing' if 'side'in meta else folder;z=load(P/where/(meta['name']+'.npz'));e=es[cid]
                if where=='missing':
                    qe=damaged(e,meta['side'],meta['kind'],np.array(meta['removed'],int));cc=C.channels(qe);rest=qe['source_rest'];ix=np.arange(len(rest));nt=native(cc,qe,reqs[cid])
                else:
                    ix=z['source_index'];cc=C.resample(cs[cid],ix);rest=rest_resample(e,ix)
                    nd=z['query_expected'][:,:,None];nt={'native_expected':np.broadcast_to(nd,cc['valid'].shape).copy(),'native_unknown':np.broadcast_to(nd,cc['valid'].shape)&~cc['valid']}
                qd=C.descriptor(C.mapped(cc,z['base_path'],len(dom[rid]),rest),dom[rid]);out=C.compare(qd,anchors[rid],cal,dom[rid])
                assert np.array_equal(out['expected'],z['expected'])
                # New sidecar never silently converts inherited absent evidence into observation.
                prior_native=np.concatenate([z['query_angle_unknown'],np.repeat(z['query_palm_unknown'][:,:,None],2,2),np.repeat(z['query_arm_unknown'],2,2)],2)
                assert np.array_equal(nt['native_unknown'],prior_native)
                arrays=out|nt|{'path':z['base_path'],'source_index':ix,'query_phase_index':qd['phase_index'],'query_span':qd['span'],'query_collapsed':qd['collapsed'],
                  'point_comparable':z['old_comparable'],'point_bad':z['old_mismatch']}
                np.savez_compressed(U/where/(meta['name']+'.npz'),**arrays)
                results[where].append({k:meta[k] for k in ('name','query','reference','kind','label','side','removed') if k in meta}|{'groups':counts(out),'whole_action_correct':None})
        write(U/'progress'/(cid+'.json'),{'at':now(),'completed':{k:len(x) for k,x in results.items()}})
        print('DONE',cid,{k:len(x) for k,x in results.items()},flush=True)
    assert [len(results[k]) for k in ('known','cross','missing')]==[297,2376,135]
    check(f['pins']);check(f['protected']);write(U/'RESULT.json',{'at':now(),**results,'training':0,'user_data_access':0,'new_dtw':0});(U/'RUNNING.json').unlink()
    print('COMPLETE',flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if not (U/'FIRST_ERROR.txt').exists():(U/'FIRST_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
