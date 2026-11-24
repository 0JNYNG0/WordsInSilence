from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import defaultdict,Counter
import sys,json,gzip,hashlib,time,traceback,importlib.util
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent;ROOT=U.parents[3]
B=OPS/'word100_repeat_boundary_consensus_20261020';P=OPS/'word100_repeat_alignment_20261020'
O=OPS/'word100_multiref_observation_compare_20261019';F=OPS/'word100_metric_evidence_20261020';M=OPS/'word100_missing_phase_validation_20261020'
sys.dont_write_bytecode=True;sys.path.insert(0,str(OPS.parent/'src'));sys.path.insert(0,str(P));sys.path.insert(0,str(B))
import alignment as A
import consensus as BC
import guard as G
spec=importlib.util.spec_from_file_location('saved_independent_votes',P/'verify.py');V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.now(timezone(timedelta(hours=9))).isoformat()
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x
def write(name,x):
    p=U/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(serial(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def gzwrite(name,x):
    p=U/name;p.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(p,'wt',encoding='utf-8') as h:json.dump(serial(x),h,ensure_ascii=False,allow_nan=False,separators=(',',':'))
def selftest():
    n=80;core=np.zeros(n,bool);core[10:70]=True
    x=np.zeros((n,4,2));x[10:70,:,0]=np.sin(np.linspace(.1,np.pi-.1,60))[:,None]
    r=dict(ts=np.arange(n)*33333,xy=x,valid=np.ones((n,4),bool),reference_core=core)
    parts=G.motion_parts(r,['L','R']);assert len(parts)==10
    test=['TRAIN motion portions']
    for k in (1,2,3):
        q=dict(ts=np.arange(n*k)*33333,xy=np.tile(x,(k,1,1)),valid=np.ones((n*k,4),bool))
        e=dict(cycles=k,path=np.column_stack([np.arange(n*k),np.arange(n*k)]),good=np.ones(n*k,bool),available=True)
        z=G.compression_trace(r,e,parts,n*k);assert not z['mask'].any();test.append('ordered traversal not collapsed '+str(k))
    e=dict(cycles=1,path=np.column_stack([np.zeros(n,int),np.arange(n)]),good=np.ones(n,bool),available=True)
    z=G.compression_trace(r,e,parts,n);assert z['mask'][0] and len(z['events'])==10;test.append('motion many reference samples to one query')
    e['good']=np.zeros(n,bool);assert not G.compression_trace(r,e,parts,n)['mask'].any();test.append('missing support separate from compression')
    flat=dict(r,xy=np.zeros_like(x));assert G.motion_parts(flat,['L','R'])==[];test.append('static pose not motion phase')
    repeat=dict(ts=np.arange(2*n)*33333,xy=np.tile(x,(2,1,1)),valid=np.ones((2*n,4),bool))
    entries=[dict(cycles=k,path=np.column_stack([np.arange(2*n),np.minimum(np.arange(2*n),n*k-1)]),good=np.ones(2*n,bool),available=True,score=float(k)) for k in (1,2,3)]
    witnesses=G.all_return_witnesses(repeat,r,entries,['L','R'])
    assert witnesses['2']['boundaries'][0] # computed even though score1 is cheapest
    test.append('alternative repeat despite cheapest single')
    w=[witnesses for _ in range(9)]
    assert G.joint_return(w,list(range(9)),2,2*n)['supported'];test.append('shared alternate boundary')
    blank=dict(repeat,valid=np.zeros_like(repeat['valid']))
    assert not G.all_return_witnesses(blank,r,entries,['L','R'])['2']['boundaries'][0];test.append('absence not return')
    varied=[]
    for i in range(9):varied.append({'2':dict(available=True,boundaries=[[i]])})
    assert not G.joint_return(varied,list(range(9)),2,2*n)['supported'];test.append('disagreeing boundary abstains')
    write('SELFTEST.json',dict(status='PASS',checks=test));print('SELFTEST',len(test),flush=True)
def freeze():
    assert not (U/'FROZEN.json').exists();prev=read(B/'FROZEN.json')
    for p,h in prev['protected'].items():assert sha(p)==h
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']:
        dest=U/'before'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    pins={str(B/n):sha(B/n) for n in ['FROZEN.json','FINAL_AUDIT.json','ROWS.json','STRESS.json','LOWERING.json','SUMMARY.json','EVIDENCE_PINS.json','consensus.py']}
    pins.update({str(P/n):sha(P/n) for n in ['INDEX.json','PROJECTION_PINS.json','alignment.py','verify.py']})
    pins.update({str(O/'FROZEN.json'):sha(O/'FROZEN.json'),str(M/'ROWS.json'):sha(M/'ROWS.json')})
    write('FROZEN.json',dict(at=now(),authorization='User approved public alternative repetition evidence and excessive short-motion compression guard; no model training or production change.',
      scope='Offline candidate, no user/mentor/holdout14-18 data, classifier inference, extraction, download, external transfer, Git, C/T3-F/web changes.',
      data='TRAIN899 IDs01-09 references, DEV300 IDs11-13,100words. Existing2391saved banks reused;900new sensor controls (0.75/1.5duration uniform resampling and0.5s midpoint pause) get new alignments in this unit.',
      rules=['Audit return boundaries for ALL available2/3traversals, independently of which cost wins. Same existing observed endpoint/core comparison, >=.5posepoint support and SAME5people across boundaries.',
        'If baseline count1 has any shared alternative2/3boundary evidence, abstain. Never force a replacement count; normal within-sign returns can also cause abstention.',
        'Partition TRAIN reference observed arm travel within annotated core into10equal-distance portions. Consecutive supported edges only, >=.5required elbow/posewrist points. No hand21/personal/per-word feature or semantic phase labels.',
        'A portion with >=2supported reference samples but only1distinct supported query sample is collapsed; missing support is separately unobserved. Relative normalized time ratios are descriptive only, not a fitted cutoff.',
        'Five distinct people must show collapse at the same query frame. For a known baseline count inspect that count; for unknown count require each person collapse under every viable hypothesis. Downgrade count confidence and strong local evidence; preserve possible envelope exactly.',
        'All outputs semantic_error/complete_performance/wrong_action remain null. No stage necessity, permissible repeats or user-error verdict inferred from a geometric warning.',
        'Same prior DTW/support parameters and baseline consensus, no DEV retuning. Failure or incomplete detection is reported without per-word correction.'],
      controls=['Original/repeat2/repeat3 and591deleted-frame prior banks, paired baseline/new decisions. Report correct/different/abstain all denominators and loss of previous correct confirmations.',
        '900new sensor-only controls: uniform duration0.75,1.5 at fixed sample period with linear pose interpolation ONLY when both endpoints observed; flagsAND. midpoint static pause15frames with copied observation. Synthetic tests, not real webcam speed.',
        '7092hand-evidence masks on original sensor, previous8case preservation. Normal lowered-time overlap strong and possible separately. Optional3words remain optional.',
        'Independent decision/compression/return reconstruction, controls/source hashes, original failure preservation. No prior scientific results overwritten.'],
      acceptance='Do not present abstention, a geometric compression flag, or synthetic count agreement as sign correctness. Report if unanimous miscounts remain, if normal controls abstain, and possible-range cost. No deployment.',
      protected=prev['protected'],previous=pins,code={str(p):sha(p) for p in [U/'guard.py',Path(__file__)]}))
    state=read(OPS/'STATE.json');state.update(current_stage='WORD100_REPEAT_COMPRESSION_GUARD_VALIDATING',activity_state='RUNNING',updated_at=now(),next_single_action='1회공동선호의대안반복근거와짧은동작압축 별도후보공개검증;학습없음.')
    state[U.name]=dict(status='FROZEN',unit=str(U));(OPS/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8');print('FROZEN',flush=True)
def load(rec):
    assert sha(rec['path'])==rec['sha256']
    with np.load(rec['path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}
def asref(d,person):
    core=d['reference_core']
    return dict(signer_id=person,pose={k:d[k] for k in ('ts','xy','valid')},core=core,
        rest={s:d[s+'_source_state']==3 for s in G.SIDES},
        expected={s:{a:core&(d[s+'_source_state']==2)&d[s+'_'+a] for a in G.AXES} for s in G.SIDES})
def query(d,row):
    ids=np.arange(len(d['ts']));ts=d['ts']
    if row['scenario'].startswith('repeat'):
        k=int(row['scenario'][-1]);period=ts[-1]-ts[0]+int(np.median(np.diff(ts)));ids=np.tile(ids,k)
        ts=np.concatenate([ts-ts[0]+j*period for j in range(k)])
    elif row['scenario'].startswith('delete'):
        ids=np.array(row['meta']['keep_rows']);ts=np.arange(len(ids))*np.median(np.diff(ts))
    return dict(ts=ts,xy=d['xy'][ids],valid=d['valid'][ids]),ids
def control(d,scenario):
    n=len(d['ts']);dt=float(np.median(np.diff(d['ts'])))
    if scenario=='pause':
        mid=n//2;left=np.r_[np.arange(mid),np.full(15,mid),np.arange(mid,n)];right=left;frac=np.zeros(len(left))
    else:
        factor=.75 if scenario=='speed075' else 1.5;grid=np.linspace(0,n-1,max(2,round((n-1)*factor)+1))
        left=np.floor(grid).astype(int);right=np.ceil(grid).astype(int);frac=grid-left
    valid=d['valid'][left]&d['valid'][right]
    xy=d['xy'][left]*(1-frac[:,None,None])+d['xy'][right]*frac[:,None,None];xy[~valid]=np.nan
    q=dict(ts=np.arange(len(left))*dt,xy=xy,valid=valid)
    flags={s:{a:d[s+'_'+a][left]&d[s+'_'+a][right] for a in G.AXES} for s in G.SIDES}
    return q,flags,dict(left=left,right=right,fraction=frac)
def loc(q,p,truth):
    w=A.weights(q['ts']);out={}
    for area in ('core','possible_core'):
        z=p[area];a=float(w@(z&truth));b=float(w@truth);c=float(w@z)
        out[area]=dict(recall=a/b if b else None,precision=a/c if c else None,intersection_s=a,annotation_s=b,predicted_s=c)
    return out
def evaluate():
    assert not (U/'SUMMARY.json').exists();f=read(U/'FROZEN.json')
    for typ in ('previous','protected','code'):
        for p,h in f[typ].items():assert sha(p)==h,p
    index=read(P/'INDEX.json');lookup={(r['clip_id'],r['view']):r for r in index};groups=defaultdict(list)
    for r in index:
        if r['signer_id']<=9:groups[r['word'],r['view']].append(r)
    cache={}
    def cached(rec):
        if rec['path'] not in cache:cache[rec['path']]=load(rec)
        return cache[rec['path']]
    raw=read(O/'FROZEN.json');metadata={r['clip_id']:r for r in raw['inputs']}
    sourcepins=read(P/'PROJECTION_PINS.json');basepins=read(B/'EVIDENCE_PINS.json')
    windows=defaultdict(list)
    for r in read(M/'ROWS.json'):windows[r['clip_id']].append(r)
    stressold={(r['clip_id'],r['side'],r['key'],r['condition']):r for r in read(B/'STRESS.json')}
    lowerold={(r['clip_id'],r['side'],r['key']):r for r in read(B/'LOWERING.json')}
    rows=[];stress=[];lower=[];savedpins={};evalpins={};counts=Counter();beg=time.time()
    tasks=read(B/'ROWS.json')
    originals=[r for r in tasks if r['view']=='sensor' and r['scenario']=='original']
    tasks+= [dict(r,scenario=sc) for r in originals for sc in ('speed075','speed150','pause')]
    for i,row in enumerate(tasks):
        cid,view=row['clip_id'],row['view'];rec=lookup[cid,view];d=cached(rec)
        rr=groups[rec['word'],view];people=[r['signer_id'] for r in rr];refs=[cached(r) for r in rr]
        scenario=row['scenario'];fresh=scenario in ('speed075','speed150','pause')
        required=raw['roles'].get(rec['word'],{}).get('required_sides',['L','R'])
        if fresh:
            q,flags,mapping=control(d,scenario);bank=[A.align_all(q,asref(r,p),required) for r,p in zip(refs,people)]
            pp={k:V.independent_votes(q,refs,bank,k,required) for k in (1,2,3)}
            baseline=BC.candidate(q,refs,bank,people,required,pp)
            source=f'new_banks/{view}/{cid}_{scenario}.json.gz'
            gzwrite(source,dict(bank=bank,baseline=baseline,required=required,mapping=mapping))
            savedpins[source]=sha(U/source)
        else:
            q,ids=query(d,row);flags={s:{a:d[s+'_'+a][ids] for a in G.AXES} for s in G.SIDES}
            assert sha(P/row['source_path'])==sourcepins[row['source_path']]
            assert sha(B/row['evidence_path'])==basepins[row['evidence_path']]
            with gzip.open(P/row['source_path'],'rt',encoding='utf-8') as h:source_data=json.load(h)
            bank=source_data['result']['bank']
            with gzip.open(B/row['evidence_path'],'rt',encoding='utf-8') as h:baseline=json.load(h)
            pp={k:V.independent_votes(q,refs,bank,k,required) for k in (1,2,3)}
            source=row['source_path'];mapping=None
        z=G.apply(q,refs,bank,people,required,pp,baseline,BC.combine);p=z['projection'];w=A.weights(q['ts'])
        # Evaluation truth is loaded into this computation only after candidate returns.
        truth=np.zeros(len(d['ts']),bool)
        for a,b in metadata[cid]['intervals']:truth|=(d['ts']>=a*1e6)&(d['ts']<=b*1e6)
        truth=(truth[mapping['left']]&truth[mapping['right']]) if fresh else truth[ids]
        oldp=baseline['projection'];oldp={k:np.asarray(v,bool) if k in ('core','possible_core','outside','unknown') else v for k,v in oldp.items()}
        assert np.array_equal(p['possible_core'],oldp['possible_core'])
        for side in G.SIDES:
            for axis in G.AXES:
                assert np.array_equal(p['possible_expected'][side][axis],oldp['possible_expected'][side][axis])
                assert np.all(~p['expected'][side][axis]|np.asarray(oldp['expected'][side][axis]))
        # Convert baseline nested masks to arrays for the shared metric formatter.
        for key in ('rest','expected','possible_expected'):
            for side in G.SIDES:
                if key=='rest':oldp[key][side]=np.asarray(oldp[key][side],bool)
                else:
                    for axis in G.AXES:oldp[key][side][axis]=np.asarray(oldp[key][side][axis],bool)
        oldmet=A.describe(q['ts'],flags,oldp,required);newmet=A.describe(q['ts'],flags,p,required)
        evidence=f'evidence/{view}/{cid}_{scenario}.json.gz';gzwrite(evidence,z);savedpins[evidence]=sha(U/evidence)
        out=dict(clip_id=cid,word=rec['word'],label=rec['label'],view=view,scenario=scenario,fresh_control=fresh,
            old_count=baseline['decision']['count'],new_count=z['count'],guard_reasons=z['guard_reasons'],
            alternative_counts=[int(k) for k,v in z['alternative_evidence'].items() if v['supported']],
            compression_frames=int(z['compression_mask'].sum()),compression_s=float(w@z['compression_mask']),
            old_loc=loc(q,oldp,truth),new_loc=loc(q,p,truth),old_metrics=oldmet,new_metrics=newmet,
            source_path=source,baseline_path=None if fresh else row['evidence_path'],evidence_path=evidence,meta=row['meta'])
        rows.append(out);counts['possible_envelope_unchanged']+=1
        if len(required)==1:
            optional=next(s for s in G.SIDES if s not in required)
            assert all(not p['expected'][optional][a].any() and not p['possible_expected'][optional][a].any() for a in G.AXES)
            counts['optional_hand_empty']+=1
        if view=='sensor' and scenario=='original':
            path=F/'features/sensor'/(cid+'.npz');meta=read(F/'inputs/sensor'/(cid+'.json'))
            assert sha(path)==meta['sha256'];evalpins[str(path)]=meta['sha256']
            with np.load(path,allow_pickle=False) as zf:
                for side in G.SIDES:
                    rest=zf['full_'+side+'_offline_full_view_state']==3
                    for axis in G.AXES:
                        prev=lowerold[cid,side,axis];missing=~flags[side][axis]
                        lower.append(dict(clip_id=cid,label=rec['label'],side=side,key=axis,
                            rest_missing_s=prev['rest_missing_s'],old_strong_s=prev['new_strong_s'],new_strong_s=float(w@(rest&missing&p['expected'][side][axis])),
                            old_possible_s=prev['new_possible_s'],new_possible_s=float(w@(rest&missing&p['possible_expected'][side][axis]))))
            for wr in windows[cid]:
                side,mode=wr['side'],wr['condition'];win=wr['window']
                mask=np.ones(len(w),bool) if mode=='whole' else (q['ts']/1e6>=win['start_s'])&(q['ts']/1e6<=win['end_s'])
                for axis in G.AXES:
                    removed=mask&flags[side][axis];prev=stressold[cid,side,axis,mode]
                    strong=float(w@(removed&p['expected'][side][axis]));possible=float(w@(removed&p['possible_expected'][side][axis]))
                    assert abs(possible-prev['new_possible_s'])<1e-7
                    stress.append(dict(clip_id=cid,label=rec['label'],side=side,key=axis,condition=mode,removed_frames=int(removed.sum()),
                        old_strong_s=prev['new_strong_s'],new_strong_s=strong,old_possible_s=prev['new_possible_s'],new_possible_s=possible,extent_unchanged=True))
        if (i+1)%150==0:
            write('STATUS.json',dict(status='RUNNING',rows=i+1,of=len(tasks),at=now()));print('rows',i+1,'seconds',round(time.time()-beg,1),flush=True)
    summary=dict(at=now(),rows=len(rows),stress_rows=len(stress),lowering_rows=len(lower),seconds=time.time()-beg,controls=dict(counts),groups={},lowering={},
        guards=dict(Counter(reason for r in rows for reason in r['guard_reasons'])))
    for view in ('legacy','sensor'):
        for sc in ('original','repeat2','repeat3','speed075','speed150','pause'):
            rs=[r for r in rows if r['view']==view and r['scenario']==sc]
            if not rs:continue
            target=int(sc[-1]) if sc.startswith('repeat') else 1
            summary['groups'][view+'_'+sc]=dict(n=len(rs),
                old_correct=sum(r['old_count']==target for r in rs),old_wrong=sum(r['old_count'] is not None and r['old_count']!=target for r in rs),old_abstain=sum(r['old_count'] is None for r in rs),
                new_correct=sum(r['new_count']==target for r in rs),new_wrong=sum(r['new_count'] is not None and r['new_count']!=target for r in rs),new_abstain=sum(r['new_count'] is None for r in rs),
                lost_correct=sum(r['old_count']==target and r['new_count'] is None for r in rs),
                alternative_witness=sum(bool(r['alternative_counts']) for r in rs),compressed_rows=sum(r['compression_frames']>0 for r in rs),
                compression_s=sum(r['compression_s'] for r in rs),
                localization={mode:{area:{metric:float(np.median([r[mode+'_loc'][area][metric] for r in rs if r[mode+'_loc'][area][metric] is not None])) for metric in ('recall','precision')} for area in ('core','possible_core')} for mode in ('old','new')})
    for axis in G.AXES:summary['lowering'][axis]={k:sum(r[k] for r in lower if r['key']==axis) for k in ('rest_missing_s','old_strong_s','new_strong_s','old_possible_s','new_possible_s')}
    for name,obj in [('ROWS.json',rows),('STRESS.json',stress),('LOWERING.json',lower),('SUMMARY.json',summary),('OUTPUT_PINS.json',savedpins),('EVALUATION_PINS.json',evalpins)]:write(name,obj)
    for typ in ('previous','protected','code'):
        for p,h in f[typ].items():assert sha(p)==h,p
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        p=U/('FIRST_ERROR_'+sys.argv[1]+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
