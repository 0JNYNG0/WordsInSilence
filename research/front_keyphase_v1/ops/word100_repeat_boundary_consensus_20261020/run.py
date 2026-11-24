from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
import sys, json, gzip, hashlib, time, traceback, importlib.util
import numpy as np
U=Path(__file__).resolve().parent; OPS=U.parent; ROOT=U.parents[3]
P=OPS/'word100_repeat_alignment_20261020'
O=OPS/'word100_multiref_observation_compare_20261019'
M=OPS/'word100_missing_phase_validation_20261020'
F=OPS/'word100_metric_evidence_20261020'
sys.dont_write_bytecode=True;sys.path.insert(0,str(OPS.parent/'src'));sys.path.insert(0,str(P))
import alignment as A
import consensus as C
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
    p=U/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(serial(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def gzwrite(name,x):
    p=U/name;p.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(p,'wt',encoding='utf-8') as f:json.dump(serial(x),f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
def selftest():
    done=[]
    ev=lambda i,k,b:dict(person=i,preferred=k,boundaries=b)
    one=[ev(i,1,{}) for i in range(9)]
    assert C.decide(one,40)['count']==1;done.append('unopposed once')
    two=[ev(i,2,{'1':[10,11,12]}) for i in range(9)]
    assert C.decide(two,40)['count']==2;done.append('five same people shared return')
    three=[ev(i,3,{'1':[10,11],'2':[30,31]}) for i in range(9)]
    assert C.decide(three,40)['count']==3;done.append('ordered shared boundaries')
    assert C.decide(one[:8]+[ev(9,2,{'1':[10]})],40)['count'] is None;done.append('count conflict abstains')
    assert C.decide([ev(i,2,{'1':[i]}) for i in range(9)],40)['count'] is None;done.append('separate boundary times abstain')
    assert C.decide([ev(i,3,{'1':[10] if i<5 else [],'2':[30] if i>=4 else []}) for i in range(9)],40)['count'] is None;done.append('cannot switch five voters between boundaries')
    assert C.decide(two[:4],40)['count'] is None;done.append('four people insufficient')
    assert C.decide([ev(i,None,{}) for i in range(9)],40)['count'] is None;done.append('no evidence abstains')
    try:C.decide(one+[one[0]],40);raise RuntimeError('duplicate passed')
    except AssertionError:done.append('duplicate people rejected')
    n=30;core=np.zeros(n,bool);core[5:25]=True
    xy=np.zeros((n,4,2));xy[5:25,:,0]=2
    q=dict(ts=np.arange(n),xy=xy,valid=np.ones((n,4),bool))
    r=dict(q,reference_core=core);rq=dict(ts=np.arange(2*n),xy=np.tile(xy,(2,1,1)),valid=np.ones((2*n,4),bool))
    entries=[dict(cycles=k,score=0 if k==2 else 1,available=True,path=[[i,i] for i in range(2*n)],good=[True]*(2*n)) for k in (1,2,3)]
    e=C.reference_evidence(rq,r,entries,['L','R'],1)
    assert e['boundaries']['1']==list(range(25,35));done.append('direct pose return witness')
    blind=dict(rq,valid=np.zeros((2*n,4),bool))
    assert not C.reference_evidence(blind,r,entries,['L','R'],1)['boundaries']['1'];done.append('missing pose never return witness')
    flat=dict(r,xy=np.zeros_like(xy));assert not C.reference_evidence(rq,flat,entries,['L','R'],1)['boundaries']['1'];done.append('flat equal core endpoint cannot prove boundary')
    write('SELFTEST.json',dict(status='PASS',checks=done));print('SELFTEST',len(done),flush=True)
def freeze():
    assert not (U/'FROZEN.json').exists()
    prev=read(P/'FROZEN.json')
    for p,h in prev['protected'].items():assert sha(p)==h
    for p in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json'):
        dest=U/'before'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    names=['FROZEN.json','FINAL_AUDIT.json','INDEX.json','PROJECTION_PINS.json','ROWS.json','STRESS.json','LOWERING.json','DELETION.json','SUMMARY.json','DIAGNOSTICS.json','POSTCHECK.json','alignment.py','verify.py']
    write('FROZEN.json',dict(at=now(),authorization='User approved reference-supported repetition boundaries with abstention and public lowering/short-action validation, not training or deployment.',
        scope='Isolated offline candidate. C/T3-F and production DTW/web unchanged. No private user/mentor/holdout14-18 reads, classifier inference, retraining, extraction, download, external transfer, Git.',
        data='Same public100 words TRAIN899 IDs01-09 references, DEV300 IDs11-13. Reuse2391 saved path banks. No new DTW paths or sampling. DEV annotations/full source states evaluation-only.',
        rules=['Each person prefers their unique minimum-cost available whole-reference traversal among1/2/3. Exact cost ties abstain.',
          'At least5people must agree, with no other person preferring another count. Disagreement or insufficient evidence =>count null, never force1.',
          'For k>1 require the SAME5people to support each ordered shared boundary at common query frames. Boundary frames lie strictly between supported core traversals and resemble BOTH reference pre-core and post-core poses better than closest core pose. Each endpoint match needs >=.5 actually observed arm points. Equality/absence cannot prove return.',
          'Reference core interval is TRAIN annotation; query annotations, full source states, known synthetic seams and scenario names never enter candidate.',
          'Confirmed count uses its strong votes; abstention uses intersection of all viable >=5-reference hypotheses. Possible core/metric envelope always union of all viable hypotheses. Reference disagreement is alignment uncertainty, not an incorrect action.',
          'Existing band.2/missingcost1.5/support.35/8corepoints/5people path rules frozen unchanged. No word/sign/person exception or DEV coefficient search. No subframe deletion or synthesized hidden hand.',
          'Count means complete reference traversals, not linguistically required repeats. Native repeating gestures and non-return repetitions may abstain. Optional hand uses existing3reviewed labels only; other88 roles unresolved.'],
        controls=['1800original/repeat2/repeat3 query-view rows and591prior deletion rows. Report confirmed correct/wrong/abstain separately; all300 in denominator.',
          '7092fixed hand-evidence masking conditions; possible-range cannot shrink relative previous chosen path; retain previous8escaped conditions.',
          '1800normal-lowered hand metric rows; compare strong overlap and uncertain possible overlap separately. No declaration of semantic correctness from offline estimated rest states.',
          'Timestamp-only speed invariance, optional hand absence, same-people boundary agreement, direct observed-return support. No actual fast-camera or sign-expert truth claimed.',
          'Independent implementation reconstruct decisions, endpoint witnesses, union/intersection, count/localization/shortmask/rest metrics and hashes.'],
        acceptance='Do not present abstention as correct classification. Report coverage and remaining false commitments, normal-lowering overlap, missed short windows. No deployment; preserve first failures, no DEV retuning.',
        protected=prev['protected'],previous={str(P/n):sha(P/n) for n in names}|{str(O/'FROZEN.json'):sha(O/'FROZEN.json'),str(M/'ROWS.json'):sha(M/'ROWS.json')},
        code={str(p):sha(p) for p in [U/'consensus.py',Path(__file__)]}))
    state=read(OPS/'STATE.json');state.update(current_stage='WORD100_REPEAT_BOUNDARY_CONSENSUS_VALIDATING',activity_state='RUNNING',updated_at=now(),next_single_action='여러 참고의 실제 반복경계 지지와 보류 규칙 공개 검증; 학습/운영 변경 없음.')
    state[U.name]=dict(status='FROZEN',unit=str(U));(OPS/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8');print('FROZEN',flush=True)
def load(rec):
    assert sha(rec['path'])==rec['sha256']
    with np.load(rec['path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}
def asref(d):
    core=d['reference_core']
    return dict(core=core,rest={s:d[s+'_source_state']==3 for s in C.SIDES},
                expected={s:{a:core&(d[s+'_source_state']==2)&d[s+'_'+a] for a in C.AXES} for s in C.SIDES})
def query(d,row):
    ids=np.arange(len(d['ts']));ts=d['ts']
    if row['scenario'].startswith('repeat'):
        k=int(row['scenario'][-1]);period=ts[-1]-ts[0]+int(np.median(np.diff(ts)))
        ids=np.tile(ids,k);ts=np.concatenate([ts-ts[0]+j*period for j in range(k)])
    elif row['scenario'].startswith('delete'):
        ids=np.array(row['meta']['keep_rows']);ts=np.arange(len(ids))*np.median(np.diff(ts))
    return dict(ts=ts,xy=d['xy'][ids],valid=d['valid'][ids]),ids
def loc(ts,p,truth):
    w=A.weights(ts);out={}
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
    cache={};cached=lambda r:cache.setdefault(r['path'],load(r)) if r['path'] not in cache else cache[r['path']]
    old=read(O/'FROZEN.json');raw={r['clip_id']:r for r in old['inputs']};paths=read(P/'PROJECTION_PINS.json')
    windows=defaultdict(list)
    for r in read(M/'ROWS.json'):windows[r['clip_id']].append(r)
    oldstress={(r['clip_id'],r['side'],r['key'],r['condition']):r for r in read(P/'STRESS.json')}
    oldlower={(r['clip_id'],r['side'],r['key']):r for r in read(P/'LOWERING.json')}
    rows=[];stress=[];lower=[];pins={};evidence_pins={};counts=Counter();beg=time.time()
    allrows=read(P/'ROWS.json')+read(P/'DELETION.json')
    for i,row in enumerate(allrows):
        cid,view=row['clip_id'],row['view'];rec=lookup[cid,view];d=cached(rec)
        rr=groups[rec['word'],view];people=[r['signer_id'] for r in rr];assert len(set(people))==len(people)
        refs=[cached(r) for r in rr];q,ids=query(d,row);n=len(ids)
        name=row['evidence_path'];assert sha(P/name)==paths[name]
        with gzip.open(P/name,'rt',encoding='utf-8') as h:oldz=json.load(h)
        bank=oldz['result']['bank'];required=oldz['required']
        for entries in bank:
            for e in entries:e['path']=np.asarray(e['path'],int);e['good']=np.asarray(e['good'],bool)
        projections={k:A.votes(q,[asref(r) for r in refs],bank,k,required) for k in (1,2,3)}
        z=C.candidate(q,refs,bank,people,required,projections);p=z['projection']
        # No query truth or source-state evaluation occurs until after candidate.
        truth=np.zeros(len(d['ts']),bool)
        for a,b in raw[cid]['intervals']:truth|=(d['ts']>=a*1e6)&(d['ts']<=b*1e6)
        flags={s:{a:d[s+'_'+a][ids] for a in C.AXES} for s in C.SIDES};w=A.weights(q['ts'])
        metrics=A.describe(q['ts'],flags,p,required)
        dest=name.replace('projections/','evidence/');gzwrite(dest,z);evidence_pins[dest]=sha(U/dest)
        out=dict(clip_id=cid,word=rec['word'],label=rec['label'],view=view,scenario=row['scenario'],
                 old_count=row['chosen_cycles'],decision=z['decision'],old_loc=row['multi_loc'],new_loc=loc(q['ts'],p,truth[ids]),
                 old_metrics=row['multi_metrics'],new_metrics=metrics,evidence_path=dest,source_path=name,meta=row['meta'])
        rows.append(out)
        oldp=projections[row['chosen_cycles']]
        assert np.all(~oldp['possible_core']|p['possible_core'])
        for s in C.SIDES:
            for a in C.AXES:assert np.all(~oldp['possible_expected'][s][a]|p['possible_expected'][s][a])
        counts['possible_envelope_preserved']+=1
        if row['scenario']=='original':
            for factor in (.75,1.5):
                assert np.allclose(A.weights((q['ts']-q['ts'][0])*factor),w*factor);counts['speed_time_weights']+=1
            # Query timestamps are not inputs to path evidence/decide except length.
            changed=dict(q,ts=np.cumsum(np.r_[0,np.diff(q['ts'])*np.where(np.arange(n-1)<n//2,.75,1.5)]))
            assert serial(C.candidate(changed,refs,bank,people,required,projections))==serial(z);counts['nonuniform_timestamp_invariance']+=1
            if len(required)==1:
                absent=next(s for s in C.SIDES if s not in required)
                ff={s:{a:v.copy() for a,v in dd.items()} for s,dd in flags.items()}
                for a in C.AXES:ff[absent][a][:]=False
                assert A.describe(q['ts'],ff,p,required)==metrics;counts['optional_hand_invariance']+=1
        if view=='sensor' and row['scenario']=='original':
            path=F/'features/sensor'/(cid+'.npz');meta=read(F/'inputs/sensor'/(cid+'.json'));assert sha(path)==meta['sha256'];pins[str(path)]=meta['sha256']
            with np.load(path,allow_pickle=False) as v:
                for s in C.SIDES:
                    rest=v['full_'+s+'_offline_full_view_state']==3
                    for a in C.AXES:
                        prev=oldlower[cid,s,a];miss=~flags[s][a]
                        lower.append(dict(clip_id=cid,label=rec['label'],side=s,key=a,rest_missing_s=prev['rest_missing_s'],old_strong_s=prev['strong_overlap_s'],
                            new_strong_s=float(w@(rest&miss&p['expected'][s][a])),old_possible_s=prev['possible_overlap_s'],
                            new_possible_s=float(w@(rest&miss&p['possible_expected'][s][a])),decision=z['decision']))
            for wr in windows[cid]:
                s,mode=wr['side'],wr['condition'];win=wr['window']
                mask=np.ones(n,bool) if mode=='whole' else (q['ts']/1e6>=win['start_s'])&(q['ts']/1e6<=win['end_s'])
                for a in C.AXES:
                    removed=mask&flags[s][a];prior=oldstress[cid,s,a,mode]
                    strong=float(w@(removed&p['expected'][s][a]));possible=float(w@(removed&p['possible_expected'][s][a]))
                    assert possible+1e-8>=prior['extra_possible_s']
                    stress.append(dict(clip_id=cid,label=rec['label'],side=s,key=a,condition=mode,removed_frames=int(removed.sum()),removed_s=float(w@removed),
                        old_strong_s=prior['extra_strong_s'],new_strong_s=strong,old_possible_s=prior['extra_possible_s'],new_possible_s=possible,
                        extent_unchanged=True))
        if (i+1)%150==0:
            write('STATUS.json',dict(status='RUNNING',rows=i+1,of=len(allrows),at=now(),seconds=time.time()-beg));print('rows',i+1,'seconds',round(time.time()-beg,1),flush=True)
    summary=dict(at=now(),rows=len(rows),stress_rows=len(stress),lowering_rows=len(lower),seconds=time.time()-beg,controls=dict(counts),groups={},lowering={},stress={})
    def median(v):return float(np.median([x for x in v if x is not None])) if any(x is not None for x in v) else None
    for view in ('legacy','sensor'):
        for scenario,truth_count in [('original',1),('repeat2',2),('repeat3',3)]:
            rs=[r for r in rows if r['view']==view and r['scenario']==scenario]
            summary['groups'][view+'_'+scenario]=dict(n=len(rs),old_matching=sum(r['old_count']==truth_count for r in rs),
                confirmed_matching=sum(r['decision']['count']==truth_count for r in rs),
                confirmed_different=sum(r['decision']['count'] is not None and r['decision']['count']!=truth_count for r in rs),
                abstained=sum(r['decision']['count'] is None for r in rs),
                reasons=dict(Counter(r['decision']['reason'] for r in rs)),
                localization={mode:{area:{key:median([r[mode+'_loc'][area][key] for r in rs]) for key in ('recall','precision')} for area in ('core','possible_core')} for mode in ('old','new')},
                nonempty_strong=sum(r['new_loc']['core']['predicted_s']>0 for r in rs))
    for a in C.AXES:
        summary['lowering'][a]={k:sum(r[k] for r in lower if r['key']==a) for k in ('rest_missing_s','old_strong_s','new_strong_s','old_possible_s','new_possible_s')}
    for mode in ('whole','center10','peak10','peak20'):
        rs=[r for r in stress if r['condition']==mode];eff=[r for r in rs if r['removed_frames']>0]
        summary['stress'][mode]=dict(total=len(rs),effective=len(eff),old_strong=sum(r['old_strong_s']>1e-8 for r in eff),new_strong=sum(r['new_strong_s']>1e-8 for r in eff),
            old_possible=sum(r['old_possible_s']>1e-8 for r in eff),new_possible=sum(r['new_possible_s']>1e-8 for r in eff))
    for name,obj in [('ROWS.json',rows),('STRESS.json',stress),('LOWERING.json',lower),('SUMMARY.json',summary),('EVIDENCE_PINS.json',evidence_pins),('EVALUATION_PINS.json',pins)]:write(name,obj)
    for typ in ('protected','previous','code'):
        for p,h in f[typ].items():assert sha(p)==h,p
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        p=U/('FIRST_ERROR_'+sys.argv[1]+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
