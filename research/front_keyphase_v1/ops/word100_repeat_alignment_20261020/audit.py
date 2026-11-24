from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter,defaultdict
import sys,json,gzip,hashlib,time,copy,traceback
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent;B=OPS.parent;ROOT=U.parents[3]
P=OPS/'word100_reference_extent_20261020';O=OPS/'word100_multiref_observation_compare_20261019';M=OPS/'word100_missing_phase_validation_20261020';E=OPS/'word100_metric_evidence_20261020'
sys.dont_write_bytecode=True;sys.path.insert(0,str(B/'src'))
from wis_lab.core.observation import Observation
import alignment as A
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.now(timezone(timedelta(hours=9))).isoformat()
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x
def write(n,x):
    p=U/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(serial(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def gzwrite(n,x):
    p=U/n;p.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(p,'wt',encoding='utf-8') as f:json.dump(serial(x),f,ensure_ascii=False,allow_nan=False,separators=(',',':'))

def selftest():
    # Synthetic shape is asymmetric; one full cycle has 2 distinct submovements.
    ts=np.arange(60)*33333;angle=np.linspace(0,2*np.pi,60)
    x=np.stack([np.sin(angle),np.cos(angle)*.6],1);xy=np.repeat(x[:,None,:],4,1);v=np.ones((60,4),bool)
    q=dict(ts=ts,xy=xy,valid=v);core=np.zeros(60,bool);core[8:52]=True
    ref=dict(clip_id='synthetic',signer_id=1,pose=q,core=core,rest={s:~core for s in A.SIDES},expected={s:{a:core.copy() for a in A.AXES} for s in A.SIDES})
    refs=[dict(ref,signer_id=i) for i in range(1,10)]
    tests=[]
    for k in (1,2,3):
        r=repeat(q,k);z=A.project(r,refs,['L','R']);assert z['chosen_cycles']==k
        assert np.array_equal(z['multi']['core'],np.tile(core,k));tests.append('correct synthetic cycles '+str(k))
    z=A.project(q,refs,['R']);assert not z['multi']['expected']['L']['location'].any();tests.append('optional hand')
    blank=dict(q,valid=np.zeros_like(v));z=A.project(blank,refs,['L','R']);assert z['multi']['unknown'].all();tests.append('no arm evidence unknown')
    z=A.project(q,refs[:4],['L','R']);assert z['multi']['unknown'].all();tests.append('fewer5 people unknown')
    try:A.project(q,refs+[refs[0]],['L','R']);raise RuntimeError('duplicate accepted')
    except AssertionError:tests.append('duplicate person rejected')
    # Independently written unbanded small DP, with production band=wide for this size.
    c=np.zeros((3,3));path,total=A.dp(c);assert path==[(0,0),(1,1),(2,2)] and total==0;tests.append('DTW tie diagonal')
    write('SELFTEST.json',dict(status='PASS',checks=tests));print('SELFTEST',len(tests),flush=True)

def freeze():
    assert not (U/'FROZEN.json').exists();prev=read(P/'FROZEN.json')
    for p,h in prev['protected'].items():assert sha(p)==h
    (U/'before').mkdir()
    for p in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json'):(U/'before'/p.name).write_bytes(p.read_bytes())
    write('FROZEN.json',dict(at=now(),authorization='User approved separate one/repeated-performance alignment and public normal lowering/repetition/short-phase checks; retain C/T3-F, no private data or training.',
        scope='Isolated offline algorithm candidate and evaluation. No production/web/model change or classifier inference, new MediaPipe extraction, downloads, external transmission, Git, private data or heldout14-18.',
        input='Same public100 words, TRAIN899 references IDs01-09, DEV300 queries IDs11-13, legacy/sensor. Full stored pose frames; no hand21 alignment or active-span trim. DEV annotations/full-view states evaluation only.',
        algorithm=['Use all existing stored frames instead of96sampling. Same body-normalized4elbow/pose-wrist points, missing cost1.5, band.2, support.35,8distinct supported core points,5people minimum.',
            'Compare1/2/3 whole-reference traversals, all people retained in median query-balanced mean alignment cost. Each cycle must have8distinct core support frames. Choose lowest common cost, lower count on exact tie. No per-word settings or repetition-correctness decision.',
            '5people core and expected votes are strong evidence; any supported reference core/expected vote is possible evidence. Mixed boundaries stay possible. At least5rest votes suppress expected observation. Uncertain possible missing intervals never become a confirmed error; original recording remains intact.',
            'Votes count people, not repeated copies. Unknown roles remain per-hand diagnostics. Optional hand only existing3reviewed singlehand words. No pose restoration or synthetic hidden observations.'],
        controls=['600original query-view rows; paired single-cycle fullframe vs1/2/3-choice isolates repetition selection. Previous96single results reused, sampling change reported separately.',
            '600two-cycle and600three-cycle whole-recording concatenations, unknown cycle boundaries to algorithm. No downsampling repeated recordings, same per-frame information. Evaluate both modes against tiled annotations after inference.',
            'Previous591hands*4maskconditions*3metrics=7092 hand-evidence masks. Same pose-based projection must remain fixed. Track strong/possible extra missing, false certainty, earlier8escaped conditions.',
            'Normal lowered-hand witness from original fullsource evaluation only. Known3singlehand optional loss checks. Timestamp-only uniform/nonuniform speed invariance, not real fast-camera tests.',
            '591peak10 windows: delete entire corresponding frame block and close timeline, then realign. This is synthetic temporal removal, not confirmed linguistically required step. Report whether remaining observation evidence can look complete; no semantic omission claim.',
            'Full prepared input hand21 invariance; separate implementation DP/selection/vote checks, pins, no private data.'],
        acceptance='Report temporal recall AND precision plus possible-envelope precision cost; repetition count estimates not linguistic truth. Do not adopt if repetition/short-phase/normal-lowering requirements remain unresolved. No coefficient/candidate retuning based onDEV results.',
        protected=prev['protected'],previous={str(p):sha(p) for p in [P/'FINAL_AUDIT.json',P/'FROZEN.json',P/'INDEX.json',P/'BASELINE.json',P/'SUMMARY.json',P/'DIAGNOSTICS.json',M/'ROWS.json',O/'FROZEN.json']},
        code={str(p):sha(p) for p in [Path(__file__),U/'alignment.py']}))
    state=read(OPS/'STATE.json');state.update(current_stage='WORD100_REPEAT_ALIGNMENT_VALIDATING',activity_state='RUNNING',updated_at=now(),next_single_action='한 번/반복 구간 정렬 별도 후보 공개 검증; 학습과 운영 적용 없음.')
    state[U.name]=dict(status='FROZEN',unit=str(U));(OPS/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8');print('FROZEN',flush=True)

def prepare():
    assert not (U/'PREPARED.json').exists();f=read(U/'FROZEN.json')
    for p,h in f['code'].items():assert sha(p)==h
    old=read(O/'FROZEN.json');raw={r['clip_id']:r for r in old['inputs']};index=read(P/'INDEX.json');pins={};out=[];beg=time.time()
    for i,r in enumerate(index):
        e=raw[r['clip_id']];path=e['inputs'][r['view']];assert sha(path)==e['input_hashes'][r['view']];pins[path]=sha(path)
        assert sha(r['path'])==r['sha256'];pins[r['path']]=r['sha256'];obs=Observation.load(path);pose=A.full_pose(obs)
        hidden=copy.deepcopy(obs);hidden.xy_px[:,33:]=np.nan;hidden.usable[:,33:]=False;h=A.full_pose(hidden)
        assert all(np.array_equal(pose[k],h[k],equal_nan=True) for k in pose)
        with np.load(r['path'],allow_pickle=False) as z:arr={k:z[k] for k in z.files if not k.startswith('pose_')}
        arr.update(pose);dest=U/'prepared'/r['view']/(r['clip_id']+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();np.savez_compressed(dest,**arr)
        out.append(dict(r,path=str(dest),sha256=sha(dest),frames=obs.n_frames))
        if (i+1)%600==0:print('prepared',i+1,flush=True)
    write('INDEX.json',out);write('INPUT_PINS.json',pins);write('PREPARED.json',dict(status='PASS',records=len(out),hand21_invariance=len(out),seconds=time.time()-beg,
        frame_min=min(r['frames'] for r in out),frame_max=max(r['frames'] for r in out),frame_median=float(np.median([r['frames'] for r in out]))));print('PREPARED',len(out),flush=True)

def load(r):
    assert sha(r['path'])==r['sha256']
    with np.load(r['path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}
def make_ref(r):
    d=load(r);core=d['reference_core'];return dict(clip_id=r['clip_id'],signer_id=r['signer_id'],pose={k:d[k] for k in ('xy','valid','ts')},core=core,
        rest={s:d[s+'_source_state']==3 for s in A.SIDES},expected={s:{a:core&(d[s+'_source_state']==2)&d[s+'_'+a] for a in A.AXES} for s in A.SIDES})
def repeat(q,k):
    ts=q['ts'];duration=ts[-1]-ts[0]+int(np.median(np.diff(ts)))
    return dict(ts=np.concatenate([ts-ts[0]+i*duration for i in range(k)]),xy=np.tile(q['xy'],(k,1,1)),valid=np.tile(q['valid'],(k,1)))
def loc(ts,p,truth):
    w=A.weights(ts);out={}
    for name in ('core','possible_core'):
        z=p[name];inter=float(w@(z&truth));total=float(w@truth);pred=float(w@z)
        out[name]=dict(recall=inter/total if total else None,precision=inter/pred if pred else None,intersection_s=inter,annotation_s=total,predicted_s=pred)
    out['unknown_s']=float(w@p['unknown']);return out
def flags_from(d):return {s:{a:d[s+'_'+a] for a in A.AXES} for s in A.SIDES}
def trial(cid,view,scenario,q,refs,required,flags,truth_fn,meta=None):
    z=A.project(q,refs,required);truth=truth_fn() # annotation used after project returns
    name=f'projections/{view}/{cid}_{scenario}.json.gz'
    gzwrite(name,dict(clip_id=cid,view=view,scenario=scenario,required=required,meta=meta,result=z))
    return dict(clip_id=cid,view=view,scenario=scenario,chosen_cycles=z['chosen_cycles'],scores=z['cycle_scores'],
        multi_loc=loc(q['ts'],z['multi'],truth),single_loc=loc(q['ts'],z['single'],truth),
        multi_metrics=A.describe(q['ts'],flags,z['multi'],required),single_metrics=A.describe(q['ts'],flags,z['single'],required),
        evidence_path=name,meta=meta),z

def run():
    assert not (U/'SUMMARY.json').exists();f=read(U/'FROZEN.json')
    for p,h in f['code'].items():assert sha(p)==h
    raw=read(O/'FROZEN.json');byid={r['clip_id']:r for r in raw['inputs']};oldbase={(r['clip_id'],r['view']):r for r in read(P/'BASELINE.json')}
    windows=defaultdict(list)
    for r in read(M/'ROWS.json'):windows[r['clip_id']].append(r)
    byword=defaultdict(list)
    for r in read(U/'INDEX.json'):byword[r['word'],r['view']].append(r)
    rows=[];stress=[];lower=[];deleted=[];counts=Counter();pins={};beg=time.time()
    for wi,((word,view),rr) in enumerate(sorted(byword.items())):
        refs=[make_ref(r) for r in rr if r['signer_id']<=9];required=raw['roles'].get(word,{}).get('required_sides',['L','R'])
        for rec in rr:
            if rec['signer_id']<=9:continue
            cid=rec['clip_id'];d=load(rec);q={k:d[k] for k in ('xy','valid','ts')};flags=flags_from(d);ts=q['ts']
            def truth_fn():
                mask=np.zeros(len(ts),bool)
                for a,b in byid[cid]['intervals']:mask|=(ts>=a*1e6)&(ts<=b*1e6)
                return mask
            row,z=trial(cid,view,'original',q,refs,required,flags,truth_fn);row.update(word=word,label=rec['label'],previous96=oldbase[cid,view]['localization']);rows.append(row);truth=truth_fn()
            proj=z['multi'];w=A.weights(ts)
            # Timestamp-only changes do not affect cost or paths; check time integration scales.
            for factor in (.75,1.5):
                scaled=(ts-ts[0])*factor;assert np.allclose(A.weights(scaled),w*factor);counts['uniform_speed']+=1
            warp=np.cumsum(np.r_[0,np.diff(ts)*np.where(np.arange(len(ts)-1)<len(ts)//2,.75,1.5)])
            assert len(warp)==len(ts) and np.all(np.diff(warp)>0);counts['nonuniform_timestamps_only']+=1
            if len(required)==1:
                ff=copy.deepcopy(flags);side=next(s for s in A.SIDES if s not in required)
                for a in A.AXES:ff[side][a][:]=False
                assert A.describe(ts,ff,proj,required)==row['multi_metrics'];counts['optional_loss_invariant']+=1
            for k in (2,3):
                rq=repeat(q,k);rf={s:{a:np.tile(v,k) for a,v in ff.items()} for s,ff in flags.items()}
                rrow,_=trial(cid,view,'repeat'+str(k),rq,refs,required,rf,lambda:np.tile(truth,k));rows.append(rrow)
            if view=='sensor':
                path=E/'features/sensor'/(cid+'.npz');meta=read(E/'inputs/sensor'/(cid+'.json'));assert sha(path)==meta['sha256'];pins[str(path)]=meta['sha256']
                with np.load(path,allow_pickle=False) as v:
                    for s in A.SIDES:
                        rest=v['full_'+s+'_offline_full_view_state']==3
                        for a in A.AXES:
                            missing=~flags[s][a]
                            lower.append(dict(clip_id=cid,side=s,key=a,rest_missing_s=float(w@(rest&missing)),
                                strong_overlap_s=float(w@(rest&missing&proj['expected'][s][a])),possible_overlap_s=float(w@(rest&missing&proj['possible_expected'][s][a]))))
                for wr in windows[cid]:
                    s=wr['side'];mode=wr['condition'];win=wr['window'];mask=np.ones(len(ts),bool) if mode=='whole' else (ts/1e6>=win['start_s'])&(ts/1e6<=win['end_s'])
                    ff=copy.deepcopy(flags)
                    for a in A.AXES:ff[s][a]&=~mask
                    metrics=A.describe(ts,ff,proj,required)
                    for a in A.AXES:
                        before=next(r for r in row['multi_metrics'] if r['side']==s and r['key']==a);after=next(r for r in metrics if r['side']==s and r['key']==a)
                        assert before['expected_s']==after['expected_s'] and before['possible_s']==after['possible_s']
                        assert after['missing_s']>=before['missing_s']-1e-8
                        removed=mask&flags[s][a]
                        stress.append(dict(clip_id=cid,word=word,label=rec['label'],side=s,key=a,condition=mode,removed_frames=int(removed.sum()),removed_s=float(w@removed),
                            extra_strong_s=float(w@(removed&proj['expected'][s][a])),extra_possible_s=float(w@(removed&proj['possible_expected'][s][a])),before=before,after=after,extent_unchanged=True))
                    if mode=='peak10':
                        keep=~mask;ids=np.flatnonzero(keep);dt=np.median(np.diff(ts));dq=dict(xy=q['xy'][keep],valid=q['valid'][keep],ts=np.arange(len(ids))*dt)
                        df={s:{a:v[keep] for a,v in ff.items()} for s,ff in flags.items()}
                        dr,_=trial(cid,view,'delete_'+s,dq,refs,required,df,lambda:truth[keep],dict(keep_rows=ids,removed_source_seconds=float(w@mask),side=s))
                        deleted.append(dr)
        write('STATUS.json',dict(status='RUNNING',word_views=wi+1,of=200,at=now()))
        if (wi+1)%10==0:print('wordviews',wi+1,'seconds',round(time.time()-beg,1),flush=True)
    def stats(vals):
        a=[v for v in vals if v is not None];return dict(n=len(a),median=float(np.median(a)) if a else None,q10=float(np.quantile(a,.1)) if a else None,q90=float(np.quantile(a,.9)) if a else None)
    summary=dict(at=now(),status='CALCULATED',seconds=time.time()-beg,query_rows=len(rows),deletion_rows=len(deleted),stress_rows=len(stress),counts=dict(counts),localization={},cycle_counts={},lowering={})
    for view in ('legacy','sensor'):
        for scenario in ('original','repeat2','repeat3'):
            rr=[r for r in rows if r['view']==view and r['scenario']==scenario];name=view+'_'+scenario
            summary['cycle_counts'][name]=dict(Counter(str(r['chosen_cycles']) for r in rr))
            summary['localization'][name]={mode:{area:{k:stats([r[mode+'_loc'][area][k] for r in rr]) for k in ('recall','precision')} for area in ('core','possible_core')} for mode in ('single','multi')}
    for a in A.AXES:summary['lowering'][a]={k:sum(r[k] for r in lower if r['key']==a) for k in ('rest_missing_s','strong_overlap_s','possible_overlap_s')}
    summary['deletion']={mode:dict(statuses=dict(Counter(m['status'] for r in deleted for m in r[mode+'_metrics'] if m['side']==r['meta']['side'])),cycles=dict(Counter(str(r['chosen_cycles']) for r in deleted))) for mode in ('single','multi')}
    for kind in ('protected','previous','code'):
        for p,h in f[kind].items():assert sha(p)==h,p
    for name,obj in [('ROWS.json',rows),('STRESS.json',stress),('LOWERING.json',lower),('DELETION.json',deleted),('EVALUATION_PINS.json',pins),('SUMMARY.json',summary)]:write(name,obj)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        p=U/('FIRST_ERROR_'+sys.argv[1]+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
