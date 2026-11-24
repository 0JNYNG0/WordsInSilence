from pathlib import Path
import sys,json,hashlib,time,traceback
from datetime import datetime,timezone,timedelta
from collections import Counter,defaultdict
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent;B=OPS.parent;ROOT=B.parents[1]
O=OPS/'word100_multiref_observation_compare_20261019';P=OPS/'word100_metric_evidence_20261020'
D=OPS/'word100_coverage_display_20261020';M=OPS/'word100_missing_phase_validation_20261020'
sys.dont_write_bytecode=True;sys.path.insert(0,str(B/'src'))
from wis_lab.core.observation import Observation
from wis_lab.service import word100_observation_coverage as C
import criterion as G
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')

def freeze():
    assert not (U/'FROZEN.json').exists()
    protected=read(D/'FROZEN.json')['protected']
    for p in [B/'src/wis_lab/service/word100_observation_coverage.py',B/'src/wis_lab/service/word100_web_analysis.py',B/'web/observation-display.js',B/'web/practice.js',B/'web/practice.html',B/'web/practice.css']:protected[str(p)]=sha(p)
    for p,h in protected.items():assert sha(p)==h,p
    (U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']:(U/'before'/p.name).write_bytes(p.read_bytes())
    previous={str(p):sha(p) for p in [D/'FINAL_AUDIT.json',D/'PUBLIC_VALIDATION.json',M/'ROWS.json',M/'ACTUAL_GAPS.json',M/'SELF_CONTROLS.json',O/'FROZEN.json']}
    write('FROZEN.json',dict(at=now(),authorization='User: validate insufficient comparison evidence using public data; distinguish normal lowering and one-hand signs; keep C/T3-F; no private training/tuning; review before new learning.',
        scope='Offline criterion validation only. No product change, inference, DTW path recomputation, scoring of original baselines, camera, downloads, training, private or holdout14-18.',
        data='Pinned100words TRAIN899 IDs01-09 and DEV300 IDs11-13, legacy and sensor views. TRAIN calibration only; DEV is previously used development data, not untouched final test.',
        rules=['G0: preserve known optional hand role; absent/fewer8frames/fewer5references => insufficient; any internal gap => partial, not full-action conclusion.',
               'G1: G0 plus global metric-specific TRAIN q05 coverage floor/q95 longest gap fraction ceiling. Pool both views and hands; no word-specific or user thresholds. Quantiles from nonoptional structurally viable TRAIN only. Freeze before DEV; no retuning.',
               'Per-hand first-last wrist observation envelope; shape uses same wrist envelope. Boundary absence is separately unassessed, never asserted normal. Wholehand absence cannot bypass gate. Unknown roles are per-hand only, not wholeword failure.',
               'Current original19-angle/body-wrist flags. Score-independent rule: low distance cannot override insufficient evidence. No sign-correctness or repetition-error label.'],
        validation=['All2398original observations. Independent gate recomputation.',
                    'DEV existing center10/peak10/peak20/whole masking windows mapped by actual timestamp. Evaluate dynamic envelope and fixed-original envelope separately. Reuse original alignment+reference masks to recompute support only, not original distances.',
                    'Reuse previous partial15-angle candidate stress distances ONLY as a separate score-fall witness; never label as new production DTW rescoring. Report different eligibility explicitly.',
                    'Timestamp factors.75/1.5; duplicated full sequence (synthetic, not lexical permission). No score or correctness invariance claim.',
                    'Add missing boundary padding, mask frozen optional hand, offline stable-lowered witness versus source-inside missing. Offline witness never available to runtime gate.',
                    'Report baseline alerts, newly caught/unflagged masks, threshold limitations. No fabricated sensitivity/specificity for real sign errors.'],
        acceptance='Validate arithmetic/role isolation/absence handling/invariance; any quantitative threshold failing short-loss detection or flagging normal-rest witnesses must not be promoted as sufficient evidence for full action. Report negative findings without changing conditions.',
        protected=protected,previous=previous,code={str(p):sha(p) for p in [Path(__file__),U/'criterion.py']}))
    state=read(OPS/'STATE.json');state.update(updated_at=now(),current_stage='WORD100_EVIDENCE_GATE_VALIDATING',activity_state='RUNNING',next_single_action='公開자료 관측부족 기준 검증; 기존 모델/웹 유지, 학습 없음.');state[U.name]=dict(status='FROZEN',unit=str(U));(OPS/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8')
    print('FROZEN',flush=True)

def selftest():
    ts=np.arange(21)*100000;ones=np.ones(21,bool);g=ones.copy();g[9:12]=False
    d=G.describe(ts,g,ones);assert abs(d['coverage']-85)<1e-7
    assert G.decide(d,9,True)['status']=='PARTIAL'
    assert G.decide(d,9,True,dict(coverage_floor=90,gap_ceiling=.2))['status']=='INSUFFICIENT'
    assert G.decide(d,0,False)['status']=='NOT_REQUIRED'
    assert G.decide(G.describe(ts,~ones,~ones),9,True)['status']=='INSUFFICIENT'
    assert G.decide(G.describe(ts,ones,ones),4,True)['status']=='INSUFFICIENT'
    assert G.decide(G.describe(ts,ones,ones),9,True)['status']=='OBSERVED_ENVELOPE'
    for factor in (.75,1.5):
        v=G.describe(ts*factor,g,ones);assert abs(v['coverage']-d['coverage'])<1e-7 and abs(v['gap_fraction']-d['gap_fraction'])<1e-7
    pad=np.r_[np.zeros(10,bool),ones,np.zeros(10,bool)];pp=G.describe(np.arange(len(pad))*100000,pad,pad);assert pp['coverage']==100 and pp['whole_coverage']<100
    assert G.decide(pp,9,True)['wrong_action'] is None
    write('SELFTEST.json',dict(status='PASS',checks=11))

def records(split):
    f=read(O/'FROZEN.json');entries=[e for e in f['inputs'] if (e['signer_id']<=9)==(split=='train')]
    wordcache={};rows=[];inputs={};maps={};raw={}
    for i,e in enumerate(entries):
        cid=e['clip_id'];known=e['word_code'] in f['roles'];roles=f['roles'].get(e['word_code'],{}).get('required_sides',[])
        assert e['signer_id'] in list(range(1,10))+[11,12,13]
        for view in ('legacy','sensor'):
            p=e['inputs'][view];assert sha(p)==e['input_hashes'][view];inputs[p]=sha(p)
            obs=Observation.load(p);flags=C.metric_flags(obs);raw[cid,view]=(obs.timestamp_us,flags)
            wp=O/'words'/view/(e['word_code']+'.json')
            if str(wp) not in wordcache:wordcache[str(wp)]=read(wp);inputs[str(wp)]=sha(wp)
            word=wordcache[str(wp)];query=next(q for q in word['queries'] if q['clip_id']==cid)
            for metric in query['metrics']:
                key=metric['key']
                for s in metric['sides']:
                    side=s['side'];required=(side in roles) if known else None
                    d=G.describe(obs.timestamp_us,flags[side][key],flags[side]['location'])
                    r=dict(clip_id=cid,word=e['word_code'],label=e['canonical_label'],signer_id=e['signer_id'],view=view,side=side,key=key,role_known=known,required=required,
                           evidence=d,valid_refs=s['valid_references'],baseline_score=s['value'],baseline_scope='Legacy19-angle saved score',split=split)
                    rows.append(r);maps[cid,view,side,key]=r
        if (i+1)%200==0:print(split,i+1,flush=True)
    return rows,maps,raw,inputs,wordcache

def calibrate():
    assert not (U/'CALIBRATION.json').exists()
    f=read(U/'FROZEN.json')
    for p,h in f['code'].items():assert sha(p)==h
    selftest();rows,_,_,inputs,_=records('train');thresholds=G.calibrate(rows)
    write('TRAIN_ROWS.json',rows);write('TRAIN_INPUT_PINS.json',inputs)
    write('CALIBRATION.json',dict(at=now(),thresholds=thresholds,clips=899,views=2,source='TRAIN899 only; not semantic-normal ground truth; global display-tail convention, not correctness cutoff',criterion_sha256=sha(U/'criterion.py'),train_rows_sha256=sha(U/'TRAIN_ROWS.json')))
    print(json.dumps(thresholds),flush=True)

def paired_refs(word,cid,side,key,window,cache,pins):
    """Keep saved alignment; recalculate existing minimum support after loss only."""
    def sample(x):
        p=x['sample_path']
        if p not in cache:
            assert sha(p)==x['sample_sha256'];pins[p]=sha(p)
            with np.load(p,allow_pickle=False) as z:cache[p]={k:z[k] for k in ('present','timestamp_us')}
        return cache[p]
    # DEV query samples live next to TRAIN samples and are pinned by previous metadata.
    view=word['view'];sp=O/'samples'/view/(cid+'.npz');mp=O/'inputs'/view/(cid+'.json')
    if str(sp) not in cache:
        meta=read(mp);assert sha(sp)==meta['sample_sha256'];pins[str(mp)]=sha(mp)
        with np.load(sp,allow_pickle=False) as z:cache[str(sp)]={k:z[k] for k in ('present','timestamp_us')}
        pins[str(sp)]=sha(sp)
    q=cache[str(sp)];group=('bend_' if key=='handshape' else 'handpos_')+side
    from wis_lab.core import features as F
    gi=list(F.GROUPS).index(group);qv=q['present'][:,gi].copy()
    if window is not None:
        if window=='whole':qv[:]=False
        else:qv&=~((q['timestamp_us']/1e6>=window[0])&(q['timestamp_us']/1e6<=window[1]))
    refs={r['clip_id']:r for r in word['references']};count=0
    for pair in word['pairs']:
        if pair['query_clip']!=cid or pair['result']['status']!='OK':continue
        r=sample(refs[pair['reference_clip']]);path=np.array(pair['diagnostics']['path'],int);qi,ri=path.T
        valid=qv[qi]&r['present'][ri,gi]
        if valid.mean()>=.5 and min(len(np.unique(qi[valid])),len(np.unique(ri[valid])))>=8:count+=1
    return count

def evaluate():
    assert not (U/'SUMMARY.json').exists();beg=time.time();f=read(U/'FROZEN.json')
    for p,h in f['code'].items():assert sha(p)==h
    cal=read(U/'CALIBRATION.json');T=cal['thresholds'];pins={};samplecache={}
    rows,maps,raw,inputs,words=records('dev');pins.update(inputs)
    counts=Counter();baseline=[];controls=[];stress=[];lowered=[]
    for r in rows:
        cid,view,side,key=r['clip_id'],r['view'],r['side'],r['key'];ts,flags=raw[cid,view]
        g0=G.decide(r['evidence'],r['valid_refs'],r['required']);g1=G.decide(r['evidence'],r['valid_refs'],r['required'],T[key])
        baseline.append(dict(r,g0=g0,g1=g1));counts['baseline_metric_hands']+=1
        for factor in (.75,1.5):
            d=G.describe((ts-ts[0])*factor,flags[side][key],flags[side]['location']);x=G.decide(d,r['valid_refs'],r['required'],T[key])
            assert x==g1;counts['speed_invariant']+=1
        dt=float(np.median(np.diff(ts)));padlen=10;n=len(ts);paddedts=np.r_[np.arange(-padlen,0)*dt,ts-ts[0],(ts[-1]-ts[0])+np.arange(1,padlen+1)*dt]
        padded={k:np.r_[np.zeros(padlen,bool),v,np.zeros(padlen,bool)] for k,v in flags[side].items()}
        dp=G.describe(paddedts,padded[key],padded['location']);xp=G.decide(dp,r['valid_refs'],r['required'],T[key]);assert xp==g1;counts['boundary_padding_invariant']+=1
        rt=np.r_[ts-ts[0],ts-ts[0]+(ts[-1]-ts[0])+dt];rd=G.describe(rt,np.tile(flags[side][key],2),np.tile(flags[side]['location'],2));rr=G.decide(rd,r['valid_refs'],r['required'],T[key])
        controls.append(dict(clip_id=cid,view=view,side=side,key=key,required=r['required'],control='repeat_sequence',before=g1['status'],after=rr['status'],before_evidence=r['evidence'],after_evidence=rd,wrong_action=None))
        if r['required'] is False:
            zz=G.decide(G.describe(ts,np.zeros(n,bool),np.zeros(n,bool)),0,False,T[key]);assert zz['status']=='NOT_REQUIRED';counts['optional_absence_ignored']+=1
        if view=='sensor':
            p=P/'features/sensor'/(cid+'.npz');meta=read(P/'inputs/sensor'/(cid+'.json'));assert sha(p)==meta['sha256'];pins[str(p)]=meta['sha256']
            with np.load(p,allow_pickle=False) as z:
                rest=(z['full_'+side+'_offline_full_view_state']==3)&~flags[side][key]
                inside=z['full_'+side+'_inside_source_but_view_missing']
                lohi=r['evidence']['bounds'];within=np.zeros(n,bool)
                if lohi:within=((ts-ts[0])/1e6>=lohi[0])&((ts-ts[0])/1e6<=lohi[1])
                lowered.append(dict(clip_id=cid,side=side,key=key,required=r['required'],rest_missing_frames=int(rest.sum()),rest_internal_frames=int((rest&within).sum()),inside_missing_frames=int(inside.sum()),g0=g0,g1=g1))
    # Original baseline supports are checked independently against saved reference paths.
    for r in rows:
        word=words[str(O/'words'/r['view']/(r['word']+'.json'))]
        refs=paired_refs(word,r['clip_id'],r['side'],r['key'],None,samplecache,pins)
        if r['required'] is not False:assert refs==r['valid_refs'],(r['clip_id'],r['side'],r['key'],refs,r['valid_refs'])
    previous=read(M/'ROWS.json')
    for j,old in enumerate(previous):
        cid,side=old['clip_id'],old['side'];ts,flags=raw[cid,'sensor'];window='whole' if old['condition']=='whole' else (old['window']['start_s'],old['window']['end_s'])
        mask=np.ones(len(ts),bool) if window=='whole' else ((ts/1e6>=window[0])&(ts/1e6<=window[1]))
        pos=flags[side]['location']&~mask
        for key in G.AXES:
            r=maps[cid,'sensor',side,key];ff=flags[side][key]&~mask;word=words[str(O/'words/sensor'/(r['word']+'.json'))]
            refs=paired_refs(word,cid,side,key,window,samplecache,pins)
            d=G.describe(ts,ff,pos);fixed=G.describe(ts,ff,pos,r['evidence']['bounds'])
            g0=G.decide(d,refs,r['required']);g1=G.decide(d,refs,r['required'],T[key]);fg=G.decide(fixed,refs,r['required'],T[key])
            b=old['baseline'][key];v=old['result'][key]
            stress.append(dict(clip_id=cid,word=r['word'],label=r['label'],side=side,key=key,condition=old['condition'],required=r['required'],removed_frames=int((flags[side][key]&mask).sum()),
                evidence=d,fixed_evidence=fixed,g0=g0,g1=g1,fixed_g1=fg,valid_refs=refs,base_g1=G.decide(r['evidence'],r['valid_refs'],r['required'],T[key]),
                witness_baseline=b,witness_masked=v,witness_score_decreased=b is not None and v is not None and v<b-1e-4,
                witness_note='Previous partial15-angle/fixed-path research scores; NOT current production19-angle DTW recalculation. Location/movement reuse prior research masks and fixed origins.'))
            if old['condition']=='whole':assert g1['status']=='INSUFFICIENT'
        if (j+1)%600==0:print('stress',j+1,flush=True)
    def totals(rr,gate):return dict(Counter(r[gate]['status'] for r in rr))
    stats={}
    for cond in ('whole','center10','peak10','peak20'):
        rr=[r for r in stress if r['condition']==cond];eff=[r for r in rr if r['removed_frames']>0];fall=[r for r in eff if r['witness_score_decreased']]
        stats[cond]=dict(n=len(rr),effective=len(eff),g0=totals(eff,'g0'),g1=totals(eff,'g1'),previous_score_fall_witnesses=len(fall),score_fall_g0=totals(fall,'g0'),score_fall_g1=totals(fall,'g1'),
            new_insufficient=sum(r['base_g1']['status']!='INSUFFICIENT' and r['g1']['status']=='INSUFFICIENT' for r in eff),
            dynamic_vs_fixed_disagree=sum(r['g1']!=r['fixed_g1'] for r in eff))
    for p,h in f['protected'].items():assert sha(p)==h,p
    for p,h in f['previous'].items():assert sha(p)==h,p
    write('BASELINE.json',baseline);write('CONTROLS.json',controls);write('STRESS.json',stress);write('LOWERING.json',lowered);write('DEV_INPUT_PINS.json',pins)
    summary=dict(status='CALCULATED',at=now(),seconds=time.time()-beg,thresholds=T,counts=dict(counts),baseline_g0=totals(baseline,'g0'),baseline_g1=totals(baseline,'g1'),
        baseline_by_view={v:totals([r for r in baseline if r['view']==v],'g1') for v in ('legacy','sensor')},stress=stats,
        repeat=dict(total=len(controls),changed=sum(r['before']!=r['after'] for r in controls),new_insufficient=sum(r['before']!='INSUFFICIENT' and r['after']=='INSUFFICIENT' for r in controls)),
        lowering=dict(metric_hands_with_rest=sum(r['rest_missing_frames']>0 for r in lowered),internal_rest=sum(r['rest_internal_frames']>0 for r in lowered),internal_rest_insufficient=sum(r['rest_internal_frames']>0 and r['g1']['status']=='INSUFFICIENT' for r in lowered)),
        role_limits='Only12words have prior role records,3 explicitly one-hand; other88 per-hand diagnostics, never a combined required-hand verdict.',
        private_data_read=0,new_training=0,model_inference=0,production_changes=0,wrong_action_verdicts=0)
    write('SUMMARY.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        p=U/('FIRST_ERROR_'+sys.argv[1]+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
