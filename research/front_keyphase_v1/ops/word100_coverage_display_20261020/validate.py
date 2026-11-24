from pathlib import Path
import sys,json,hashlib,time,traceback
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;B=U.parents[1];O=U.parent/'word100_multiref_observation_compare_20261019';P=U.parent/'word100_metric_evidence_20261020'
sys.dont_write_bytecode=True;sys.path.insert(0,str(B/'src'))
from wis_lab.service import word100_observation_coverage as C
from wis_lab.core import features as F
from wis_lab.core.observation import Observation
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')

def scalar_duration(ts,flags):
    t=[float(x-ts[0])/1e6 for x in ts];n=len(t)
    if n<2:return 0.,0.
    gap=sorted([t[i+1]-t[i] for i in range(n-1)]);mid=len(gap)//2
    median=gap[mid] if len(gap)%2 else (gap[mid-1]+gap[mid])/2
    covered=0.
    for i in range(n):
        a=max(0. if i==0 else (t[i-1]+t[i])/2,t[i]-1.5*median)
        b=min(t[-1] if i==n-1 else (t[i]+t[i+1])/2,t[i]+1.5*median)
        if flags[i]:covered+=max(0,b-a)
    return covered,t[-1]

def check_timeline(x):
    if x['status']=='UNAVAILABLE':return
    intervals=x['intervals'];assert abs(sum(v['end_s']-v['start_s'] for v in intervals)-x['duration_s'])<1e-7
    assert all(abs(a['end_s']-b['start_s'])<1e-7 for a,b in zip(intervals,intervals[1:]))
    assert all(v['end_s']>v['start_s'] for v in intervals)
    assert 0<=x['observed_percent']<=100
    assert abs(sum(v['duration_s'] for v in x['missing_intervals'])+x['observed_s']-x['duration_s'])<1e-7

def selftest():
    ts=np.arange(11)*100000;ok=np.ones(11,bool)
    a=C.timeline(ts,ok);assert a['observed_percent']==100 and not a['missing_intervals']
    ok[4:6]=False;a=C.timeline(ts,ok);assert abs(a['observed_percent']-80)<1e-8
    assert abs(a['longest_unobserved_s']-.2)<1e-8
    assert C.timeline(ts,np.zeros(11,bool))['observed_percent']==0
    assert C.timeline([],[])['observed_percent'] is None
    assert C.timeline([0],[True])['status']=='UNAVAILABLE'
    for rate in (.75,1.5):
        x=C.timeline(ts*rate,ok);assert abs(x['observed_percent']-a['observed_percent'])<1e-8
        assert abs(x['longest_unobserved_s']-a['longest_unobserved_s']*rate)<1e-8
    gap=np.r_[np.arange(5)*100000,np.arange(5)*100000+2000000]
    x=C.timeline(gap,np.ones(10,bool));assert any(v['state']=='timing_gap' for v in x['intervals']) and x['observed_percent']<100
    try:C.timeline([0,0],[True,True]);raise AssertionError('duplicate timestamps accepted')
    except ValueError:pass
    sub=C.timeline(ts,ok,(.2,.8));check_timeline(sub)
    assert abs(sub['observed_percent']-100*2/3)<1e-8
    write('SELFTEST.json',{'status':'PASS','checks':10})

def run():
    assert not (U/'PUBLIC_VALIDATION.json').exists()
    f=read(U/'FROZEN.json');old=read(O/'FROZEN.json');c=Counter();rows=[];examples=[];pins={};beg=time.time();maxerr=0.;scalarframes=0
    impl=[B/'src/wis_lab/service/word100_observation_coverage.py',B/'web/observation-display.js',B/'src/wis_lab/service/word100_web_analysis.py',B/'web/practice.js',B/'web/practice.html',B/'web/practice.css',Path(__file__)]
    write('IMPLEMENTATION.json',{str(p):sha(p) for p in impl})
    for i,entry in enumerate(old['inputs']):
        assert entry['signer_id'] in list(range(1,10))+[11,12,13]
        for view in ('legacy','sensor'):
            path=entry['inputs'][view];assert sha(path)==entry['input_hashes'][view];pins[path]=sha(path)
            obs=Observation.load(path);flags=C.metric_flags(obs);full=C.summarize(obs,old['roles'].get(entry['word_code'],{}).get('required_sides') if isinstance(old.get('roles'),dict) else None)
            ts=obs.timestamp_us;active=None
            try:
                lo,hi=F.active_span(obs)
                if hi>lo:active=tuple(float((ts[k]-ts[0])/1e6) for k in (lo,hi))
            except F.FeatureError:pass
            assert full['accepted'] is None and full['is_a_verdict'] is False
            if entry['signer_id']==1:
                try:origin,scale,_=F.body_frame(obs)
                except F.FeatureError:origin=None
                for j in range(obs.n_frames):
                    actual=F.frame_features(obs,j,origin,scale)[1] if origin is not None else {}
                    for side in ('L','R'):
                        for key,group in [('handshape','bend_'),('location','handpos_'),('movement','handpos_')]:
                            assert bool(flags[side][key][j])==bool(actual.get(group+side,False)),(entry['clip_id'],view,j,key,side)
                    scalarframes+=1
                c['scalar_clips']+=1
            for metric in full['metrics']:
                for side in metric['sides']:
                    x=side['whole'];check_timeline(x);check_timeline(side['analysis_span'])
                    numerator,denominator=scalar_duration(ts,flags[side['side']][metric['key']])
                    if denominator:
                        err=abs(x['observed_percent']-100*numerator/denominator);maxerr=max(maxerr,err);assert err<1e-7
                    c['metric_hand_checks']+=1
            for factor in (.75,1.5):
                t=(ts-ts[0])*factor
                transformed=C.summarize_flags(t,flags,tuple(v*factor for v in active) if active else None)
                for m,original in zip(transformed['metrics'],full['metrics']):
                    for side,oside in zip(m['sides'],original['sides']):
                        for scope in ('whole','analysis_span'):
                            x,y=side[scope],oside[scope];check_timeline(x)
                            if x['observed_percent'] is not None:
                                assert abs(x['observed_percent']-y['observed_percent'])<1e-7
                                assert abs(x['longest_unobserved_s']-y['longest_unobserved_s']*factor)<1e-7
                assert transformed['is_a_verdict'] is False and transformed['accepted'] is None;c['speed_controls']+=1
            # Repeat the same public observation sequence twice. This is an observation
            # control, not a claim that every word permits an added linguistic repetition.
            t=ts-ts[0];dt=np.median(np.diff(t));repeat_ts=np.r_[t,t+t[-1]+dt]
            repeat_flags={s:{k:np.tile(v,2) for k,v in ss.items()} for s,ss in flags.items()}
            repeated=C.summarize_flags(repeat_ts,repeat_flags,None)
            assert repeated['is_a_verdict'] is False and repeated['accepted'] is None
            for m in repeated['metrics']:
                for side in m['sides']:
                    check_timeline(side['whole']);num,den=scalar_duration(repeat_ts,repeat_flags[side['side']][m['key']]);assert abs(side['whole']['observed_percent']-100*num/den)<1e-7
            c['repeat_controls']+=1
            if view=='sensor':
                cp=P/'features/sensor'/(entry['clip_id']+'.npz');cm=read(P/'inputs/sensor'/(entry['clip_id']+'.json'));assert sha(cp)==cm['sha256']
                with np.load(cp,allow_pickle=False) as z:
                    for s in ('L','R'):
                        count=int(((z['full_'+s+'_offline_full_view_state']==3)&~z['full_'+s+'_any_hand_points']).sum())
                        c['offline_lowered_handframes']+=count;c['clipsides_with_lowered_witness']+=count>0
            rows.append({'clip_id':entry['clip_id'],'word':entry['word_code'],'label':entry['canonical_label'],'role':entry['role'],'view':view,'coverage':full})
            c['public_clips_views']+=1
            if entry['clip_id'] in ('NIA_SL_WORD1139_REAL12_F','NIA_SL_WORD0298_REAL12_F','NIA_SL_WORD0003_REAL11_F'):
                examples.append(rows[-1])
        if (i+1)%200==0:write('STATUS.json',{'status':'PUBLIC_VALIDATING','clips':i+1});print('validated',i+1,flush=True)
    assert c['public_clips_views']==2398 and c['speed_controls']==4796 and c['repeat_controls']==2398
    for p,h in f['protected'].items():assert sha(p)==h,p
    for n,h in f['previous_results'].items():assert sha(U.parent/'word100_missing_phase_validation_20261020'/n)==h
    write('INPUT_PINS.json',pins);write('COVERAGE.json',rows);write('PUBLIC_EXAMPLES.json',examples)
    write('PUBLIC_VALIDATION.json',{'status':'PASS','counts':dict(c),'scalar_eligibility_frames':scalarframes,'max_coverage_percent_error':maxerr,'protected_unchanged':True,'previous_results_unchanged':True,'error_verdicts_created':0,'score_changes':0,'new_training':0,'private_data_read':0,'seconds':time.time()-beg,'limits':'Temporal transforms are constructed controls, not certified normal lexical repetitions. No semantic correctness or recognition accuracy measured.'})
    print('PUBLIC PASS',flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        p=U/('FIRST_ERROR_'+sys.argv[1]+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
