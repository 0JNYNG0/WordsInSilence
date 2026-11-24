import sys, json, hashlib, zipfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np

U=Path(__file__).resolve().parent; LAB=U.parents[1]; ROOT=LAB.parents[1]
sys.path[:0]=[str(LAB/'src'),str(ROOT/'research/sign_model_pilot_v1/src')]
from wis_lab.core.observation import Observation
from wis_lab.service import neural
from pilot.web_input import to_native_like,active_span
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def profile(obs):
    nat,pr=to_native_like(obs.xy_px,obs.usable); span=active_span(pr)
    if span is None:return {'status':'NO_HAND'}
    lo,hi=span;p=pr[lo:hi+1];l=p[:,25:46].any(1);r=p[:,46:67].any(1)
    clip,why,stats=neural._channels(obs)
    ts=obs.timestamp_us[lo:hi+1]
    return dict(status=why or 'OK',lo=int(lo),hi=int(hi),span_frames=len(p),span_seconds=float((ts[-1]-ts[0])/1e6),
      median_dt_ms=float(np.median(np.diff(ts))/1000),max_dt_ms=float(np.max(np.diff(ts))/1000),
      left=float(l.mean()),right=float(r.mean()),both=float((l&r).mean()),neither=float((~(l|r)).mean()),
      shoulders=float((p[:,5]&p[:,2]).mean()),body_scale=clip['body_scale'],
      hand_scales={k:float(v) if np.isfinite(v) else None for k,v in clip['hand_scales'].items()},
      body_valid_mean=clip['body_valid'][:,:9].mean(0).tolist(),
      left_channel_valid=float(clip['hand_valid'][:,:21].mean()),right_channel_valid=float(clip['hand_valid'][:,21:].mean()))

def main():
    assert not (U/'AUDIT.json').exists(),'Preserve first audit'
    config=read(LAB/'config/word100_practice.json');labels=dict(zip(config['word_codes'],config['labels']))
    session=LAB/'runs/ses_083c3a62d345';snapshot=U/'snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['session.json','timing_events.jsonl','trial_reviews.jsonl','decisions.jsonl']:
        (snapshot/name).write_bytes((session/name).read_bytes())
    events=[json.loads(s) for s in (snapshot/'timing_events.jsonl').read_text('utf-8').splitlines() if s.strip()]
    reviews=[json.loads(s) for s in (snapshot/'trial_reviews.jsonl').read_text('utf-8').splitlines() if s.strip()]
    rm={r['trial_id']:r for r in reviews};rows=[];guard={}
    paths=sorted((session/'trials').glob('*.result.json'),key=lambda p:p.stat().st_mtime_ns)
    for p in paths:
        result=read(p);tid=result['trial_id'];base=p.with_name(tid);npz=base.with_suffix('.npz')
        assert sha(npz)==result['observation_sha256'];assert result['practice100']['checkpoint_sha256']==config['checkpoint_sha256']
        logits=np.array(result['practice100']['logits'],float);order=np.argsort(-logits,kind='stable');idx=config['word_codes'].index(result['requested_word_id'])
        prob=np.exp(logits-logits.max());prob/=prob.sum();rank=int(np.where(order==idx)[0][0])+1
        obs=Observation.load(str(npz));prof=profile(obs)
        for f in (session/'trials').glob(tid+'.*'):guard[str(f)]=sha(f)
        dtw=list((session/'trials').glob(tid+'.training_dtw_*.json'))
        rows.append(dict(trial_id=tid,requested=result['requested_word_id'],label=labels[result['requested_word_id']],rank=rank,
          target_score_percent=float(prob[idx]*100),top1=config['labels'][order[0]],top1_score_percent=float(prob[order[0]]*100),
          top5=[config['labels'][i] for i in order[:5]],profile=prof,end_reason=result['end_reason'],self_report=rm.get(tid,{}).get('followed_guidance'),
          npz=str(npz),capture=str(base.with_suffix('.capture.zip')),dtw_sidecar=str(dtw[0]) if dtw else None))
    public=[];cache=Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013/observations')
    for item in config['items']:
        for signer in list(range(1,10))+[11,12,13]:
            clip=f"NIA_SL_{item['word_id']}_REAL{signer:02d}_F";p=cache/(clip+'.npz')
            public.append(dict(clip_id=clip,word_id=item['word_id'],label=item['label'],signer=signer,role='train' if signer<=9 else 'development',observation=str(p),sha256=sha(p),profile=profile(Observation.load(str(p)))))
    trials_from_events=sorted({e['trial_id'] for e in events if e.get('trial_id')})
    failed=[e for e in events if e.get('event')=='trial_closed']
    output=dict(at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),session=session.name,participant=read(snapshot/'session.json')['participant_id'],
      result_count=len(rows),unique_words=len({r['requested'] for r in rows}),review_count=len(rm),event_trial_ids=trials_from_events,closed_events=failed,
      session_json_trial_ids=read(snapshot/'session.json')['trial_ids'],rows=rows,public_profiles=public,
      topk={str(k):dict(correct=sum(r['rank']<=k for r in rows),total=len(rows),percent=100*sum(r['rank']<=k for r in rows)/len(rows)) for k in [1,3,5]},
      model_sha256=sha(config['checkpoint']),no_user_inference=True)
    guard[config['checkpoint']]=sha(config['checkpoint']);guard[str(LAB/'config/word100_multiref.json')]=sha(LAB/'config/word100_multiref.json')
    write(U/'ORIGINAL_GUARD.json',guard);write(U/'AUDIT.json',output)
    for r in rows:print(json.dumps({k:r[k] for k in ['label','rank','target_score_percent','top1','profile']},ensure_ascii=False),flush=True)
    print(json.dumps(output['topk']))

if __name__=='__main__':main()
