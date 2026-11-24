"""Separate saved-logit/raw-NPZ/hash recount. No model inference or user transformations."""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;LAB=U.parents[1]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=read(U/'AUDIT.json');p=read(U/'PUBLIC_SENSITIVITY.json');config=read(LAB/'config/word100_practice.json')
guard=read(U/'ORIGINAL_GUARD.json');assert all(sha(path)==pin for path,pin in guard.items())
session=LAB/'runs'/a['session'];files=list((session/'trials').glob('*.result.json'))
assert len(files)==len(a['event_trial_ids'])==len(a['rows'])==12
assert len({r['requested'] for r in a['rows']})==12
events=[json.loads(x) for x in (U/'snapshot/timing_events.jsonl').read_text('utf-8').splitlines() if x.strip()]
assert set(x['trial_id'] for x in events if x.get('trial_id'))==set(x['trial_id'] for x in a['rows'])
counts={1:0,3:0,5:0};dtw=[]
for r in a['rows']:
    base=session/'trials'/r['trial_id'];result=read(base.with_suffix('.result.json'))
    logits=result['practice100']['logits'];target=config['word_codes'].index(result['requested_word_id'])
    ranking=sorted(range(100),key=lambda i:(-logits[i],i));rank=ranking.index(target)+1
    assert rank==r['rank'] and config['labels'][ranking[0]]==r['top1']
    for k in counts:counts[k]+=rank<=k
    with np.load(base.with_suffix('.npz'),allow_pickle=False) as z:
        pr=z['usable'];lo,hi=r['profile']['lo'],r['profile']['hi']
        for s,sl in [('left',slice(33,54)),('right',slice(54,75))]:
            rate=float(pr[lo:hi+1,sl].any(1).mean());assert abs(rate-r['profile'][s])<1e-9
    with zipfile.ZipFile(r['capture']) as z:
        m=json.loads(z.read('manifest.json'));assert m['trial_id']==r['trial_id'] and m['observation_sha256']==sha(r['npz'])
        assert len(m['frames'])==m['frame_count']
        assert all(hashlib.sha256(z.read(f['image'])).hexdigest()==f['sha256'] for f in m['frames'])
    d=read(r['dtw_sidecar']);assert d['word_id']==r['requested'] and d['query_observation_sha256']==sha(r['npz'])
    dtw.append({'label':r['label'],'status':d['status'],'sides':[{'metric':m['key'],'side':s['side'],'valid_references':s['valid_references'],'value':s['value'],'reference_max':m['baseline']['sides'][s['side']]['max']} for m in d['metrics'] for s in m['sides']]})
assert counts=={int(k):v['correct'] for k,v in a['topk'].items()}
assert len(p['rows'])==36 and len({x['clip_id'] for x in p['rows']})==36
baseline={r['clip_id']:r for r in read(LAB/'runs/word100_tcn_v1/whole_context/PREDICTIONS.json')}
byword=[]
for label in [r['label'] for r in a['rows']]:
    ps=[r for r in p['rows'] if r['label']==label];assert {r['signer'] for r in ps}=={11,12,13}
    ranks={}
    for name in ['original','remove_left','remove_right']:
        ranks[name]=[]
        for r in ps:
            target=config['word_codes'].index(r['word_id']);v=r['variants'][name];lg=v['logits']
            rank=sorted(range(100),key=lambda i:(-lg[i],i)).index(target)+1
            assert rank==v['target_rank'];ranks[name].append(rank)
            if name=='original':assert np.max(np.abs(np.array(lg)-baseline[r['clip_id']]['logits']))<1e-4
    byword.append(dict(label=label,ranks=ranks))
onehand=[r for r in byword if r['label'] in ['멀다','공항','별']]
o=dict(status='PASS',implementer='Codex',verifier='Codex separate raw recount; not a second agent or expert linguistic review',
    checked_original_hash_count=len(guard),model_unchanged=True,all_replay_frames_hash_checked=True,
    all_started_trials_accounted_for=True,raw_hand_rates_recounted=True,topk=counts,total=12,
    public_control_parity_max_abs=max(r['variants']['original']['saved_control_logit_max_abs_diff'] for r in p['rows']),
    public_cases=36,public_forward_count=108,public_summary=byword,
    confirmed_onehand_reference_words=['멀다','공항','별'],
    onehand_control_correct=sum(x==1 for r in onehand for x in r['ranks']['original']),
    onehand_remove_left_correct=sum(x==1 for r in onehand for x in r['ranks']['remove_left']),
    onehand_total=9,dtw_saved_summary=dtw,
    unchanged_channels='Every intervention asserted exact equality of body, other hand, timestamps and original active span; only126 channels belonging to the removed21-point hand were zero/masked.',
    limits=['Human reference contact sheets inspected for hand participation, not expert correctness labels',
    'User trial records were read only; no user inference/training/tuning/candidate selection',
    'Artificial whole-hand removal is a sensitivity diagnostic, not a real webcam accuracy estimate',
    'Remaining two-handed errors and specific latent mechanism are not causally established'])
dest=U/'INDEPENDENT_CHECK.json';assert not dest.exists();dest.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps({k:v for k,v in o.items() if k not in ['dtw_saved_summary','public_summary']},ensure_ascii=False))
