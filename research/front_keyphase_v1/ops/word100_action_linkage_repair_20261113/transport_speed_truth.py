"""Exact provenance mapping only; does not infer or amend old evaluation gates."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
E=O/'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/full_bag_rate_rank_learning/evaluation'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
j=read(E/'COMPARISON.json');frozen=read(E/'FROZEN.json');raws=read(O/'word100_completion_fullpath_20261110/REPLAY_INPUTS.json')['rows']
rawmap={(r['clip_id'],r['kind']):r for r in raws};fm={(r['clip_id'],r['kind']):r for r in frozen['scenarios']}
normal={r['clip_id']:r for r in j['rows'] if r['seed']==42 and r['kind']=='full_normal'}
mapped=[];checks=[]
for cid,base in normal.items():
    rm=rawmap[cid,'full_normal'];assert sha(rm['path'])==rm['sha256'];src=np.load(rm['path']);origin=np.asarray(rm['source_indices'])
    for kind in ['speed075','speed125']:
        meta=rawmap[cid,kind];assert sha(meta['path'])==meta['sha256']
        raw=np.load(meta['path']);indices=np.asarray(meta['source_indices']);cache=fm[cid,kind]
        assert sha(cache['source'])==cache['source_sha256'];features=np.load(cache['source'])
        assert np.array_equal(raw['timestamp_us'],features['ts']) and len(indices)==len(features['x'])
        assert meta['raw_source_sha256']==rm['raw_source_sha256']
        # Indices refer to the same original array. No guessed resampling rule.
        positions={int(v):i for i,v in enumerate(origin)};lookup=np.array([positions[int(v)] for v in indices])
        assert np.array_equal(raw['xy_px'],src['xy_px'][lookup],equal_nan=True)
        assert np.array_equal(raw['score'],src['score'][lookup],equal_nan=True)
        for c in base['conditions']['full_bag_rate_rank']['cards']:
            truth=c.get('truth') or {};w=truth.get('witness') or []
            if not w:continue
            entries=[]
            for k in w:
                source_index=int(origin[k]);dest=np.flatnonzero(indices==source_index).astype(int).tolist()
                entries.append(dict(normal_index=k,source_index=source_index,query_indices=dest,state='retained' if dest else 'dropped_by_resampling'))
            mapped.append(dict(clip_id=cid,kind=kind,card=c['card'],family=c['family'],witnesses=entries,
                retained_source_points=sum(bool(e['query_indices']) for e in entries),
                no_completion_target=True,raw_path=meta['path'],raw_sha256=meta['sha256'],feature_source=cache['source'],feature_sha256=cache['source_sha256']))
        checks.append(dict(clip_id=cid,kind=kind,source_map_exact=True,raw_and_model_timestamps_equal=True))
out=dict(passed=True,checks=checks,mapped_cards=mapped,original_witnesses=sum(len(r['witnesses']) for r in mapped),
    retained_witnesses=sum(r['retained_source_points'] for r in mapped),
    original_comparison_sha256=sha(E/'COMPARISON.json'),plan_sha256=sha(U/'SPEED_TRUTH_PLAN.json'),
    inference=False,training=False,new_accuracy=False,old_gates_unchanged=True)
with (U/'SPEED_TRUTH_TRANSPORT.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(passed=True,cases=len(checks),mapped_cards=len(mapped),original_witnesses=out['original_witnesses'],retained_witnesses=out['retained_witnesses'],dropped=out['original_witnesses']-out['retained_witnesses']),ensure_ascii=False))
