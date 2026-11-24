from pathlib import Path
import json
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def save(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
rows=read(U/'ROWS.json');st=read(U/'STRESS.json');deletion=read(U/'DELETION.json');summary=read(U/'SUMMARY.json')
base={(r['clip_id'],r['view']):r for r in rows if r['scenario']=='original'}
idx={(r['clip_id'],r['view']):r for r in read(U/'INDEX.json')}
old={r['clip_id']:r for r in read(OPS/'word100_multiref_observation_compare_20261019/FROZEN.json')['inputs']}
oldesc=read(OPS/'word100_reference_extent_20261020/DIAGNOSTICS.json')['previous_escaped'];key=lambda r:(r['clip_id'],r['side'],r['key'],r['condition']);by={key(r):r for r in st}
esc=[]
for r in oldesc:
    n=by[key(r)];esc.append(dict(clip_id=r['clip_id'],label=r['label'],side=r['side'],key=r['key'],condition=r['condition'],
        removed_s=n['removed_s'],previous_extra_missing_s=r['new_missing_expected_s'],new_strong_s=n['extra_strong_s'],new_possible_s=n['extra_possible_s'],
        after=n['after']['status']))
stress={}
for mode in ('whole','center10','peak10','peak20'):
    rr=[r for r in st if r['condition']==mode]
    stress[mode]=dict(total=len(rr),effective=sum(r['removed_frames']>0 for r in rr),strong_revealed=sum(r['extra_strong_s']>1e-8 for r in rr),
        possible_revealed=sum(r['extra_possible_s']>1e-8 for r in rr),unrevealed_effective=sum(r['removed_frames']>0 and r['extra_possible_s']<=1e-8 for r in rr))
deletion_counts=Counter();examples=[]
for r in deletion:
    b=base[r['clip_id'],r['view']];side=r['meta']['side'];rec=idx[r['clip_id'],r['view']]
    with np.load(rec['path'],allow_pickle=False) as z:ts=z['ts']
    t=(ts-ts[0])/1e6;w=np.diff(np.r_[t[0],(t[:-1]+t[1:])/2,t[-1]]);truth=np.zeros(len(ts),bool)
    for a,z in old[r['clip_id']]['intervals']:truth|=(ts>=a*1e6)&(ts<=z*1e6)
    removed=np.ones(len(ts),bool);removed[np.array(r['meta']['keep_rows'])]=False;removed_core=float(w@(removed&truth))
    if removed_core>0:deletion_counts['windows_with_annotated_time_removed']+=1
    for m in r['multi_metrics']:
        if m['side']!=side:continue
        before=next(x for x in b['multi_metrics'] if x['side']==side and x['key']==m['key'])
        clear=m['status']=='OBSERVED_SUPPORTED_INTERVAL_ONLY';bclear=before['status']=='OBSERVED_SUPPORTED_INTERVAL_ONLY'
        deletion_counts['metric_hand_conditions']+=1
        deletion_counts['after_observed_intervals_only']+=int(clear)
        deletion_counts['clear_before_and_after']+=int(clear and bclear)
        deletion_counts['after_clear_with_annotated_time_removed']+=int(clear and removed_core>0)
        if clear and removed_core>0 and len(examples)<12:examples.append(dict(clip_id=r['clip_id'],side=side,key=m['key'],removed_annotation_seconds=removed_core,before=before['status'],after=m['status']))
report=dict(stress=stress,previous8escaped=esc,previous8_revealed_strong=sum(r['new_strong_s']>1e-8 for r in esc),previous8_revealed_possible=sum(r['new_possible_s']>1e-8 for r in esc),
    deletion_counts=dict(deletion_counts),deletion_examples=examples,
    interpretation='Synthetic peak windows are not expert-labeled essential phases. Observed intervals only is NOT complete or correct sign. Possible union can increase recall while including normal lowering; do not treat uncertain missing as wrong action.',
    original_cycle_gt='Each source recording is one supplied full performance; differing selected traversal counts may also reflect intrinsic repeat differences between people, not certified linguistic mistakes.')
save('DIAGNOSTICS.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
