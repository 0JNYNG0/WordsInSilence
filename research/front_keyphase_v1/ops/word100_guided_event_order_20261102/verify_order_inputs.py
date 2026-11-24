"""Reconstruct query labels from saved source witnesses and mappings."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    a=read(U/'ORDER_SUPERVISION.json');d=read(O/'word100_action_episode_supervision_20261102/ACTION_TARGETS.json')
    e=read(O/'word100_guide_conditioned_episode_20261102/CONTEXT_TARGETS.json')
    b=read(O/'word100_learned_action_process_20261102/PREPARED.json')['rows']
    extra=read(O/'word100_seated_fit_transfer_20261102/PAIR_MANIFEST.json')['conditions']['seated_variation']
    sources={(r['clip_id'],r['kind']):r for r in b+extra}
    direct={r['clip_id']:{int(p['frame']):int(p['action_family']) for p in r['points'] if p['mask']} for r in d['rows']}
    context={r['clip_id']:{int(p['frame']):int(p['action_family']) for p in r['points'] if p['mask']} for r in e['rows']}
    known=unknown=0;clips=set();ids=set()
    for q in a['rows']:
        meta=sources[(q['clip_id'],q['kind'])];assert sha(meta['cache'])==meta['cache_sha256']
        mapping=np.asarray(meta['mapping']) if 'mapping' in meta else np.load(meta['cache'])['mapping']
        assert len(mapping)==q['frames'];ids.add(q['recorded_id']);clips.add(q['clip_id'])
        for name,target in [('first',q['asked_before']),('second',q['asked_after'])]:
            f=q[name+'_witnesses'];sf=q['source_'+name+'_witnesses']
            assert mapping[f].tolist()==sf and all(direct[q['clip_id']][x]==target for x in sf)
        assert all(context[q['clip_id']][int(mapping[f])]>=8 for f in q['reviewed_nonmain'])
        first=q['first_witnesses'];second=q['second_witnesses']
        expected=1 if first and second and max(first)<min(second) else 0 if first and second and max(second)<min(first) else -1
        assert expected==q['target']
        mask={x['frame']:x['label'] for x in q['occurrence_targets']}
        assert set(mask)==set(first)|set(second)|set(q['reviewed_nonmain'])
        assert all(mask[x]==1 for x in first) and all(mask[x]==2 for x in second) and all(mask[x]==0 for x in q['reviewed_nonmain'])
        if q['kind'] in ['original','speed_0.8','speed_1.25','pose_only_3','seated_variation']:
            assert q['target']==int(not q['swapped_query'])
        if expected<0:unknown+=1
        else:known+=1
    assert ids==set(range(1,10)) and known==a['known_queries']==576 and unknown==a['unknown_queries']==54
    for i in range(0,len(a['rows']),2):
        x,y=a['rows'][i:i+2];assert x['flow_key']==y['flow_key'] and x['asked_before']==y['asked_after']
        assert x['target']<0 and y['target']<0 or x['target']+y['target']==1
    for p,h in a['pins'].items():assert sha(p)==h
    result=dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),status='PASS_SPARSE_OCCURRENCE_QUERY_INPUTS',
        queries=630,known_queries=known,held_queries=unknown,unique_source_videos=len(clips),recorded_ids=sorted(ids),
        supported_words=sorted({q['word_id'] for q in a['rows']}),all_four_word_role_inputs_retained=True,
        repetition_word_without_supported_edges='WORD1567: no order truth invented; existing hand/motion scope remains.',
        masks_and_source_mapping_verified=True,normal_speed_and_pose_gap_orders_preserved=True,swapped_query_counterexamples_verified=True,
        independent_expert_truth=False,new_training=0,new_predictions=0,whole_goal_complete=False)
    with (U/'INPUTS_VERIFIED.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(result)

if __name__=='__main__':main()
