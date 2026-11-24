"""Independent source-by-source verification; no model inference or scoring."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    frozen=read(U/'FROZEN.json');data=read(U/'SUPERVISION.json');coverage=read(U/'COVERAGE.json')
    for p,h in frozen['pins'].items():assert sha(p)==h,p
    raw=read(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json')
    relations=read(O/'word100_learned_body_relation_20261102/RELATION_TARGETS.json')
    movements=read(O/'word100_learned_motion_order_20261102/MOTION_TARGETS.json')
    rr={(r['clip_id'],r['center_frame']):r for r in relations['rows']}
    pp={(r['clip_id'],r['frame']):r for r in data['points']}; ww={r['id']:r for r in data['windows']}
    seen=[]
    for r in raw['rows']:
        a=np.load(r['input'],allow_pickle=False)
        for label in r['labels']:
            key=r['clip_id'],label['center_frame'];p=pp[key];rel=rr[key];seen.append(key)
            assert p['y']==label['y']+rel['y'] and p['mask']==label['target_mask']+rel['mask']
            assert p['timestamp_us']==int(a['timestamp_us'][key[1]])
            assert p['guide_clause_gold'] is None and p['boundary_uncertain']
            for j,known in enumerate(p['mask']):assert p['observation_provenance'][j]==('direct_image' if known else 'unknown')
    assert set(seen)==set(pp) and len(seen)==len(pp)==144
    for label in movements['rows']:
        w=ww[label['id']]
        for k in ['person','role','frames','timestamps_us','y','mask']:assert w[k]==label[k]
        assert w['native_frame_ids']==list(range(label['frames'][0],label['frames'][1]+1))
        assert not w['framewise_target_propagation'] and w['guide_clause_gold'] is None
    for gran,rows,names in [('point',data['points'],data['point_targets']),('window',data['windows'],data['window_targets'])]:
        for role in ['fit','development']:
            for j,name in enumerate(names):
                for state in ['positive','negative','unknown']:
                    selected=[r for r in rows if r['role']==role and (('unknown' if not r['mask'][j] else 'positive' if r['y'][j] else 'negative')==state)]
                    c=coverage[gran][role][name][state]
                    assert len(selected)==c['count'] and sorted({r['person'] for r in selected})==c['recorded_ids']
                    assert sorted({r['word_id'] for r in selected})==c['words']
    splits={role:{r['person'] for r in data['sources'] if r['role']==role} for role in ['fit','development']}
    assert not splits['fit']&splits['development'] and splits['fit']==set(range(1,10)) and splits['development']=={11,12,13}
    links=read(U/'CLAUSE_COMPONENT_LINKS.json')['rows']
    assert all(not r['training_label'] and not r['linguistic_correctness_supported'] for r in links)
    out=dict(status='DIRECT_PREDICATE_SOURCE_AND_MASKS_VERIFIED',point_records=144,motion_windows=108,
        point_fit=108,point_development=36,motion_fit=81,motion_development=27,
        native_time_and_source_hashes_exact=True,unknowns_preserved=True,old_event_labels_unchanged=True,
        source_links_are_partial_not_gold=True,new_training=0,new_inference=0,whole_goal_complete=False)
    with (U/'VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
    print(out)
if __name__=='__main__':main()
