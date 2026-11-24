from collections import Counter
import numpy as np
from common import G,P,R,V,read,sha,exclusive

def main():
    plan=read(G/'FROZEN.json'); data=read(G/'SENSOR_GAP_RESULTS.json'); rows=data['rows'];assert len(rows)==36
    source={r['word_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows'] if r['person']==12}
    counts={}
    for k in plan['gap_conditions']:
        rr=[r for r in rows if r['kind']==k];assert len(rr)==12
        counts[k]=dict(baseline_correct=sum(r['baseline_correct'] for r in rr),current_correct=sum(r['current_correct'] for r in rr),
            known=sum(r['known_count'] for r in rr),case_regressions=sum(r['current_correct']<r['baseline_correct'] for r in rr),
            motion_statuses=dict(Counter(m['status'] for r in rr for m in r['motions'])))
    for r in rows:
        label=next(l for l in source[r['word']]['labels'] if l['center_frame']==r['center'])
        assert label['y']==r['y'] and label['target_mask']==r['known']
        p=np.load(G/'gap_predictions'/f"{r['case']}.npz")['retained_probability'][r['center']]
        np.testing.assert_allclose(p,r['probability'],rtol=0,atol=0,equal_nan=True)
        m=np.asarray(r['known'],bool); y=np.asarray(r['y']);valid=np.isfinite(p)&m
        assert r['current_correct']==int(((p>=.5)==y)[valid].sum())
        out=read(G/'gap_feedback'/f"{r['case']}.json");assert out['whole_sign_correctness'] is None
        assert out['motion_observations']==r['motions']
        if r['kind']=='pose_and_image_7':assert set(r['frames'])<=set(i for g in out['evidence_gaps'] for i in g['frames'])
    assert all(x['observations_identical'] for x in data['goal_blind_checks']) and len(data['goal_blind_checks'])==4
    order=read(G/'MAIN_ORDER_RESULTS.json')['rows'];assert len(order)==4
    order_counts=[]
    for r in order:
        assert sha(r['source'])==r['source_sha256'] and sha(r['video'])==r['video_sha256']
        mapping=np.asarray(r['source_frame_indices']);lo,hi=r['reverse_frames']
        expected=np.arange(len(mapping));expected[lo:hi+1]=expected[lo:hi+1][::-1];np.testing.assert_array_equal(mapping,expected)
        out=read(r['result']);a=np.load(r['input']);assert np.all(np.diff(a['timestamp_us'])>0)
        assert out['whole_sign_correctness'] is None and out['motion_observations']==r['motions']
        for c in r['physical_stage_checks']:assert mapping[c['edited_frame']]==c['source_frame']
        checks=r['physical_stage_checks'];order_counts.append(dict(word=r['word'],baseline_configuration_correct=sum(c['baseline_configuration_correct'] for c in checks),
            edited_configuration_correct=sum(c['edited_configuration_correct'] for c in checks),count=len(checks),
            directed_motion_confirmed=sum(m['status']=='DIRECTED_MOTION_OBSERVED' for m in r['motions']),
            stage_statuses=[s['status'] for s in r['stages']],no_whole_sign_correct_claim=True))
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(G/'DIAGNOSTIC_VERIFICATION.json',dict(status='PASS_ARITHMETIC_WITH_REMAINING_SEQUENCE_LIMIT',sensor_counts=counts,order_counts=order_counts,
        short3_center_performance_preserved=counts['pose_only_3']['current_correct']==counts['pose_only_3']['baseline_correct'],
        seven_frame_gaps_not_fully_robust=True,goal_blind_observations_verified=True,blank_times_preserved=True,
        critical_limit='Main-order edits retain/splice some expected-direction movement. Positive direction is occurrence evidence, not correct ordering. Existing phase/flow warnings must be shown; no whole-sign pass.',
        no_new_inference=True,whole_goal_complete=False,verifier='Same Codex separate saved-result checker; not independent human expert'))
    print(counts);print(order_counts)

if __name__=='__main__':main()
