"""Independent arithmetic of newly generated inputs and24 predictions."""
from prepare_coherent import *
def metric(points):
    return dict(known=len(points),correct=sum(p['pred']==p['y'] for p in points))
def main():
    assert not (K/'VERIFIED.json').exists();plan=read(K/'EXECUTION_FREEZE.json');manifest=read(K/'INPUT_MANIFEST.json');diag=read(K/'DIAGNOSTIC.json')
    all_checks=[];summaries={};arrays=0
    for r in manifest['rows']:
        assert sha(r['input'])==r['input_sha256'] and sha(r['old_input'])==r['old_input_sha256'] and sha(r['source'])==r['source_sha256']
        a=np.load(r['input']);b=np.load(r['old_input']);result=read(r['result'])
        assert np.array_equal(a['frame_id'],b['frame_id']) and np.array_equal(a['timestamp_us'],b['timestamp_us'])
        assert a['image_available'].all() and result['actual_sensor_extent']==r['actual_sensor_extent']
        assert result['source_sha256']==r['source_sha256']
        all_checks.append(dict(clip_id=r['clip_id'],source_and_axis=True,frames=len(a['frame_id']),new_input_differs=sha(r['input'])!=sha(r['old_input'])))
    for name in plan['conditions']:
        r=read(K/'evaluations'/f'{name}.json');points=[p for p in r['points'] if p['y']<8];raw=[p for p in r['raw_points'] if p['y']<8]
        for k,pts in [('major',points),('raw_major',raw)]:
            mm=metric(pts);assert mm['known']==r[k]['known'] and mm['correct']==r[k]['correct']
        rest=[p for p in r['role_points'] if p['y']==2]
        assert len(rest)==r['rest_known'] and sum(p['pred']==2 for p in rest)==r['rest_correct']
        for m,sq in zip(manifest['rows'],r['sequences']):
            a=np.load(K/'predictions'/name/f"{m['clip_id']}.npz");n=m['frames'];assert sq['key']==m['clip_id']+'_coherent_native'
            assert np.array_equal(a['frame_id'],np.arange(n)) and sq['comparison_frames']==n and sq['comparison_frame_range']==[0,n-1]
            assert np.array_equal(a['role_probability'][:,2],a['current_probability'][:,9])
            for key,w in [('probability',9),('raw_action_probability',9),('role_probability',3),('activity_probability',2),('current_probability',10)]:
                assert a[key].shape==(n,w) and np.isfinite(a[key]).all() and np.allclose(a[key].sum(1),1,atol=1e-6)
            ix=np.flatnonzero((a['current_probability'][:,9]>=.8)&(a['activity_probability'][:,1]>=.8)&a['image_available']).tolist()
            assert ix==sq['rest_action_conflict_frames'];assert sq['whole_sign_correctness'] is None
            arrays+=1
        assert sum(bool(q['rest_action_conflict_frames']) for q in r['sequences'])==r['conflict_cases']
        summaries[name]=dict(major=metric(points),raw_major=metric(raw),rest_correct=r['rest_correct'],rest_known=r['rest_known'],supported_order=r['supported_order'],conflict_cases=r['conflict_cases'],
            correct_content_lost=sum(a['pred']==a['y'] and b['pred']!=b['y'] for a,b in zip(raw,points)))
    assert arrays==24 and len(all_checks)==12
    for p,h in plan['pins'].items():assert sha(p)==h
    for tag in plan['conditions']:
        for r in manifest['rows']:
            a=np.load(K/'predictions'/tag/f"{r['clip_id']}.npz");b=np.load(K/'predictions'/plan['conditions'][0]/f"{r['clip_id']}.npz")
            assert np.array_equal(a['current_probability'],b['current_probability'])
    old=read(J/'DIAGNOSIS.json')['conditions'][plan['primary']]['native']
    previous=dict(major_known=sum(r['direct_major_known'] for r in old),major_correct=sum(r['direct_major_correct'] for r in old),
        supported_order=sum(r['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED' for r in old),conflict_cases=sum(bool(r['rest_conflicts']) for r in old),
        scope='Sum already stored native diagnostics, no old model inference or changed label/score rule.')
    assert previous['major_known']==summaries[plan['primary']]['major']['known']
    outcome=dict(at=clock(),status='PASS_ARITHMETIC_INPUT_PROVENANCE_NOT_MODEL_ACCEPTANCE',rows=all_checks,conditions=summaries,old_saved_native_summary=previous,
        compared_new_arrays=arrays,original_results_unchanged=True,model_hashes_unchanged=True,
        whole_frontend_contrast=True,isolated_rgb_cause_proven=False,independent_final_test=False,
        verification='Same Codex separate arithmetic/source hashes; not independent KSL expert truth.',whole_goal_complete=False)
    exclusive(K/'VERIFIED.json',outcome);print('verified',summaries,'previous',previous)
if __name__=='__main__':main()
