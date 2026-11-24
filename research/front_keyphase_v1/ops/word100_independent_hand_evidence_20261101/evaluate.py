"""Evaluate saved readouts only; no model, coordinate or path recomputation."""
import sys
sys.dont_write_bytecode=True
from collections import Counter, defaultdict
import json
import numpy as np
from pipeline import UNIT, read, write, digest

def main():
    manifest=read(UNIT/'RESULT_MANIFEST.json');inputs=read(UNIT/'INPUTS.json')
    tasks={t['id']:t for t in inputs['natural']+inputs['technical']}
    assert len(manifest['results'])==len(tasks)==927
    tally=defaultdict(Counter); conditions=defaultdict(Counter); failures=[];checks=0
    per_record=[];natural_nonzero=Counter();numerical_tol=1e-8
    for item in manifest['results']:
        assert digest(item['path'])==item['sha256']
        r=read(item['path']); t=tasks[r['id']];ev=r['evidence']
        assert ev['path_full_scope'] and not ev['position_used_for_gating'] and ev['whole_sign_verdict'] is None
        path=np.asarray(ev['path'],int)
        for key,g in ev['groups'].items():
            for side in ('query','reference'):
                tally[(r['kind'],key,side)].update(g[side]['counts'])
                counts=g[side]['counts'];checks+=1
                assert counts['possible']==counts['direct_complete']+counts['partial_shared']+counts['not_comparable']
            if r['kind']=='natural':
                peaks=[s['max_abs_deg'] for s in g['summary_by_dimension'] if s['max_abs_deg'] is not None]
                natural_nonzero[key+'_supported_records']+=bool(peaks)
                natural_nonzero[key+'_positive_records']+=bool(peaks and max(peaks)>numerical_tol)
                continue
            condition=t['edit']['condition']; hand=t['edit']['hand']; selected=key.startswith(hand+'_')
            actual=g['links']['absolute_difference_deg']
            for k,row in enumerate(actual):
                for d,value in enumerate(row):
                    if value is None:continue
                    expected=0.
                    if selected and condition=='finger_change' and key.endswith('finger_angles14') and d==2:expected=18.
                    if selected and not key.endswith('finger_angles14'):
                        expected=abs(t['edit']['axis_delta_degrees'][path[k,0]])
                        if condition=='common_arm_hand_rotation' and key.endswith('relative2d'):expected=0.
                    checks+=1
                    if abs(value-expected)>numerical_tol:
                        if len(failures)<50:failures.append([r['id'],key,k,d,value,expected])
            if selected and condition=='one_finger_missing' and key.endswith('finger_angles14'):
                for row in actual:
                    checks+=1
                    if row[2] is not None:failures.append([r['id'],'missing angle became comparable'])
            if selected and condition=='axis_missing' and not key.endswith('finger_angles14'):
                checks+=1
                if any(v is not None for row in actual for v in row):failures.append([r['id'],'missing axis became comparable'])
        if r['kind']=='technical':
            c=t['edit']['condition'];hand=t['edit']['hand'];conditions[c]['opportunities']+=1
            key=hand+('_finger_angles14' if c in ('finger_change','one_finger_missing') else '_hand_axis2d')
            g=ev['groups'][key]; dim=2 if key.endswith('finger_angles14') else 0
            changed=np.asarray(t['edit']['axis_delta_degrees'])!=0
            relevant=[]
            for k,(i,j) in enumerate(path):
                if c=='short_axis_change' and not changed[i]:continue
                value=g['links']['absolute_difference_deg'][k][dim]
                if value is not None:relevant.append(value)
            conditions[c]['has_shared_target_evidence']+=bool(relevant)
            conditions[c]['no_shared_target_evidence']+=not bool(relevant)
            conditions[c]['has_measured_target_difference']+=bool(relevant and max(relevant)>numerical_tol)
            per_record.append({'id':r['id'],'condition':c,'hand':hand,'shared_target_cells':len(relevant),
                'target_peak_degrees':max(relevant) if relevant else None,'real_motion_error':None})
    assert not failures,failures[:5]
    result={'status':'PASS_FIXED_SIGNAL_CONTRACT_NOT_EDUCATIONAL_ACCURACY',
        'natural_records':117,'technical_records':810,'checks':checks,'failures':failures,
        'axis_counts':[{'kind':k[0],'group':k[1],'axis':k[2],**dict(v)} for k,v in tally.items()],
        'technical_conditions':dict(conditions),'technical_record_summary':per_record,
        'natural_positive_readout_records':dict(natural_nonzero),
        'new_inference':0,'new_training':0,'new_dtw':0,'timing':{k:manifest[k] for k in ('compute_seconds','wall_seconds','end_to_end_latency')},
        'educational_accuracy':None,'normal_rotation_crossing_accuracy':None,
        'interpretation':'Natural positive differences are between public reference performances; no wrong-sign labels. Technical edits test channel independence only. Unobserved modified cells are not recovered or passed.',
        'prior_position_results_reused':585,'prior_shape_seed_links':351,'no_automatic_correction':True}
    write(UNIT/'EVALUATION.json',result)
    print(json.dumps({k:result[k] for k in ('status','checks','natural_records','technical_records','technical_conditions')},ensure_ascii=False))

if __name__=='__main__':main()
