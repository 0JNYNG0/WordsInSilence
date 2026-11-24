"""Root independent saved-output checks; never imports the guard/encoder."""
from pathlib import Path
import json, hashlib
from collections import Counter
U=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
counts=Counter()
def check(kind, condition):
    assert condition,kind
    counts[kind]+=1
def main():
    data=read(U/'DATASET.json');guard=read(U/'DATASET_MOTION_GUARDED.json');audit=read(U/'MOTION_GUARD_AUDIT.json')
    first=read(U/'LABEL_AUDIT.json');primary={}
    for f in ('LABELS_A.json','LABELS_B.json','LABELS_C.json'):
        primary.update({r['sample_id']:r for r in read(U/'review'/f)['rows']})
    secondary={r['sample_id']:r for r in read(U/'review/LABELS_SECONDARY.json')['rows']}
    check('original_failed_gate_retained',first['status']=='NOT_READY_SUPPORT_OR_REVIEW' and audit['training_allowed'] is False)
    check('row_and_target_order',len(data['rows'])==len(guard['rows'])==288 and data['targets']==guard['targets'])
    changes=[]
    for old,new in zip(data['rows'],guard['rows']):
        sid=old['sample_id']
        for key in old:
            if key not in ('y','target_mask','reasons'):
                check('all_original_metadata_unchanged',old[key]==new[key])
        withheld={h:False for h in 'RL'}
        if sid in secondary:
            for h in 'RL':
                p,q=primary[sid],secondary[sid]
                withheld[h]=(p['motion'][h] is None or q['motion'][h] is None or
                    p['motion'][h]!=q['motion'][h] or p['roi_window'][h] is not True or q['roi_window'][h] is not True)
        for j,name in enumerate(data['targets']):
            drop='_motion_' in name and withheld[name[0]]
            expect_mask=False if drop else old['target_mask'][j]
            expect_value=0.0 if drop else old['y'][j]
            check('all_feature_cells',new['target_mask'][j]==expect_mask and new['y'][j]==expect_value)
            if old['target_mask'][j]!=new['target_mask'][j]:
                changes.append((sid,name,old['y'][j]))
    check('exact_withdrawal',len(changes)==audit['changed_cells']==84 and all(v==0 for _,_,v in changes))
    check('known_denominators',sum(sum(r['target_mask']) for r in data['rows'])==3768 and sum(sum(r['target_mask']) for r in guard['rows'])==3684)
    check('original_pins_retained',data['pins']==guard['pins'])
    for p,h in data['pins'].items():check('first_artifact_hash',sha(p)==h)
    diagnosis=read(U/'MOTION_SEQUENCE_DIAGNOSIS.json')
    disputed={(r['id'],r['hand']) for r in read(U/'review/MOTION_DISAGREEMENT_FIRST.json')}
    by={r['sample_id']:r for r in data['rows']};keys=set();direct=unknown=0
    for out in diagnosis['rows']:
        key=(out['sample_id'],out['side']);check('diagnostic_unique',key not in keys);keys.add(key)
        row=by[key[0]]
        check('diagnostic_frame_scope',[t['frame'] for t in out['timeline']]==row['frame_ids'])
        check('diagnostic_timestamp_scope',[t['timestamp_us'] for t in out['timeline']]==row['timestamps_us'])
        for t in out['timeline']:
            cell=t['components']['translation'];direct+=cell['direct'];unknown+=not cell['direct']
            check('unknown_never_numeric_label',cell['direct']==(cell['value'] is not None))
        check('no_semantic_or_training_truth',out['training_eligible'] is False and out['semantic_importance'] is None and out['semantic_phase'] is None and out['educational_correctness'] is None)
        check('actual_source_image',sha(out['original_image'])==out['original_image_sha256'])
    check('all_disputed_cases_preserved',keys==disputed and len(keys)==28)
    check('all_original_hand_frames',direct==226 and unknown==138 and direct+unknown==364)
    for p,h in diagnosis['source_pins'].items():check('diagnostic_pins',sha(p)==h)
    for p,h in read(U/'SELECTION.json')['protected'].items():check('protected_artifacts',sha(p)==h)
    check('no_training_artifacts',not any((U/n).exists() for n in ('TRAIN_FROZEN.json','TRAIN_RUNNING.json','TRAINING.json','SCORES.json')))
    check('no_result_html',(U/'RESULT_KO.html').exists() is False)
    result=dict(status='PASS_SAVED_REMEDIATION_ARITHMETIC_NOT_LABEL_RELIABILITY',checks=dict(counts),check_count=sum(counts.values()),
        withdrawn_negative_cells=84,all_windows_preserved=288,disputed_hand_windows_preserved=28,
        direct_translation_frame_cells=direct,unresolved_translation_frame_cells=unknown,
        original_reliability_failure_preserved=True,training_started=False,
        independent_human_or_ksl_review=False,reviewer='Root separate scalar decoder; implementation written by agent and root',
        code_sha256=sha(__file__),artifacts={str(U/n):sha(U/n) for n in ['DATASET.json','DATASET_MOTION_GUARDED.json','MOTION_GUARD_AUDIT.json','MOTION_SEQUENCE_DIAGNOSIS.json']})
    with (U/'review/ROOT_REMEDIATION_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','checks')},ensure_ascii=False))
if __name__=='__main__':main()
