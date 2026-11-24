"""Supplement saved-score verification: complete pins and descriptive counts."""
from pathlib import Path
import hashlib
import json
import numpy as np

U=Path(__file__).resolve().parent.parent
OLD=U.parent/'word100_stage_feedback_cycle_20261031'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    checks=0;failures=[]
    def check(ok,name,detail=None):
        nonlocal checks
        checks+=1
        if not bool(ok):failures.append(dict(check=name,detail=detail))
    metadata=read(U/'FRESH_METADATA_FREEZE.json');freeze=read(U/'FRESH_EVALUATION_FREEZE.json');scores=read(U/'FRESH_SCORES.json');inputs=read(U/'fresh/INPUTS.json');proto=read(U/'QUALITY_PROTOCOL.json')
    required=[U/'fresh'/n for n in ['SELECTION.json','INPUTS.json','VISUAL_INDEX.json','LABEL_AUDIT.json']]+[OLD/'VISUAL_INDEX.json',OLD/'NORMALIZATION.json']+[U/n for n in ['QUALITY_PROTOCOL.json','NORMALIZATION.json','quality.py','build_fresh_labels.py','evidence_adapter.py']]
    check(set(metadata['pins'])=={str(p) for p in required},'complete_metadata_pin_set')
    for p,h in metadata['pins'].items():check(sha(p)==h,'metadata_pin/'+Path(p).name)
    expected_predictions={str(U/'predictions'/f'fresh_{condition}_{seed}.npy') for seed in [42,43,44] for condition in proto['comparison']+['candidate_availability']}
    check(set(scores['prediction_pins'])==expected_predictions,'complete_12_prediction_pins')
    for p,h in scores['prediction_pins'].items():check(sha(p)==h,'prediction_pin/'+Path(p).name)
    for r in inputs['rows']:
        for p in [Path(r['input']),U/'fresh/targets'/f"{r['alias']}.npz"]:
            check(str(p) in freeze['pins'],'evaluation_input_target_pin/'+r['alias'])
            if str(p) in freeze['pins']:check(sha(p)==freeze['pins'][str(p)],'evaluation_input_target_hash/'+r['alias'])
    for seed in [42,43,44]:
        new=read(U/'runs'/str(seed)/'RUN.json');old=read(OLD/'runs'/f'pose_capacity_matched_{seed}'/'RUN.json')
        check(new['parameters']==9990 and old['parameters']==47914,'composite_parameters/'+str(seed))
        check(new['parameters']+old['parameters']==57904,'composite_parameter_sum/'+str(seed))
    all_targets=read(OLD/'LEARNING_INPUTS.json')['all_targets'];targets=scores['targets'];rows=inputs['rows'];labels=[]
    for r in rows:
        with np.load(U/'fresh/targets'/f"{r['alias']}.npz",allow_pickle=False) as z:labels.append((z['y'],z['target_mask']))
    diagnostics=read(U/'SAVED_OUTPUT_DIAGNOSTICS.json');recomputed=[]
    for condition in proto['comparison']:
        for seed in [42,43,44]:
            pred=np.load(U/'predictions'/f'fresh_{condition}_{seed}.npy',allow_pickle=False)
            counts={k:0 for k in ['shape_cross_known','shape_cross_correct','shape_known','shape_correct','shape_positive_known','shape_positive_correct','motion_known','motion_correct']}
            for i,(y,mask) in enumerate(labels):
                for frame in range(len(y)):
                    cross_column=all_targets.index('hands_overlap');cross=bool(mask[frame,cross_column] and y[frame,cross_column]==1)
                    for j,target in enumerate(targets):
                        column=all_targets.index(target)
                        if not mask[frame,column]:continue
                        correct=int((float(pred[i,frame,j])>=.5)==bool(y[frame,column]))
                        if target in ['R_index_only','R_open_four','L_open_four']:
                            counts['shape_known']+=1;counts['shape_correct']+=correct
                            if cross:counts['shape_cross_known']+=1;counts['shape_cross_correct']+=correct
                            if y[frame,column]==1:counts['shape_positive_known']+=1;counts['shape_positive_correct']+=correct
                        if '_motion_' in target:counts['motion_known']+=1;counts['motion_correct']+=correct
            expected=dict(condition=condition,seed=seed,**counts)
            actual=[r for r in diagnostics['rows'] if r['condition']==condition and r['seed']==seed]
            check(len(actual)==1 and actual[0]==expected,'diagnostic_saved_prediction_counts/'+condition+'/'+str(seed))
            recomputed.append(expected)
    result=dict(status='PASS' if not failures else 'FAIL',reviewer='Codex rgb_labels_b; separate agent from implementation',model_inference_calls=0,project_or_torch_imports=False,checks=checks,failures=failures,metadata_pins=len(metadata['pins']),prediction_pins=len(expected_predictions),evaluated_input_target_pairs=len(rows),additional_trained_parameters=9990,required_existing_pose_parameters=47914,composite_parameters=57904,descriptive_counts=recomputed,source_pins={str(U/n):sha(U/n) for n in ['FRESH_METADATA_FREEZE.json','FRESH_EVALUATION_FREEZE.json','FRESH_SCORES.json','SAVED_OUTPUT_DIAGNOSTICS.json']},verifier_sha256=sha(__file__),scope='No label reconstruction, no visual label correctness judgment, no inference. Separate metadata freeze supplies missing metadata pins identified in static review.')
    out=U/'review/QUALITY_VERIFY_FRESH_PINS.json'
    with out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],checks=checks,failures=len(failures),output=str(out)),ensure_ascii=False))
    return int(bool(failures))

if __name__=='__main__':raise SystemExit(main())
