"""Once-only candidate inference through Claude APIs; saved comparators reused."""
import datetime
import json
import shutil
import time
import numpy as np
import torch
from runtime_binding import U,W,RUN,OLD_RUN,read,sha,setup,gw,gp,ra
from window_bridge import window_valid_from_indices


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert (U/'TRAINING_COMPLETION.json').is_file()
    trace=read(U/'TRAINING_TRACE_AUDIT.json')
    assert trace['status']=='PASS' and trace['finalized']
    assert not (RUN/'EVALUATION_SCORES.npz').exists(), 'Scores already exist: never repeat'
    assert not (RUN/'EVALUATION_STARTED.json').exists(), 'Interrupted inference requires review'
    ctx=read(U/'EXECUTION_CONTEXT.json')
    for path,expected in ctx['library_pins'].items():
        assert sha(path)==expected,path
    frozen=gw.validate_frozen(str(RUN),ctx)
    frozen_sha=sha(RUN/'FROZEN_MODEL.json')
    plan,cfg,meta,factory,_,tf=setup()
    assert sha(OLD_RUN/'EVALUATION_SCORES.npz')==plan['comparison_source']['scores_sha256']
    assert sha(OLD_RUN/'EVALUATION_ROWS.json')==plan['comparison_source']['rows_sha256']
    baseline_path=W/'runs/partial_encoder_pair_v1/checkpoint_control_final.pt'
    assert sha(baseline_path)==plan['pins']['baseline832']
    old_rows=read(OLD_RUN/'EVALUATION_ROWS.json')
    rows=old_rows['rows']
    assert len(rows)==855 and old_rows['classes']==ctx['classes']==meta['classes']
    assert all(r['role']!='train' and '/runs/ses_' not in r['path'].replace('\\','/') for r in rows)
    with np.load(OLD_RUN/'EVALUATION_SCORES.npz',allow_pickle=False) as z:
        baseline=z['baseline_scores'].copy();old_scores=z['candidate_scores'].copy();labels=z['y'].copy()
    assert np.array_equal(labels,np.asarray([r['y'] for r in rows],dtype=np.int64))
    model=gw.load_final_state(factory(),str(RUN),ctx)
    before=gw.state_digest(model.state_dict())
    normalize=tf.CenterAndScaleNormalize(reference_points_preset=ra.NORMALIZE_PRESET,frame_level=False)
    prepare=lambda xy,mask,layout:gp.prepare_windows(xy,mask,ra.LAYOUTS[layout][1],[3,4],60,10,lambda t:normalize({'frames':t})['frames'])
    loaded=[gp.load_row(r,meta['layout_specs'],prepare) for r in rows]
    x=torch.stack([sample for sample,_ in loaded])
    diagnostics=[dict(id=r['id'],group=r['group'],condition=r['condition'],**diag) for r,(_,diag) in zip(rows,loaded)]
    assert diagnostics==read(OLD_RUN/'EVALUATION_INPUT_DIAGNOSTICS.json'), 'Old/new input diagnostics differ'
    assert gp.tensor_sha256(x)==read(OLD_RUN/'EVALUATION_EXECUTION.json')['input_sha256']
    valid=window_valid_from_indices([d['prepare_diag']['indices'] for d in diagnostics])
    with np.load(U/'WINDOW_METADATA.npz',allow_pickle=False) as z:
        assert np.array_equal(valid.numpy(),z['eval_valid'])
    started=time.perf_counter()
    (RUN/'EVALUATION_STARTED.json').write_text(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'frozen_model_sha256':frozen_sha,'input_sha256':gp.tensor_sha256(x),'window_valid_sha256':gp.tensor_sha256(valid)},indent=2),encoding='utf-8')
    scores=gw.score_batches(model,x,16,window_valid=valid)
    assert scores.shape==(855,93) and np.isfinite(scores).all()
    assert before==gw.state_digest(model.state_dict())
    assert frozen_sha==sha(RUN/'FROZEN_MODEL.json')
    np.savez(RUN/'EVALUATION_SCORES.npz',candidate_scores=scores,baseline_scores=baseline,old_stgcn_scores=old_scores,y=labels)
    shutil.copyfile(OLD_RUN/'EVALUATION_ROWS.json',RUN/'EVALUATION_ROWS.json')
    (RUN/'EVALUATION_INPUT_DIAGNOSTICS.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    record={'status':'SCORED_NOT_YET_INDEPENDENTLY_AUDITED','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'final_model_sha256':frozen['final_model_sha256'],'frozen_model_sha256':frozen_sha,
       'input_sha256':gp.tensor_sha256(x),'window_valid_sha256':gp.tensor_sha256(valid),'rows':855,'classes':93,
       'score_sha256':sha(RUN/'EVALUATION_SCORES.npz'),'baseline_model_sha256':plan['pins']['baseline832'],
       'old_comparator_scores_sha256':plan['comparison_source']['scores_sha256'],
       'model_state_unchanged':True,'no_training_or_selection':True,'user_data_used':False,
       'old_input_exact':True,'comparator_inferences':0,'seconds':time.perf_counter()-started}
    (RUN/'EVALUATION_EXECUTION.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
