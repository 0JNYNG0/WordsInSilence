"""Independent mask oracle and masked-hand raw input audit; no model inference."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import torch

U=Path(__file__).resolve().parent
OLD=U.parent/'stgcn_execution_ready_20261005'
RUN=U.parent.parent/'runs/stgcn_valid_window_v1'
OLD_RUN=RUN.parent/'stgcn_full_path_v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    old_verifier=OLD/'check_evaluation_inputs.py'
    pinned=read(U/'EVALUATION_AUDIT_PINS.json')
    assert sha(old_verifier)==pinned['old_independent_verifier_sha256']
    assert sha(OLD/'EVALUATION_INPUT_INDEPENDENT_CHECK.json')==pinned['old_input_audit_sha256']
    spec=importlib.util.spec_from_file_location('independent_input_oracle',old_verifier)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    plan=read(U/'FROZEN_EXPERIMENT.json')
    assert sha(RUN/'EVALUATION_ROWS.json')==plan['comparison_source']['rows_sha256']
    rows=read(RUN/'EVALUATION_ROWS.json')['rows']
    diags=read(RUN/'EVALUATION_INPUT_DIAGNOSTICS.json')
    assert diags==read(OLD_RUN/'EVALUATION_INPUT_DIAGNOSTICS.json')
    ex=read(RUN/'EVALUATION_EXECUTION.json')
    assert ex['input_sha256']==read(OLD_RUN/'EVALUATION_EXECUTION.json')['input_sha256']
    assert ex['model_state_unchanged'] and ex['no_training_or_selection'] and not ex['user_data_used']
    assert ex['frozen_model_sha256']==sha(RUN/'FROZEN_MODEL.json')
    masks=[]
    for row,d in zip(rows,diags):
        t=row['end_exclusive']-row['start']
        assert t==d['prepare_diag']['T']
        ix=[-1]*(60-t)+list(range(t)) if t<=60 else [j*(t-1)//59 for j in range(60)]
        assert ix==d['prepare_diag']['indices']
        masks.append([10*(j+1)>max(0,60-t) for j in range(6)])
    mask=np.asarray(masks,dtype=bool)
    assert mask.shape==(855,6) and mask.any(axis=1).all()
    assert oracle.tensor_sha(torch.from_numpy(mask))==ex['window_valid_sha256']
    with np.load(U/'WINDOW_METADATA.npz',allow_pickle=False) as z:
        assert np.array_equal(mask,z['eval_valid'])
    checks=[]
    for i,row in enumerate(rows):
        if row['mask_hand'] is None:
            continue
        path=Path(row['path']);assert sha(path)==row['sha256']
        xkey,mkey,tkey,_,_=oracle.LAYOUTS[row['layout']]
        with np.load(path,allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in (xkey,mkey,tkey)}
        x,distance,missing=oracle.reconstruct(row,arrays)
        assert oracle.tensor_sha(x)==diags[i]['input_tensor_sha256']
        assert distance==diags[i]['prepare_diag']['shoulder_distance']
        checks.append({'row':i,'id':row['id'],'mask_hand':row['mask_hand'],'missing_points':missing})
    assert len(checks)==5
    report={'status':'PASS','all_metadata_rows':855,'input_exact_to_previously_independent_verified':True,
       'window_metadata_independent_oracle':True,'masked_hand_raw_reconstruction':checks,
       'model_inferences':0,'limitation':'Other raw inputs reuse prior independent audit because all row tensor SHA and full tensor SHA are exact; not new independent performance evidence.'}
    (U/'EVALUATION_INPUT_INDEPENDENT_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
