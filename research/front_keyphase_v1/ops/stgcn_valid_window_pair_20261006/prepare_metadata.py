"""Codex independent input audit and thin binding to Claude's index metadata API.

Reuses the frozen input cache. No model inference, extraction or learning.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from window_bridge import window_valid_from_indices

U = Path(__file__).resolve().parent
OLD = U.parent / 'stgcn_execution_ready_20261005'
OLD_RUN = U.parent.parent / 'runs/stgcn_full_path_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def indices_for(t):
    assert isinstance(t, int) and t > 0
    return [-1] * (60-t) + list(range(t)) if t <= 60 else [j*(t-1)//59 for j in range(60)]


def main():
    plan = json.loads((U/'FROZEN_EXPERIMENT.json').read_text())
    meta = json.loads((OLD/'prepared/TRAINING_META.json').read_text(encoding='utf-8'))
    assert sha(OLD/'prepared/training.npz') == plan['comparison_source']['training_cache_sha256'] == meta['cache_sha256']
    assert sha(OLD/'TRAIN_SPLIT.json') == plan['pins']['train_split']
    assert sha(OLD_RUN/'EVALUATION_ROWS.json') == plan['comparison_source']['rows_sha256']
    assert len(meta['rows']) == len(meta['ids']) == len(meta['origins']) == 2346
    indices = [indices_for(int(r['end_exclusive']-r['start'])) for r in meta['rows']]
    valid = window_valid_from_indices(indices)
    # Independent mask oracle: a window is real iff its right edge exceeds
    # the prepended padding length. For long samples all six windows are real.
    oracle = np.asarray([[10*(j+1)>max(0,60-(r['end_exclusive']-r['start'])) for j in range(6)] for r in meta['rows']])
    assert np.array_equal(valid.numpy(), oracle)
    with np.load(OLD/'prepared/training.npz', allow_pickle=False) as z:
        assert z['x'].shape == (2346,6,2,10,27)
        assert np.isfinite(z['x']).all()
        assert np.array_equal(z['y'], np.array([r['y'] for r in meta['rows']]))
        assert np.array_equal(z['weight'], np.array([r['weight_count'] for r in meta['rows']]))
        assert meta['ids'] == [r['id'] for r in meta['rows']]
        assert meta['origins'] == [r['origin'] for r in meta['rows']]
        fit = np.flatnonzero(z['fit']); val = np.flatnonzero(z['internal_val'])
        assert len(fit)==2089 and len(val)==257 and int(z['primary'][val].sum())==233
        assert len(set(fit)&set(val))==0 and len(set(fit)|set(val))==2346
        assert all(meta['rows'][i]['partition']=='fit' for i in fit)
        assert all(meta['rows'][i]['partition']=='internal_val' for i in val)
        assert set(meta['origins'][i] for i in fit).isdisjoint(meta['origins'][i] for i in val)
    old_rows = json.loads((OLD_RUN/'EVALUATION_ROWS.json').read_text(encoding='utf-8'))['rows']
    diagnostics = json.loads((OLD_RUN/'EVALUATION_INPUT_DIAGNOSTICS.json').read_text())
    assert len(old_rows)==len(diagnostics)==855
    eval_indices=[]
    for r,d in zip(old_rows,diagnostics):
        assert (r['id'],r['group'],r['condition'])==(d['id'],d['group'],d['condition'])
        ix=indices_for(d['prepare_diag']['T'])
        assert ix==d['prepare_diag']['indices']
        eval_indices.append(ix)
    eval_valid = window_valid_from_indices(eval_indices)
    eval_oracle = np.asarray([[10*(j+1)>max(0,60-d['prepare_diag']['T']) for j in range(6)] for d in diagnostics])
    assert np.array_equal(eval_valid.numpy(),eval_oracle)
    assert valid.dtype==eval_valid.dtype==torch.bool and valid.all(dim=1).sum()>=1
    out=U/'WINDOW_METADATA.npz'
    assert not out.exists(), 'Metadata already prepared; verify stored audit instead of overwriting'
    np.savez(out, train_indices=np.asarray(indices,np.int64), train_valid=valid.numpy(),
             eval_indices=np.asarray(eval_indices,np.int64), eval_valid=eval_valid.numpy())
    result={'status':'PASS','training_rows':2346,'fit_rows':2089,'internal_rows':257,
            'final_rows':855,'metadata_sha256':sha(out),'source_cache_sha256':meta['cache_sha256'],
            'source_metadata_sha256':sha(OLD/'prepared/TRAINING_META.json'),
            'old_evaluation_diagnostics_sha256':sha(OLD_RUN/'EVALUATION_INPUT_DIAGNOSTICS.json'),
            'fit_empty_windows':int((~valid[fit]).sum()),'all_empty_windows':int((~valid).sum()),
            'final_empty_windows':int((~eval_valid).sum()),'oracle_equal':True,
            'coordinates_labels_weights_ids_origins_unchanged':True,'coordinate_or_hand_mask_inference':False}
    (U/'METADATA_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
