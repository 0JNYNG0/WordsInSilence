"""Body-progression-only input contract; no change to the action extent ranker."""
from pathlib import Path
import sys
_unit=Path(__file__).resolve().parent
sys.path.insert(0,str(_unit.parent/'progression_presence_learning'))
import presence_core as upstream
from presence_core import *
B=_unit
S=B.parent/'paired_static_progression_learning'
# P, N, HERE, PRIOR, OPS, old, ProgressionHead, FAMILIES retain upstream meanings.
source_labeled_data=upstream.labeled_data

def check_pins(values):
    for path,expected in values.items():
        assert Path(path).is_file() and sha(path)==expected, path

def check_body_features(x):
    """Explicit hand features must be zero; hidden64 comes from masked encoding."""
    assert x.ndim==2 and x.shape[1]==1160 and np.isfinite(x).all()
    parts=x.reshape(-1,4,290)
    q=parts[:,:,64:164].reshape(-1,4,10,10)
    assert np.count_nonzero(q[:,:,:,4:8])==0
    assert np.count_nonzero(q[:,:,:,9])==0
    assert np.count_nonzero(parts[:,:,164:])==0

def body_relation_indices():
    return np.array([part*290+64+axis*10+j for part in range(4) for axis in range(10) for j in [0,1,2,3,8]])

def labeled_data():
    """Preserve original rows/labels/masks/spans; replace only observation x."""
    rows,data=source_labeled_data();meta=read(B/'INPUTS_COMPLETE.json');items=meta['body_bags']
    assert len(rows)==len(items)==286
    for i,(r,d,m) in enumerate(zip(rows,data,items)):
        assert m['bag_index']==i
        assert all(m[k]==r[k] for k in ['clip_id','person','split','family','view','reverse','origin','source_sha256'])
        assert sha(m['path'])==m['sha256']
        z=np.load(m['path']);x=z['local_body'];check_body_features(x)
        assert x.shape==d['x'].shape and np.array_equal(z['spans'],d['spans'])
        assert np.array_equal(z['y'],d['full'])
        assert np.array_equal(x[:,body_relation_indices()],d['x'][:,body_relation_indices()])
        d['x']=x
    return rows,data

def stop_if_requested():
    # Cooperative pause without overwriting any completed artifacts.
    assert not(B/'STOP_REQUESTED').exists(), 'STOP_REQUESTED present; preserve state and stop'

def preserve_failure(stage):
    import time,traceback
    B.mkdir(parents=True,exist_ok=True)
    text=traceback.format_exc()
    with (B/f'{stage}_FAILURE_{time.time_ns()}.txt').open('x',encoding='utf-8') as f:f.write(text)
    first=B/'FIRST_FAILURE.txt'
    if not first.exists():
        with first.open('x',encoding='utf-8') as f:f.write(stage+'\n'+text)
