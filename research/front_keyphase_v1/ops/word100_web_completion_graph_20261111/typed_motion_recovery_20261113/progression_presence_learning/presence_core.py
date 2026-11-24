from pathlib import Path
import sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
from common import *
N=HERE/'null_candidate_learning'
FAMILIES=[3,4,5,7]

class ProgressionHead(old.LocalRankHead):
    """Same observation input; a separate binary progression output per family.

    Never modifies the old extent ranker or the common100 encoder.
    """
    pass

def bucket(n):
    return 0 if n<=6 else 1 if n<=10 else 2 if n<=15 else 3 if n<=21 else 4

def labeled_data():
    rows=read(N/'TRAINING_INPUT_INDEX.json')['rows']
    masks=read(N/'local_bag_evaluation/RULES_FROZEN.json')['partial_masks']
    assert len(rows)==len(masks)
    data=[]
    for i,(r,m) in enumerate(zip(rows,masks)):
        assert m['bag_index']==i and m['clip_id']==r['clip_id']
        assert all(m[k]==r[k]for k in ['family','split','view','reverse'])
        assert sha(r['path'])==r['sha256']
        z=np.load(r['path']); spans=z['spans']; full=z['y'].astype(bool)
        assert len(z['local_full'])==len(spans)==len(full) and np.isfinite(z['local_full']).all()
        partial=np.zeros(len(full),bool)
        for c in m['candidates']:
            assert 0<=c['candidate_index']<len(full)
            partial[c['candidate_index']]=True
        assert not (full&partial).any()
        hold=r['origin']=='reviewed_local_hold'
        known=np.ones(len(full),bool) if hold else full|partial
        y=np.zeros(len(full),np.float32) if hold else (full|partial).astype(np.float32)
        data.append(dict(x=z['local_full'],y=y,known=known,full=full,partial=partial,
                         spans=spans,buckets=np.array([bucket(b-a+1)for a,b in spans])))
    return rows,data
