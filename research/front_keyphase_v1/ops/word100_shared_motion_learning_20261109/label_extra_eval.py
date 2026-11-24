"""Fixed visual review; never imported by training. No model outputs consulted."""
from core import *
N=None
# Row-major: IDs11,12,13; times35%,50%,65%; features R_index_only,R_open_four,L_open_four.
labels={
'WORD0003':[[N,N,N],[1,0,0],[1,0,0], [1,0,0],[1,0,0],[1,0,0], [1,0,0],[1,0,0],[1,0,0]],
'WORD0071':[[0,1,1],[1,0,N],[1,0,0], [N,N,N],[N,0,0],[1,0,0], [N,N,N],[0,1,1],[1,0,0]],
'WORD0500':[[N,0,N],[1,0,0],[1,0,0], [N,N,N],[1,0,0],[1,0,0], [1,0,N],[1,0,0],[1,0,0]],
'WORD0678':[[0,N,N],[N,N,N],[1,0,N], [0,N,N],[0,N,N],[1,0,N], [N,N,N],[0,N,N],[1,0,N]],
'WORD0949':[[N,0,N],[1,0,N],[N,0,N], [N,N,N],[1,0,N],[N,0,N], [N,0,N],[1,0,N],[N,0,N]],
'WORD1139':[[N,0,1],[N,0,1],[N,0,1], [N,N,N],[N,0,1],[N,0,1], [N,N,N],[N,0,N],[N,0,1]],
'WORD1290':[[0,1,1],[0,1,1],[0,1,1], [0,1,1],[0,1,1],[0,1,1], [N,N,N],[0,1,1],[0,1,1]],
'WORD2551':[[0,0,N],[0,0,N],[0,N,N], [N,0,N],[0,N,N],[0,N,N], [N,N,N],[0,0,N],[0,N,N]],
}
assert not (U/'EVALUATION_COMPLETE.json').exists()
rows=read(U/'EXTRA_EVAL_RGB.json')['rows'];counts={w:0 for w in labels}
for r in rows:
 w=r['word_id'];r['targets']=labels[w][counts[w]];counts[w]+=1
assert all(n==9 for n in counts.values())
write(U/'EXTRA_EVAL_LABELS.json',dict(scope='72 public images from24 DEV clips, reviewed before candidate predictions; same Codex visual review, not independent expert gold',features=['R_index_only','R_open_four','L_open_four'],rows=rows,policy='Null excludes ambiguous or resting/out-of-frame hands from annotation truth. Foreshortened index, partially flexed opening, and edge-only indistinct fingers stay null. Open-four false is not a fist label. Targets are visual features, not educational sign correctness.',potential_cells=216,known_cells=sum(x is not None for r in rows for x in r['targets']),null_cells=sum(x is None for r in rows for x in r['targets']),training_use=False,model_selection_use=False))
print('Evaluation visual labels frozen before predictions.')
