"""Sparse pixel review by Codex; not independent linguistic annotation."""
from pathlib import Path
import sys
T=Path(__file__).resolve().parent;O=T.parent
sys.path.insert(0,str(O/'word100_learned_motion_invariance_20261102'))
from prepare_real_fit import read,sha,exclusive,clock

def main():
    review=read(T/'BACKGROUND_REVIEW.json');rows=[]
    # Viewed16sheets. This point has elevated/bent arms relative to adjacentrest;
    # do not turn an ambiguous transition into a rest training label.
    uncertain={('NIA_SL_WORD1133_REAL01_F',0)}
    for r in review['rows']:
        points=[]
        for p in r['samples']:
            assert sha(p['path'])==p['sha256'];known=(r['clip_id'],p['frame']) not in uncertain
            points.append(dict(**p,group=9 if known else None,mask=known,
              reason='Visible relaxed lowered arms/hands at body sides; broad physical rest only' if known else 'Possible movement/transition; no rest target'))
        rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],role=r['role'],points=points))
    exclusive(T/'BACKGROUND_TARGETS.json',dict(at=clock(),rows=rows,reviewer='Codex visual inspection of16sheets192sourceframes',
      independent_expert=False,fit_known=sum(p['mask'] for r in rows if r['role']=='fit' for p in r['points']),
      development_known=sum(p['mask'] for r in rows if r['role']=='development' for p in r['points']),
      no_interpolation=True,no_detector_based_rest_labels=True,no_requirement_to_show_rest_hands_at_runtime=True,
      source_review_sha256=sha(T/'BACKGROUND_REVIEW.json')))
    print('191known sparse rest frames;1uncertain masked')

if __name__=='__main__':main()
