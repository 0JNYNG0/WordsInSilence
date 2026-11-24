from audit import *

# All twelve sheets were directly inspected before any D prediction.
# Only frames actually displayed in the sheet are selected by these bounds.
BOUNDS={
('0739',11):[(0,52,68),(1,71,81),(2,84,106)],
('0739',12):[(0,43,59),(1,63,77),(2,79,101)],
('0739',13):[(0,60,76),(1,80,93),(2,96,114)],
('1133',11):[(3,48,62),(4,65,75)],
('1133',12):[(3,59,78),(4,81,85)],
('1133',13):[(3,48,68),(4,71,85)],
('1567',11):[(5,45,96)],('1567',12):[(5,45,91)],('1567',13):[(5,35,90)],
('2523',11):[(6,43,57),(7,59,76)],
('2523',12):[(6,44,56),(7,58,78)],
('2523',13):[(6,54,62),(7,64,81)],
}

def main():
    assert not (D/'EVALUATION.json').exists() and not (D/'predictions').exists()
    manifest=read(D/'DEV_REVIEW_MANIFEST.json');bg={r['clip_id']:r for r in read(T/'BACKGROUND_TARGETS.json')['rows']};rows=[]
    for r in manifest['rows']:
        cid=r['clip_id'];indices=set(r['indices']);points={}
        for g,lo,hi in BOUNDS[(r['word_id'][4:],r['person'])]:
            for f in sorted(indices):
                if lo<=f<=hi:assert f not in points;points[f]=dict(frame=f,action_family=g,mask=True,provenance='direct_evaluation_only_chronological_sheet_review')
        rest={p['frame'] for p in bg[cid]['points'] if p['mask']}|{0,r['frames']-1}
        assert not rest&points.keys(),(cid,rest&points.keys())
        for f in rest:points[f]=dict(frame=f,action_family=8,mask=True,provenance='previous_direct_rest_review_or_current_sheet_endpoint')
        assert sha(r['sheet'])==r['sheet_sha256'] and r['person'] in [11,12,13]
        rows.append(dict(clip_id=cid,word_id=r['word_id'],person=r['person'],points=sorted(points.values(),key=lambda p:p['frame']),unassigned_reviewed_frames=sorted(indices-points.keys()),sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],full_sheet_visually_reviewed=True))
    exclusive(D/'DEV_ACTION_TARGETS.json',dict(at=clock(),rows=rows,known=sum(len(r['points']) for r in rows),action_known=sum(p['action_family']<8 for r in rows for p in r['points']),
      evaluation_only=True,candidate_predictions_seen=False,old_labels_modified=False,new_training_use=False,
      limitation='Same Codex sparse physical event-family review, not independent expert truth. Exposed DEV cannot measure final generalization. Unassigned is unknown. No dense interpolation or mandatory hold/count.',
      pins={str(p):sha(p) for p in [D/'DEV_REVIEW_INTENT.json',D/'DEV_REVIEW_MANIFEST.json',D/'ACTION_TARGETS.json',D/'label_dev_review.py']}))
    print('Frozen12evaluation-only reviews; known',sum(len(r['points']) for r in rows))
if __name__=='__main__':main()
