"""Direct chronological FIT review, sparse witnesses only; no interpolation."""
from audit import *
SELECTED={
1:[[48,51,53,55,56,59,60],[62,64,67,68,70,72],[75,76,78,80,81,83,86,87,89,91]],
2:[[43,46,49,53,56,57,59,62],[64,66,69,71,72,74,76],[79,82,84,86,89,90,91,92,95,97,99]],
3:[[53,56,59,62,65,68,69],[71,74,75,77,78,80,81,82],[85,88,91,94,97,100,103]],
4:[[70,73,77,80,84,86,87],[91,94,98,100,101,103,105],[108,112,114,115,119,121,122,126,128,129]],
5:[[41,44,46,49,50,52],[55,58,61,64,65,67,68,69],[72,75,78,81,84,85,86,87,90]],
6:[[52,56,59,62,63,66,69],[71,73,76,78,79,81,83],[86,89,92,93,96,98,99,100,103,105,106]],
7:[[37,40,42,45,47,48,50,52],[53,55,58,61,63],[66,69,71,74,76,79,80]],
8:[[43,46,49,52,56,57,59,62],[65,69,72],[75,78,81,82,85,87,88,91,93,95,98]],
9:[[39,42,45,48,51,52],[54,57,58,60,63,64],[69,71,72,75,77,78,81,84]],
}
DEFINITIONS=[
 dict(id=0,name='right_open_hand_face_presentation',description='Open right hand presented in front of face; no precise contact/location linguistic rule inferred.'),
 dict(id=1,name='right_face_torso_relocation_and_cross_forearm_configuration',description='Relocation between face and torso with configuration change and crosswise forearm endpoint. A momentarily upright-looking fist inside this movement remains in this action family.'),
 dict(id=2,name='right_forearm_reorientation_and_upright_configuration',description='Subsequent forearm reorientation and upright endpoint/hold. Not identified solely by seeing an upright fist.')]

def main():
    manifest=read(D/'REVIEW_MANIFEST.json');rows=[]
    for person,sets in SELECTED.items():
        cid=f'NIA_SL_WORD0739_REAL{person:02}_F';r=next(r for r in manifest['rows'] if r['clip_id']==cid);reviewed={p['frame'] for p in r['frames']};used=set();pts=[]
        for group,frames in enumerate(sets):
            for f in frames:
                assert f in reviewed and f not in used;used.add(f);pts.append(dict(frame=f,action_family=group,mask=True))
        assert sha(r['sheet'])==r['sheet_sha256']
        rows.append(dict(clip_id=cid,person=person,word_id='WORD0739',sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],points=sorted(pts,key=lambda p:p['frame']),
          unassigned_reviewed_frames=sorted(reviewed-used),full_sheet_visually_reviewed=True))
    exclusive(D/'REVIEW_0739.json',dict(at=clock(),rows=rows,action_families=DEFINITIONS,reviewer='Same Codex direct chronological RGB, not independent KSL expert',
      labels='Sparse directly viewed action-family witnesses. Unassigned is unknown here, not an error/meaningless frame. No dense onset/offset truth.',
      direction='Families identify the participating action process; direction correctness is separate. Reverse-input target transformation must be reviewed, not assumed from a frame mapping.',
      text_support='Canonical visual descriptions + partial sourced word clauses. Do not claim every clause is linguistically required or every spatial tolerance known.',whole_goal_complete=False))
    print('Reviewed9FIT clips, sparse action witnesses',sum(len(r['points']) for r in rows))
if __name__=='__main__':main()
