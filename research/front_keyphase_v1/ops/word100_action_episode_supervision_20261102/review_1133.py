from audit import *
SELECTED={
1:[[49,51,54,56,58],[60,63,64,65,67,68,69,72,73,74,76,77,78,81]],
2:[[45,47,50,52,53,54,56,59],[61,63,64,65,66,68,70,72,74]],
3:[[50,51,53,55,57,58,60],[64,65,67,68,69,70,72,74,77,78,79]],
4:[[74,76,78,80,81,82,84],[88,91,93,95,97,98,99,101,102,103]],
5:[[44,46,48,51,52,53,55,57],[60,62,64,66,68,70]],
6:[[40,42,44,46,48,49],[54,55,57,59,60]],
7:[[46,47,48,49,50,52,54,56,58,60],[62,63,65,67,69,71,72,73]],
8:[[72,74,77,79,80,82,84,87],[89,91,92,93,94,97,99,102]],
9:[[49,50,51,53,54,55,57],[59,60,62,63,64,66,68]],
}
def main():
    manifest=read(D/'REVIEW_MANIFEST.json');rows=[]
    for person,sets in SELECTED.items():
        cid=f'NIA_SL_WORD1133_REAL{person:02}_F';r=next(r for r in manifest['rows'] if r['clip_id']==cid);reviewed={p['frame'] for p in r['frames']};used=set();pts=[]
        for group,frames in enumerate(sets,3):
            for f in frames:
                assert f in reviewed and f not in used;used.add(f);pts.append(dict(frame=f,action_family=group,mask=True))
        assert sha(r['sheet'])==r['sheet_sha256'];rows.append(dict(clip_id=cid,person=person,word_id='WORD1133',sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],points=sorted(pts,key=lambda p:p['frame']),unassigned_reviewed_frames=sorted(reviewed-used),full_sheet_visually_reviewed=True))
    exclusive(D/'REVIEW_1133.json',dict(at=clock(),rows=rows,action_families=[dict(id=3,name='right_hand_traversal_along_projected_left_arm',description='Observed arm-traversal process; actual direction is separate.'),dict(id=4,name='hand_near_left_wrist_after_traversal',description='Endpoint evidence after travel. A hold duration is NOT a language requirement.')],
      review='Same Codex chronological RGB, not independent KSL expert truth',labels='Sparse directly seen process witnesses, no dense interpolation or exact onset/offset. Boundaries and return frames remain unassigned.',
      source_limit='Supports projected arm travel and endpoint relation. Does not prove palm surface contact, pressure, exact length or mandatory pause. No user/DEV corrections.',whole_goal_complete=False))
    print('Reviewed9FIT clips; witnesses',sum(len(r['points']) for r in rows))
if __name__=='__main__':main()
