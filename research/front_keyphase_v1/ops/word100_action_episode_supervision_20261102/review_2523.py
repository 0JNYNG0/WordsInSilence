from audit import *

# Bounds select actually inspected sheet frames, never the intervening frames.
BOUNDS={1:[(31,38),(40,56)],2:[(47,56),(58,81)],3:[(55,68),(69,84)],4:[(73,85),(90,109)],5:[(54,63),(65,80)],6:[(54,66),(70,84)],7:[(34,45),(47,61)],8:[(42,56),(58,75)],9:[(55,65),(68,82)]}

def main():
    manifest=read(D/'REVIEW_MANIFEST.json');rows=[]
    for person,bounds in BOUNDS.items():
        cid=f'NIA_SL_WORD2523_REAL{person:02}_F';r=next(r for r in manifest['rows'] if r['clip_id']==cid)
        reviewed={p['frame'] for p in r['frames']};used=set();pts=[]
        for group,(lo,hi) in enumerate(bounds,6):
            for f in sorted(reviewed):
                if lo<=f<=hi:
                    assert f not in used;used.add(f);pts.append(dict(frame=f,action_family=group,mask=True))
        assert sha(r['sheet'])==r['sheet_sha256']
        rows.append(dict(clip_id=cid,person=person,word_id='WORD2523',sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],points=pts,unassigned_reviewed_frames=sorted(reviewed-used),full_sheet_visually_reviewed=True))
    exclusive(D/'REVIEW_2523.json',dict(at=clock(),rows=rows,action_families=[
      dict(id=6,name='bilateral_approximation_and_converged_presentation',description='Approximating hands/fingertips and their converged presentation, without claiming exact skin contact.'),
      dict(id=7,name='bilateral_distance_change_and_outward_configuration',description='Change from converged to separated projected hand configuration, including the endpoint. Direction correctness remains separate.')],
      review='Same Codex chronological RGB, not independent KSL expert truth',labels='Sparse directly inspected process witnesses. Ambiguous edges, preparation and return stay unassigned. No dense interpolation.',
      limits='No required hold duration, exact depth/contact, anthropometric position or full language correctness inferred.',whole_goal_complete=False))
    print('Reviewed9FIT clips; witnesses',sum(len(r['points']) for r in rows))
if __name__=='__main__':main()
