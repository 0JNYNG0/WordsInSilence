from audit import *

# Every sheet was actually inspected chronologically. These bounds SELECT
# already displayed witnesses only; they do not generate dense frame labels.
BOUNDS={1:(55,98),2:(45,100),3:(44,82),4:(67,117),5:(49,90),6:(52,91),7:(37,75),8:(65,111),9:(60,102)}

def main():
    manifest=read(D/'REVIEW_MANIFEST.json');rows=[]
    for person,(lo,hi) in BOUNDS.items():
        cid=f'NIA_SL_WORD1567_REAL{person:02}_F';r=next(r for r in manifest['rows'] if r['clip_id']==cid)
        reviewed={p['frame'] for p in r['frames']};used={f for f in reviewed if lo<=f<=hi}
        assert sha(r['sheet'])==r['sheet_sha256']
        rows.append(dict(clip_id=cid,person=person,word_id='WORD1567',sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],
          points=[dict(frame=f,action_family=5,mask=True) for f in sorted(used)],unassigned_reviewed_frames=sorted(reviewed-used),full_sheet_visually_reviewed=True))
    exclusive(D/'REVIEW_1567.json',dict(at=clock(),rows=rows,action_families=[dict(id=5,name='alternating_two_hand_process_and_configurations',description='Two closed-hand/thumb configurations alternate vertically through crossing and turning; includes observed endpoint configuration without requiring a hold.')],
      review='Same Codex chronological RGB, not independent KSL expert truth',labels='Sparse directly seen process witnesses. No dense interpolation, prescribed first side, fixed repeat count, or exact timing.',
      separation='Direction and observed high/low configurations remain separate outputs. This family alone does not certify a full cycle, sufficient movement, exact finger shape or correct language.',whole_goal_complete=False))
    print('Reviewed9FIT clips; witnesses',sum(len(r['points']) for r in rows))
if __name__=='__main__':main()
