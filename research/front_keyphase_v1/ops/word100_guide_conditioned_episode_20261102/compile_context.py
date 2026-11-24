"""Compile actually reviewed public FIT witnesses; never fill temporal gaps."""
from pathlib import Path
import sys,copy,collections
E=Path(__file__).resolve().parent; D=E.parent/'word100_action_episode_supervision_20261102'
sys.path.insert(0,str(D));from audit import read,exclusive,sha,clock

# Each tuple contains directly reviewed approach/return and rest frames.
# Unlisted candidates remain unknown, including possible late main actions.
REVIEW={
'WORD0739':[
([40,43,94,97,99],[]),([36,39,102,105,109],[112]),
([47,50,106,109,112],[115]),([63,136,140],[143]),
([35,95,98],[]),([42,45,49,110,113,117],[120]),
([32,35,82,84,87],[92]),([36,39,101,104],[108,111]),
([30,33,87,90],[96,99])],
'WORD1133':[
([40,42,45,85,87,90],[92]),([38,41,43,81,83,86],[36,88]),
([43,46,86,89,91],[41,94,96]),([65,67,69,110,112,114],[]),
([37,39,77,79,82],[35,84,86]),([33,35,37,68,70,71,73],[75]),
([35,37,39,78,80,82,84],[]),([59,62,64,67,107,109,112],[114,117]),
([37,39,41,43,72,74,76],[35,80])],
'WORD1567':[
([47,50,106,109],[]),([38,107,110],[35]),([33,36,90,93,96],[30]),
([53,56,128,131],[]),([40,43,99,102],[37,105]),([41,44,46,99,102],[38]),
([29,32,83,85],[88]),([52,55,58,121,124],[]),([48,51,111,114],[45])],
'WORD2523':[
([25,27,61,63,65],[67,69]),([40,42,86,88,91],[93]),([49,51,90,92,94],[96]),
([68,70,116,118,121],[123]),([48,50,86,88,90],[92]),([50,90,92,94],[96]),
([30,67,69,71],[73]),([81,83,85],[87]),([51,88,90,92],[49,94,96])]
}

def main():
    assert not (E/'CONTEXT_TARGETS.json').exists()
    old=read(D/'ACTION_TARGETS.json'); manifest=read(E/'CONTEXT_REVIEW_MANIFEST.json')
    lookup={r['clip_id']:r for r in old['rows']}; rows=[];reviewed=[]
    pins={str(p):sha(p) for p in [D/'ACTION_TARGETS.json',E/'PLAN.json',E/'CONTEXT_REVIEW_MANIFEST.json',E/'compile_context.py']}
    for sheet in manifest['sheets']:
        assert sha(sheet['path'])==sheet['sha256'];pins[sheet['path']]=sheet['sha256']
    for m in manifest['rows']:
        cid=m['clip_id']; original=lookup[cid];assert 1<=m['person']<=9
        context,rest=REVIEW[m['word_id']][m['person']-1]
        candidates=set(m['before']+m['after']);assigned=set(context+rest)
        assert len(assigned)==len(context)+len(rest) and assigned<=candidates
        previous={p['frame']:p for p in original['points']};assert not assigned&previous.keys()
        frames={p['frame']:p for p in m['frames']}
        for p in frames.values():assert sha(p['path'])==p['sha256']
        points=copy.deepcopy(original['points'])
        for label,ids in [(9,context),(8,rest)]:
            for f in ids:points.append(dict(frame=f,action_family=label,mask=True,provenance='direct_chronological_context_sheet_review',image=frames[f]['path'],image_sha256=frames[f]['sha256']))
        result=dict(original,points=sorted(points,key=lambda p:p['frame']))
        result['unassigned_reviewed_frames']=sorted(candidates-assigned)
        rows.append(result)
        reviewed.append(dict(clip_id=cid,person=m['person'],word=m['word_id'],approach_or_return=context,rest=rest,unknown=sorted(candidates-assigned)))
        assert all(next(p for p in points if p['frame']==q['frame'])==q for q in original['points'])
    families=copy.deepcopy(old['families'])+[dict(id=9,name='observed_approach_or_return',description='Observed approach/return movement around the sampled action; not rest, not an error, not proof of linguistic irrelevance. Direction-neutral for reverse source mapping.')]
    result=dict(at=clock(),rows=rows,families=families,pins=pins,review=reviewed,
      reviewer='Same Codex actual chronological RGB-sheet review; not independent KSL expert annotation.',
      scope='FIT01-09 only,36 clips12sheets. Preserve all850old sparse labels. Candidate temporal bounds are NOT labels. No dense interpolation; unknown remains unknown. No DEV/H/private review.',
      observations='Some candidate frames after the last prior action witness visibly continue the action. They remain unknown, never automatically return. Main/return boundary and near-rest settling are withheld.',
      known=sum(len(r['points']) for r in rows),added_context=sum(len(r['approach_or_return']) for r in reviewed),added_rest=sum(len(r['rest']) for r in reviewed),unknown_candidates=sum(len(r['unknown']) for r in reviewed),whole_goal_complete=False)
    exclusive(E/'CONTEXT_TARGETS.json',result)
    exclusive(E/'TARGETS_VERIFIED.json',dict(at=clock(),status='PASS',reviewed_clips=36,reviewed_sheets=12,old_known_preserved=old['known'],old_action_preserved=old['action_known'],new_known=result['known'],added_context=result['added_context'],added_rest=result['added_rest'],unknown_candidates=result['unknown_candidates'],candidate_count=sum(len(m['before'])+len(m['after']) for m in manifest['rows']),new_dense_labels=0,collisions=0,person_split=True,all_review_images_hash_checked=True,verification='Same Codex separate structural check, not independent annotation'))
    print({k:result[k] for k in ['known','added_context','added_rest','unknown_candidates']})
if __name__=='__main__':main()
