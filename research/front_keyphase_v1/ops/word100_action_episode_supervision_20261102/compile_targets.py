from audit import *

def main():
    assert not (D/'ACTION_TARGETS.json').exists()
    manifests={r['clip_id']:r for r in read(D/'REVIEW_MANIFEST.json')['rows']}
    bg={r['clip_id']:r for r in read(T/'BACKGROUND_TARGETS.json')['rows']}
    source={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows'] if r['role']=='fit'}
    rows=[];families=[];pins={}
    for word in ['0739','1133','1567','2523']:
        path=D/f'REVIEW_{word}.json';v=read(path);pins[str(path)]=sha(path);families+=v['action_families']
        for r in v['rows']:
            cid=r['clip_id'];sr=source[cid];mr=manifests[cid]
            assert r['full_sheet_visually_reviewed'] and r['person'] in range(1,10)
            assert sha(r['sheet'])==r['sheet_sha256']==mr['sheet_sha256']
            pts={p['frame']:dict(p,provenance='direct_chronological_sheet_review') for p in r['points']}
            assert len(pts)==len(r['points'])
            rest={p['frame'] for p in bg[cid]['points'] if p['mask']}
            rest|={mr['frames'][0]['frame'],mr['frames'][-1]['frame']}
            assert not rest&pts.keys(),('rest/action contradiction',cid,rest&pts.keys())
            for f in rest:pts[f]=dict(frame=f,action_family=8,mask=True,provenance='previous_direct_rest_review_or_current_sheet_endpoint')
            assert all(0<=f<sr['frames'] for f in pts)
            for p in mr['frames']:assert sha(p['path'])==p['sha256']
            pins[r['sheet']]=r['sheet_sha256'];pins[sr['source']]=sr['source_sha256']
            rows.append(dict(clip_id=cid,person=r['person'],word_id=r['word_id'],label=sr['label'],source=sr['source'],source_sha256=sr['source_sha256'],frames=sr['frames'],points=sorted(pts.values(),key=lambda p:p['frame']),unassigned_reviewed_frames=r['unassigned_reviewed_frames']))
    families.append(dict(id=8,name='observed_rest',description='Directly observed resting posture, not an assumption from unknown/missing data.'))
    assert len(rows)==36 and sorted(f['id'] for f in families)==list(range(9))
    for g in range(8):assert len({r['person'] for r in rows if any(p['action_family']==g for p in r['points'])})==9
    # Source descriptions remain evidence with their original limitations.
    gids={0:[0],1:[1],2:[2],3:[3],4:[4],5:[5,6,7,8,9],6:[10],7:[11,12]}
    gs=read(P/'GUIDE_SUPPORT.json')['guides']
    for f in families:
        f['guide_indices']=gids.get(f['id'],[])
        f['guide_ids']=[gs[i]['canonical_stage_id'] for i in f['guide_indices']]
        f['source_descriptions']=[dict(description=gs[i]['description'],source_cards=gs[i]['source_cards']) for i in f['guide_indices']]
    obj=dict(at=clock(),rows=rows,families=families,pins=pins,known=sum(len(r['points']) for r in rows),action_known=sum(p['action_family']<8 for r in rows for p in r['points']),
      supervision='Sparse physical action-family identity, not expert language correctness, exact phase boundaries, mandatory pause/repetition or all-frame truth.',
      reverse_contract='Action-family denotes the content of a source event independent of its playback direction. The wrist endpoint family does not assert after-traversal in edited clips. Forward/reverse correctness is judged separately; no reversed verb is labeled correct. Ambiguous edit seams receive no action label.',
      scope='36 public FIT recordings, recorded IDs01-09; same Codex visual review, not independent annotation; no DEV/H/private input. Unassigned means unknown, never a negative/meaningless frame.',
      new_dense_labels=0,whole_goal_complete=False)
    exclusive(D/'ACTION_TARGETS.json',obj)
    exclusive(D/'TARGETS_VERIFIED.json',dict(at=clock(),status='PASS',actual_reviewed=36,action_witnesses=obj['action_known'],known=obj['known'],families=9,each_action_family_people=9,rest_action_collisions=0,all_image_hashes_checked=True,source_frame_bounds=True,reviewer='Same Codex; separate structural validation, not independent expert truth'))
    print({k:obj[k] for k in ['known','action_known']})
if __name__=='__main__':main()
