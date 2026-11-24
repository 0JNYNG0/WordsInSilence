"""A versioned FIT-only annotation overlay; never rewrites old evidence."""
from prepare_review import *

def main():
    assert not (U/'ANNOTATION_OVERLAY.json').exists()
    checked=read(U/'VERIFIED.json'); decisions=read(U/'REVIEW_DECISIONS.json')
    assert checked['status']=='NATIVE_FRAME_AND_ANNOTATION_STRUCTURE_VERIFIED'
    for p,h in checked['pins'].items(): assert sha(p)==h
    olda=read(D/'ACTION_TARGETS.json'); oldr=read(E/'CONTEXT_TARGETS.json')
    byrole={r['clip_id']:r for r in oldr['rows']}; additions={r['clip_id']:r for r in decisions['rows']}
    c=decisions['correction_candidates'][0]
    assert sha(c['full_image'])==c['full_image_sha256']
    # Explicit visual judgement recorded after full-resolution source inspection.
    correction=dict(c, status='RESOLVED_FOR_SEPARATE_PHYSICAL_EVENT_TRAINING_OVERLAY',
        evidence='Full native f62: fingers flexed, hand beside chin, rather than the open-hand face presentation of family0. Chronological f63-67 continue relocation. Family1 includes this process. Source-index/JPEG identity independently checked in VERIFIED.json.',
        reviewer='Same Codex visual review, not independent KSL teacher or official dataset correction.',
        uncertainty='No exact fingertip/contact/linguistic correctness label. Other ambiguous boundary frames remain action-family unknown.')
    rows=[]; nnew=0; corrected=0
    for a in olda['rows']:
        assert a['person'] in range(1,10)
        fam={p['frame']:p['action_family'] for p in a['points'] if p['mask']}
        rol={p['frame']:(1 if p['action_family']<8 else 2 if p['action_family']==8 else 0) for p in byrole[a['clip_id']]['points'] if p['mask']}
        points={f:dict(frame=f,action_family=fam.get(f,-1),role=rol.get(f,-1),provenance='existing_derived_FIT_annotation') for f in set(fam)|set(rol)}
        for p in additions.get(a['clip_id'],{}).get('points',[]):
            assert p['frame'] not in points
            points[p['frame']]=dict(p);nnew+=1
        if a['clip_id']==c['clip_id']:
            assert points[62]['action_family']==0 and points[62]['role']==1
            points[62].update(action_family=1,provenance='resolved_native_RGB_correction_0_to_1');corrected+=1
        rows.append(dict(clip_id=a['clip_id'],word_id=a['word_id'],person=a['person'],source=a['source'],source_sha256=a['source_sha256'],points=[points[f] for f in sorted(points)]))
    assert nnew==172 and corrected==1
    save(U/'CORRECTION_RESOLUTION.json',correction)
    save(U/'ANNOTATION_OVERLAY.json',dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),rows=rows,
        new_points=172,corrected_old_points=1,old_files_unchanged=True,unknown_is_not_negative=True,
        scope='Sparse physical event/process supervision only, FIT01-09. No KSL educational acceptability or exact onset/offset claims.',
        pins={str(p):sha(p) for p in [D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',U/'REVIEW_DECISIONS.json',U/'VERIFIED.json',U/'CORRECTION_RESOLUTION.json',U/'build_overlay.py']},
        whole_goal_complete=False))
    print('Separate overlay:36 FIT clips,172 new points,1 documented correction;old records unchanged')

if __name__=='__main__':main()
