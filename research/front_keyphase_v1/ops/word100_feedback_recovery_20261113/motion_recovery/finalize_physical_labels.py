"""Record direct RGB judgments after all 32 first panels + 16 native-completion panels.

No pose value, word identity, model output or numeric movement cutoff generates truth.
The small explicit overrides below are SOURCE ANNOTATIONS, not runtime word exceptions.
"""
from audit_saved import *
from collections import Counter

def main():
    draft=read(U/'PHYSICAL_LABEL_DRAFT.json');rows=[]
    # Both reviewers saw the beginning of the return movement in the last part of
    # these windows. This is not a label that the centre frame itself is moving.
    reviewed_overrides={
      'NIA_SL_WORD2523_REAL03_F:held_posture_review:82':'MM',
      'NIA_SL_WORD2523_REAL06_F:held_posture_review:82':'MM'}
    for r0 in draft['rows']:
        r=dict(r0);pair=reviewed_overrides.get(r['key'],r['provisional_right_left'])
        sampled=sorted(set(r['review_frames']+r.get('completion_native_frames',[])))
        r.update(target='window_contains_wrist_translation',right=pair[0],left=pair[1],
          reviewed_unique_frames=sampled,
          evidence_seconds=[sampled[0]/30,sampled[-1]/30],
          reviewed_every_native_frame=sampled==list(range(sampled[0],sampled[-1]+1)),
          native_window=[max(0,r['frame']-4),r['frame']+4],
          annotation_author='Codex direct RGB review; not human KSL expert annotation',
          scope='Any clearly observed wrist translation relative to torso within this short window; not central-frame velocity, clause membership, sign correctness, rotation or finger state',
          visibility={'R':'wrist and torso visible at reviewed times','L':'wrist and torso visible at reviewed times'},
          boundary_policy='A clearly moving suffix makes the WINDOW M; do not project this label onto every native frame or infer an exact onset',
          rationale={side:('Visible change in wrist placement relative to torso at multiple reviewed native times.'if lab=='M'else'Wrist placement remains visibly stable relative to torso across all available native times; small bodily sway is not called an action.'if lab=='H'else'Small right-wrist change cannot be separated reliably from sway/transition in RGB; not forced into M or H.')for side,lab in zip(['R','L'],pair)},
          label_selected_without_pose_threshold=True)
        if r['key']in reviewed_overrides:
            r['rationale']={s:'Window begins with maintained placement; later frames 84-86 show the start of downward return. M applies to the whole window, not necessarily frame82.'for s in ['R','L']}
            r['independent_agent_review']='Root inspected original evidence and agreed whole-window M/M, no centre-frame claim. Native-completion review by annotation author agrees.'
        if pair=='UM':r['independent_agent_review']='Root independently agreed R=U and L=M on original RGB; later native-completion review preserves this uncertainty.'
        if r['frame']==0:
            r['rationale']={s:'Actual unique native frames0-4 show the wrist maintained at rest; absent negative-time frames and duplicated preview images do not add evidence.'for s in ['R','L']}
        assert all(lab!='H'or r['reviewed_every_native_frame']for lab in pair)
        assert r['person']in range(2,8)
        rows.append(r)
    counts=Counter(lab for r in rows for lab in [r['right'],r['left']]);support=[]
    for side in ['right','left']:
        for label in ['M','H','U']:
            q=[r for r in rows if r[side]==label]
            support.append(dict(side=side,label=label,n=len(q),record_ids=sorted(set(r['person']for r in q)),expressions=sorted(set(r['word_id']for r in q))))
    pins={str(U/'PHYSICAL_REVIEW_MANIFEST.json'):sha(U/'PHYSICAL_REVIEW_MANIFEST.json'),str(U/'PHYSICAL_LABEL_DRAFT.json'):sha(U/'PHYSICAL_LABEL_DRAFT.json')}
    for r in rows:
        for k in ['panel','completion_panel']:
            if k in r:pins[r[k]]=sha(r[k])
    write('PHYSICAL_SOURCE_LABELS.json',dict(status='DIRECT_RGB_LABELS_READY_FOR_ROOT_SPEC_REVIEW',target='window_contains_wrist_translation',rows=rows,counts=dict(counts),support=support,
      scope=dict(windows=len(rows),hand_cells=2*len(rows),source_clips=len(set(r['clip_id']for r in rows)),record_ids=list(range(2,8)),expressions=4,
      frames='Most windows9 native frames/0.267s; clip-start windows5 native frames/0.133s; witnesses are correlated, not independent new videos'),
      annotation_process='32 original panels inspected; all H/U candidates additionally inspected at missing odd native frames in16full-image panels. M only needs an observed movement witness within window; no claim all its intermediate frames have a dense label.',
      legacy_labels_unchanged=True,old_clause_negatives_not_physical_hold=True,physical_U_not_trained_as_H=True,
      panel_pins=pins))
    print(json.dumps(dict(counts=dict(counts),support=support),ensure_ascii=False))

if __name__=='__main__':main()
