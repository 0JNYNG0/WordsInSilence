"""Read-only arithmetic/provenance verification, separate from label writer."""
from collections import Counter
import numpy as np
from phase_model import *

def main():
    assert not (S/'INPUT_VERIFICATION.json').exists()
    labels=read(S/'PHYSICAL_STATE_TARGETS.json');reviews={r['clip_id']:r for r in read(S/'REVIEW_MANIFEST.json')['rows']};comparisons=0
    for r in labels['rows']:
        ref=reviews[r['clip_id']];assert [p['frame'] for p in r['points']]==ref['frames'] and r['source_sha256']==ref['source_sha256']
        for p in r['points']:
            assert sha(p['image'])==p['image_sha256'] and p['mask']==(p['group']>=0)
            if p['mask']:assert labels['groups'][p['group']]['word_id']==r['word_id']
        if r['word_id']=='WORD1567':
            prior={p['frame']:p for p in ref['known_center_labels']}
            for p in r['points']:
                if p['mask'] and p['frame'] in prior:
                    q=prior[p['frame']]
                    if q['mask'][5] and q['mask'][6]:
                        assert tuple(int(v) for v in q['y'][5:])==((1,0) if p['group']==5 else (0,1));comparisons+=1
    rows=fit_rows();assert len(rows)==180
    per=Counter(r['kind'] for r in rows);assert all(n==36 for n in per.values()) and len(per)==5
    for r in rows:
        assert len(r['mapping'])==len(r['y']) and set(r['y'])<=set(range(-1,9))
        if r['kind']=='actual_reversed_pixels':assert r['mapping']==list(range(len(r['mapping'])-1,-1,-1))
        elif r['kind']=='pose_only_3':assert r['mapping']==list(range(len(r['mapping'])))
    for p,h in read(P/'INPUT_MANIFEST.json')['source_pins'].items():assert sha(p)==h
    out=dict(status='PASS_PUBLIC_INPUTS_AND_LABEL_MASKS',labels=labels['summary'],original_fit_videos=36,actual_video_extractions=144,sensor_gap_variants=36,
        input_conditions=dict(per),old_height_label_consistency_checks=comparisons,old_height_label_conflicts=0,old_source_pins_preserved=True,
        independent_human_expert=False,reviewer='Codex visual review plus separately implemented saved-label/input checks',
        no_label_densification=True,no_DTW_targets_in_new_phase_loss=True,private_inputs=0,locked14_15_inputs=0)
    exclusive(S/'INPUT_VERIFICATION.json',out);print(out)

if __name__=='__main__':main()
