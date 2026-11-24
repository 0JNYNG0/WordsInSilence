"""Separate provenance and map checks before any new training."""
import numpy as np
from state_model import *

def main():
    assert not (T/'INPUT_VERIFICATION.json').exists();assert read(T/'PREP_PROGRESS.json')['status']=='COMPLETE'
    b=read(T/'BACKGROUND_TARGETS.json');assert b['fit_known']==143 and b['development_known']==48
    review=read(T/'BACKGROUND_REVIEW.json');assert sha(T/'BACKGROUND_REVIEW.json')==b['source_review_sha256']
    for r in review['rows']:
        assert sha(r['source'])==r['source_sha256']
        for p in r['samples']:assert sha(p['path'])==p['sha256']
    for sheet in review['sheets']:assert sha(sheet['path'])==sheet['sha256']
    plans={r['clip_id']:r for r in read(T/'SOURCE_PLAN.json')['rows']}
    for r in read(T/'LOCAL_FIT_MANIFEST.json')['rows']:
        p=plans[r['clip_id']];n=p['frames'];lo,hi=p['local_reverse_range'];expected=list(range(n));expected[lo:hi+1]=expected[lo:hi+1][::-1]
        assert expected==r['source_frame_indices']==p['source_frame_indices']
        assert sha(r['source'])==r['source_sha256'] and sha(r['path'])==r['video_sha256'] and sha(r['input'])==r['analysis_sha256']
        with np.load(r['input']) as a:
            assert len(a['latent'])==n and np.array_equal(a['frame_id'],np.arange(n)) and np.all(np.diff(a['timestamp_us'])>0)
            assert np.isfinite(a['latent']).all()
    rows=fit_rows();assert len(rows)==216 and {r['person'] for r in rows}==set(range(1,10))
    dev={r['person'] for r in b['rows'] if r['role']=='development'};assert dev=={11,12,13}
    assert not dev&{r['person'] for r in rows}
    counts={str(g):sum(int((r['y']==g).sum()) for r in rows) for g in range(10)}
    for g in range(10):assert {r['person'] for r in rows if (r['y']==g).any()}==set(range(1,10))
    plan=read(T/'FROZEN_TRAINING.json');assert all(sha(p)==h for p,h in plan['pins'].items())
    exclusive(T/'INPUT_VERIFICATION.json',dict(at=clock(),status='PASS',actual_new_local_inputs=36,fit_inputs=216,known_targets=counts,
      source_mappings_preserved=True,unknown_labels_ignored=True,old_pins_preserved=True,fit_dev_record_ids_disjoint=True,
      caveat='Recorded ID split only; does not assert biometric identity independence across every source video. Visual review and arithmetic both Codex, not independent expert labels.'))
    print('PASS216FITinputs191newrestlabels;old pins and maps preserved')

if __name__=='__main__':main()
