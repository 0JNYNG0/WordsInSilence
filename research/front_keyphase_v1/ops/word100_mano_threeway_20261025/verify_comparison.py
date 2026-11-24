"""Separate stored-output audit, by same Codex author; not an independent human review."""
from pathlib import Path
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def main():
    f=json.loads((U/'FROZEN_PROTOCOL.json').read_text('utf-8'));summary=json.loads((U/'COMPARISON.json').read_text('utf-8'))
    frames=base_points=cross_points=hp=htp=0
    for m,row in zip(f['manifest'],summary['rows']):
        with np.load(m['prepared']) as b, np.load(U/'comparisons'/(m['clip_id']+'.npz')) as c:
            assert np.array_equal(b['frame_id'],c['frame_id'])
            assert np.array_equal(b['timestamp_us'],c['timestamp_us'])
            v=b['baseline_valid'];frames+=len(v);base_points+=int(v.sum());cross_points+=int(v[m['crossing_frames']].sum())
            for name in ('H','HT'):
                assert np.array_equal(c[name+'_xy'][v],b['baseline_xy'][v])
                assert np.array_equal(c[name+'_evidence'],v)
                added=c[name+'_inferred'];valid=c[name+'_valid'];assert not (added&v).any()
                assert np.array_equal(valid, v|added)
                q=c[name+'_xy'];r=c['roi_box'];inside=(q[...,0]>=r[0])&(q[...,0]<r[2])&(q[...,1]>=r[1])&(q[...,1]<r[3])
                assert inside[valid].all() and np.isfinite(q[valid]).all()
                assert np.array_equal(np.where(valid[...,None],q,0)[v],b['baseline_xy'][v])
            assert not (c['HT_inferred']&~c['H_inferred']).any()
            assert int(c['H_inferred'].sum())==row['H_added_estimate_point_frames']
            assert int(c['HT_inferred'].sum())==row['HT_added_estimate_point_frames']
            hp+=int(c['H_inferred'].sum());htp+=int(c['HT_inferred'].sum())
    assert frames==812
    pins=[]
    for p,h in f['protected'].items():
        assert sha(p)==h,p;pins.append(p)
    initial=json.loads((U/'ASSET_RECEIPT.json').read_text('utf-8'))
    # Keep receipt structure intact; archive identity is additionally pinned explicitly here.
    archive=Path(r'C:\Users\local-user\Downloads\mano_v1_2.zip')
    assert sha(archive)=='50976831790ea9657d8110e0c94e50e90eaf35cd76169f0b27e5d32f3fcd951f'
    report={'status':'PASS_NUMERIC_PRESERVATION_NOT_ANATOMICAL_ACCURACY','frames_preserved':frames,
            'baseline_point_frames_exact':base_points,'crossing_point_frames_exact':cross_points,
            'H_additional_estimated_points':hp,'HT_additional_estimated_points':htp,
            'protected_unchanged':len(pins),'original_MANO_archive_unchanged':True,
            'no_inferred_points_promoted_to_original_evidence':True,
            'authorship':'Separate saved-output checks by same Codex; no independent expert annotation',
            'training':0,'C_T3F_changed':False}
    (U/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
