"""Read-only numerical/source verification; writes only its new fresh report."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
from collections import Counter
import numpy as np
from PIL import Image
import prepare_fresh as prep


def main():
    u=prep.U; selected=prep.read(u/'SELECTION.json'); inputs=prep.read(u/'INPUTS.json')
    visual={r['alias']:r for r in prep.read(u/'VISUAL_INDEX.json')['rows']}
    done=prep.read(u/'PREP_COMPLETED.json'); adapter=prep.adapters(); checks=0; source_hashes=[]; entries=[]
    assert len(inputs['rows'])==len(selected['rows'])==32; checks+=1
    assert not (u/'EXECUTION_LOCK.json').exists() and prep.read(u/'CURRENT_EXECUTION.json')['active'] is None; checks+=1
    for path,digest in selected['pins'].items():
        assert prep.sha(path)==digest,path;checks+=1
    old=prep.read(prep.PRIOR/'SELECTION.json'); old_hashes={r['archive_source']['sha256'] for r in old['rows']}
    for row,frozen in zip(inputs['rows'],selected['rows']):
        alias=row['alias']; assert alias==frozen['alias']; checks+=1
        assert all(row[k]==frozen[k] for k in ('clip_id','word_id','person','role','record','record_sha256','observation','observation_sha256','box'));checks+=1
        checkpoint=prep.read(u/'prepared'/(alias+'.json'))
        for path,digest in checkpoint['pins'].items():
            assert prep.sha(path)==digest,path;checks+=1
        assert row['source_sha256']==frozen['archive_source']['sha256'] and row['source_sha256'] not in old_hashes;checks+=1
        source_hashes.append(row['source_sha256'])
        with np.load(row['input'],allow_pickle=False) as z:
            assert set(z.files)=={'x','mask','rgb','window_features','window_starts','window_center_us','frame_id','timestamp_us'};checks+=1
            n=row['frames'];assert z['x'].shape==z['mask'].shape==(n,69) and z['rgb'].shape==(n,768);checks+=1
            assert z['mask'].dtype==bool and np.isfinite(z['x']).all() and np.isfinite(z['rgb']).all();checks+=1
            obs=adapter.read_observation(row['observation']); b=row['box'];xy=obs['xy_px']
            assert np.array_equal(z['frame_id'],obs['frame_id']) and np.array_equal(z['timestamp_us'],obs['timestamp_us']);checks+=1
            assert np.array_equal(z['frame_id'],np.arange(n));checks+=1
            obs['usable']&=(xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3])
            geo=adapter.extract_features(obs);assert geo['feature_names']==inputs['feature_names']==row['feature_names'];checks+=1
            assert np.array_equal(z['x'],np.nan_to_num(geo['x']).astype(np.float32)) and np.array_equal(z['mask'],geo['mask']);checks+=1
            from common import starts
            expected=np.asarray(starts(n));assert np.array_equal(z['window_starts'],expected);checks+=1
            mid=(z['timestamp_us'][expected]+z['timestamp_us'][expected+15])/2
            assert np.array_equal(mid,z['window_center_us']);checks+=1
            assert z['window_features'].shape==(len(expected),768) and np.isfinite(z['window_features']).all();checks+=1
            rgb=np.stack([np.interp(z['timestamp_us'],mid,z['window_features'][:,j]) for j in range(768)],1).astype(np.float32)
            assert np.array_equal(rgb,z['rgb']);checks+=1
            v=visual[alias];assert v['static_ordinals']==[3,6,9] and len(v['frames'])==13;checks+=1
            for frame in v['frames']:
                fid=frame['frame_id'];assert abs(frame['seconds']-obs['timestamp_us'][fid]/1e6)<1e-9;checks+=1
                with Image.open(frame['file']) as im:assert im.size==(1920,1080);checks+=1
            with Image.open(v['sequence']) as im:assert im.size==(1600,1640);checks+=1
            with Image.open(v['states']) as im:assert im.size==(1500,1110);checks+=1
        entries.append(dict(alias=alias,frames=row['frames'],windows=row['windows'],input_sha256=row['input_sha256'],source_sha256=row['source_sha256']))
    assert len(set(source_hashes))==32;checks+=1
    assert dict(Counter(r['role'] for r in inputs['rows']))=={'fresh_known':24,'fresh_expression':8};checks+=1
    assert sum(r['frames'] for r in inputs['rows'])==done['frames']==4045;checks+=1
    assert sum(r['windows'] for r in inputs['rows'])==done['windows'];checks+=1
    report=dict(status='PASS_IMPLEMENTER_SELF_VERIFICATION',at=prep.now(),checks=checks,clips=32,
                frames=done['frames'],windows=done['windows'],sources_unique_disjoint_prior80=True,
                all_pinned_old_models_code_records_observations_unchanged=True,
                no_training_or_target_labels=True,gpu_reexecuted_for_verification=False,
                scope='Source hashes, full-frame identities, same69 feature/mask reconstruction, window schedule/768 interpolation and13+3 visual layout; not semantic labels or independent quality verification.',
                independent_review='Parent pending; this verifier is authored by preparation implementer.',
                pins={str(p):prep.sha(p) for p in [Path(__file__),u/'prepare_fresh.py',u/'SELECTION.json',u/'INPUTS.json',u/'VISUAL_INDEX.json',u/'PREP_COMPLETED.json']},rows=entries)
    prep.write(u/'PREP_VERIFICATION.json',report)
    print(json.dumps({k:report[k] for k in ('status','checks','clips','frames','windows')},ensure_ascii=False))


if __name__=='__main__':main()
