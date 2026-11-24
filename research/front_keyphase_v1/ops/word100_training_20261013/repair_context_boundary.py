"""Repair only three derived contexts before any classifier training.

Original observations/whole inputs/development inputs are preserved. Original
context files and records are retained in the project evidence directory.
"""
import shutil
from pathlib import Path
import numpy as np
from common import U,OUT,RUN,read,write,sha,pin_check
from wis_lab.core.observation import Observation
from pilot.web_input import to_native_like
from pilot.data import build_channels

def main():
    pin_check()
    assert not (U/'CONTEXT_BOUNDARY_REPAIR.json').exists()
    assert not list(RUN.glob('**/*.pt')), 'Never repair inputs after model training starts'
    audit=read(U/'CONTEXT_BOUNDARY_AUDIT.json');assert audit['affected']==3
    manifest={r['clip_id']:r for r in read(U/'MANIFEST.json')}
    before={str(p):sha(p) for p in (OUT/'records').glob('*.json')}
    changed=[]
    for item in audit['issues']:
        cid=item['clip_id'];row=manifest[cid];assert row['role']=='train_candidate'
        p=OUT/'records'/f'{cid}.json';rec=read(p);cp=Path(rec['channel_file'])
        assert sha(rec['observation'])==rec['observation_sha256']==item['observation_sha256']
        assert sha(cp)==rec['channel_sha256']==item['channel_sha256']
        backup=U/'before_boundary_repair'/cid;backup.mkdir()
        shutil.copy2(p,backup/p.name);shutil.copy2(cp,backup/cp.name)
        with np.load(cp,allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
        original_whole={k:v.copy() for k,v in arr.items() if k.startswith('whole_') or k=='x_whole'}
        obs=Observation.load(rec['observation']);nat,pr=to_native_like(obs.xy_px,obs.usable)
        lo,hi=item['new_span'];a,b=row['intervals'][0];ts=obs.timestamp_us
        assert ts[lo]>=round((a-.3)*1e6) and ts[hi]<=round((b+.3)*1e6)
        assert hi==item['old_span'][1]+1 and lo==item['old_span'][0]
        arr.update(x_context=build_channels(nat[lo:hi+1],pr[lo:hi+1],ts[lo:hi+1])['x'],
            context_frame_id=obs.frame_id[lo:hi+1],context_ts=ts[lo:hi+1])
        tmp=cp.with_name(cp.stem+'.boundary_repair.npz');np.savez_compressed(tmp,**arr)
        with np.load(tmp,allow_pickle=False) as z:
            for key,v in original_whole.items():np.testing.assert_array_equal(v,z[key])
        tmp.replace(cp)
        rec['channel_sha256']=sha(cp);rec['feature_check']['context_span']=[lo,hi]
        rec['boundary_repair']='Exact microsecond endpoint; original files in ops before_boundary_repair. No new extraction.'
        write(p,rec)
        assert sha(rec['observation'])==item['observation_sha256']
        changed.append(dict(clip_id=cid,before_channel_sha256=item['channel_sha256'],after_channel_sha256=rec['channel_sha256'],
            before_span=item['old_span'],after_span=item['new_span'],whole_input_unchanged=True,raw_observation_unchanged=True))
    changed_paths={str(OUT/'records'/f"{x['clip_id']}.json") for x in changed}
    assert all(sha(p)==h for p,h in before.items() if p not in changed_paths)
    write(U/'CONTEXT_BOUNDARY_REPAIR.json',dict(status='REPAIRED_PENDING_FULL_CHECK',reason='Exact0.30-second boundary implemented in integer microseconds; no threshold relaxation.',
        new_landmark_extractions=0,training_before_repair=0,affected_training_contexts=3,development_changes=0,
        unchanged_other_records=len(before)-3,changes=changed,frozen_experiment_sha256=sha(U/'FROZEN_EXPERIMENT.json')))
    print('Three context endpoints repaired; observations, whole inputs and development inputs preserved.')

if __name__=='__main__':main()
