"""Prepare one prospective numerical comparison; never regenerate old results."""
from pathlib import Path
import json, hashlib, numpy as np
U=Path(__file__).resolve().parent
OLD=U.parent/'word100_phase_correspondence_20261101'
NORMAL={'identity','speed_075','speed_125','hold15','repeat2_same','repeat3_same','early_slow','late_slow'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
def main():
    assert not (U/'PAIR_INPUTS.json').exists()
    old=read(OLD/'INPUTS.json');pins={str(OLD/'INPUTS.json'):sha(OLD/'INPUTS.json')}
    controls=[{**c,'normal_control':c['kind'] in NORMAL,'archived_path_not_recomputed':True} for c in old['controls'] if c['kind']!='swap_adjacent_core_blocks']
    checks=0
    for alias,rec in old['inputs'].items():
        source=rec['original'];a=load(source);n=len(a['x']);mid=n//2
        for kind in ['early_slow','late_slow']:
            repeats=np.where(np.arange(n)<mid,2 if kind=='early_slow' else 1,1 if kind=='early_slow' else 2)
            idx=np.repeat(np.arange(n),repeats);q={k:v[idx].copy() for k,v in a.items()}
            q['frame_id']=np.arange(len(idx),dtype=np.int64);q['timestamp_us']=q['frame_id']*33333
            path=U/'controls'/f'{alias}_{kind}.npz';path.parent.mkdir(exist_ok=True)
            assert not path.exists();np.savez_compressed(path,**q)
            assert np.array_equal(q['x'],a['x'][idx],equal_nan=True) and np.array_equal(q['mask'],a['mask'][idx]);checks+=2
            assert set(idx)==set(range(n));checks+=1
            controls.append(dict(id=f'{alias}_{kind}',alias=alias,kind=kind,input=str(path),reference=source,metadata=dict(original_indices=idx.tolist(),all_original_frames_preserved=True,synthetic_time_only=True),normal_control=True,archived_path_not_recomputed=True))
    assert len(controls)==192 and sum(c['normal_control'] for c in controls)==128;checks+=2
    for c in controls:
        for p in [c['input'],c['reference']]:
            pins[p]=sha(p);a=load(p)
            assert a['x'].shape==a['mask'].shape and a['x'].shape[1]==69 and a['mask'].dtype==bool
            assert (np.diff(a['frame_id'])>0).all() and (np.diff(a['timestamp_us'])>0).all();checks+=2
    for rec in old['inputs'].values():pins[rec['original']]=sha(rec['original'])
    data=dict(controls=controls,natural=old['natural'],inputs={k:{'original':v['original']} for k,v in old['inputs'].items()},pins=pins,
              matcher_keys=['x','mask','frame_id','timestamp_us'],source_truth_is_evaluation_only=True,
              archived_velocity_columns_preserved_but_replaced_by_common_sidecar_for_both_conditions=True)
    write(U/'PAIR_INPUTS.json',data)
    write(U/'PREPARATION_CHECKS.json',dict(status='PASS',checks=checks,controls=192,normal=128,natural_pairs=12,new_time_only_arrays=32,old_results_run=0))
if __name__=='__main__':main()
