from pathlib import Path
import sys,json,hashlib,datetime
from collections import Counter
import numpy as np
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf8')
U=Path(__file__).resolve().parent;O=U.parent;sys.path.insert(0,str(O.parent/'src'))
from wis_lab.core import features as F
from wis_lab.core.observation import Observation
import prepare as prep
import time_transform as T
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    r=read(U/'PREPARED.json');s=read(U/'SCOPE.json');assert len(r['rows'])==300 and not r['flags']
    for p,h in {**r['input_pins'],**r['output_pins'],**s['protected']}.items():assert sha(p)==h,p
    counts=Counter()
    for row in r['rows']:
        z=load(row['output']);a=load(row['source_observation']);e=load(row['source_evidence'])
        for key in ['xy_px','observed','usable','conf','hand_score','timestamp_us','frame_id']:assert np.array_equal(z[key],a[key],equal_nan=True)
        assert (np.diff(z['timestamp_us'])>0).all() and np.array_equal(z['values'],e['values'][0],equal_nan=True) and np.array_equal(z['available'],e['available'][0])
        core=np.array([any(lo<=float(t)/1e6<=hi for lo,hi in row['core_intervals']) for t in z['timestamp_us']]);assert np.array_equal(core,z['core'])
        assert np.array_equal(z['quality_eligible'],e['eligible'][0]);assert not(z['usable']&~z['observed']).any()
        counts['clips']+=1;counts['frames']+=len(core);counts['coordinate_scalars_preserved']+=a['xy_px'].size;counts['original_angle_quality_states_preserved']+=e['angle_state'][0].size
    offsets={};cursor=0
    for name,(width,_) in F.GROUPS.items():offsets[name]=cursor;cursor+=width
    rows=sorted(r['rows'],key=lambda x:x['clip_id']);checks=[]
    for row in rows[:3]:
        z=load(row['output'])
        for kind in ['fast075','slow15','pace','pause','skip3_50']:
            ix=T.mapping(len(z['core']),z['core'],kind);obs=T.observation(z,ix);features,present=prep.own_features(obs,T.body_origin(z,ix),float(z['scale']));pairs=0;changed=0
            for side in ['L','R']:
                gi=F.GROUP_ORDER.index('motion_'+side);pg=F.GROUP_ORDER.index('handpos_'+side);off=offsets['motion_'+side];po=offsets['handpos_'+side]
                assert not present[0,gi]
                for i in range(1,len(ix)):
                    good=present[i,pg] and present[i-1,pg] and 0<(obs.timestamp_us[i]-obs.timestamp_us[i-1])/1e6<=.2
                    assert bool(present[i,gi])==bool(good)
                    if not good:continue
                    dt=(int(obs.timestamp_us[i])-int(obs.timestamp_us[i-1]))/1e6
                    expected=(features[i,po:po+2].astype(float)-features[i-1,po:po+2].astype(float))/dt*F.MOTION_SCALE
                    np.testing.assert_allclose(features[i,off:off+2],expected,atol=2e-6,rtol=1e-5);pairs+=1
                    a=int(np.floor(ix[i]));b=int(np.ceil(ix[i]));w=ix[i]-a
                    old=(z['features'][a,off:off+2]*(1-w)+z['features'][b,off:off+2]*w)
                    if np.isfinite(old).all() and not np.allclose(features[i,off:off+2],old,atol=1e-6):changed+=1
            assert pairs>0
            checks.append(dict(clip_id=row['clip_id'],kind=kind,valid_velocity_pairs=pairs,different_from_stale_feature_interpolation=changed))
    assert sum(x['different_from_stale_feature_interpolation'] for x in checks)>0
    # Synthetic missing/occlusion bracket cannot become observed by interpolation.
    z=load(rows[0]['output']);z['observed'][4,33]=False;z['usable'][4,33]=False;z['xy_px'][4,33]=np.nan
    obs=T.observation(z,np.array([3.,3.5,4.,4.5,5.]));assert not obs.observed[1:4,33].any() and np.isnan(obs.xy_px[1:4,33]).all()
    out=dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS_INPUT_CONTRACT_AND_TIME_MOTION_FIXTURES',counts=dict(counts),public_transform_fixtures=checks,missing_interpolation_preserved=True,original_body_scale_fixed=True,original_per_frame_origin_resampled=True,
      feature_check='Original reconstruction by same production feature function; independent raw-array/core contract and velocity formula on15 transformed public fixtures. Not independent landmark truth.',
      pending='Actual100word time-alignment comparison not executed. Freeze all transforms/evaluation metrics and compare DTW path+evidence conditions next. Do not treat15 fixtures as300video trial.',
      original_prepared_sha256=sha(U/'PREPARED.json'),pins={str(U/n):sha(U/n) for n in ['prepare.py','time_transform.py','verify_prepare.py','PREPARED.json','SCOPE.json']},model_changed=False)
    with (U/'VERIFIED_PREPARE.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print('VERIFIED_PREPARE',dict(counts),sum(x['valid_velocity_pairs'] for x in checks),sum(x['different_from_stale_feature_interpolation'] for x in checks),flush=True)
if __name__=='__main__':main()
