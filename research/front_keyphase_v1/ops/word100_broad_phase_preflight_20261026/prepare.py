from pathlib import Path
import sys,json,hashlib,datetime,time,traceback
import numpy as np
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf8')
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];B=O/'word100_body_boundary_20261017';E=O/'word100_recovery_evidence_contract_20261026';V=O/'word100_recovery_dev300_20261026'
sys.path.insert(0,str(ROOT/'src'))
# ROOT above is repository root; actual package is beneath front_keyphase_v1.
sys.path.insert(0,str(O.parent/'src'))
from wis_lab.core.observation import Observation
from wis_lab.core import features as F
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def own_features(obs,origin,scale):
    rows=[];masks=[]
    for i in range(obs.n_frames):
        x,m=F.frame_features(obs,i,origin,scale);rows.append(x);masks.append([m[g] for g in F.GROUP_ORDER])
    offsets={};offset=0
    for g in F.GROUP_ORDER:offsets[g]=offset;offset+=F.GROUPS[g][0]
    F._fill_motion(rows,masks,obs.timestamp_us,offsets)
    return np.asarray(rows,np.float32),np.asarray(masks,bool)
def main():
    assert read(E/'VERIFIED.json')['anatomical_accuracy_verified'] is False
    f=read(E/'FROZEN.json');meta={r['clip_id']:r for r in read(B/'PREP_CONTEXT.json')['ordered_rows']};rows=f['rows'];assert len(rows)==300 and len({x['word_code'] for x in rows})==100 and {x['signer_id'] for x in rows}=={11,12,13}
    pins={str(p):sha(p) for p in [Path(__file__),Path(F.__file__),Path(sys.modules[Observation.__module__].__file__),E/'FROZEN.json',E/'VERIFIED.json',B/'PREP_CONTEXT.json']};protected=f['protected']
    for p,h in protected.items():assert sha(p)==h,p
    write(U/'SCOPE.json',dict(at=now(),authorization='Existing autonomous goal; prepare already-developed public100-word DEV300 known-time diagnostic only. No learning, candidate selection, user/mentor/extra199/200 access or operating edit.',
      plan='Use original C fullframe→ROI observations and existing geometry evidence. Fixed original body origin/scale. For later timing variants resample coordinates and masks at observed endpoints, synthesize elapsed timestamps at original median cadence, recompute feature motion from coordinates/time. Known mapping/evaluation core only to evaluator. No new detector. Compare oldDTW and frozen fractional+evidence-preserving candidate, with both gains and lost short events. Candidate remains unadopted.',
      separation='This is100word development coverage, not new unseen people or semantic correctness validation. Angle available differs from angle verified. No rawRGB recovery accuracy claim. Core annotation marks evaluation activity, not mandatory sign phases.',
      pending='This script prepares originals and verifies exact feature reconstruction only. Later transformation fixtures and fixed all-condition trial remain to implement and validate.',rows=rows,pins=pins,protected=protected))
    (U/'inputs').mkdir();out=[];outputpins={};started=time.perf_counter();totalframes=0
    for i,row in enumerate(rows):
        cid=row['clip_id'];r=meta[cid];assert r['role']=='development_only'
        op=B/'data/clips'/cid/'masked/observation.npz';ep=E/'inputs'/(cid+'.npz');obs=Observation.load(str(op));obs.validate()
        with np.load(ep,allow_pickle=False) as z:e={k:z[k].copy() for k in z.files}
        assert np.array_equal(obs.frame_id,e['frame_id']) and np.array_equal(obs.timestamp_us,e['timestamp_us'])
        assert np.array_equal(obs.xy_px[:,33:].reshape(-1,2,21,2),e['xy'][0],equal_nan=True)
        assert np.array_equal(obs.usable[:,33:].reshape(-1,2,21),e['valid'][0])
        q=F.sequence_features(obs,trim=False);origin,scale,_=F.body_frame(obs);x,p=own_features(obs,origin,scale)
        assert np.array_equal(x,q['features'],equal_nan=True) and np.array_equal(p,q['present'])
        core=np.zeros(obs.n_frames,bool)
        for begin,end in r['intervals']:core|=(obs.timestamp_us/1e6>=begin)&(obs.timestamp_us/1e6<=end)
        active=np.flatnonzero(core);flags=[]
        if len(active)<5:flags.append('core_too_short_for_fixed_skip')
        if len(active)>1 and not (np.diff(active)==1).all():flags.append('multiple_core_segments_need_explicit_transform')
        outpath=U/'inputs'/(cid+'.npz')
        np.savez_compressed(outpath,xy_px=obs.xy_px,observed=obs.observed,usable=obs.usable,conf=obs.conf,hand_score=obs.hand_score,timestamp_us=obs.timestamp_us,frame_id=obs.frame_id,
          features=x,present=p,core=core,origin=origin,scale=np.array(scale),values=e['values'][0],available=e['available'][0],quality_eligible=e['eligible'][0],identity_conflict=e['identity_conflict'],angle_state=e['angle_state'][0])
        outputpins[str(outpath)]=sha(outpath);pins[str(op)]=sha(op);pins[str(ep)]=sha(ep);totalframes+=obs.n_frames
        out.append(dict(clip_id=cid,word_code=r['word_code'],label=r['canonical_label'],signer_id=r['signer_id'],role='development_only',frames=obs.n_frames,core_frames=len(active),core_intervals=r['intervals'],cadence_us=float(np.median(np.diff(obs.timestamp_us))),flags=flags,original_features_exact=True,source_observation=str(op),source_evidence=str(ep),output=str(outpath),available_angle_cells=int(e['available'][0,:,:,:15].sum()),quality_eligible_angle_cells=int(e['eligible'][0,:,:,:15].sum())))
        if (i+1)%50==0:print('PREPARED',i+1,300,flush=True)
    for p,h in protected.items():assert sha(p)==h,p
    write(U/'PREPARED.json',dict(at=now(),status='ORIGINAL_INPUTS_PREPARED_NO_TIMING_TRIAL_YET',rows=out,input_pins=pins,output_pins=outputpins,frames=totalframes,all_original_feature_reconstructions_exact=True,flags=[dict(clip_id=r['clip_id'],flags=r['flags']) for r in out if r['flags']],seconds=time.perf_counter()-started,training=0,new_detector=0,model_inference=0,production_changed=False))
    print('PREPARED_COMPLETE',len(out),totalframes,sum(bool(x['flags']) for x in out),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        write(U/('ERROR_'+now().replace(':','_')+'.json'),dict(traceback=traceback.format_exc()));raise
