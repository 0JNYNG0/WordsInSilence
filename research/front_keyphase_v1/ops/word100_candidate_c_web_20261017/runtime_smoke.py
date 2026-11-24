"""One fixed TRAIN input forward and blank RGB detector lifecycle; no camera/DEV/training."""
from pathlib import Path
import json, sys, time, traceback
import numpy as np
U=Path(__file__).resolve().parent
LAB=U.parents[1]
sys.path.insert(0,str(LAB/'src'))
from wis_lab.service import word100, word100_candidate as C, word100_multiref
from wis_lab.core.observation import Observation, empty_frames, sha256_file

def run():
    paths=[C.CHECKPOINT,Path(word100.config()['checkpoint'])]
    before={str(p):sha256_file(p) for p in paths}
    contract=json.loads((LAB/'config/contract.json').read_text('utf-8'))
    start=time.perf_counter()
    with C.CandidateLandmarker(contract) as lm:
        out=empty_frames(4);out['frame_id'][:]=np.arange(4)
        rgb=np.zeros((240,320,3),dtype=np.uint8)
        for i in range(4):lm.process(rgb,i*40000,out,i)
        ob=lm.finalize_observation(Observation(**out))
        assert not ob.observed.any() and ob.n_frames==4
        assert ob.sidecar['extractor_id']==lm.extractor_id()
        detector_id=lm.extractor_id()
    detector_seconds=time.perf_counter()-start
    record=json.loads((C.UNIT/'data/records/NIA_SL_WORD0106_REAL01_F.json').read_text('utf-8'))
    assert record['role']=='train_candidate' and record['signer_id']==1
    views=record.get('views',record.get('modes'))
    co=C.finalize_observation(Observation.load(views['sensor']['observation']))
    mo=Observation.load(views['m0_sensor']['observation'])
    c=C.recognise(co);m=word100.recognise(mo)
    assert c['model_status']==m['model_status']=='READY',(c.get('reason'),m.get('reason'))
    assert len(c['logits'])==len(m['logits'])==100
    assert {str(p):sha256_file(p) for p in paths}==before
    bank,pin=word100_multiref.bank()
    return dict(status='PASS',detector_blank_frames=4,detector_seconds=detector_seconds,
        candidate_extractor_id=detector_id, actual_model_forward=2,
        smoke_source=record['clip_id'],source_role='TRAIN',C_status=c['model_status'],M0_status=m['model_status'],
        C_elapsed_ms=c['elapsed_ms'],M0_elapsed_ms=m['elapsed_ms'],dtw_bank_sha256=pin,
        checkpoint_hashes_unchanged=before,camera_runs=0,DEV_reads=0,training=0,
        performance_claim='No accuracy estimate; one preselected TRAIN clip only checks runtime plumbing.')

if __name__=='__main__':
    try: result=run()
    except Exception: result=dict(status='FAIL',traceback=traceback.format_exc())
    n=1
    while (U/f'RUNTIME_SMOKE_{n:03}.json').exists():n+=1
    (U/f'RUNTIME_SMOKE_{n:03}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    sys.exit(0 if result['status']=='PASS' else 1)
