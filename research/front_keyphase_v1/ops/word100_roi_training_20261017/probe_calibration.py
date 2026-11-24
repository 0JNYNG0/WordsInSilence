"""Two fixed TRAIN readiness probes; no crop, classifier or training."""
from pathlib import Path
import json, sys, time, traceback
import prep

U=Path(__file__).resolve().parent
IDS=('NIA_SL_WORD2732_REAL02_F','NIA_SL_WORD0500_REAL03_F')

def main():
    prep.authorize()
    context=prep.read(prep.CONTEXT)
    prep.check_pins(context['pins'])
    preflight=prep.read(U/'root_audit_legacy_preflight_all_RESULT.json')
    assert set(IDS)<=set(preflight['failure_clip_ids'])
    rows=[next(r for r in context['ordered_rows'] if r['clip_id']==cid) for cid in IDS]
    assert all(r['role']=='train_candidate' for r in rows)
    scope={'status':'FROZEN_BEFORE_TWO_READINESS_PROBES','clip_ids':list(IDS),
           'selection':'First TRAIN geometry failure and first TRAIN stability failure from legacy preflight, before any new classification score',
           'context_sha256':prep.sha256_file(prep.CONTEXT),
           'code_sha256':prep.sha256_file(__file__),
           'preflight_sha256':prep.sha256_file(U/'root_audit_legacy_preflight_all_RESULT.json'),
           'purpose':'Confirm whether old-observation calibration failures persist with frozen joint_v3; no threshold changes',
           'training':0,'classifier_inference':0,'maximum_new_detector_runs':2}
    prep.write_new(U/'CALIBRATION_PROBE_SCOPE.json',scope)
    results=[]; started=time.perf_counter()
    try:
        for row in rows:
            cid=row['clip_id'];d=U/'calibration_probe'/cid
            assert not d.exists();d.mkdir(parents=True)
            old=prep.read(row['legacy_record_path'])
            video,temporary=prep.source_file(row,old,d/'temporary')
            capture=prep.capture(video,cid,d/'native_capture',prep.read(prep.LAB/'config/contract.json'))
            obs=prep.Observation.load(capture['candidate'])
            cal=prep.cal.calibrate_rest(obs,int(obs.sidecar['width']),int(obs.sidecar['height']))
            prep.write_new(d/'calibration.json',cal)
            result={'clip_id':cid,'label':row['canonical_label'],'role':row['role'],
                    'capture':capture,'source_path':str(video),'source_temporary':temporary,
                    'source_sha256':prep.sha256_file(video),'calibration':cal,
                    'training':0,'classifier_inference':0}
            prep.write_new(d/'RESULT.json',result)
            results.append(result)
            print(json.dumps({'clip':cid,'calibration':cal['status']},ensure_ascii=False),flush=True)
        prep.check_pins(context['pins'])
        prep.write_new(U/'CALIBRATION_PROBE_RESULT.json',{'status':'READINESS_PROBES_COMPLETE',
            'rows':results,'new_full_detector_runs':2,'new_crop_runs':0,'seconds':time.perf_counter()-started,
            'training':0,'classifier_inference':0,'failed_calibrations':sum(r['calibration']['status']!='CALIBRATED' for r in results),
            'thresholds_unchanged':True,'full_preparation_authorized_but_training_gate_not_passed':True})
    except Exception:
        with (U/'CALIBRATION_PROBE_FIRST_ERROR.txt').open('x',encoding='utf-8') as f:f.write(traceback.format_exc())
        raise

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
