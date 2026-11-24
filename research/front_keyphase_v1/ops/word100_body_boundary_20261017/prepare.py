"""Versioned preparation using frozen legacy helpers and common body fallback."""
from pathlib import Path
import argparse
import copy
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
PREVIOUS = U.with_name('word100_roi_training_20261017')
sys.path.insert(0, str(PREVIOUS))
import prep as legacy
sys.path.insert(0, str(U))
import boundary

COEFFICIENT = U / 'design_data_COEFFICIENT.json'
CONTEXT = U / 'PREP_CONTEXT.json'
DATA = U / 'data'


def configure():
    legacy.U, legacy.DATA, legacy.CONTEXT = U, DATA, CONTEXT
    coefficient = legacy.read(COEFFICIENT)
    legacy.cal = SimpleNamespace(calibrate_rest=lambda obs, w, h: boundary.calibrate(obs, w, h, coefficient))


def freeze():
    assert legacy.read(U/'AUTHORIZATION.json')['approved'] is True
    assert not CONTEXT.exists()
    previous = legacy.read(PREVIOUS/'PREP_CONTEXT.json')
    legacy.check_pins(previous['pins'])
    context = copy.deepcopy(previous)
    recovery = legacy.read(PREVIOUS/'PREP_RECOVERY_CONTEXT.json')
    probe = legacy.read(PREVIOUS/'CALIBRATION_PROBE_RESULT.json')
    probes = {r['clip_id']: r for r in probe['rows']}
    oldflags = legacy.read(PREVIOUS/'root_audit_legacy_preflight_CAUSE_SUMMARY.json')
    failed_train = {r['clip_id'] for group in oldflags['causes'].values() for r in group if r['role']=='train_candidate'}
    assert len(failed_train) == 10
    for row in context['ordered_rows']:
        cid = row['clip_id']
        if cid in recovery['patches']:
            row['retained_video'] = recovery['patches'][cid]['new']
        if cid in probes:
            row['reuse_native'] = copy.deepcopy(probes[cid]['capture'])
            row['reuse_native']['reused'] = True
            row['retained_video'] = probes[cid]['source_path']
    pins = context['pins']
    for path in [Path(__file__), U/'boundary.py', COEFFICIENT, U/'AUTHORIZATION.json',
                 U/'FROZEN_PROTOCOL.json', PREVIOUS/'PREP_CONTEXT.json',
                 PREVIOUS/'PREP_RECOVERY_CONTEXT.json', PREVIOUS/'CALIBRATION_PROBE_RESULT.json']:
        pins[str(path.resolve())] = legacy.sha256_file(path)
    for row in context['ordered_rows']:
        if row.get('retained_video'):
            p = Path(row['retained_video']); expected = legacy.read(row['legacy_record_path'])['source_sha256']
            assert legacy.sha256_file(p) == expected
            pins[str(p.resolve())] = expected
        if row.get('reuse_native'):
            legacy.check_pins(row['reuse_native']['pins'])
            pins.update(row['reuse_native']['pins'])
    context.update(schema='word100_body_boundary_preparation_v1',
                   status='FROZEN_BEFORE_BODY_FALLBACK_PREPARATION',
                   output_root=str(DATA.resolve()), coefficient_path=str(COEFFICIENT),
                   boundary_code=str(U/'boundary.py'), protocol=str(U/'FROZEN_PROTOCOL.json'),
                   early_ids=sorted(set(context['prior20_ids']) | failed_train),
                   failure_train_ids=sorted(failed_train),
                   previous_context=str(PREVIOUS/'PREP_CONTEXT.json'))
    legacy.write_new(CONTEXT, context)
    print(json.dumps({'status':'FROZEN','rows':len(context['ordered_rows']),
        'early_rows':len(context['early_ids']),'native_cache':sum(bool(r.get('reuse_native')) for r in context['ordered_rows'])}),flush=True)


def preflight():
    """Fixed rule on all legacy observations, not a new detection result."""
    context = legacy.read(CONTEXT); legacy.check_pins(context['pins'])
    rows = []; train_feature_failures = []; calibration_failures = []
    for row in context['ordered_rows']:
        old = legacy.read(row['legacy_record_path'])
        obs = legacy.Observation.load(old['observation'])
        cal = boundary.calibrate(obs, int(obs.sidecar['width']), int(obs.sidecar['height']), legacy.read(COEFFICIENT))
        result = {'clip_id':row['clip_id'],'label':row['canonical_label'],'role':row['role'],
                  'method':cal['method'],'calibration':cal['status'], 'bottom':cal.get('bottom')}
        if cal['status'] != 'CALIBRATED':
            calibration_failures.append(row['clip_id'])
        else:
            masked = legacy.roi.apply_roi(obs, cal['box_source_pixels'])
            _, info = legacy.features(masked, row)
            result['feature_info'] = info
            result['geometry_flags'] = [f for f in legacy.visibility_flags(row, obs, masked, masked) if f['comparison']=='native_to_mask']
            if info['reason'] is not None and row['role']=='train_candidate':
                train_feature_failures.append(row['clip_id'])
        rows.append(result)
    legacy.write_new(U/'LEGACY_PREFLIGHT.json', {'status':'LEGACY_FORECAST_ONLY',
        'context_sha256':legacy.sha256_file(CONTEXT),'rows':rows,
        'calibration_failures':calibration_failures,'train_feature_failures':train_feature_failures,
        'fallback_rows':sum(r['method']=='train_median_body_fallback' for r in rows),
        'detector':0,'training':0,'fixed_coefficient_unchanged':True})
    print(json.dumps({'rows':len(rows),'calibration_failures':calibration_failures,
        'train_feature_failures':train_feature_failures,
        'fallback_rows':sum(r['method']=='train_median_body_fallback' for r in rows)}),flush=True)


def run(stage, limit, seconds):
    context = legacy.read(CONTEXT); legacy.check_pins(context['pins'])
    for p, stat in context['archive_stat'].items():
        now = Path(p).stat(); assert [now.st_size, now.st_mtime_ns] == stat
    assert not (U/'PREP_FIRST_FAILURE.json').exists(), 'First failure must be reviewed; no blind retry'
    lock=U/'PREP_RUNNING.json'; legacy.write_new(lock,{'pid':os.getpid(),'stage':stage,'context_sha256':legacy.sha256_file(CONTEXT)})
    start=time.perf_counter(); count=0; cid=None
    try:
        contract=legacy.read(legacy.LAB/'config/contract.json')
        selected=[r for r in context['ordered_rows'] if stage=='all' or r['clip_id'] in context['early_ids']]
        for row in selected:
            cid=row['clip_id']
            if (DATA/'records'/(cid+'.json')).exists():
                legacy.prepare_row(row,legacy.sha256_file(CONTEXT),contract)
                continue
            if count >= limit or time.perf_counter()-start >= seconds: break
            assert legacy.shutil.disk_usage(U).free >= 10_000_000_000
            legacy.event(stage='body_row_start',clip_id=cid)
            rec,reused=legacy.prepare_row(row,legacy.sha256_file(CONTEXT),contract)
            count+=1
            legacy.event(stage='body_row_complete',clip_id=cid,method=rec['calibration']['method'],
                         flags=len(rec['visual_flags']),seconds=time.perf_counter()-start)
        completed=sum((DATA/'records'/(r['clip_id']+'.json')).exists() for r in selected)
        result={'status':'PREPARED_PENDING_INDEPENDENT_AND_VISUAL_REVIEW' if completed==len(selected) else 'CHUNK_COMPLETE',
                'stage':stage,'selected':len(selected),'completed':completed,'new_records':count,
                'seconds':time.perf_counter()-start,'training':0}
        legacy.write_status(result); print(json.dumps(result),flush=True)
    except Exception as exc:
        failure={'status':'FAILED_NO_TRAINING','clip_id':cid,'type':type(exc).__name__,
                 'message':str(exc),'traceback':traceback.format_exc(),'training':0,
                 'context_sha256':legacy.sha256_file(CONTEXT)}
        if not (U/'PREP_FIRST_FAILURE.json').exists(): legacy.write_new(U/'PREP_FIRST_FAILURE.json',failure)
        legacy.write_status(failure)
        raise
    finally:
        if lock.exists() and legacy.read(lock)['pid']==os.getpid():lock.unlink()


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','preflight','run'])
    p.add_argument('--stage',choices=['early','all'],default='early')
    p.add_argument('--limit',type=int,default=1200);p.add_argument('--seconds',type=float,default=1800)
    a=p.parse_args()
    if a.action=='freeze':freeze()
    else:
        configure()
        if a.action=='preflight':preflight()
        else:run(a.stage,a.limit,a.seconds)
