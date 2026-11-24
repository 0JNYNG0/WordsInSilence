"""Extend frozen legacy forecast to the exact1199 roles; reuse representative100."""
import sys
sys.dont_write_bytecode = True
import time
from collections import Counter
from pathlib import Path
import root_audit_legacy_preflight as audit


def main():
    started = time.perf_counter()
    report_path = audit.U / 'root_audit_legacy_preflight_all_RESULT.json'
    assert not report_path.exists(), 'Reuse completed forecast'
    prior_path = audit.U / 'root_audit_legacy_preflight_RESULT.json'
    prior = audit.read(prior_path)
    assert prior['status'] == 'LEGACY_PREFLIGHT_COMPLETE_NOT_CANDIDATE_READINESS'
    assert all(audit.sha(p) == h for p, h in prior['source_pins'].items())
    cache = {r['clip_id']: r for r in prior['results']}
    manifest = audit.read(audit.OLD / 'MANIFEST.json')
    assert len(manifest) == 1199 and Counter(r['role'] for r in manifest) == {'train_candidate': 899, 'development_only': 300}
    checked = audit.read(audit.OLD / 'INPUT_CHECK.json')
    pins = {str(p): audit.sha(p) for p in (prior_path, audit.OLD / 'MANIFEST.json',
            audit.OLD / 'INPUT_CHECK.json', audit.SRC, Path(audit.__file__), Path(__file__))}
    audit.write(audit.U / 'root_audit_legacy_preflight_all_PLAN.json', dict(status='FROZEN_BEFORE_FORECAST',
        selection='All1199 rows of original frozen training manifest, without new role selection; TRAIN899/DEV300',
        reuse100='Prior lowest-TRAIN-ID representative results with verified source pins',
        scope='Legacy observations only; frozen calibration, diagnostics on failed rows, no new detector/model execution', source_pins=pins))
    results = []
    for r in manifest:
        rp = audit.DATA / 'records' / f"{r['clip_id']}.json"
        assert audit.sha(rp) == checked['record_hashes'][r['clip_id']]
        rec = audit.read(rp)
        op = Path(rec['observation'])
        sp = op.with_suffix('.sidecar.json')
        assert audit.sha(op) == rec['observation_sha256']
        pins.update({str(rp): audit.sha(rp), str(op): audit.sha(op), str(sp): audit.sha(sp)})
        if r['clip_id'] in cache:
            c = cache[r['clip_id']]
            assert c['observation_sha256'] == rec['observation_sha256']
            cal = c['calibration']
            detail = c['diagnostics']
        else:
            obs = audit.Observation.load(str(op))
            cal = audit.stable.calibrate_rest(obs, int(obs.sidecar['width']), int(obs.sidecar['height']))
            detail = audit.diagnose(obs) if cal['status'] != 'CALIBRATED' else None
        results.append(dict(clip_id=r['clip_id'], label=r['canonical_label'], role=r['role'],
            signer_id=r['signer_id'], observation=str(op), observation_sha256=rec['observation_sha256'],
            legacy_preflight=True, calibration=cal, diagnostics=detail, reused_representative=r['clip_id'] in cache))
    failures = [r for r in results if r['calibration']['status'] != 'CALIBRATED']
    by_role = {role: dict(total=sum(r['role'] == role for r in results),
        failed=sum(r['role'] == role for r in failures)) for role in ('train_candidate', 'development_only')}
    assert all(audit.sha(p) == h for p, h in pins.items())
    report = dict(status='LEGACY_PREFLIGHT_COMPLETE_NOT_CANDIDATE_READINESS', total=1199,
        calibrated=1199-len(failures), not_calibrated=len(failures), by_role=by_role,
        failure_clip_ids=[r['clip_id'] for r in failures], results=results, source_pins=pins,
        elapsed_seconds=time.perf_counter()-started, reused_representatives=100, all_source_pins_unchanged=True,
        meaning='Forecast from legacy assignment only. No joint_v3 detection or threshold change. Not an actual training-input readiness pass/fail.',
        next='Inspect newjoint observation and first-second source/raw candidates for listed rows before inferring actual calibration failure.',
        detector_execution=0, classifier_execution=0, training=0, heldout_or_user_access=0)
    audit.write(report_path, report)
    import json
    print(json.dumps({k: report[k] for k in ('status','total','calibrated','not_calibrated','by_role','failure_clip_ids','elapsed_seconds')}, ensure_ascii=False))


if __name__ == '__main__': main()
