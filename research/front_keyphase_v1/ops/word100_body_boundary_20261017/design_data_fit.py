"""Freeze one TRAIN-only median body coefficient using boundary.shoulder_anchor.

No detector/classifier/evaluation access. No coefficient search. Existing stable
rest boundaries are preservation targets, not hand-position ground truth.
"""
from pathlib import Path
import importlib.util
import json
import sys
from datetime import datetime, timezone
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;OPS=U.parent
sys.path.insert(0,str(U))
import boundary
from wis_lab.core.observation import Observation, sha256_file

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def main():
    target=U/'design_data_COEFFICIENT.json'
    assert not target.exists(),'Frozen coefficient is immutable; reuse it'
    manifest=OPS/'word100_training_20261013/MANIFEST.json'
    checked=OPS/'word100_training_20261013/INPUT_CHECK.json'
    checked_data=read(checked)
    rows=[r for r in read(manifest) if r['role']=='train_candidate']
    assert len(rows)==899 and {r['signer_id'] for r in rows}==set(range(1,10))
    fixed=[Path(__file__),U/'boundary.py',boundary.OLD,manifest,checked]
    pins={str(p.resolve()):sha256_file(p) for p in fixed};fit_rows=[];failures=[]
    legacy=Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013')
    for row in rows:
        cid=row['clip_id'];rp=legacy/'records'/f'{cid}.json';rh=sha256_file(rp)
        assert rh==checked_data['record_hashes'][cid]
        rec=read(rp);op=Path(rec['observation']);oh=sha256_file(op)
        assert oh==rec['observation_sha256']
        side=op.with_suffix('.sidecar.json')
        pins.update({str(rp.resolve()):rh,str(op.resolve()):oh,str(side.resolve()):sha256_file(side)})
        obs=Observation.load(str(op));a=boundary.shoulder_anchor(obs)
        old=boundary.stable.calibrate_rest(obs,int(obs.sidecar['width']),int(obs.sidecar['height']))
        assert a['status']=='CALIBRATED',cid+' unavailable shoulder anchor'
        if old['status']!='CALIBRATED':
            failures.append({'clip_id':cid,'reason':old['reason'],'shoulder_anchor':a});continue
        ratio=(old['bottom']-a['median_shoulder_center'][1])/a['shoulder_width']
        assert np.isfinite(ratio) and ratio>0
        fit_rows.append({'clip_id':cid,'signer_id':row['signer_id'],'role':'train_candidate',
                         'old_bottom':old['bottom'],'shoulder_anchor':a,'ratio':float(ratio),
                         'observation':str(op.resolve()),'observation_sha256':oh})
    assert len(fit_rows)==889 and len(failures)==10
    ratio=float(np.median([r['ratio'] for r in fit_rows]))
    # Independent prior arithmetic must agree, but never changes the chosen median rule.
    prior=read(U/'design_data_shoulder_window_RESULT.json')
    assert abs(ratio-prior['coefficient'])<1e-12
    result={'schema':boundary.SCHEMA,'status':'FROZEN_TRAIN_MEDIAN','ratio':ratio,
            'created_utc':datetime.now(timezone.utc).isoformat(),'training_source_rows':899,'fit_rows':889,
            'train_calibration_failures_not_in_ratio_fit':10,
            'selection_rule':'Median chosen by root before matching-window output. One pooled TRAIN coefficient; no word/person/model/dev-score-specific coefficients.',
            'formula':'ratio_i=(old_stable_bottom_i-shoulder_anchor_i.median_shoulder_center[1])/shoulder_anchor_i.shoulder_width; ratio=median(ratio_i)',
            'fallback_formula':'floor(shoulder_anchor.median_shoulder_center[1]+ratio*shoulder_anchor.shoulder_width)',
            'preservation':'Existing stable calibration successes retain the exact original lower boundary; only failures call body fallback.',
            'statistical_limit':'A descriptive median of training framing, not an anatomical constant, confidence bound, or guaranteed action-preserving boundary.',
            'source_pins':pins,'population':fit_rows,'excluded_fit_rows':failures,
            'dev_arrays_read':0,'user_data':0,'new_detector_conditions':0,'training':0}
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'ratio':ratio,'fit_rows':889,'source_pins':len(pins),'coefficient_sha256':sha256_file(target)}))
if __name__=='__main__':main()
