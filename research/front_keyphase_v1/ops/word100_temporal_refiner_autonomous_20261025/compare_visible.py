"""Frozen image-only diagnostic rectangles; never used to tune any candidate."""
from pathlib import Path
import json,importlib.util,hashlib
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;B=U.parent/'word100_temporal_public_bundle_20261025';OLD=U.parent/'word100_temporal_refiner_pilot_20261025'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    spec=importlib.util.spec_from_file_location('safe',B/'load_inputs.py');safe=importlib.util.module_from_spec(spec);spec.loader.exec_module(safe)
    manifest={r['clip_id']:r for r in read(B/'MANIFEST.json')}
    original=read(OLD/'VISIBLE_SAMPLE_SELECTION.json')['cases'];labels={r['sample_id']:r for r in read(OLD/'VISIBLE_LABELS_FROZEN.json')['labels']}
    cases=[]
    for row in original:
        label=labels[row['sample_id']]
        cases.append(dict(sample_id=row['sample_id'],clip_id=row['clip_id'],frame_id=row['frame_id'],side=row.get('side_index',row.get('side')),rectangle=label['rectangle'],scorable=label['scorable'],cohort='previous24'))
    new=read(U/'VISIBLE_INDEPENDENT_NEW.json');index={r['sample_id']:r for r in new['image_only_labels']}
    for row in new['selection']:
        label=index[row['sample_id']]
        cases.append(dict(sample_id=row['sample_id'],clip_id=row['clip_id'],frame_id=row['frame_id'],side=row['anatomical_side'],rectangle=label['wrist_rectangle_source_pixels'],scorable=label['scorable_coarse_wrist'],cohort='new12',hand_relation=label['hand_relation'],image_posture=label['image_posture']))
    results=[]
    for case in cases:
        row=manifest[case['clip_id']]
        with np.load(row['arrays']) as z:raw={k:z[k].copy() for k in z.files}
        frame=int(np.flatnonzero(raw['frame_id']==case['frame_id'])[0]);side=case['side']
        if isinstance(side,str):side={'left':0,'right':1}[side]
        assert side in (0,1),case
        joint=33+21*side;box=case['rectangle'];roi=np.asarray(raw['roi_box']).reshape(-1)
        within_roi=None if box is None else bool(box[0]>=roi[0] and box[1]>=roi[1] and box[2]<roi[2] and box[3]<roi[3])
        payloads={k:safe.from_arrays(raw,k) for k in ('B0','B1')}
        conditions={k:(p['xy_px'],p['usable'],np.zeros_like(p['usable'])) for k,p in payloads.items()}
        for cond in ('B2','B3'):
            for seed in (17,29,43):
                name=f'{cond}_seed{seed}'
                with np.load(U/'runs'/name/'natural_estimates'/(row['clip_id']+'.npz')) as z:
                    conditions[name]=(z['xy_px'].copy(),z['usable_observed']|z['estimated'],z['estimated'].copy())
        for name,(xy,usable,estimated) in conditions.items():
            available=bool(usable[frame,joint]);point=xy[frame,joint].astype(float) if available else None
            distance=None
            if point is not None and case['scorable']:
                distance=float(np.linalg.norm(np.maximum(np.asarray(box[:2])-point,0)+np.maximum(point-np.asarray(box[2:]),0)))
            results.append(case|dict(condition=name,within_fixed_roi=within_roi,available=available,estimated=bool(estimated[frame,joint]),xy=None if point is None else point.tolist(),outside_rectangle_px=distance))
    counts={}
    for cohort in ('previous24','new12'):
        counts[cohort]={}
        for name in conditions:
            rs=[r for r in results if r['cohort']==cohort and r['condition']==name]
            eligible=[r for r in rs if r['scorable'] and r['within_fixed_roi']]
            counts[cohort][name]=dict(total=len(rs),in_roi_scorable=len(eligible),available=sum(r['available'] for r in eligible),inside_box=sum(r['outside_rectangle_px']==0 for r in eligible),outside_box=sum(r['outside_rectangle_px'] is not None and r['outside_rectangle_px']>0 for r in eligible),estimates=sum(r['estimated'] for r in eligible))
    output=dict(status='COMPLETE_COARSE_IMAGE_DIAGNOSTIC_NOT_ANATOMICAL_ACCURACY',counts=counts,rows=results,
                labels_pins={str(p):sha(p) for p in (OLD/'VISIBLE_LABELS_FROZEN.json',U/'VISIBLE_INDEPENDENT_NEW.json')},
                notes=['Previously reviewed24 and newly image-reviewed12 are separate','Coarse rectangles by AI image review, no expert21point or hidden3D truth','Outside box may be rounding or uncertain anatomy; do not equate every outside point with error','No threshold/model selection from these diagnostics','Down hands outside fixed ROI excluded from localization metric, not treated as errors'])
    with (U/'VISIBLE_COMPARISON.json').open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2)
    print(json.dumps(counts,ensure_ascii=False))
if __name__=='__main__':main()
