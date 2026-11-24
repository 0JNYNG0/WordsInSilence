"""Read-only TRAIN899 body-geometry availability audit; no detector/model calls."""
from pathlib import Path
import json
import math
import sys
import numpy as np

sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent
OPS=U.parent
LEGACY=Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def quant(v):
    v=np.asarray([x for x in v if x is not None and math.isfinite(x)],float)
    return {'n':len(v),'min':float(v.min()),'p10':float(np.quantile(v,.1)),
            'median':float(np.median(v)),'p75':float(np.quantile(v,.75)),
            'p90':float(np.quantile(v,.9)), 'max':float(v.max())} if len(v) else {'n':0}
def q(v,p=.5):return float(np.quantile(v,p)) if len(v) else None

def main():
    all_rows=read(OPS/'word100_training_20261013/MANIFEST.json')
    rows=[r for r in all_rows if r['role']=='train_candidate']
    assert len(rows)==899 and {r['signer_id'] for r in rows}==set(range(1,10))
    # Use only already reported TRAIN calibration metadata, never DEV coordinate arrays.
    old=read(OPS/'word100_roi_training_20261017/root_audit_legacy_preflight_all_RESULT.json')
    old={r['clip_id']:r['calibration'] for r in old['results'] if r['role']=='train_candidate'}
    out=[]
    for row in rows:
        cid=row['clip_id'];rec=read(LEGACY/'records'/f'{cid}.json')
        with np.load(rec['observation'],allow_pickle=False) as z:
            xy=z['xy_px'].astype(float); valid=z['observed'] & z['usable'] & np.isfinite(xy).all(-1)
            ts=z['timestamp_us']; frames=len(ts)
        sc=read(Path(rec['observation']).with_suffix('.sidecar.json'))
        w,h=sc['width'],sc['height']
        valid &= (xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<h)
        sh=valid[:,11]&valid[:,12]
        scales=np.linalg.norm(xy[:,11]-xy[:,12],axis=-1)
        sh &= scales>1
        scale=q(scales[sh]); cy=q(((xy[:,11,1]+xy[:,12,1])*.5)[sh])
        prefix=(ts-ts[0])<=1_000_000
        psh=sh&prefix
        item={'clip_id':cid,'signer_id':row['signer_id'],'frames':frames,'source_role':'train_candidate',
              'shoulder_frames':int(sh.sum()),'prefix_shoulder_frames':int(psh.sum()),
              'shoulder_scale':scale,'shoulder_center_y':cy,'prefix_scale':q(scales[psh]),
              'prefix_center_y':q(((xy[:,11,1]+xy[:,12,1])*.5)[psh]),'source_height':h}
        side_results=[]
        for name,si,ei,wi in [('left',11,13,15),('right',12,14,16)]:
            up=sh&valid[:,si]&valid[:,ei];fo=sh&valid[:,ei]&valid[:,wi]
            ul=np.linalg.norm(xy[:,si]-xy[:,ei],axis=-1);fl=np.linalg.norm(xy[:,ei]-xy[:,wi],axis=-1)
            entry={'side':name,'upper_frames':int(up.sum()),'fore_frames':int(fo.sum()),
                   'upper_median':q(ul[up]),'fore_median':q(fl[fo]),'upper_p90':q(ul[up],.9),'fore_p90':q(fl[fo],.9)}
            if up.any() and fo.any() and scale is not None:
                entry['length_ratio_median']=(entry['upper_median']+entry['fore_median'])/scale
                entry['length_ratio_p90']=(entry['upper_p90']+entry['fore_p90'])/scale
            side_results.append(entry)
        item['sides']=side_results
        if all('length_ratio_median' in e for e in side_results):
            item['virtual_median_bottom']=cy+min(e['upper_median']+e['fore_median'] for e in side_results)-.02*scale
            item['virtual_p90_bottom']=cy+min(e['upper_p90']+e['fore_p90'] for e in side_results)-.02*scale
        cal=old[cid];item['old_calibrated']=cal['status']=='CALIBRATED'
        if item['old_calibrated']:
            item['old_bottom']=cal['bottom']
            item['old_rest_ratio_fullclip']=(cal['bottom']-cy)/scale
            item['old_rest_ratio']=(cal['bottom']-item['prefix_center_y'])/item['prefix_scale']
        out.append(item)
    # One descriptive training-only coefficient; not selected using class/recognition scores.
    k=q([r['old_rest_ratio'] for r in out if r['old_calibrated']])
    ks={name:q([r['old_rest_ratio'] for r in out if r['old_calibrated']], p)
        for name,p in [('median',.5),('p75',.75),('p90',.9)]}
    for r in out:
        if r['shoulder_scale'] is not None:
            r['fixed_ratio_descriptive_bottom']=r['prefix_center_y']+k*r['prefix_scale']
            r['fixed_1p5_bottom']=r['prefix_center_y']+1.5*r['prefix_scale']
            if not r['old_calibrated']:
                r['fallback_first1s_bodies']={name:r['prefix_center_y']+v*r['prefix_scale'] for name,v in ks.items()}
    summary={'status':'TRAIN_ONLY_GEOMETRY_DESIGN_AUDIT_NOT_NEW_BOUNDARY_IMPLEMENTATION',
             'rows':len(out),'coordinate_arrays_read_train':899,'coordinate_arrays_read_dev':0,'detector':0,'training':0,
             'shoulder_available':sum(r['shoulder_scale'] is not None for r in out),
             'both_arm_lengths_available':sum('virtual_p90_bottom' in r for r in out),
             'old_calibrated':sum(r['old_calibrated'] for r in out),
             'old_rest_ratio':quant([r.get('old_rest_ratio') for r in out]),
             'descriptive_global_train_rest_ratio_k':k,
             'first1s_descriptive_coefficients':ks,
             'length_ratios_median':quant([s.get('length_ratio_median') for r in out for s in r['sides']]),
             'length_ratios_p90':quant([s.get('length_ratio_p90') for r in out for s in r['sides']]),
             'prefix_vs_full_shoulder_scale_ratio':quant([r['prefix_scale']/r['shoulder_scale'] for r in out if r['prefix_scale'] and r['shoulder_scale']]),
             'by_signer':{str(s):{'n':sum(r['signer_id']==s for r in out),'rest_ratio':quant([r.get('old_rest_ratio') for r in out if r['signer_id']==s]),
                 'arm_length_ratio_p90':quant([ss.get('length_ratio_p90') for r in out if r['signer_id']==s for ss in r['sides']])} for s in range(1,10)},
             'candidate_geometry_deltas_vs_old_boundary':{},'not_calibrated_train_rows':[r for r in out if not r['old_calibrated']]}
    for mode in ['virtual_median_bottom','virtual_p90_bottom','fixed_ratio_descriptive_bottom','fixed_1p5_bottom']:
        eligible=[r for r in out if r['old_calibrated'] and mode in r]
        summary['candidate_geometry_deltas_vs_old_boundary'][mode]={'delta_pixels':quant([r[mode]-r['old_bottom'] for r in eligible]),
             'above_old_by_more_than20px':sum(r[mode]<r['old_bottom']-20 for r in eligible),
             'below_old_by_more_than20px':sum(r[mode]>r['old_bottom']+20 for r in eligible)}
    result={'summary':summary,'rows':out,
            'limitation':'Legacy pose observations, no new joint or RGB extraction. Existing rest boundary is a comparison reference not truth. All-clip arm projection remains gesture-dependent. k is descriptive training geometry, not an adopted constant.'}
    with (U/'design_data_audit_RESULT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
