"""After RGB freeze, inspect generic coordinate computability. No model scoring."""
from pathlib import Path
import json,hashlib,importlib.util,sys
from collections import Counter
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent; O=U.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def angle(o,a,b):
    va=a-o;vb=b-o;den=float(np.linalg.norm(va)*np.linalg.norm(vb))
    if den<=1e-6:return None
    return float(np.degrees(np.arccos(np.clip(np.dot(va,vb)/den,-1,1))))
def features(p):
    scale=float(np.median(np.linalg.norm(p[[5,9,13,17]]-p[0],axis=1)))
    if scale<=1e-6:return None
    return [float(np.linalg.norm(p[4]-p[j])/scale) for j in [8,12,16,20]]+[angle(p[0],p[4],p[8]),angle(p[5],p[4],p[8])]

def main():
    target=U/'EXPANDED_COORDINATE_VALIDITY.json'; assert not target.exists()
    freeze=read(U/'EXPANDED_RGB_FREEZE.json'); labels=read(freeze['label_file'])
    assert sha(freeze['label_file'])==freeze['sha256']
    ids={q['clip_id'] for q in labels['rows']}
    selected={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['clip_id'] in ids}
    assert len(selected)==9 and all(r['role']=='fit' and r['person'] in [1,2,3] for r in selected.values())
    spec=importlib.util.spec_from_file_location('_expanded_requirement',O/'word100_guide_requirement_learning_20261111/requirement_core.py')
    rc=importlib.util.module_from_spec(spec);spec.loader.exec_module(rc)
    cache={};raws={};pins=[]
    for cid,meta in selected.items():
        assert sha(meta['path'])==meta['sha256']
        with np.load(meta['path'],allow_pickle=False) as z:
            cache[cid]=dict(x=rc.prior.physical(z['x_full']),frame_id=z['frame_id'].copy(),timestamp_us=z['timestamp_us'].copy())
        rec=read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')
        assert rec['role']=='fit' and rec['source_sha256']==meta['source_sha256']
        assert sha(rec['raw_path'])==rec['raw_sha256']
        with np.load(rec['raw_path'],allow_pickle=False) as z:
            raws[cid]={k:z[k].copy() for k in ['xy_px','frame_id','timestamp_us']}
        pins.append(dict(clip_id=cid,path=meta['path'],sha256=meta['sha256'],raw_path=rec['raw_path'],raw_sha256=rec['raw_sha256'],source_sha256=meta['source_sha256']))
    out=[];needed=[0,4,5,8,9,12,13,16,17,20]
    for lab in labels['rows']:
        z=cache[lab['clip_id']]; meta=selected[lab['clip_id']]
        ii=np.flatnonzero(z['frame_id']==lab['frame_id']);assert len(ii)==1
        i=int(ii[0]);x=z['x'][i];base=9 if lab['hand']=='L' else 30;side=0 if lab['hand']=='L' else 1
        assert meta['source_sha256']==lab['source_sha256']
        # PNG exporter integer-rounded timestamps can differ by one microsecond.
        delta=int(z['timestamp_us'][i])-int(lab['timestamp_us']);assert abs(delta)<=1
        p=x[:102].reshape(51,2)[base:base+21]
        bv=x[186:237][base:base+21]>.5;hv=x[237+side*21:258+side*21]>.5
        good=bv&hv&np.isfinite(p).all(1);missing=[k for k in needed if not good[k]]
        vals=features(p) if not missing else None
        reasons=[]
        if missing:reasons.append('missing_or_masked_points:'+','.join(map(str,missing)))
        if vals is None or any(v is None for v in vals): reasons.append('feature_unavailable')
        raw=raws[lab['clip_id']];ri=np.flatnonzero(raw['frame_id']==lab['frame_id']);assert len(ri)==1
        assert int(raw['timestamp_us'][ri[0]])==int(z['timestamp_us'][i])
        checks=None
        if not reasons:
            start=91 if lab['hand']=='L' else 112
            pv=raw['xy_px'][ri[0],start:start+21].astype(np.float64);expected=features(pv)
            diff=np.abs(np.asarray(expected)-np.asarray(vals))
            assert diff[:4].max()<1e-4 and diff[4:].max()<.002
            checks=dict(max_distance_error=float(diff[:4].max()),max_angle_error_deg=float(diff[4:].max()),source_frame_timestamp_exact=True)
        out.append(dict(label_id=lab['label_id'],clip_id=lab['clip_id'],record_id=lab['record_id'],hand=lab['hand'],frame_id=lab['frame_id'],
            rgb_state=lab['observed_state'],rgb_applicability=lab['applicability'],source_timestamp_us=int(z['timestamp_us'][i]),
            timestamp_delta_from_rgb_metadata_us=delta,body_hand_valid=bv.tolist(),handlocal_valid=hv.tolist(),
            coordinate_computable=not reasons,coordinate_exclusion_reasons=reasons,features=vals,
            label_eligible_for_aperture_only=not reasons and lab['applicability']=='applicable' and lab['observed_state']!='unknown',
            raw_pixel_arithmetic_check=checks))
    groups=[]
    for state in ['wide_projected_aperture','narrow_projected_aperture','unknown']:
        rs=[r for r in out if r['rgb_state']==state]
        groups.append(dict(rgb_state=state,denominator=len(rs),computable=sum(r['coordinate_computable'] for r in rs),
            label_eligible=sum(r['label_eligible_for_aperture_only'] for r in rs)))
    result=dict(status='RGB_FIRST_FIT_OBSERVABILITY_ONLY',label_file=freeze['label_file'],label_sha256=freeze['sha256'],
        rows=out,counts=dict(denominator=len(out),clips=9,expressions=3,record_ids=[1,2,3],coordinate_computable=sum(r['coordinate_computable'] for r in out),
          aperture_label_eligible=sum(r['label_eligible_for_aperture_only'] for r in out),rgb_unknown=sum(r['rgb_state']=='unknown' for r in out)),
        state_groups=groups,pins=pins,formulas=dict(scale='median wrist0 to MCP5,9,13,17 distance',features=['thumb4-index8 / scale','thumb4-middle12 / scale','thumb4-ring16 / scale','thumb4-little20 / scale','wrist0 apex thumb4-index8 angle','indexMCP5 apex thumb4-index8 angle']),
        policy='Identical formulas for both hands, no word-ID branches. Unknown or not-applicable RGB remains unknown even if coordinates compute.',
        conclusion='Only three additional positive wide labels; no additional narrow labels. This does not support a generic learned narrow classifier. Masks or distances must not convert curled, index-only or thumb-hidden examples into fingertip-gathering truth.',
        labels_unchanged=sha(freeze['label_file'])==freeze['sha256'],new_training=False,dev_loaded=False,model_scored=False,private_access=False,
        independent_human_ground_truth=False,verification='Separate raw-pixel and normalized-coordinate arithmetic by same Codex, not independent anatomical annotation',code_sha256=sha(__file__))
    assert result['labels_unchanged'];write(target,result)
    print(json.dumps(dict(counts=result['counts'],groups=groups),ensure_ascii=False))

if __name__=='__main__':main()
