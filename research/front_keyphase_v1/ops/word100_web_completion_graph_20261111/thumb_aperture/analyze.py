"""FIT-only thumb-aperture observability; no deployed model inference or edits."""
from pathlib import Path
import sys, json, hashlib, importlib.util
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
G = U.parent
O = G.parent

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def angle(origin, a, b):
    x, y = a-origin, b-origin
    den = np.linalg.norm(x)*np.linalg.norm(y)
    if den <= 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(x,y)/den, -1, 1))))

def main():
    assert not (U/'RESULT.json').exists(), 'Completed observability result is immutable.'
    contract=read(U/'FROZEN.json')
    labels=read(G/'reviews/egypt_fit_shape_audit.json')
    assert sha(G/'reviews/egypt_fit_shape_audit.json') == contract['sources'][0]['sha256']
    rows=[r for r in labels['rows'] if r['observed_state'] in ['wide_projected_aperture','narrow_projected_aperture']]
    assert len(rows)==36 and {r['record_id'] for r in rows}=={1,4,7}
    # Metadata is filtered before loading arrays: do not load DEV rows via public_rows().
    ready=read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']
    selected={r['clip_id']:r for r in ready if r['clip_id'] in {a['clip_id'] for a in rows}}
    assert len(selected)==3 and all(r['role']=='fit' for r in selected.values())
    spec=importlib.util.spec_from_file_location('_thumb_requirement', O/'word100_guide_requirement_learning_20261111/requirement_core.py')
    rc=importlib.util.module_from_spec(spec);spec.loader.exec_module(rc)
    cache={};pins=[]
    for cid,r in selected.items():
        assert sha(r['path'])==r['sha256']
        with np.load(r['path'],allow_pickle=False) as z:
            xx=rc.prior.physical(z['x_full'])
            cache[cid]=(xx,z['frame_id'].copy(),z['timestamp_us'].copy())
        pins.append(dict(path=r['path'],sha256=r['sha256'],source_sha256=r['source_sha256'],clip_id=cid))
    out=[]
    tips=[4,8,12,16,20];mcps=[5,9,13,17];needed=sorted(set([0]+tips+mcps))
    for lab in rows:
        cid=lab['clip_id'];x,fids,ts=cache[cid];ix=np.flatnonzero(fids==lab['frame_id'])
        assert len(ix)==1 and lab['source_sha256']==selected[cid]['source_sha256']
        i=int(ix[0]);base=9 if lab['hand']=='L' else 30;side=0 if lab['hand']=='L' else 1
        p=x[i,:102].reshape(51,2)[base:base+21]
        bv=x[i,186:237][base:base+21]>.5
        hv=x[i,237+side*21:258+side*21]>.5
        good=bv&hv&np.isfinite(p).all(1)
        reasons=[]
        missing=[k for k in needed if not good[k]]
        if missing:reasons.append('missing_or_masked_hand_points:'+','.join(map(str,missing)))
        scale=float(np.median(np.linalg.norm(p[mcps]-p[0],axis=1))) if good[[0]+mcps].all() else None
        if scale is None or scale<=1e-6:reasons.append('palm_scale_unavailable_or_degenerate')
        distances=[None]*4;wa=None;ma=None
        if not reasons:
            distances=[float(np.linalg.norm(p[4]-p[j])/scale) for j in tips[1:]]
            wa=angle(p[0],p[4],p[8]);ma=angle(p[5],p[4],p[8])
            if wa is None or ma is None:reasons.append('angle_degenerate')
        out.append(dict(lab,sequence_index=i,source_timestamp_us=int(ts[i]),mask_points_needed=needed,
                        body_hand_valid=[bool(v) for v in bv],handlocal_valid=[bool(v) for v in hv],
                        eligible=not reasons,ineligible_reasons=reasons,palm_scale_in_body_units=scale,
                        thumb_to_fingertips_norm=distances,thumb_to_index_wrist_angle_deg=wa,
                        thumb_to_index_mcp_angle_deg=ma,hand_xy_body_normalized=p.tolist(),
                        not_ground_truth_coordinates=True))
    summary=[]
    for person in [1,4,7]:
        for hand in ['R','L']:
            states={}
            for state in ['wide_projected_aperture','narrow_projected_aperture']:
                rs=[r for r in out if r['record_id']==person and r['hand']==hand and r['observed_state']==state]
                good=[r for r in rs if r['eligible']]
                vals=np.asarray([[*r['thumb_to_fingertips_norm'],r['thumb_to_index_wrist_angle_deg'],r['thumb_to_index_mcp_angle_deg']] for r in good],float)
                states[state]=dict(n=len(rs),eligible=len(good),median=np.median(vals,axis=0).tolist() if len(good) else None,
                    min=vals.min(0).tolist() if len(good) else None,max=vals.max(0).tolist() if len(good) else None)
            a=states['wide_projected_aperture']['median'];b=states['narrow_projected_aperture']['median']
            summary.append(dict(record_id=person,hand=hand,states=states,wide_minus_narrow=[q-r for q,r in zip(a,b)] if a is not None and b is not None else None))
    result=dict(status='FIT_OBSERVABILITY_ONLY',rows=out,summary=summary,features=['thumb_index_dist','thumb_middle_dist','thumb_ring_dist','thumb_little_dist','wrist_apex_angle','index_mcp_apex_angle'],
        counts=dict(public_fit_videos=3,hand_frame_denominator=len(out),eligible=sum(r['eligible'] for r in out),unknown=sum(not r['eligible'] for r in out)),
        pins=pins,protocol_sha256=sha(U/'FROZEN.json'),code_sha256=sha(__file__),dev_loaded=False,training_ran=False,model_inference_ran=False,
        caution='Valid mask only means coordinates were retained; it does not verify their actual pixel accuracy. All labels are single-Codex RGB drafts, not contact/fist truth.')
    write(U/'RESULT.json',result)
    print(json.dumps(dict(counts=result['counts'],summary=summary),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
