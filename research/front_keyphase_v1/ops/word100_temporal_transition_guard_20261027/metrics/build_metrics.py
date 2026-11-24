"""Frozen common RGB/pose transition metrics on saved native observations.
No handshape labels, target gloss, new detector, training, or source modification.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,time,traceback
import numpy as np
import cv2
from PIL import Image

U=Path(__file__).resolve().parents[1]; O=U/'metrics'; OPS=U.parent
OLD=OPS/'word100_temporal_shape_context_20261027'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(name,value):
    p=O/name
    if p.exists():raise FileExistsError(str(p))
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def missing(reason,**kw):return dict(valid=False,score=None,reason=reason,**kw)

def angle_score(a,b,va,vb,conf):
    values=[]; indices=[]
    for k,tr in enumerate(conf['triplets']):
        if not np.all(va[tr]&vb[tr]):continue
        angles=[]
        for p in (a,b):
            x=p[tr[0]]-p[tr[1]];y=p[tr[2]]-p[tr[1]]
            nx=np.linalg.norm(x);ny=np.linalg.norm(y)
            if min(nx,ny)<conf['minimum_segment_px']:break
            angles.append(float(np.degrees(np.arccos(np.clip(np.dot(x,y)/(nx*ny),-1,1)))))
        if len(angles)==2:values.append(abs(angles[1]-angles[0]));indices.append(k)
    detail=dict(valid_angle_count=len(values),triplet_indices=indices,absolute_angle_changes_deg=values,score_unit='degrees')
    if len(values)<conf['minimum_angles']:return missing('insufficient_observed_nondegenerate_angle_pairs',**detail)
    return dict(valid=True,score=float(np.percentile(values,90)),reason='observed_2d_angle_pairs',**detail)

def image_score(gray0,gray1,a,b,va,vb,conf):
    h,w=gray0.shape
    # Both observed MCP spans are necessary; one hidden palm is not normalized using the other alone.
    if not np.all(va[[5,17]]&vb[[5,17]]):return missing('palm_width_not_observed_in_both_frames')
    widths=[float(np.linalg.norm(p[5]-p[17])) for p in (a,b)]
    width=float(np.median(widths))
    if width<10:return missing('palm_width_below_10px',palm_widths_px=widths,palm_width_px=width)
    pts0=a[va];pts1=b[vb]
    if len(pts0)<3 or len(pts1)<3:return missing('insufficient_observed_hand_hull_points',palm_width_px=width)
    mask=np.zeros_like(gray0,dtype=np.uint8)
    for pts in (pts0,pts1):
        hull=cv2.convexHull(np.round(pts).astype(np.int32))
        if abs(cv2.contourArea(hull))==0:return missing('degenerate_hand_hull',palm_width_px=width)
        cv2.fillConvexPoly(mask,hull,255)
    radius=conf['hull_dilation_px']
    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    p0=cv2.goodFeaturesToTrack(gray0,maxCorners=conf['max_corners'],qualityLevel=conf['quality_level'],minDistance=conf['min_distance_px'],mask=mask)
    detail=dict(palm_widths_px=widths,palm_width_px=width,hull_mask_pixels=int(np.count_nonzero(mask)),initial_corners=0 if p0 is None else len(p0))
    if p0 is None or len(p0)<conf['minimum_tracks']:return missing('insufficient_corners',**detail)
    args=dict(winSize=(conf['lk_window'],conf['lk_window']),maxLevel=conf['lk_levels'],criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
    p1,st1,_=cv2.calcOpticalFlowPyrLK(gray0,gray1,p0,None,**args)
    if p1 is None:return missing('forward_tracking_failed',**detail)
    back,st0,_=cv2.calcOpticalFlowPyrLK(gray1,gray0,p1,None,**args)
    if back is None:return missing('backward_tracking_failed',**detail)
    q0=p0.reshape(-1,2);q1=p1.reshape(-1,2);qb=back.reshape(-1,2)
    fb=np.linalg.norm(qb-q0,axis=1)
    okay=(st0.reshape(-1)>0)&(st1.reshape(-1)>0)&np.isfinite(q1).all(1)&np.isfinite(qb).all(1)&(fb<=conf['fb_max_px'])
    inside=(q1[:,0]>=0)&(q1[:,0]<w)&(q1[:,1]>=0)&(q1[:,1]<h)
    okay&=inside
    target_in_hull=np.zeros(len(q1),bool)
    inds=np.flatnonzero(inside & np.isfinite(q1).all(1))
    ix=np.minimum(w-1,np.maximum(0,np.round(q1[inds,0]).astype(int)))
    iy=np.minimum(h-1,np.maximum(0,np.round(q1[inds,1]).astype(int)))
    target_in_hull[inds]=mask[iy,ix]>0
    okay&=target_in_hull
    q0=q0[okay];q1=q1[okay]
    detail.update(accepted_tracks=len(q0),accepted_fb_errors_px=fb[okay].astype(float).tolist())
    if len(q0)<conf['minimum_tracks']:return missing('insufficient_forward_backward_hull_tracks',**detail)
    cv2.setRNGSeed(conf['seed'])
    matrix,inliers=cv2.estimateAffinePartial2D(q0,q1,method=cv2.RANSAC,ransacReprojThreshold=conf['affine_ransac_px'],maxIters=2000,confidence=0.99,refineIters=10)
    if matrix is None or inliers is None:return missing('partial_affine_fit_failed',**detail)
    fraction=float(np.mean(inliers))
    detail.update(partial_affine=matrix.tolist(),affine_inlier_count=int(inliers.sum()),affine_inlier_fraction=fraction,
                  from_points_px=q0.astype(float).tolist(),to_points_px=q1.astype(float).tolist())
    if fraction<0.5:return missing('partial_affine_inliers_below_half',**detail)
    predicted=q0@matrix[:,:2].T+matrix[:,2]
    residual=np.linalg.norm(predicted-q1,axis=1)
    p90=float(np.percentile(residual,90))
    detail.update(residuals_px=residual.tolist(),residual_p90_px=p90,score_unit='palm_width',residual_population='all_accepted_FB_tracks_not_only_RANSAC_inliers')
    return dict(valid=True,score=p90/width,reason='global_similarity_compensated_local_residual',**detail)

def main():
    started=time.perf_counter()
    if (O/'RAW_EDGES.json').exists() or (O/'CALIBRATION.json').exists():raise FileExistsError('Saved metrics/calibration exist; do not rescore.')
    f=read(U/'FROZEN.json');rc=f['rgb_metrics'];pc=f['pose_metrics']
    pins={str(U/'FROZEN.json'):sha(U/'FROZEN.json'),str(Path(__file__).resolve()):sha(__file__)}
    for p,h in f['pins'].items():assert sha(p)==h;pins[p]=h
    contract=dict(frozen_sha256=pins[str(U/'FROZEN.json')],coefficient_source='FROZEN.json',
       observation='native.xy_px; require observed AND usable AND finite AND inside original image per point, no interpolation',
       rgb='RGB originals converted to grayscale; union of both observed hand convex hulls, 8px dilation; both endpoint tracks inside union; ShiTomasi, forward/backward LK, partial-affine RANSAC, p90 of ALL accepted track residuals / median of two observed MCP5-to17 widths',
       optical_flow_defaults=dict(lk_criteria_count=30,lk_criteria_epsilon=0.01,ransac_max_iters=2000,ransac_confidence=0.99,ransac_refine_iters=10),
       calibration='Evaluate the 24 T07/T10 edges first. Each metric needs >=12 valid; fixed threshold=max(valid). No label reads or score-driven tuning.',
       provenance_limit='Projection, camera/view change, blur, occlusion, bad saved keypoints and true articulation can all affect these scores. They do not certify linguistic correctness.',
       no_labels_loaded=True,no_new_detector=True,no_training=True)
    write('IMPLEMENTATION_CONTRACT.json',contract);pins[str(O/'IMPLEMENTATION_CONTRACT.json')]=sha(O/'IMPLEMENTATION_CONTRACT.json')
    oldidx=read(OLD/'visual_inputs/INDEX.json');newidx=read(U/'visual_inputs/INDEX.json')
    shared=read(OPS/'word100_shared_feature_bundle_20261027/FROZEN.json')
    records={r['clip_id']:r for r in shared['rows']}
    for p in (OLD/'visual_inputs/INDEX.json',U/'visual_inputs/INDEX.json',OPS/'word100_shared_feature_bundle_20261027/FROZEN.json'):pins[str(p)]=sha(p)
    allcases={r['case_id']:(r,'old_diagnostic') for r in oldidx['rows']}
    allcases.update({r['case_id']:(r,'new_evaluation') for r in newidx['rows']})
    calibration_ids=[r['case_id'] for r in f['calibration']['sources']]
    assert calibration_ids==['T07','T10']
    sequence_order=calibration_ids+[cid for cid in sorted(allcases) if cid not in calibration_ids]
    rows=[];case_summaries=[];source_frames=0
    for case_number,cid in enumerate(sequence_order):
        case,role=allcases[cid]; sr=records[case['clip_id']]
        rp=Path(sr['record']);assert sha(rp)==sr['record_sha256'];record=read(rp)
        assert record['source']['sha256']==case['source_sha256']
        assert sha(case['source'])==case['source_sha256'];pins[case['source']]=case['source_sha256']
        op=Path(record['views']['native']['observation']);sp=op.with_suffix('.sidecar.json');sc=read(sp)
        assert sha(op)==sc['observation_sha256'] and sc['source_sha256']==case['source_sha256']
        for p in (rp,op,sp):pins[str(p)]=sha(p)
        z=np.load(op,allow_pickle=False);start=33+({'L':0,'R':1}[case['side']])*21
        xy=z['xy_px'][:,start:start+21].astype(np.float64)
        valid=z['observed'][:,start:start+21]&z['usable'][:,start:start+21]&np.isfinite(xy).all(-1)
        frames=sorted(case['frames'],key=lambda x:x['actual_pts_us']);images=[]
        assert len(frames)==13
        for fr in frames:
            arr=np.asarray(Image.open(fr['full_file']).convert('RGB'))
            assert hashlib.sha256(arr.tobytes()).hexdigest()==fr['rgb_sha256']
            assert int(z['frame_id'][fr['frame_id']])==fr['frame_id'] and abs(int(z['timestamp_us'][fr['frame_id']])-fr['actual_pts_us'])<=1
            images.append(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY));source_frames+=1
        h,w=images[0].shape
        valid&=(xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<h)
        current=[]
        for i in range(12):
            a=frames[i]['frame_id'];b=frames[i+1]['frame_id'];assert b==a+1
            entry=dict(case_id=cid,edge_index=i,from_frame_id=a,to_frame_id=b,from_pts_us=frames[i]['actual_pts_us'],to_pts_us=frames[i+1]['actual_pts_us'],
                from_image_key=frames[i]['image_key'],to_image_key=frames[i+1]['image_key'],side=case['side'],evaluation_role=role,
                calibration_edge=cid in calibration_ids,recorded_person_id=case['recorded_person_id'],
                observed_points_from=int(valid[a].sum()),observed_points_to=int(valid[b].sum()),
                rgb=image_score(images[i],images[i+1],xy[a],xy[b],valid[a],valid[b],rc),
                pose=angle_score(xy[a],xy[b],valid[a],valid[b],pc))
            current.append(entry);rows.append(entry)
        summary=dict(case_id=cid,role=role,edges=12,rgb_valid=sum(e['rgb']['valid'] for e in current),pose_valid=sum(e['pose']['valid'] for e in current))
        case_summaries.append(summary)
        print(json.dumps(summary),flush=True)
        if case_number==1:
            calibration={metric:[e[metric]['score'] for e in rows if e[metric]['valid']] for metric in ('rgb','pose')}
            ok=all(len(v)>=12 for v in calibration.values())
            cal=dict(status='PASS_FIXED_MAX' if ok else 'FAIL_INSUFFICIENT_CALIBRATION',calibration_case_ids=calibration_ids,
                calibration_edges=24,rgb_valid=len(calibration['rgb']),pose_valid=len(calibration['pose']),
                rgb_threshold=max(calibration['rgb']) if len(calibration['rgb'])>=12 else None,
                pose_threshold=max(calibration['pose']) if len(calibration['pose'])>=12 else None,
                rgb_scores=calibration['rgb'],pose_scores=calibration['pose'],threshold_rule='maximum_valid_score_no_floor_no_search',
                frozen_sha256=pins[str(U/'FROZEN.json')],implementation_sha256=pins[str(Path(__file__).resolve())],
                scope='Only two previous stable recordedID01 windows, not a population normal range.',calibration_rows=rows.copy())
            write('CALIBRATION.json',cal)
            if not ok:
                write('RAW_EDGES_PARTIAL_CALIBRATION_FAILURE.json',dict(rows=rows,pins=pins))
                raise RuntimeError('Frozen calibration has fewer than 12 valid edges for a metric; no threshold relaxation or candidate comparison.')
    assert len(rows)==432 and len(case_summaries)==36 and source_frames==468
    rows.sort(key=lambda r:(r['case_id'],r['edge_index']))
    assert all(sha(p)==h for p,h in pins.items())
    write('RAW_EDGES.json',dict(status='COMPLETE_UNLABELLED_METRICS',rows=rows,case_summaries=case_summaries,pins=pins,
         calibration_file=str(O/'CALIBRATION.json'),no_labels_loaded=True,no_new_detector=True,no_training=True,
         fixed_parameters=f['rgb_metrics']|{'pose':f['pose_metrics']},scope='216 new edges plus216 old diagnostic/calibration edges. Not correctness labels.'))
    write('VERIFY_SELF.json',dict(status='PASS',source_pins_preserved=len(pins),windows=36,edges=432,decoded_saved_RGB_hashes_checked=468,
        observations_match_actual_frame_and_pts=True,coarse_labels_read=False,coefficient_search=False,new_detector=False,training=False,
        execution_seconds=time.perf_counter()-started,numpy=np.__version__,opencv=cv2.__version__,source_outputs_overwritten=False))

if __name__=='__main__':
    try:main()
    except Exception:
        p=O/'FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
