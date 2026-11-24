"""Separate scalar arithmetic check, no image/pose inference rerun."""
from pathlib import Path
import json, math, hashlib, collections

u=Path(__file__).resolve().parents[1]; o=u/'metrics'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
raw=read(o/'RAW_EDGES.json'); cal=read(o/'CALIBRATION.json')
def pct(v,p=.9):
    a=sorted(v); i=(len(a)-1)*p; k=int(i)
    return a[k]+(a[min(k+1,len(a)-1)]-a[k])*(i-k)
def near(a,b):
    assert math.isfinite(a) and math.isfinite(b)
    err=abs(a-b); assert err<1e-8,(a,b,err)
    return err
maxerr=0.; status=collections.Counter(); reasons=collections.Counter(); comparisons=collections.Counter(); joint=collections.Counter()
for r in raw['rows']:
    rgb=r['rgb']; pose=r['pose']; role=r['evaluation_role']
    status[role+'_rgb_'+str(rgb['valid'])]+=1; status[role+'_pose_'+str(pose['valid'])]+=1
    if rgb['valid']:
        A=rgb['partial_affine']
        vals=[math.hypot(A[0][0]*p[0]+A[0][1]*p[1]+A[0][2]-q[0],A[1][0]*p[0]+A[1][1]*p[1]+A[1][2]-q[1]) for p,q in zip(rgb['from_points_px'],rgb['to_points_px'])]
        assert len(vals)==rgb['accepted_tracks']==len(rgb['residuals_px']) and len(vals)>=8
        for a,b in zip(vals,rgb['residuals_px']): maxerr=max(maxerr,near(a,b))
        width=sum(rgb['palm_widths_px'])/2
        maxerr=max(maxerr,near(width,rgb['palm_width_px']),near(pct(vals),rgb['residual_p90_px']),near(pct(vals)/width,rgb['score']),near(rgb['affine_inlier_count']/len(vals),rgb['affine_inlier_fraction']))
        assert rgb['affine_inlier_fraction']>=.5 and width>=10 and max(rgb['accepted_fb_errors_px'])<=1.5
    else:
        assert rgb['score'] is None
        reasons['rgb:'+rgb['reason']]+=1
    if pose['valid']:
        assert len(pose['absolute_angle_changes_deg'])==pose['valid_angle_count']==len(pose['triplet_indices']) and pose['valid_angle_count']>=8
        maxerr=max(maxerr,near(pct(pose['absolute_angle_changes_deg']),pose['score']))
    else:
        assert pose['score'] is None
        reasons['pose:'+pose['reason']]+=1
    if rgb['valid'] and pose['valid']:
        rg=rgb['score']<=cal['rgb_threshold']; po=pose['score']<=cal['pose_threshold']
        joint[role+':rgb_'+str(rg)+':pose_'+str(po)]+=1
        comparisons[role+(':no_change_signal' if rg and po else ':change_or_view_signal')]+=1
    else: comparisons[role+':insufficient']+=1
for m in ('rgb','pose'):
    vals=[r[m]['score'] for r in raw['rows'] if r['case_id'] in ['T07','T10'] and r[m]['valid']]
    assert len(vals)==cal[m+'_valid']>=12
    maxerr=max(maxerr,near(max(vals),cal[m+'_threshold']))
    assert sorted(vals)==sorted(cal[m+'_scores'])
result={
    'status':'PASS_SAVED_VALUE_ARITHMETIC',
    'review_type':'Implementer separate scalar arithmetic recheck, not independent human/algorithm accuracy',
    'edges':len(raw['rows']),'max_numeric_abs_error':maxerr,
    'checks':['Similarity residual recomputed from saved source/target points and matrix','p90 computed by independent sorted linear interpolation','Palm normalization and affine inlier fraction recomputed','Pose p90 recomputed from stored absolute angle changes','Both calibration maxima independently recovered from old T07/T10 raw edges','Invalid scores remain null'],
    'coverage':dict(status),'raw_failure_reasons':dict(reasons),
    'edge_guard_counts':dict(comparisons),'two_metric_cross_table':dict(joint),
    'limitations':['No RGB or detector inference was rerun','This verifies stored arithmetic, not correctness of observed landmarks or tracked image points','Both 2D projected-angle changes and residual image changes respond to view rotation/occlusion as well as articulation','Union of endpoint hand hulls plus dilation may include background or other-hand tracks; forward/backward agreement cannot establish physical hand identity','An every-edge all-pass temporal guard compounds abstention with path length','Threshold is maximum of two stable ID01 windows, not normal motion population tolerance'],
    'source_pins':{str(o/n):hashlib.sha256((o/n).read_bytes()).hexdigest() for n in ['RAW_EDGES.json','CALIBRATION.json','build_metrics.py']},
    'no_labels_loaded':True,'existing_outputs_changed':False,
    'prior_check_attempt_error':'A transient stdin checker omitted utf8 encoding and hit Windows cp949 UnicodeDecodeError before numeric checks or writes. No data or metric file was modified.'}
p=o/'ARITHMETIC_REVIEW.json'; assert not p.exists()
p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
