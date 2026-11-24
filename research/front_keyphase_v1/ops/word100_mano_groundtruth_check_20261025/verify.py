"""Separate saved-output arithmetic/preservation checks; same Codex, not expert review."""
from pathlib import Path
import math,json,hashlib
import numpy as np
from scipy.io import loadmat
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=read(U/'FROZEN.json');r=read(U/'CORRECTED_INFERENCE.json');s=read(U/'CORRECTED_SCORES.json');v=read(U/'PREINFERENCE_VISUAL_REVIEW.json')
with np.load(U/'truth.npz') as z:gt=z['xy_px'];ids=z['ids']
assert len(r['index'])==len(ids)==32 and list(ids)==[x['id'] for x in r['index']]
for p,h in f['prior_result_pins'].items():assert sha(p)==h,p
for p,h in f['protected']['protected'].items():assert sha(p)==h,p
for p,h in read(U/'CODE_PINS.json').items():assert sha(p)==h,p
for p,h in read(U/'SIDE_CORRECTION.json')['pins'].items():assert sha(p)==h,p
params=loadmat(U/'params.mat');xyz=loadmat(U/'pose_gt.mat')['pose_gt'];max_projection_error=0
means={k:{'all':[],'wrist':[],'fingers':[],'tips':[],'visible':[],'pck':0} for k in ['mp','hamer','fallback']}
improve=0
for j,(i,row) in enumerate(zip(ids,r['index'])):
    assert sha(row['prediction'])==row['sha256']
    fx,fy,cx,cy=params['cam_param'][i];x,y,w,h=params['bbox'][i]
    # Independent scalar projection, rather than importing the preparation or scoring functions.
    for k in range(21):
        X,Y,Z=map(float,xyz[i,k]);u=((X/Z)*fx+cx-x)*256/w;vv=((Y/Z)*fy+cy-y)*256/h
        max_projection_error=max(max_projection_error,abs(u-gt[j,k,0]),abs(vv-gt[j,k,1]))
    with np.load(row['prediction']) as z:
        with np.load(U/'predictions'/f'{i:05d}.npz') as old:assert np.array_equal(z['mp_xy'],old['mp_xy'])
        assert np.array_equal(z['fallback_xy'],z['mp_xy'])
        errs={}
        for m in means:
            a=z[m+'_xy'];assert np.isfinite(a).all();e=[]
            for k in range(21):
                dx=float(a[k,0])-float(gt[j,k,0]);dy=float(a[k,1])-float(gt[j,k,1]);d=math.sqrt(dx*dx+dy*dy);e.append(d)
                q=means[m];q['all'].append(d)
                if k==0:q['wrist'].append(d)
                else:q['fingers'].append(d)
                if k in [4,8,12,16,20]:q['tips'].append(d)
                if k in v['clearly_visible_fingertips'][str(i)]:q['visible'].append(d)
                q['pck']+=int(d<=math.hypot(256,256)*.05)
            errs[m]=math.fsum(e)/21
        improve+=int(errs['hamer']<errs['mp'])
assert max_projection_error<1e-8
for m,q in means.items():
    for a,b in [('all','all21_mean_px'),('wrist','wrist_mean_px'),('fingers','finger20_mean_px'),('tips','tip5_mean_px'),('visible','visible_tip_mean_px')]:assert abs(math.fsum(q[a])/len(q[a])-s['results'][m][b])<1e-8
    assert abs(100*q['pck']/672-s['results'][m]['PCK_0p05_diag_pct'])<1e-8
assert improve==s['paired_HaMeR_lower_mean_error']==10
summary={'status':'PASS_SAVED_NUMERICS_AND_PRESERVATION','images':32,'source_projection_max_difference_px':max_projection_error,'scored_points_each_condition':672,'visible_tip_points':len(means['mp']['visible']),'previous_result_files_unchanged':len(f['prior_result_pins']),'protected_project_files_unchanged':len(f['protected']['protected']),'MP_predictions_reused_exact':32,'fallback_identical_to_MP':32,'code_and_correction_pins_pass':True,'original_wrong_side_results_retained_not_used':True,'C_T3F_changed':False,'training':0,'user_data_access':0,'reviewer':'Separate check by same Codex; no independent expert annotation','limits':['No missing detection in these32','No time-sequence or two-hand crossing validation','Single dataset subset with unknown pretraining overlap','Visibility labels are conservative Codex review, not source expert labels']}
(U/'VERIFICATION.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
