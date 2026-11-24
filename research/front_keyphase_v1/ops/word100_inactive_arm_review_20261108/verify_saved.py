"""Separate arithmetic/hash verification; no model inference or private input."""
from pathlib import Path
import json, hashlib
import numpy as np

U=Path(__file__).resolve().parent; O=U.parent
J=O/'word100_rtmw_joint_learning_20261108'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks=[]
    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))
    f=read(J/'FROZEN.json')
    cf=read(U/'COUNTERFACTUAL.json')
    for view,key in [('full','baseline'),('upper_mask','saved_upper_mask')]:
        saved={r['clip_id']:r for r in read(J/f'tcn_runs/finetune_C_20261013/PREDICTIONS_{view}.json')}
        for r in cf['rows']:
            v=np.asarray(saved[r['clip_id']]['logits']); truth=f['classes'].index(r['word'])
            check(view+' '+r['clip_id'],int(np.sum(v>v[truth]))+1==r[key]['truth_rank']
                  and [f['labels'][i] for i in np.argsort(-v)[:3]]==r[key]['top3'])
    for w,summary in cf['summary'].items():
        rows=[r for r in cf['rows'] if r['word']==w]
        check(w+' summary N',len(rows)==summary['clips']==3)
        for k,v in summary['conditions'].items():
            check(w+' '+k,v['top1_correct']==sum(r['conditions'][k]['truth_rank']==1 for r in rows)
                  and v['truth_ranks']==[r['conditions'][k]['truth_rank'] for r in rows])
    crops=read(U/'CROP_PROBE.json'); plan=read(U/'CROP_FROZEN.json')
    for r,p in zip(crops['rows'],plan['rows']):
        z=np.load(U/(r['clip_id']+'_crop.npz'),allow_pickle=False); old=np.load(p['raw'],allow_pickle=False)
        check(r['clip_id']+' time/frame identity',np.array_equal(z['frame_id'],old['frame_id']) and np.array_equal(z['timestamp_us'],old['timestamp_us']) and len(z['frame_id'])==r['frames'])
        l,t,rr,b=p['crop']
        for side,start in [('L',91),('R',112)]:
            q=old['xy_px'][:,start:start+21]; a=z['xy_px'][:,start:start+21]
            outside=np.count_nonzero((q[:,:,0]<l)|(q[:,:,0]>=rr)|(q[:,:,1]<t)|(q[:,:,1]>=b),axis=1)>=18
            inside=np.count_nonzero(np.isfinite(a).all(2)&(a[:,:,0]>=0)&(a[:,:,0]<640)&(a[:,:,1]>=0)&(a[:,:,1]<480),axis=1)>=18
            stats=r['geometric_comparison'][side]
            check(r['clip_id']+' '+side+' proxy recount',int(outside.sum())==stats['baseline_hand_outside_crop_frames'] and int((outside&inside).sum())==stats['new_inframe_hand_in_those_frames'])
        check(r['clip_id']+' source hash',sha(p['source'])==p['source_sha256'])
    attr=read(U/'ATTRIBUTION.json')
    for r in attr['rows']:
        truth=f['classes'].index(r['word'])
        for k,v in r['conditions'].items():
            logits=np.asarray(v['logits']); rank=int(np.sum(logits>logits[truth]))+1
            probs=np.exp(logits-logits.max()); probs/=probs.sum()
            check(r['clip_id']+' '+k+' ranking',rank==v['truth_rank'] and (rank==1)==v['top1_correct'] and [f['labels'][i] for i in np.argsort(-logits)[:3]]==v['top3'] and abs(probs[truth]-v['truth_probability'])<1e-6)
    pins={**read(O/'word100_action_feedback_demo_20261108/FROZEN.json')['protected'],**f['pins'],**read(U/'FROZEN.json')['pins']}
    for p,h in pins.items(): check('preserved '+p,sha(p)==h)
    check('public crop frame total',sum(r['frames'] for r in crops['rows'])==505)
    out=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,
             check_count=len(checks),protected_hash_count=len(pins),
             verifier='Same Codex, separate saved-array/logit arithmetic script; not an independent person or expert ground truth',
             inference_repeated=False,training=False,private_pixels_or_keypoints_read=False,
             visual_review=dict(clip='NIA_SL_WORD0739_REAL11_F',frames=[63,80,98],finding='Left hand not visible in cropped RGB; left-hand skeleton spans toward visible right hand and across empty image regions.',scope='Three diagnostic public frames only; not a 505-frame human annotation'))
    (U/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='checks'},ensure_ascii=False))
    assert out['status']=='PASS'

if __name__=='__main__': main()
