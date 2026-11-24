"""Separate arithmetic reconstruction, same Codex operator, no inference."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def run():
    f=read(U/'FROZEN.json');idx=read(U/'INDEX.json');scores=read(U/'SCORED_ROWS.json');summary=read(U/'SUMMARY.json');count=0
    for p,h in {**f['pins'],**f['protected']}.items():assert digest(p)==h;count+=1
    rdir=O/'word100_recovery_image_benchmark_20261025';vdir=O/'word100_temporal_refiner_pilot_20261025'
    aa={q['sample_id']:q for q in read(rdir/'BLIND_REVIEW_A.json')['rows']};bb={q['sample_id']:q for q in read(rdir/'BLIND_REVIEW_B.json')['rows']}
    vv={q['sample_id']:q for q in read(vdir/'VISIBLE_LABELS_FROZEN.json')['labels']}
    manifest={q['clip_id']:q for q in f['rows']};saved={q['clip_id']:q for q in idx};cache={};tim=[];preserved=Counter()
    for cid,r in manifest.items():
        t=saved[cid];assert digest(t['arrays'])==t['arrays_sha256'];assert digest(t['records'])==t['records_sha256'];count+=2
        z=load(r['arrays']);q=load(t['arrays']);n=r['frames'];bv=z['baseline_roi_usable'][:,33:].reshape(n,2,21);bx=z['baseline_roi_xy_px'][:,33:].reshape(n,2,21,2)
        assert np.array_equal(q['my_xy'][bv],bx[bv]);assert q['my_valid'][bv].all();count+=int(bv.sum())
        assert np.array_equal(q['frame_id'],z['frame_id']);assert np.array_equal(q['timestamp_us'],z['timestamp_us']);count+=2
        preserved['MY_original_joint_slots']+=int(bv.sum());tim.append(q['timing_ms'])
        cache[cid]=(z,q)
    for r in scores:
        z,q=cache[r['clip_id']];i=r['frame_id'];s=r['side'];k=33+21*s
        if r['condition'] in ('M0','M1'):
            pref='baseline_roi' if r['condition']=='M0' else 'redetected_roi';pt=z[pref+'_xy_px'][i,k];av=bool(z[pref+'_usable'][i,k])
        else:
            pre='y' if r['condition']=='Y' else 'my';pt=q[pre+'_xy'][i,s,0];av=bool(q[pre+'_valid'][i,s,0])
        sid=r['id']
        if sid.startswith('R'):
            a,b=aa[sid],bb[sid];boxes=[a['wrist_box'],b['wrist_box']]
            ok=a['visibility']==b['visibility']=='visible' and a['identity_confident'] and b['identity_confident'] and all(x is not None for x in boxes)
            if ok:ok=max(boxes[0][0],boxes[1][0])<min(boxes[0][2],boxes[1][2]) and max(boxes[0][1],boxes[1][1])<min(boxes[0][3],boxes[1][3])
        else:ok=vv[sid]['scorable'];boxes=[vv[sid]['rectangle']]*2
        if not ok:status='unscorable'
        elif not av:status='unavailable'
        else:
            inside=[b[0]<=pt[0]<=b[2] and b[1]<=pt[1]<=b[3] for b in boxes]
            status='inside_both' if all(inside) else 'inside_one_ambiguous' if any(inside) else 'outside_both'
        assert status==r['status'] and av==r['available'],r;count+=1
    for g,conditions in summary['wrist_image_consistency'].items():
        for c,stats in conditions.items():
            rr=[r for r in scores if r['condition']==c and (g=='ALL42' or r['group']==g)]
            assert dict(Counter(r['status'] for r in rr))==stats['counts'];assert len(rr)==stats['total'];count+=2
    t=np.concatenate(tim)
    for c,lat in {'M0':t[:,1]+t[:,2],'M1':t[:,1]+t[:,2]+t[:,3],'Y':t[:,1]+t[:,4],'MY':t[:,1]+t[:,2]+t[:,4]+t[:,5]}.items():
        a=summary['latency_cpu_detector_components'][c];assert abs(float(lat.mean())-a['mean_ms'])<1e-9;assert abs(float(np.percentile(lat,95))-a['p95_ms'])<1e-9;count+=2
    assert preserved['MY_original_joint_slots']==summary['preservation']['MY_original_joint_slots']
    result={'status':'PASS','assertions_including_preserved_joint_slots':count,'operator':'Codex separate arithmetic script, not independent human/agent','rows_reconstructed':len(scores),'protected':len(f['protected']),'clips':len(manifest),'source_and_output_hashes_verified':True,'training':0,'model_inference':0,'user_recordings':0}
    (U/'VERIFY_SAVED.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':run()
