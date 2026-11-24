"""Separate scalar calculation and saved-array/pin verification; not expert GT."""
from pathlib import Path
from math import hypot, isfinite
from statistics import median
import json, hashlib, time
import numpy as np

U=Path(__file__).resolve().parent
OLD=U.parent/'word100_mano_threeway_20261025'
def read(p): return json.loads(Path(p).read_text('utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b): return np.array_equal(a,b,equal_nan=True)
def distance(a,b): return hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))
def scalar(a,b,va,vb,s):
    if not isfinite(s) or s<=1:return False,False
    valid=[bool(va[j] and vb[j] and all(isfinite(float(x)) for x in (*a[j],*b[j]))) for j in range(21)]
    agree=[]
    for f in range(5):
        ix=list(range(1+4*f,5+4*f)); d=[distance(a[j],b[j]) for j in ix if valid[j]]
        agree.append(len(d)>=3 and valid[ix[-1]] and median(d)<=.08*s and distance(a[ix[-1]],b[ix[-1]])<=.08*s)
    v1=sum(agree)>=4
    tips=[4,8,12,16,20]
    if not v1 or not all(valid[j] for j in tips):return v1,False
    d=[[distance(a[i],b[j]) for j in tips] for i in tips]
    count=sum(agree[i] and all(d[i][i]<d[i][j] and d[i][i]<d[j][i] for j in range(5) if i!=j) for i in range(5))
    return v1,count>=4

def main():
    started=time.perf_counter(); f=read(U/'FROZEN.json'); v=read(U/'FROZEN_V2.json'); r=read(U/'RESULT_V2.json'); r1=read(U/'RESULT.json')
    assert sha(U/'candidate.py')==f['candidate_sha256']==v['candidate_v1_sha256']
    assert sha(U/'candidate_v2.py')==v['candidate_sha256']
    assert sha(U/'RESULT.json')==v['v1_results_sha256'] and sha(U/'FROZEN.json')==v['base_scope_sha256']
    for group in ('protected','prior_pins'):
        for path,h in f[group].items():assert sha(path)==h,path
    frames=points=cross=crosspoints=normal=changes=0
    requests=read(U/'REQUESTED_DIAGNOSTIC_V2.json'); checked_pairs=0
    by_clip={k:[x for x in requests if x['clip_index']==k] for k in range(7)}
    for k,m in enumerate(f['manifest']):
        for path_key,hash_key in [('prepared','prepared_sha256'),('arrays','arrays_sha256'),('rgb_path','rgb_sha256')]:assert sha(m[path_key])==m[hash_key]
        z=np.load(m['prepared']); p=np.load(OLD/'comparisons'/(m['clip_id']+'.npz')); n=np.load(U/'outputs_v2'/(m['clip_id']+'.npz')); n1=np.load(U/'outputs'/(m['clip_id']+'.npz'))
        for key in n.files:assert same(n[key],n1[key]),('visuals_not_reusable',m['clip_id'],key)
        # Reconstruct expected arrays from old results and independently calculated conflicts.
        ex={key:p[old].copy() for key,old in [('G_xy','H_xy'),('G_valid','H_valid'),('G_inferred','H_inferred'),('GT_xy','HT_xy'),('GT_valid','HT_valid'),('GT_inferred','HT_inferred')]}
        conflict=np.zeros(z['trigger'].shape,np.uint8)
        pose=z['pose'];pv=z['pose_valid']; full=z['full_xy'];fv=z['full_valid']; trig=z['trigger']
        for row in by_clip[k]:
            t=row['frame']; s=row['side']; a=np.load(OLD/f'raw_predictions/{k:02d}_{t:04d}_{s}.npz')['xy_px']; va=np.isfinite(a).all(-1)
            scale=distance(pose[t,11],pose[t,12]) if pv[t,[11,12]].all() else 0
            holds=[]
            for q in row['pairs']:
                side=q['existing_side']; _,flag=scalar(a,full[t,side],va,fv[t,side],scale)
                assert flag==q['hold_new'];checked_pairs+=1
                if flag:holds.append(side)
            assert bool(holds)==row['conflict']
            if not trig[t,s] or not holds:continue
            conflict[t,s]=1
            for side in holds:conflict[t,side]=1
            for target,old in [('G','H'),('GT','HT')]:
                if p[old+'_inferred'][t,s].any():
                    ex[target+'_xy'][t,s]=p['B_xy'][t,s];ex[target+'_valid'][t,s]=p['B_valid'][t,s];ex[target+'_inferred'][t,s]=False
        for key,a in ex.items():assert same(n[key],a),(m['clip_id'],key)
        assert same(n['identity_conflict'],conflict)
        b=p['B_valid'];frames+=len(b);points+=int(b.sum())
        for target in ['G','GT']:
            assert same(n[target+'_xy'][b],p['B_xy'][b])
            assert n[target+'_valid'][b].all()
            assert same(n[target+'_evidence'],p['B_evidence'])
        for key in ['frame_id','timestamp_us','roi_box']:assert same(n[key],p[key])
        for t in m['crossing_frames']:
            cross+=1;crosspoints+=int(p['B_valid'][t].sum())
            for target,old in [('G','H'),('GT','HT')]:
                for suffix in ['xy','valid','inferred']:assert same(n[target+'_'+suffix][t],p[old+'_'+suffix][t])
        if m['role']=='existing_observation_control':
            normal+=len(b)
            for target,old in [('G','H'),('GT','HT')]:
                for suffix in ['xy','valid','inferred']:assert same(n[target+'_'+suffix],p[old+'_'+suffix])
        changes+=int((p['H_inferred'].any(-1)&~n['G_inferred'].any(-1)).sum())
    all_flags=[[],[]]; broad_frames=complete=0
    for m in f['controls']:
        assert sha(m['path'])==m['sha256'] and sha(m['record'])==m['record_sha256']
        z=np.load(m['path']); xy=z['xy_px'];usable=z['usable'];ids=z['frame_id'];broad_frames+=len(ids)
        for t in range(len(ids)):
            if not usable[t,33:75].all() or not usable[t,[11,12]].all():continue
            sc=distance(xy[t,11],xy[t,12])
            if not isfinite(sc) or sc<=1:continue
            complete+=1
            flags=scalar(xy[t,33:54],xy[t,54:75],usable[t,33:54],usable[t,54:75],sc)
            for j,flag in enumerate(flags):
                if flag:all_flags[j].append((m['clip_id'],t,int(ids[t])))
    for j,rr in enumerate([r1,r]):
        expected=[(x['clip_id'],x['frame_index'],x['frame_id']) for x in rr['broad_proxy']['flags']]
        assert expected==all_flags[j]
        assert rr['broad_proxy']['complete_pairs']==complete and rr['broad_proxy']['frames']==broad_frames
    assert changes==7 and points==14149 and cross==20 and crosspoints==840 and frames==812
    assert normal==94,(normal,[x['role'] for x in f['manifest']])
    out={'status':'PASS','method':'Separate scalar math + bitwise saved array and SHA256 verification; same Codex author, not independent human or expert ground truth','primary_frames':frames,'existing_points_unchanged':points,'crossing_frames_unchanged':cross,'crossing_points_unchanged':crosspoints,'normal_control_frames_unchanged':normal,'changed_new_additions':changes,'requested_pair_decisions_verified':checked_pairs,'broad_frames':broad_frames,'broad_pairs':complete,'v1_flags':all_flags[0],'v2_flags':all_flags[1],'protected_files_unchanged':len(f['protected']),'prior_files_unchanged':len(f['prior_pins']),'v1_v2_primary_arrays_identical':True,'time_seconds':time.perf_counter()-started}
    dest=U/'VERIFICATION.json';assert not dest.exists();dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
