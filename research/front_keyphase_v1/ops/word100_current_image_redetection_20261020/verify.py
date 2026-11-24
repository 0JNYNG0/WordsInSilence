"""Saved-output reconstruction; no detector/candidate imports or inference."""
import json,hashlib,itertools
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def inside(p,w,h):
    p=np.asarray(p);return np.isfinite(p).all(-1)&(p[...,0]>=0)&(p[...,0]<w)&(p[...,1]>=0)&(p[...,1]<h)
P=[0,5,9,13,17]

def main():
    f=read(U/'FROZEN.json');idx=read(U/'INDEX.json');counts=Counter();reasons=Counter()
    additions=[];speed=[];lat={'image':[],'crop':[]};normal=0;errors=0;crossing=0
    for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
    ann=read(U.parent/'word100_image_arm_connectivity_20261018/ANNOTATED.json')
    cross=read(U.parent/'word100_image_arm_connectivity_20261018/CROSSING.json')
    rows={r['clip_id']:r for r in f['rows']}
    for item in idx:
        assert sha(item['path'])==item['sha256']
        r=rows[item['clip_id']];x=read(item['path']);z=load(r['views']['sensor']['raw'])
        ass=load(r['views']['sensor']['assignment'])['sides'];obs=load(r['views']['sensor']['observation'])
        assert len(x['frames'])==len(z['frame_id'])
        for i,e in enumerate(x['frames']):
            n=int(z['raw_count'][i]);assert e['baseline_sides']==ass[i,:n].tolist()
            w,h=int(z['width'][i]),int(z['height'][i]);pose=z['pose_xy_px'][i];good=z['pose_usable'][i]&inside(pose,w,h)
            scale=float(np.linalg.norm(pose[11]-pose[12])) if good[[11,12]].all() else 0
            for mode in ['image','crop']:
                lat[mode].append(e[f'{"full" if mode=="image" else "crop"}_ms'])
                for d in e[mode]:
                    if mode=='crop':
                        b=d['box'];assert 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h
                checks=e[mode+'_checks'];expected=[]
                for q in checks:
                    d=e[mode][q['index']];p=np.array(d['xy']);reason='invalid_geometry';s=-1
                    if inside(p[P],w,h).all() and scale>1:
                        distances=[np.linalg.norm(p[0]-pose[k])/scale if good[[j,k]].all() else np.inf for j,k in [(13,15),(14,16)]]
                        s=int(np.argmin(distances))
                        if distances[s]>.25 or distances[1-s]-distances[s]<.10:reason='arm_uncertain'
                        elif s in e['baseline_sides']:reason='preserve_existing_side'
                        elif any(np.isfinite(b[P]).all() and np.linalg.norm(b[P].mean(0)-p[P].mean(0))<.16*scale for b in z['hand_xy_px'][i,:n]):reason='duplicate_existing_detection'
                        elif 'box' in d and not inside(p[P]-np.array(d['box'][:2]),d['box'][2]-d['box'][0],d['box'][3]-d['box'][1]).all():reason='crop_boundary_uncertain'
                        else:reason='current_image_and_arm'
                    expected.append([s,reason])
                for s in [0,1]:
                    same=[j for j,t in enumerate(expected) if t==[s,'current_image_and_arm']]
                    if len(same)>1:
                        for j in same:expected[j][1]='multiple_candidates_same_side'
                ok=[j for j,t in enumerate(expected) if t[1]=='current_image_and_arm']
                if len(ok)==2:
                    centers=[np.array(e[mode][j]['xy'])[P].mean(0) for j in ok]
                    if np.linalg.norm(centers[0]-centers[1])<.16*scale:
                        for j in ok:expected[j][1]='new_candidates_overlap_uncertain'
                for j,q in enumerate(checks):
                    s,reason=expected[j]
                    if mode=='crop' and reason=='current_image_and_arm' and s!=e[mode][j]['requested_side']:reason='different_arm_than_crop_request'
                    assert q['reason']==reason,(x['clip_id'],i,j,mode,q['reason'],reason)
                    assert q['accepted']==(reason=='current_image_and_arm')
                    reasons[mode+'_'+reason]+=1;counts['decisions_checked']+=1
                    if q['accepted']:
                        assert s not in e['baseline_sides'];counts[mode+'_additions']+=1
                        counts[mode+('_core' if e['core'] else '_outside_annotation')]+=1
                        additions.append(dict(clip_id=x['clip_id'],label=x['label'],index=i,frame_id=e['frame_id'],
                            time=e['timestamp_us']/1e6,mode=mode,side=s,core=e['core'],center=q['center'],candidate_index=j))
            counts['frames']+=1
        for a in ann:
            if a['clip_id']==x['clip_id'] and a['view']=='sensor':
                actual=int(ass[a['index'],a['raw_index']]);assert x['frames'][a['index']]['baseline_sides'][a['raw_index']]==actual
                if a['verdict']=='normal_identity':normal+=1;assert actual==a['actual_side']
                elif a['verdict']=='error_identity':errors+=1
        crossing+=sum(a['clip_id']==x['clip_id'] and a['view']=='sensor' for a in cross)
        if x['speed']:
            base=x['speed'][0]['frames']
            for run in x['speed'][1:]:
                count_changed=0;label_changed=0;maxdiff=0;diff_frames=0
                for b,t in zip(base,run['frames']):
                    bh,th=b['hands'],t['hands']
                    if len(bh)!=len(th):count_changed+=1;continue
                    if not bh:continue
                    perms=list(itertools.permutations(range(len(th))))
                    order=min(perms,key=lambda perm:sum(np.linalg.norm(np.array(bh[j]['xy'])-np.array(th[k]['xy'])) for j,k in enumerate(perm)))
                    delta=max(float(np.abs(np.array(bh[j]['xy'])-np.array(th[k]['xy'])).max()) for j,k in enumerate(order))
                    maxdiff=max(maxdiff,delta);diff_frames+=delta>1e-5
                    label_changed+=any(bh[j]['label']!=th[k]['label'] for j,k in enumerate(order))
                speed.append(dict(clip_id=x['clip_id'],factor=run['factor'],pass_index=run['pass_index'],
                    frames=len(base),count_changed=count_changed,label_changed=label_changed,
                    max_xy_delta_px=maxdiff,xy_changed_frames=diff_frames))
    report=dict(status='SAVED_RECONSTRUCTION_PASS_PENDING_VISUAL',clips=len(idx),counts=dict(counts),
        reasons=dict(reasons),normal_preserved=normal,known_errors_unchanged=errors,crossing_preserved=crossing,
        speed=speed,latency_ms={k:dict(median=float(np.median(v)),p95=float(np.quantile(v,.95)),sum=float(sum(v))) for k,v in lat.items()},
        protected_files=len(f['protected']),user_data=0,training=0,production_change=0,
        limitations='Preserved observations by design; additions require image review. Existing MediaPipe detections do not establish correct detailed finger angles. Same TRAIN engineering comparison; no classifier accuracy measured.')
    write(U/'VERIFY.json',report);write(U/'ADDITIONS.json',additions)
    print(json.dumps({k:v for k,v in report.items() if k not in ['reasons','speed']},ensure_ascii=False));print('SPEED',speed)
if __name__=='__main__':main()
