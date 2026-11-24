from common import *
from collections import Counter,defaultdict
import time
f=read(U/'FROZEN.json');idx=read(U/'INDEX.json');oldix={r['clip_id']:r for r in read(OLD/'INDEX.json')}
manifest={r['clip_id']:r for r in f['old_rows']+f['new_rows']}
score=module('selective_wrist_score',O/'word100_recovery_image_benchmark_20261025/evaluate_benchmark.py')
ra={r['sample_id']:r for r in read(O/'word100_recovery_image_benchmark_20261025/BLIND_REVIEW_A.json')['rows']}
rb={r['sample_id']:r for r in read(O/'word100_recovery_image_benchmark_20261025/BLIND_REVIEW_B.json')['rows']}
vl={r['sample_id']:r for r in read(O/'word100_temporal_refiner_pilot_20261025/VISIBLE_LABELS_FROZEN.json')['labels']}
oldscore=[r for r in read(OLD/'SCORED_ROWS.json') if r['condition']=='M1'];byclip=defaultdict(list)
for r in oldscore:byclip[r['clip_id']].append(r)
scored=[];adds=[];reasons=Counter();cohorts={};timing=[];oldfull=[];perclip=[];preserve=Counter();knowncross=read(OLD/'CROSSING_PRESERVATION.json');crosscount=0
for ix in idx:
    r=manifest[ix['clip_id']];z=load(r['arrays']);a=load(ix['arrays']);rec=read(ix['records'])['frames'];cid=ix['clip_id'];n=r['frames'];co=cohorts.setdefault(ix['cohort'],Counter())
    co.update(clips=1,frames=n,called_frames=int(a['called'].sum()),called_hand_slots=int(a['called_sides'].sum()),position_only_hand_frames=int(a['position_only'].sum()))
    base=z['redetected_roi_xy_px'][:,33:].reshape(n,2,21,2);bv=z['redetected_roi_usable'][:,33:].reshape(n,2,21)
    for c in ['s','p']:
        assert np.array_equal(a[c+'_xy'][bv],base[bv]) and a[c+'_valid'][bv].all()
        preserve[c]+=int(bv.sum())
    for q in knowncross:
        if q['clip_id']==cid and q['condition']=='M1':
            i=q['frame_id'];assert np.array_equal(a['p_xy'][i][bv[i]],base[i][bv[i]]);crosscount+=1
    for rr in rec:
        i=rr['i'];reasons.update(q['reason'] for q in rr['decisions'])
        for q in rr['decisions']:
            if q['accepted']:
                s=q['side'];nkeep=int(a['p_valid'][i,s].sum());entry={'clip_id':cid,'frame_id':i,'side':s,'position_only':q['position_only'],'roi_joints':nkeep,'cohort':ix['cohort'],'label':r['display_label']}
                adds.append(entry);co['new_native_hands']+=1;co['new_roi_hands']+=nkeep>0
    if cid in oldix:
        ot=load(oldix[cid]['arrays'])['timing_ms'];m1=ot[:,1]+ot[:,2]+ot[:,3];timing.append(np.stack([m1,a['yolo_ms'],a['geometry_ms']],axis=-1));oldfull.append(ot[:,4]+ot[:,5])
    for lab in byclip[cid]:
        sid=lab['id'];i=lab['frame_id'];s=lab['side']
        if sid.startswith('R'):aa,bb=ra[sid],rb[sid]
        else:
            l=vl[sid];aa=bb={'identity_confident':l['scorable'],'visibility':'visible' if l['scorable'] else 'ambiguous','wrist_box':l['rectangle']}
        for c in ('M1','S','P'):
            v=bool(bv[i,s,0]) if c=='M1' else bool(a[c.lower()+'_valid'][i,s,0]);p=base[i,s,0] if c=='M1' else a[c.lower()+'_xy'][i,s,0]
            out=score.score_point(p,v,aa,bb)
            if c=='M1':assert out['status']==lab['status']
            scored.append({**{k:lab[k] for k in ['id','group','clip_id','frame_id','side','stratum']},'condition':c,'available':v,'point':p.tolist() if v else None,'position_only':bool(a['position_only'][i,s]) if c=='P' else False,**out})
    perclip.append({'clip_id':cid,'cohort':ix['cohort'],'frames':n,'calls':int(a['called'].sum()),'position_only':int(a['position_only'].sum()),'yolo_time_total_ms':float(a['yolo_ms'].sum())})
summary={};paired={};lookup={(r['id'],r['condition']):r for r in scored}
for group in ['R24','V18','ALL42']:
    summary[group]={}
    for c in ['M1','S','P']:
        rows=[r for r in scored if r['condition']==c and (group=='ALL42' or r['group']==group)];cnt=Counter(r['status'] for r in rows)
        summary[group][c]={'total':len(rows),'scorable':sum(r['status']!='unscorable' for r in rows),'counts':dict(cnt)}
for c in ('S','P'):
    gains=[];loss=[];wrong=[]
    for r in oldscore:
        a=lookup[r['id'],'M1'];b=lookup[r['id'],c]
        if a['status']=='unscorable':continue
        if a['status']!='inside_both' and b['status']=='inside_both':gains.append(r['id'])
        if a['status']=='inside_both' and b['status']!='inside_both':loss.append(r['id'])
        if a['status']=='unavailable' and b['status'] in ('outside_both','inside_one_ambiguous'):wrong.append(r['id'])
    paired[c]={'gained':gains,'lost':loss,'new_outside_or_ambiguous':wrong}
t=np.concatenate(timing);of=np.concatenate(oldfull)
def stats(v):return dict(mean_ms=float(v.mean()),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),sum_ms=float(v.sum()))
latency={'M1_prior_components':stats(t[:,0]),'new_selective_yolo_over_all_frames':stats(t[:,1]),'new_selective_yolo_on_called_frames':stats(t[:,1][t[:,1]>0]),'M1_plus_selective_components':stats(t[:,0]+t[:,1]),'previous_yolo_all_frames_component':stats(of)}
res={'status':'EVALUATED_RESEARCH_ONLY','cohorts':{k:dict(v) for k,v in cohorts.items()},'scoring':summary,'paired_vs_M1':paired,'preserved_M1_joint_frames':dict(preserve),'historical_crossing_frames_preserved':crosscount,
 'new_added_hand_frames':len(adds),'position_only_hand_frames':sum(r['position_only'] for r in adds),'reasons':dict(reasons),'latency':latency,
 'note':'Prior M1 timing+fresh conditional YOLO components, excludes decode/geometry/TCN/DTW. Same known diagnostic42 coarse wrist labels; 07 extension not annotated21GT. No training/user-data/runtime replacement.',
 'five_decisions':[{**lookup[s,'P'],'baseline_status':lookup[s,'M1']['status']} for s in ['R01','R02','R14','R21','R22']]}
write(U/'SUMMARY.json',res);write(U/'SCORED_ROWS.json',scored);write(U/'ADDITIONS.json',adds);write(U/'PER_CLIP.json',perclip)
print(json.dumps(res,ensure_ascii=False,indent=2))
