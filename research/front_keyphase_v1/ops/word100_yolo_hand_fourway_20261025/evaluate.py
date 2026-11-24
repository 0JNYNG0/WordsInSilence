"""Saved-output scoring; no detector calls or threshold tuning."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,importlib.util
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    frozen=read(U/'FROZEN.json');index=read(U/'INDEX.json');manifest={r['clip_id']:r for r in frozen['rows']}
    assert len(index)==len(manifest)==39
    spec=importlib.util.spec_from_file_location('old_wrist_scorer',O/'word100_recovery_image_benchmark_20261025/evaluate_benchmark.py');score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
    R=O/'word100_recovery_image_benchmark_20261025';V=O/'word100_temporal_refiner_pilot_20261025'
    labels=[];ra={r['sample_id']:r for r in read(R/'BLIND_REVIEW_A.json')['rows']};rb={r['sample_id']:r for r in read(R/'BLIND_REVIEW_B.json')['rows']}
    for s in read(R/'SELECTION.json')['selection']:
        labels.append({'id':s['sample_id'],'clip_id':s['clip_id'],'frame_id':s['frame_id'],'side':s['anatomical_side'],'group':'R24','stratum':s['stratum'],'a':ra[s['sample_id']],'b':rb[s['sample_id']]})
    vl={r['sample_id']:r for r in read(V/'VISIBLE_LABELS_FROZEN.json')['labels']}
    for s in read(V/'VISIBLE_SAMPLE_SELECTION.json')['cases']:
        if s['clip_id'] not in manifest:continue
        lab=vl[s['sample_id']];a={'identity_confident':lab['scorable'],'visibility':'visible' if lab['scorable'] else 'ambiguous','wrist_box':lab['rectangle']}
        labels.append({'id':s['sample_id'],'clip_id':s['clip_id'],'frame_id':s['frame_id'],'side':s['side'],'group':'V18','stratum':s['selection_stratum'],'crossing':lab['crossing'],'a':a,'b':a})
    # V uses existing coarse consensus rectangles, not two newly independent labels.
    group_labels=defaultdict(list)
    for l in labels:group_labels[l['clip_id']].append(l)
    scored=[];raw_local=[];totals={c:Counter() for c in ['M0','M1','Y','MY']};timing=[];alladds=[];reasons=Counter();yreason=Counter();perclip=[];preserve=Counter();cross=[]
    crossing=read(O/'word100_image_arm_connectivity_20261018/CROSSING.json')
    crosssets=defaultdict(set)
    for q in crossing:
        if q['clip_id'] in manifest:crosssets[q['clip_id']].add(q['index'])
    for idx in index:
        assert sha(idx['arrays'])==idx['arrays_sha256'];assert sha(idx['records'])==idx['records_sha256']
        cid=idx['clip_id'];r=manifest[cid];z=load(r['arrays']);out=load(idx['arrays']);records=read(idx['records'])['frames'];n=r['frames']
        assert len(records)==n and np.array_equal(out['frame_id'],z['frame_id']) and np.array_equal(out['timestamp_us'],z['timestamp_us'])
        cond={'M0':(z['baseline_roi_xy_px'][:,33:].reshape(n,2,21,2),z['baseline_roi_usable'][:,33:].reshape(n,2,21)),
              'M1':(z['redetected_roi_xy_px'][:,33:].reshape(n,2,21,2),z['redetected_roi_usable'][:,33:].reshape(n,2,21)),
              'Y':(out['y_xy'],out['y_valid']),'MY':(out['my_xy'],out['my_valid'])}
        for c,(xy,valid) in cond.items():
            assert xy.shape==(n,2,21,2) and valid.shape==(n,2,21)
            assert np.isfinite(xy[valid]).all()
            x0,y0,x1,y1=z['roi_box'];p=xy[valid];assert ((p[:,0]>=x0)&(p[:,0]<x1)&(p[:,1]>=y0)&(p[:,1]<y1)).all()
            totals[c].update(frames=n,joint_frames=int(valid.sum()),wrist_frames=int(valid[:,:,0].sum()),complete_hand_frames=int(valid.all(-1).sum()))
        for c in ('M1','MY'):
            base,bv=cond['M0'];xy,v=cond[c]
            assert v[bv].all() and np.array_equal(xy[bv],base[bv])
            preserve[c+'_original_joint_slots']+=int(bv.sum())
            for i in crosssets[cid]:
                if i<n:
                    assert np.array_equal(xy[i][bv[i]],base[i][bv[i]])
                    cross.append({'clip_id':cid,'frame_id':i,'condition':c,'preserved_joints':int(bv[i].sum())})
        cc=Counter()
        for rec in records:
            i=rec['i'];assert i==rec['frame_id'];reasons.update(q['reason'] for q in rec['merge']);yreason.update(rec['y_reasons'])
            for q in rec['merge']:
                if q['accepted']:
                    side=q['side'];kept=int(cond['MY'][1][i,side].sum())
                    entry={'clip_id':cid,'frame_id':i,'side':side,'index':q['index'],'roi_kept_joints':kept,'label':r['display_label']}
                    alladds.append(entry);cc['accepted_native_hands']+=1;cc['added_roi_hands']+=kept>0
            cc['fresh_saved_hand_count_mismatch']+=rec['fresh_mp_hand_count']!=rec['saved_mp_hand_count']
            cc['raw_yolo_detections']+=len(rec['full'])
        perclip.append({'clip_id':cid,**cc});timing.append(out['timing_ms'])
        for lab in group_labels[cid]:
            i=lab['frame_id'];s=lab['side']
            for c,(xy,v) in cond.items():
                q=score.score_point(xy[i,s,0],bool(v[i,s,0]),lab['a'],lab['b'])
                scored.append({'id':lab['id'],'group':lab['group'],'stratum':lab['stratum'],'crossing':lab.get('crossing','metadata' if i in crosssets[cid] else 'unknown'),'clip_id':cid,'frame_id':i,'side':s,'condition':c,'available':bool(v[i,s,0]),'point':xy[i,s,0].tolist() if v[i,s,0] else None,**q})
            hits=[]
            for j,d in enumerate(records[i]['full']):
                if d['valid'][0]:
                    q=score.score_point(d['xy'][0],True,lab['a'],lab['b'])
                    hits.append({'index':j,'status':q['status'],'assigned_side':records[i]['y_sides'][j]})
            raw_local.append({'id':lab['id'],'group':lab['group'],'scorable':score.agreement(lab['a'],lab['b'])['scorable'],'hits':hits,'any_supported':any(q['status']=='inside_both' for q in hits)})
    summaries={}
    for group in ('R24','V18','ALL42'):
        summaries[group]={}
        for c in totals:
            rr=[q for q in scored if q['condition']==c and (group=='ALL42' or q['group']==group)]
            counts=Counter(q['status'] for q in rr);scorable=sum(q['status']!='unscorable' for q in rr);available=sum(q['status'] not in ('unscorable','unavailable') for q in rr)
            summaries[group][c]={'total':len(rr),'scorable':scorable,'available_scorable':available,'counts':dict(counts),'supported_of_scorable_pct':100*counts['inside_both']/scorable if scorable else None,
                                 'supported_of_available_pct':100*counts['inside_both']/available if available else None}
    t=np.concatenate(timing)
    def stats(v):return {'mean_ms':float(np.mean(v)),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95)),'sum_ms':float(v.sum())}
    # Measured detector components; shared saved-body identity costs NOT included.
    detector_time={'M0':t[:,1]+t[:,2],'M1':t[:,1]+t[:,2]+t[:,3], 'Y':t[:,1]+t[:,4], 'MY':t[:,1]+t[:,2]+t[:,4]+t[:,5]}
    timing_summary={c:stats(v) for c,v in detector_time.items()}
    paired={}
    base_by={(q['id'],q['condition']):q for q in scored}
    for c in ('M1','Y','MY'):
        gains=[];loss=[];newwrong=[]
        for lab in labels:
            a=base_by[(lab['id'],'M0')];b=base_by[(lab['id'],c)]
            if a['status']=='unscorable':continue
            if a['status']!='inside_both' and b['status']=='inside_both':gains.append(lab['id'])
            if a['status']=='inside_both' and b['status']!='inside_both':loss.append(lab['id'])
            if a['status']=='unavailable' and b['status'] in ('outside_both','inside_one_ambiguous'):newwrong.append(lab['id'])
        paired[c]={'new_supported_vs_M0':gains,'lost_supported_vs_M0':loss,'new_outside_or_ambiguous_vs_missing_M0':newwrong}
    summary={'status':'SCORED_NOT_ADOPTED','scope':{'clips':len(index),'frames':len(t),'words':frozen['words'],'people':frozen['people'],'coarse_samples':len(labels),'new_independent_people':0},
             'wrist_image_consistency':summaries,'coverage_not_accuracy':{c:dict(v) for c,v in totals.items()},'latency_cpu_detector_components':timing_summary,
             'latency_note':'Sums of separately measured serial detector calls, same fresh CPU session; excludes initialization, file decode, ROI/identity, TCN/DTW/web. Y still needs shared MediaPipe body for left/right assignment. Not measured webcam FPS.',
             'component_timing':{k:stats(t[:,i]) for i,k in enumerate(read(U/'RUNTIME.json')['timing_columns'])},
             'paired':paired,'preservation':dict(preserve),'crossing_metadata_frames':len(cross)//2,
             'accepted_native_hand_frames':len(alladds),'accepted_roi_hand_frames':sum(x['roi_kept_joints']>0 for x in alladds),'merge_reasons':dict(reasons),'y_side_reasons':dict(yreason),
             'raw_yolo_supported_localizations':{g:sum(q['any_supported'] and q['scorable'] for q in raw_local if q['group']==g) for g in ('R24','V18')},
             'fresh_saved_mp_hand_count_mismatches':sum(q['fresh_saved_hand_count_mismatch'] for q in perclip),
             'limitations':['Known development clips and coarse AI wrist ranges, not precise21joint truth','Finger accuracy and full occlusion restoration not measured','V18 is a separate older coarse range protocol','M0/M1 accuracy caches reused; latency is fresh component measurement','No TCN accuracy, DTW feedback accuracy or new-user performance claim'],
             'training':0,'classifier_inference':0,'user_mentor_access':0,'production_changes':0}
    for p,h in {**frozen['pins'],**frozen['protected']}.items():assert sha(p)==h,p
    write(U/'SCORED_ROWS.json',scored);write(U/'RAW_LOCALIZATION.json',raw_local);write(U/'ADDITIONS.json',alladds);write(U/'CROSSING_PRESERVATION.json',cross);write(U/'PER_CLIP.json',perclip);write(U/'SUMMARY.json',summary)
    print(json.dumps({'scope':summary['scope'],'wrist':summaries,'latency':timing_summary,'paired':paired,'added':len(alladds)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
