from common import *
import adapter as A
from collections import Counter
import shutil,traceback

def freeze():
    assert not (U/'FROZEN.json').exists()
    yf=read(Y/'FROZEN.json');hf=read(H/'FROZEN.json');df=read(D/'FROZEN.json')
    check(hf['protected']);rows=yf['old_rows']+yf['new_rows'];ids={r['clip_id'] for r in rows}
    assert all(r['signer_id'] in range(1,10) for r in rows)
    pairs=[p for p in read(D/'RESULT.json')['cross'] if p['kind']=='identity' and p['query'] in ids and p['reference'] in ids]
    paths=[Y/'FROZEN.json',Y/'INDEX.json',H/'FROZEN.json',H/'SEPARATED_EVIDENCE.json',D/'FROZEN.json',D/'RESULT.json',
           Y/'ADDITIONS.json',O/'word100_finger_evidence_20261020/candidate.py',O/'word100_angle_evidence_20261021/candidate.py',
           O/'word100_yolo_hand_fourway_20261025/CROSSING_PRESERVATION.json']+list(U.glob('*.py'))
    for r in rows:paths+=[Path(r['arrays'])]+[Path(p) for p in r['source_assets'].values()]
    for r in read(Y/'INDEX.json'):paths+=[Path(r['arrays']),Path(r['records'])]
    for r in hf['manifest']:paths +=[H/'outputs'/(r['clip_id']+'.npz')]
    for p in pairs:paths +=[D/'cross'/(p['name']+'.npz'),I/(p['reference']+'.npz')]
    (U/'before').mkdir()
    for p in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json'):shutil.copyfile(p,U/'before'/p.name)
    write(U/'FROZEN.json',dict(at=now(),authorization='Autonomous public evidence integration and comparison, existing models preserved.',
        rows=rows,pairs=pairs,hamer_manifest=hf['manifest'],protected=hf['protected'],pins={str(p.resolve()):sha(p) for p in paths},
        conditions=['native MP','MP retry','MP retry + selective YOLO'],
        preparation='Full-source native -> accepted full-source MP retry -> accepted YOLO -> same original ROI. Preserve all old valid coordinates. Fixed original pose-derived shoulder scale for all three; no hand-size recomputation. Existing21 quality sidecars exactly reused; other25 get same frozen two-prediction bone/angle geometry. Same word labels used only for reference linkage.',
        identity='Preserve all raw observations; existing HaMeR identity_eligible=false side-frames marked conflict in all3 conditions. No forced assignment. New YOLO accepted current-arm provenance recorded. Region proposals and HaMeR inferred coordinates separate, never compared as current observed evidence.',
        measurement='All46 public known TRAIN clips, all frames. Same previously saved cross-person DTW paths for available21 clips/3words; reference always fixed MP retry with same conflict sidecar. Reuse old21-channel expected domain and append hand-wrist2 using palm domain. No query-driven domain shrink, no new alignment. Average only when entire contributing query span supported. This measures evidence plumbing, not alignment or anatomical/recognition accuracy.',
        eligibility='Angle4 means two-prediction geometry not conflicted, not ground truth. Current wrist/palm present and no known identity conflict means position comparable. Independent arm information remains. Unknown difference NaN. No correctness verdict or correction direction.',
        acceptance='All valid old coordinates and body channels preserved; inherited21 angle states exact; false identity not promoted; HaMeR/region not observed; fixed denominator; wrist-only cannot create palm/finger; crossing preserved. Report actual added/lost evidence and comparison, no forced improvement.',
        forbidden=['user/mentor inputs','extra199/200','new training','new detector inference','operating changes','Git','download','upload']))
    write(U/'SELFTEST.json',A.selftest());print('FROZEN',len(rows),len(pairs),flush=True)

def prepare():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected'])
    assert not (U/'RUNNING.json').exists();write(U/'RUNNING.json',dict(at=now()))
    (U/'inputs').mkdir();(U/'comparisons').mkdir()
    Q=module('recovery_geometry',O/'word100_finger_evidence_20261020/candidate.py')
    E=module('recovery_angle',O/'word100_angle_evidence_20261021/candidate.py')
    yi={r['clip_id']:r for r in read(Y/'INDEX.json')};hi={r['clip_id']:r for r in f['hamer_manifest']}
    regions=read(H/'SEPARATED_EVIDENCE.json');rows=[];counts=Counter()
    for r in f['rows']:
        cid=r['clip_id'];z=load(r['arrays']);yo=load(yi[cid]['arrays']);n=len(z['frame_id']);box=z['roi_box']
        rec=read(r['source_assets']['detections'])['frames'];assert len(rec)==n
        yrec={p['i']:p for p in read(yi[cid]['records'])['frames']}
        nx=z['native_xy_px'][:,33:].reshape(n,2,21,2);nv=z['native_usable'][:,33:].reshape(n,2,21)
        full=nx.copy();fv=nv.copy();origin=np.where(nv,1,0).astype(np.uint8)
        for i in range(n):
            for q in rec[i]['decisions']:
                if not q['accepted']:continue
                s=q['side'];assert not fv[i,s].any();p=np.asarray(rec[i]['full'][q['index']]['xy']);v=np.asarray(q['joint_supported'],bool)
                full[i,s]=np.where(v[:,None],p,np.nan);fv[i,s]=v;origin[i,s]=np.where(v,2,0)
        m1,mv=A.roi(full,fv,box);base=z['redetected_roi_xy_px'][:,33:].reshape(n,2,21,2);bv=z['redetected_roi_usable'][:,33:].reshape(n,2,21)
        assert same(mv,bv) and np.allclose(m1[bv],base[bv],atol=.0002,rtol=0)
        # Save original stored float32 values exactly after reconstruction proof.
        m1=base.copy();pfull=full.copy();pfv=fv.copy();porigin=origin.copy()
        for i,rr in yrec.items():
            for q in rr['decisions']:
                if not q['accepted']:continue
                s=q['side'];assert not pfv[i,s].any();v=np.asarray(q['joint_supported'],bool);p=np.asarray(rr['full'][q['index']]['xy'])
                pfull[i,s]=np.where(v[:,None],p,np.nan);pfv[i,s]=v;porigin[i,s]=np.where(v,3,0)
        pp,pv=A.roi(pfull,pfv,box);assert same(pv,yo['p_valid']) and np.allclose(pp[pv],yo['p_xy'][pv],atol=.0002,rtol=0)
        pp=yo['p_xy'].copy();assert same(pp[bv],base[bv]) and pv[bv].all()
        xs=[];vs=[];x0,v0=A.roi(nx,nv,box);xs=[x0,m1,pp];vs=[v0,mv,pv]
        angle_states=[];angle_values=[];quality_pairs=[]
        for condition,(x,v) in enumerate(zip(xs,vs)):
            pair_ok=np.zeros((n,2),bool);av=np.zeros((n,2,15),bool);au=av.copy()
            for i in range(n):
                for s in range(2):
                    b,bb=A.pair(rec[i],s,bool(nv[i,s].any()),v[i,s],box)
                    if condition==0 and not nv[i,s].any():b=bb=None
                    if condition==2 and np.any(pv[i,s]&~mv[i,s]):
                        qq=[q for q in yrec[i]['decisions'] if q['accepted'] and q['side']==s];assert len(qq)==1
                        b=np.asarray(yrec[i]['crop'][qq[0]['crop_index']]['xy']);bb=v[i,s]&A.inside(b,box)
                    q=Q.compare_shape(x[i,s],v[i,s],b,bb)
                    pair_ok[i,s]=q['pair_available'];av[i,s]=q['angle_available'];au[i,s]=q['angle_unresolved']
            evidence=E.angle_evidence(x,v,pair_ok,av,au)
            if 'angle_evidence' in r['source_assets']:
                inherited=load(r['source_assets']['angle_evidence'])
                if condition==1:assert same(evidence['angle_state'],inherited['angle_state'])
            angle_states.append(evidence['angle_state']);angle_values.append(evidence['angle_deg']);quality_pairs.append(pair_ok)
        conflict=np.zeros((n,2),bool);inferred=np.full((n,2,21,2),np.nan,np.float32);iv=np.zeros((n,2,21),bool)
        region=np.zeros((n,2),bool)
        if cid in hi:
            hz=load(H/'outputs'/(cid+'.npz'));conflict=~hz['identity_eligible']
            iv=hz['R_inferred'];assert iv.shape==(n,2,21)
            inferred=np.where(iv[...,None],hz['R_xy'],np.nan)
            for a in regions:
                if a['clip']==cid and a['new_region_identity_proposal']==a['side']:region[a['frame'],a['side']]=True
        pose=z['native_xy_px'][:,:33];pok=z['native_usable'][:,:33]&A.inside(pose,box)
        vals=[];raw=[];eligible=[];scales=[]
        for x,v,st in zip(xs,vs,angle_states):
            val,ava,eli,scale=A.channel_view(x,v,st,pose,pok,conflict);vals.append(val);raw.append(ava);eligible.append(eli);scales.append(scale)
        assert scales[0]==scales[1]==scales[2]
        oo=np.stack([np.where(v0,1,0),np.where(mv,origin,0),np.where(pv,porigin,0)]).astype(np.uint8)
        output=dict(xy=np.stack(xs),valid=np.stack(vs),point_origin=oo,full_final_xy=pfull,full_final_valid=pfv,
            angle_state=np.stack(angle_states),angle_degree=np.stack(angle_values),quality_pair=np.stack(quality_pairs),
            values=np.stack(vals),available=np.stack(raw),eligible=np.stack(eligible),identity_conflict=conflict,
            inferred_xy=inferred,inferred_valid=iv,region_advisory=region,pose_xy=pose,pose_valid=pok,
            frame_id=z['frame_id'],timestamp_us=z['timestamp_us'],roi_box=box,body_scale=np.asarray(scales))
        np.savez_compressed(U/'inputs'/(cid+'.npz'),**output)
        counts.update(clips=1,frames=n,old_valid_points=int(mv.sum()),yolo_new_points=int((pv&~mv).sum()),
            identity_conflict_hand_frames=int(conflict.sum()),inferred_points_separate=int(iv.sum()),region_advisory_hand_frames=int(region.sum()))
        row=dict(clip_id=cid,word_code=r['word_code'],signer_id=r['signer_id'],frames=n,
            observed_joint_frames=[int(v.sum()) for v in vs],angle_evidence=[int((s==4).sum()) for s in angle_states],
            feedback_eligible=[int(e.sum()) for e in eligible],yolo_wrist_only=int(yo['position_only'].sum()),
            identity_conflicts=int(conflict.sum()),file_sha256=sha(U/'inputs'/(cid+'.npz')))
        rows.append(row)
    write(U/'INPUT_RESULT.json',dict(at=now(),counts=dict(counts),rows=rows,training=0,user_access=0,operating_changes=0))
    (U/'RUNNING.json').unlink();print('INPUTS',json.dumps(dict(counts)),flush=True)

def compare():
    f=read(U/'FROZEN.json');assert not (U/'COMPARE_RUNNING.json').exists();write(U/'COMPARE_RUNNING.json',dict(at=now()))
    inputs={r['clip_id']:load(U/'inputs'/(r['clip_id']+'.npz')) for r in f['rows']};rows=[];totals={}
    for p in f['pairs']:
        q=inputs[p['query']];r=inputs[p['reference']];old=load(D/'cross'/(p['name']+'.npz'))
        domain=np.concatenate([old['expected'],old['expected'][:,:,15:17]],-1)
        assert len(q['frame_id'])==int(old['path'][:,0].max()+1) and len(r['frame_id'])==len(domain)
        anchor=dict(values=r['values'][1],eligible=r['eligible'][1]);result=[]
        for k in range(3):result.append(A.compared(dict(values=q['values'][k],eligible=q['eligible'][k]),anchor,domain,old['path']))
        out=dict(path=old['path'],expected=domain,compared=np.stack([a['compared'] for a in result]),
            difference=np.stack([a['difference'] for a in result]),query_span=result[0]['query_span'],timestamp_us=r['timestamp_us'])
        np.savez_compressed(U/'comparisons'/(p['name']+'.npz'),**out)
        groups={}
        for group,(lo,hi) in A.GROUPS.items():
            masks=out['compared'][:,:,:,lo:hi];cnt=[int(v.sum()) for v in masks]
            g=dict(expected=int(domain[:,:,lo:hi].sum()),compared=cnt,
                   added_retry=int((masks[1]&~masks[0]).sum()),lost_retry=int((masks[0]&~masks[1]).sum()),
                   added_yolo=int((masks[2]&~masks[1]).sum()),lost_yolo=int((masks[1]&~masks[2]).sum()))
            groups[group]=g;t=totals.setdefault(group,Counter())
            for key,value in g.items():
                if key=='compared':
                    for k,v in enumerate(value):t['compared_'+str(k)]+=v
                else:t[key]+=value
        rows.append(dict(name=p['name'],query=p['query'],reference=p['reference'],groups=groups))
    check(f['pins']);check(f['protected'])
    write(U/'COMPARE_RESULT.json',dict(at=now(),pairs=len(rows),groups={k:dict(v) for k,v in totals.items()},rows=rows,
        whole_action_correct=None,classification_accuracy_measured=False,new_alignment=False))
    (U/'COMPARE_RUNNING.json').unlink();print('COMPARE',json.dumps({k:dict(v) for k,v in totals.items()}),flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        p=U/('FAILURE_'+now().replace(':','_')+'.txt');p.write_text(traceback.format_exc(),encoding='utf8');raise
