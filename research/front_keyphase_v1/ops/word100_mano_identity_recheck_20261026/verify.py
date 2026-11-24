"""Saved flow quality and identity decision verification without importing the candidate."""
from pathlib import Path
from math import hypot,isfinite
from statistics import median
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_mano_duplicate_guard_20261025';OLD=O/'word100_mano_threeway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):return np.array_equal(a,b,equal_nan=True)
def dist(a,b):return hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))
def check_assessment(hand,pose,pv,tracks,actual):
    scale=dist(pose[11],pose[12]) if pv[[11,12]].all() else 0
    ds=[dist(hand[0],pose[15+s])/scale if scale>1 and pv[[11+s,13+s,15+s]].all() and np.isfinite(hand[0]).all() else float('inf') for s in (0,1)]
    arm=int(ds[1]<ds[0]);arm=arm if ds[arm]<=.25 and ds[1-arm]-ds[arm]>=.10 else -1
    assert actual['current_arm']==arm
    qs=[]
    for q in actual['temporal']:
        details=[]
        for d in q['details']:
            tr=tracks[(d['anchor'],d['target'],q['side'])];xy,valid=tr
            ids=[i for i in [4,8,12,16,20] if valid[i] and np.isfinite(hand[i]).all()]
            errs=[dist(xy[i],hand[i])/scale for i in ids] if scale>1 else []
            score=median(errs) if len(errs)>=4 else None
            wrist=dist(xy[0],hand[0])/scale if scale>1 and valid[0] and np.isfinite(hand[0]).all() else None
            assert ids==d['tips']
            for a,b in [(score,d['score']),(wrist,d['wrist'])]:
                assert (a is None)==(b is None)
                if a is not None:assert abs(a-b)<1e-6,(a,b)
            details.append((d['anchor'],d['target'],score,wrist))
        both=len(details)==2 and details[0][0]<details[0][1]<details[1][0]
        qualified=both and all(d[2] is not None and d[2]<=.12 and d[3] is not None and d[3]<=.12 for d in details)
        cost=max(d[2] for d in details) if both and all(d[2] is not None for d in details) else None
        assert qualified==q['qualified'] and both==q['two_sided']
        qs.append((qualified,cost))
    supported=[s for s in (0,1) if qs[s][0] and (not qs[1-s][0] or qs[s][1]<.5*qs[1-s][1]) and arm==s]
    assert actual['supported_side']==(supported[0] if len(supported)==1 else -1)

def main():
    f=read(U/'FROZEN.json');v2=read(U/'FROZEN_V2.json');v3=read(U/'FROZEN_V3.json')
    assert sha(U/'identity.py')==f['candidate_sha256']==v3['v1_candidate_sha256']
    assert sha(U/'identity_v2.py')==v2['candidate_sha256']==v3['v2_candidate_sha256']
    assert sha(U/'identity_v3.py')==v3['candidate_sha256']
    assert sha(U/'RESULT.json')==v2['v1_result_sha256'] and sha(U/'RESULT_V2.json')==v3['v2_result_sha256']
    for group in ['protected','prior_pins']:
        for p,h in f[group].items():assert sha(p)==h,p
    idx=read(U/'TRACK_INDEX_V3.json');endpoints={};steps=0;trackcount=0
    for clip,rows in idx.items():
        endpoints[clip]={}
        for r in rows:
            if r.get('failed'):continue
            assert sha(r['path'])==r['sha256'];z=np.load(r['path']);valid=np.ones(21,bool)
            for n,(ta,tb) in enumerate(r['frames']):
                assert abs(ta-tb)==1
                prev=z['previous'][n];q=z['forward'][n];back=z['backward'][n];e=z['error'][n];status=z['status'][n];bs=z['back_status'][n]
                ok=np.array([status[j]==1 and bs[j]==1 and np.isfinite(q[j]).all() and isfinite(float(e[j])) and e[j]<=20 and dist(prev[j],back[j])<=1.5 and 0<=q[j,0]<640 and 0<=q[j,1]<360 for j in range(21)])
                valid &= ok;assert np.array_equal(valid,z['valid'][n]);steps+=1
            assert np.array_equal(valid,z['final_valid']);assert np.allclose(z['xy'],z['forward'][-1]*3,equal_nan=True,atol=1e-4)
            endpoints[clip][r['anchor'],r['target'],r['side']]=(z['xy'],valid);trackcount+=1
    diag=read(U/'DIAGNOSTICS_V3.json');previous=read(G/'REQUESTED_DIAGNOSTIC_V2.json');res=read(U/'RESULT_V3.json')
    framecount=points=cross=crosspoints=normal=heldslots=checked=suppressedanchors=0
    for k,m in enumerate(f['manifest']):
        assert sha(m['prepared'])==m['prepared_sha256'] and sha(m['rgb_path'])==m['rgb_sha256']
        z=np.load(m['prepared']);g=np.load(G/'outputs_v2'/(m['clip_id']+'.npz'));p=np.load(U/'outputs_v3'/(m['clip_id']+'.npz'));p1=np.load(U/'outputs'/(m['clip_id']+'.npz'))
        suppressed=np.zeros(p['anchors'].shape,bool)
        for r in previous:
            if r['clip_index']!=k:continue
            for q in r['pairs']:
                if q['hold_new'] and q['existing_side']!=r['side']:
                    suppressed[r['frame'],r['side']]=True;suppressed[r['frame'],q['existing_side']]=True
        assert np.array_equal(suppressed,p['anchor_suppressed'])
        assert np.array_equal(p['anchors'],p1['anchors']&~suppressed)
        suppressedanchors+=int((p1['anchors']&suppressed).sum())
        eligible=np.ones((len(z['frame_id']),2),bool)
        for r in [d for d in diag if d['clip_index']==k]:
            t,s=r['frame'],r['side'];raw=np.load(OLD/f'raw_predictions/{k:02d}_{t:04d}_{s}.npz')['xy_px']
            check_assessment(raw,z['pose'][t],z['pose_valid'][t],endpoints[m['clip_id']],r['raw_assessment']);checked+=1
            for side in (0,1):
                for d in r['raw_assessment']['temporal'][side]['details']:
                    a=d['anchor'];assert p['anchors'][a,side] and not suppressed[a,side]
                    assert abs(float(z['timestamp_us'][a]-z['timestamp_us'][t]))<=500000.01
            if r['duplicate_conflict']:
                e=r['existing_side'];check_assessment(z['full_xy'][t,e],z['pose'][t],z['pose_valid'][t],endpoints[m['clip_id']],r['existing_assessment']);checked+=1
                assert r['decision']['chosen']==-1;eligible[t,[e,s]]=False
        assert np.array_equal(eligible,p['identity_eligible']);heldslots+=int((~eligible).sum())
        assert not p['identity_supported'].any()
        for target,source in [('R_xy','G_xy'),('R_valid','G_valid'),('R_inferred','G_inferred'),('R_observed','G_evidence'),('raw_G_xy','G_xy'),('raw_G_valid','G_valid'),('raw_G_evidence','G_evidence')]:assert same(p[target],g[source]),target
        for key in ['frame_id','timestamp_us','roi_box']:assert same(p[key],g[key])
        framecount+=len(z['frame_id']);points+=int(g['G_evidence'].sum());cross+=len(m['crossing_frames']);crosspoints+=int(g['G_evidence'][m['crossing_frames']].sum())
        if m['role']=='existing_observation_control':normal+=len(z['frame_id']);assert p['identity_eligible'].all()
        assert p['identity_eligible'][m['crossing_frames']].all()
    for r in res['broad_diagnostics']:
        m=next(x for x in f['controls'] if x['clip_id']==r['clip']);assert sha(m['path'])==m['sha256']
        z=np.load(m['path']);t=r['frame']
        for s in (0,1):check_assessment(z['xy_px'][t,33+s*21:54+s*21],z['xy_px'][t,:33],z['usable'][t,:33],endpoints[r['clip']],r['assessments'][s]);checked+=1
    assert (framecount,points,cross,crosspoints,normal,heldslots)==(812,14149,20,840,94,20)
    a=read(U/'RESULT.json');old_affirm=sum(x['existing_assessment']['supported_side']>=0 for x in a['rows']);new_affirm=sum(x['existing_assessment']['supported_side']>=0 for x in res['rows'])
    out={'status':'PASS','operator':'Codex separate scalar verification; same author, not expert GT','flow_tracks':trackcount,'flow_steps':steps,'assessment_checks':checked,'primary_frames':framecount,'original_points_preserved':points,'crossing_frames_preserved':cross,'crossing_points_preserved':crosspoints,'normal_control_frames_preserved':normal,'identity_ineligible_slots':heldslots,'quarantined_previously_eligible_anchors':suppressedanchors,'wrong_existing_identity_affirmations_v1':old_affirm,'wrong_existing_identity_affirmations_v3':new_affirm,'new_identity_replacements':0,'protected_files':len(f['protected']),'prior_files':len(f['prior_pins']),'all_raw_and_nonconflict_arrays_unchanged':True,'limitations':'No expert identity/3D GT; RGB tracking failures remain; same diagnostic materials reused.'}
    with (U/'VERIFICATION.json').open('x',encoding='utf-8') as q:json.dump(out,q,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
