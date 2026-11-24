"""Public 36-video raw-observation/assignment diagnosis; no inference or training."""
from pathlib import Path
import json,hashlib
from collections import Counter,defaultdict
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def runs(frames):
    result=[]
    for f in frames:
        if not result or f!=result[-1][-1]+1:result.append([f])
        else:result[-1].append(f)
    return result
audit=read(OPS/'word100_mind_public_audit_20261017/AUDIT.json')
sources=audit['records']
assert len(sources)==36 and all(1<=r['signer_id']<=9 for r in sources)
SCOPE={
 'task':'Public overlap, side-assignment and sequence diagnosis only',
 'authorization':'2026-10-17 user request: public material, no user fitting, report before training.',
 'clips':[r['clip_id'] for r in sources],'words':['WORD0500','WORD0003','WORD1642','WORD1884'],
 'data':'Existing TRAIN01-09 only; original RGB and cached raw detections/assignment traces.',
 'fixed_measurements':['all annotated core frames: raw hand count, assigned sides, palm5 support, reason categories',
 'missing-side categories distinguish no raw observation vs withheld assignment vs point invalidity',
 'assigned != raw handedness is a disagreement, NOT a confirmed swap/error',
 'all raw-label-disagreement intervals and two-raw-unassigned intervals inspected visually',
 'longest sensor palm gap per word plus mind06 inspected in full source RGB',
 'ordered RGB core samples for all36 reused from previous audit; new detailed chronological strips'],
 'sequence_scope':'Describe observed movement stages, not expert sign decomposition. No model order-invariance conclusion without probe.',
 'no':['training','classifier inference','new MediaPipe','user recording reads','DEV/holdout new reads',
 'production web/model/DTW changes','downloads','Git','coefficient tuning'],
 'operator':'Codex root; separate recount follows',
 'pins':{str(OPS/'word100_mind_public_audit_20261017/AUDIT.json'):sha(OPS/'word100_mind_public_audit_20261017/AUDIT.json'),str(Path(__file__)):sha(__file__)}
}
if not (U/'SCOPE.json').exists():write(U/'SCOPE.json',SCOPE)
else:
    old=read(U/'SCOPE.json')
    assert old['clips']==SCOPE['clips'] and old['fixed_measurements']==SCOPE['fixed_measurements']
    write(U/'RECOVERY.json',{'cause':'Reused raw and candidate reside in different directories',
     'scope_sha256':sha(U/'SCOPE.json'),'corrected_code_sha256':sha(__file__),
     'condition_change':False,'excluded_clips':0})
out=[];pins={};flags=[];global_counts={}
PALM=np.array([0,5,9,13,17])
for r in sources:
    rp=OPS/'word100_body_boundary_20261017/data/records'/f"{r['clip_id']}.json";rec=read(rp)
    assert rec['role']=='train_candidate' and rec['signer_id']==r['signer_id']
    pins[str(rp)]=sha(rp);assert sha(r['video'])==r['source']['sha256'];pins[r['video']]=sha(r['video'])
    a,b=r['annotation']['data'][0]['start'],r['annotation']['data'][0]['end']
    rr={k:r[k] for k in ['clip_id','label','word_code','signer_id','video']}
    rr['annotation']=[a,b];rr['views']={}
    for view,capkey in [('native','native_capture'),('sensor','sensor_capture')]:
        cp=Path(rec[capkey]['candidate']).parent
        rawpath=Path(rec[capkey]['raw']);ap=cp/'assignment.npz';tp=cp/'trace.json';op=Path(rec[capkey]['candidate'])
        for p in [rawpath,ap,tp,op]:
            expected=rec[capkey]['pins'].get(str(p))
            if expected:assert sha(p)==expected,p
            pins[str(p)]=sha(p)
        raw=np.load(rawpath);ass=np.load(ap)['sides'];obs=np.load(op);trace=read(tp)
        ts=raw['timestamp_us']/1e6;core=np.flatnonzero((ts>=a)&(ts<=b))
        assert len(ts)==len(trace)==len(ass)==len(obs['xy_px'])
        assert np.array_equal(raw['timestamp_us'],obs['timestamp_us'])
        counts=Counter();reasons=Counter();missing=defaultdict(Counter);disagreement=[];withheld=[];frame_rows=[]
        palm=np.stack([obs['usable'][:,33+PALM].all(1),obs['usable'][:,54+PALM].all(1)],axis=1)
        for f0 in core:
            f=int(f0);n=int(raw['raw_count'][f]);sides=ass[f,:n].tolist();tr=trace[f]
            assert [None if s<0 else s for s in sides]==tr['assigned_sides']
            counts['core_frames']+=1;counts[f'raw{n}']+=1;counts[f'assigned{sum(s>=0 for s in sides)}']+=1
            if n==2 and any(s<0 for s in sides):withheld.append(f)
            diffs=[]
            for i,s in enumerate(sides):
                reasons[tr['reasons'][i]]+=1
                if s>=0 and s!=int(raw['raw_label'][f,i]):
                    diffs.append(i);counts['raw_label_disagreement_detections']+=1
            if diffs:disagreement.append(f)
            for s in [0,1]:
                if palm[f,s]:cat='palm_observed'
                elif s in sides:cat='assigned_but_palm_unusable'
                elif n==0:cat='no_raw_hands'
                elif n==1 and all(x>=0 for x in sides):cat='only_other_hand_detected'
                elif any(x<0 for x in sides):cat='unassigned_detection_present'
                else:cat='other'
                missing['LR'[s]][cat]+=1
            frame_rows.append({'frame':f,'time':float(ts[f]),'raw_count':n,'raw_labels':raw['raw_label'][f,:n].tolist(),
             'raw_scores':raw['raw_score'][f,:n].tolist(),'sides':sides,'reasons':tr['reasons'],
             'palm_usable':palm[f].tolist(),'label_disagreement_raw_indices':diffs,
             'same_side_competition':tr['same_side_competition'],'close_palms':tr['close_palms']})
        gaps={}
        for s in [0,1]:
            gs=runs([int(f) for f in core if not palm[f,s]])
            gaps['LR'[s]]=[{'frames':g,'start':float(ts[g[0]]),'end':float(ts[g[-1]]),'frames_count':len(g),
                        'nominal_seconds':len(g)/r['video_fps']} for g in gs]
        info={'raw_path':str(rawpath),'assignment_path':str(ap),'trace_path':str(tp),'candidate_path':str(op),
              'counts':dict(counts),'reasons':dict(reasons),'missing_categories':{k:dict(v) for k,v in missing.items()},
              'gaps':gaps,'disagreement_runs':runs(disagreement),'two_raw_withheld_runs':runs(withheld),'frames':frame_rows}
        for kind,rruns in [('raw_label_disagreement',runs(disagreement)),('two_raw_withheld',runs(withheld))]:
            for g in rruns:flags.append({'clip_id':r['clip_id'],'word_code':r['word_code'],'view':view,'kind':kind,'frames':g})
        rr['views'][view]=info
    out.append(rr)
for v in ['native','sensor']:
    cc=Counter();mm=defaultdict(Counter);rs=Counter()
    for row in out:
        cc.update(row['views'][v]['counts']);rs.update(row['views'][v]['reasons'])
        for s,d in row['views'][v]['missing_categories'].items():mm[s].update(d)
    global_counts[v]={'counts':dict(cc),'missing_categories':{s:dict(x) for s,x in mm.items()},'reasons':dict(rs)}
result={'scope_sha256':sha(U/'SCOPE.json'),'records':out,'summary':global_counts,'flags':flags,'pins':pins,
        'new_training':0,'user_recording_reads':0,'inference':0,'new_detection':0}
write(U/'AUDIT.json',result)
print(json.dumps({'summary':global_counts,'flag_intervals':len(flags),'flag_frames':sum(len(x['frames']) for x in flags),
 'flags':[{'clip':x['clip_id'],'view':x['view'],'kind':x['kind'],'frames':[x['frames'][0],x['frames'][-1]]} for x in flags]},ensure_ascii=False,indent=2))

