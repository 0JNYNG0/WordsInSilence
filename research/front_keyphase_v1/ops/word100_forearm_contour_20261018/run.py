import json,gzip,hashlib,time,traceback,shutil,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import cv2,numpy as np
from candidate import propose,PARAMS
U=Path(__file__).resolve().parent;P=U.parent/'word100_image_arm_connectivity_20261018';ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def freeze():
    assert not (U/'CONTEXT.json').exists() and read(U/'CONTRACTS.json')['status']=='PASS'
    pins=read(P/'CONTEXT.json')['pins'];pins.update(read(P/'OUTPUT_PINS.json'))
    for name in ['RESULTS.json','ANNOTATED.json','CROSSING.json','CONTEXT.json','INDEPENDENT.json','FINAL_AUDIT.json','RESULT.json','candidate.py','run.py','verify.py']:
        pins[str(P/name)]=sha(P/name)
    for p in [U/'candidate.py',U/'contracts.py',U/'CONTRACTS.json',U/'run.py']:pins[str(p)]=sha(p)
    for p,h in pins.items():assert sha(p)==h,p
    (U/'before').mkdir(exist_ok=True)
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',ROOT/'research/front_keyphase_v1/ops/STATE.json']:shutil.copyfile(p,U/'before'/p.name)
    write(U/'CONTEXT.json',{'at':now(),'parameters':PARAMS,'pins':pins,'source':str(P),
      'authorization':'User approved forearm contour and direction validation using publicTRAIN; retain normal/crossing, C and user data; no training until results reviewed.',
      'scope':'Same frozen785 selected frames/367publicTRAIN clips from prior RGB unit:461hand annotations (441normal,13originalerrors,2uncertain,5additional),315crossingframes,and remaining9neighbors. Not all899RGB replay; known development cases, not heldout accuracy.',
      'method':'Cropped grayscaleGaussian3x3; Sobel/8;12transverse profiles at .15..85 along BOTH elbow-to-Hand-wrist directions. Opposite aligned edge peaks, width .06..40shoulders, center .08,10/12support,maxgap1,widthCV.35,maxadjacentjump.08. All coefficients fixed before public results. No fixedskincolor, no nearestelbow rule, no labels/future/hidden pixels.',
      'acceptance':'Only source-verified new corrections count. Any newly wrong normal/crossing assignment rejects this candidate globally; no postresult tuning. Keep geometry/masks and confirmed/withheld assignments. If competing paths or duplicate destination, abstain. Review all actual changes.',
      'independence':'Codex implementation and source review; separate scalar reference code. Not separate agent/human reviewer.',
      'prohibitions':['user/mentor/DEV/heldout use','training or classifier inference','C/web/TCN/DTW changes','camera/download/external transfer/Git','original or previous frozen output edits']})
    print('FROZEN',len(pins),'pins',flush=True)
def run():
    c=read(U/'CONTEXT.json');assert c['parameters']==PARAMS
    assert not (U/'RUNNING.json').exists() and not (U/'SUMMARY.json').exists()
    for p,h in c['pins'].items():assert sha(p)==h,p
    write(U/'RUNNING.json',{'at':now()});start=time.monotonic();(U/'trials').mkdir(exist_ok=True)
    results=[];changes=[];counts=Counter();pins={};sources=read(P/'RESULTS.json')
    for k,x in enumerate(sources):
        image=cv2.imread(x['image']);assert image is not None
        out=propose(image,x['elbows'],x['wrists'],x['scale'],x['old_sides'],x['old_identity'])
        rev=propose(image,x['elbows'],x['wrists'][::-1],x['scale'],x['old_sides'][::-1],x['old_identity'][::-1])
        assert rev['sides'][::-1]==out['sides'] and rev['evidence_side'][::-1]==out['evidence_side']
        blank=propose(np.zeros_like(image),x['elbows'],x['wrists'],x['scale'],x['old_sides'],x['old_identity'])
        assert blank['sides']==x['old_sides'] and all(s<0 for s in blank['evidence_side'])
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        inv=propose(255-gray,x['elbows'],x['wrists'],x['scale'],x['old_sides'],x['old_identity'])
        assert inv['sides']==out['sides'] and inv['evidence_side']==out['evidence_side'],('contrast_invariance',k)
        f=U/'trials'/f'{k:04}.json.gz'
        with gzip.open(f,'wt',encoding='utf8') as z:json.dump(out['trials'],z,separators=(',',':'))
        pins[str(f)]=sha(f)
        compact=[[{key:val for key,val in t.items() if key not in ['slices','offsets']} for t in h] for h in out['trials']]
        results.append({**{key:x[key] for key in ['clip_id','word','view','index','frame_id','raw_count','old_sides','old_identity','observed_points','image','source_video','source_sha256','elbows','wrists','scale']},**{key:out[key] for key in ['sides','evidence_side','reason','threshold']},'trials_file':str(f),'trials_summary':compact})
        for i,(old,new) in enumerate(zip(x['old_sides'],out['sides'])):
            if old!=new:changes.append({'record_index':k,'raw_index':i,'before':old,'after':new,'clip_id':x['clip_id'],'view':x['view'],'index':x['index']})
        counts.update(out['reason']);counts['frames']+=1;counts['hand_candidates']+=x['raw_count'];counts['observed_points_preserved']+=sum(x['observed_points'])
        counts['order_checks']+=1;counts['blank_checks']+=1;counts['contrast_checks']+=1
        if (k+1)%100==0:print('CONTOUR',k+1,'/',len(sources),'changes',len(changes),flush=True)
    lookup={(x['clip_id'],x['view'],x['index']):i for i,x in enumerate(results)}
    ann=read(P/'ANNOTATED.json')
    for a in ann:
        k=lookup[(a['clip_id'],a['view'],a['index'])];x=results[k];i=a['raw_index'];a.update(before=x['old_sides'][i],after=x['sides'][i],evidence_side=x['evidence_side'][i],reason=x['reason'][i],record_index=k)
    cross=read(P/'CROSSING.json')
    for a in cross:
        k=lookup[(a['clip_id'],a['view'],a['index'])];a['before']=results[k]['old_sides'];a['after']=results[k]['sides'];a['record_index']=k
    for p,h in c['pins'].items():assert sha(p)==h,p
    normal=[a for a in ann if a['verdict']=='normal_identity'];rem=[a for a in ann if a['remaining9']]
    write(U/'RESULTS.json',results);write(U/'CHANGES.json',changes);write(U/'ANNOTATED.json',ann);write(U/'CROSSING.json',cross);write(U/'OUTPUT_PINS.json',pins)
    summary={'status':'EXECUTED_PENDING_VERIFICATION_VISUAL','counts':dict(counts),'changed_events':len(changes),'normal441_preserved':sum(a['before']==a['after']==a['actual_side'] for a in normal),'normal_regressions':sum(a['after']!=a['actual_side'] for a in normal),'remaining9_correct':sum(a['after']==a['actual_side'] for a in rem),'crossing315_preserved':sum(a['before']==a['after'] for a in cross),'crossing_changed_frames':sum(a['before']!=a['after'] for a in cross),'elapsed_seconds':time.monotonic()-start,'training':0,'classifier_inference':0,'user_reads':0,'production_changed':False}
    write(U/'SUMMARY.json',summary);(U/'RUNNING.json').unlink();print(json.dumps(summary),flush=True)
if __name__=='__main__':
    try:freeze() if sys.argv[1]=='freeze' else run()
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'at':now(),'error':traceback.format_exc()})
        raise
