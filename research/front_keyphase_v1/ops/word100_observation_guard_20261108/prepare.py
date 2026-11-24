"""Public RGB framing inputs only. Never opens a camera or trains a model."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;J=O/'word100_rtmw_joint_learning_20261108';P=O/'word100_inactive_arm_review_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
    assert not (U/'INPUT_PLAN.json').exists()
    cmp=read(O/'word100_rtmw_mediapipe_compare_20261108/FROZEN.json')['rows']
    core=read(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json')['rows']
    chosen=[dict(r,role='calibration' if r['person'] in [1,2] else 'regression') for r in cmp]
    chosen += [dict(r,role='development_regression') for r in core if r['person'] in [11,12,13]]
    rows=[]
    for r in chosen:
        rec=read(J/'records'/f"{r['clip_id']}.json");z=np.load(rec['raw_path']);body=np.nanmedian(z['xy_px'][:12],0);s=float(np.linalg.norm(body[5]-body[6]));top=max(0,int(body[0,1]-.7*s));bottom=min(int(z['source_height']),int(max(body[7,1],body[8,1])+.15*s));height=bottom-top;width=int(height*4/3);cx=(body[5,0]+body[6,0])/2;left=max(0,min(int(z['source_width'])-width,int(cx-width/2)))
        cache=P/(r['clip_id']+'_crop.npz')
        rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],label=r['label'],person=r['person'],role=r['role'],source=r['source'],source_sha256=r['source_sha256'],raw=rec['raw_path'],raw_sha256=rec['raw_sha256'],crop=[left,top,left+width,bottom],intervals=r['intervals'],cache=str(cache) if cache.exists() else None))
    write(U/'INPUT_PLAN.json',dict(rows=rows,geometry='Same first12 shoulder/nose/elbow crop as previous diagnosis, fixed before outputs',original_full_observations='Reuse existing raw arrays',privacy='No private videos/arrays, camera or external transfer',baseline_model='fixed finetune_C_20261013',criteria='Design/calibrate on ID01/02 observation geometry only, never TCN predictions; freeze before regression scoring. Require no normal full-view correct-to-wrong TCN regression, preserve pre-reviewed normal crossing, improve absent-hand contamination and crop recognition without word-specific logic. If a gate fails, no silent relaxation or deployment.',protected=read(O/'word100_action_feedback_demo_20261108/FROZEN.json')['protected']))
def extract(role):
    import cv2,av
    sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'));from stream_runtime import Engine
    engine=None;rows=read(U/'INPUT_PLAN.json')['rows'];events=[]
    for r in rows:
        if (r['role']=='calibration') != (role=='calibration'):continue
        dest=U/'crops'/(r['clip_id']+'.npz');dest.parent.mkdir(exist_ok=True)
        if r['cache'] or dest.exists():continue
        assert sha(r['source'])==r['source_sha256'];assert sha(r['raw'])==r['raw_sha256']
        if engine is None:engine=Engine()
        l,t,rr,b=r['crop'];old=np.load(r['raw']);xy=[];sc=[];ts=[]
        first_time=None
        with av.open(r['source']) as c:
            for frame in c.decode(video=0):
                frame_time=frame.pts*frame.time_base
                if first_time is None:first_time=frame_time
                im=frame.to_ndarray(format='bgr24');im=cv2.resize(im[t:b,l:rr],(640,480));q=engine.infer(im);xy.append(q['xy']);sc.append(q['score']);ts.append(round(float(frame_time-first_time)*1e6))
        assert np.array_equal(ts,old['timestamp_us'])
        np.savez_compressed(dest,xy_px=np.stack(xy),score=np.stack(sc),timestamp_us=np.array(ts),frame_id=old['frame_id'],source_width=640,source_height=480)
        msg=dict(clip_id=r['clip_id'],frames=len(ts),output=str(dest));events.append(msg);print(json.dumps(msg),flush=True)
        with (U/'EXTRACTION_EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(msg)+'\n')
    write(U/('EXTRACT_'+role+'.json'),dict(status='COMPLETE',new=events))
if __name__=='__main__':
    if sys.argv[1]=='plan':plan()
    else:extract(sys.argv[1])
