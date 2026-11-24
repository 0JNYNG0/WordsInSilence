"""Public FIT-only physical camera-framing variation, same predeclared policy."""
from pathlib import Path
from fractions import Fraction
from contextlib import ExitStack
from datetime import datetime,timezone,timedelta
import sys,json,hashlib,math,time,os,shutil,traceback
import numpy as np,av
from PIL import Image,ImageDraw
L=Path(__file__).resolve().parent;O=L.parent
P=O/'word100_learned_guide_feedback_20261102';D=O/'word100_action_episode_supervision_20261102';E=O/'word100_guide_conditioned_episode_20261102';N=O/'word100_learned_motion_invariance_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clock():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def exclusive(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with Path(p).open('x',encoding='utf8') as f:f.write(text)
def capacity():
    free=shutil.disk_usage(L).free
    if free<1024**3:raise RuntimeError('Reserve1GiB; never remove previous runs.')
    return free
def frame_picks(points,n):
    groups=sorted({p['action_family'] for p in points if p['mask'] and p['action_family']<8})
    picks=[0]
    for g in groups:
        ids=[p['frame'] for p in points if p['mask'] and p['action_family']==g]
        picks.append(ids[len(ids)//2])
    if len(picks)<5:
        ids=[p['frame'] for p in points if p['mask'] and p['action_family']<8]
        picks+=list(np.asarray(ids)[np.linspace(0,len(ids)-1,5-len(picks)).astype(int)])
    return list(map(int,picks[:5]+[n-1]))

def main():
    assert not (L/'CROP_MANIFEST.json').exists();capacity()
    rows=[r for r in read(P/'INPUT_MANIFEST.json')['rows'] if r['role']=='fit'];assert len(rows)==36 and {r['person'] for r in rows}==set(range(1,10))
    targets={r['clip_id']:r for r in read(D/'ACTION_TARGETS.json')['rows']};contexts={r['clip_id']:r for r in read(E/'CONTEXT_TARGETS.json')['rows']}
    files=[P/'INPUT_MANIFEST.json',D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',O/'word100_seated_web_framing_20261102/validate_prepare.py',L/'prepare_seated.py']
    plan=dict(at=clock(),previous_turn='Progress: J2training176comparisons; K12newinputs24comparisons; input consistency benefit verified; no live jobs.',
        inputs=36,recorded_ids=list(range(1,10)),policy='Same existing public webcam proxy: initial30usable shoulder centers median y plus median shoulder width; floor even crop bottom. Crop pixels only, no resize, retain source frame/timestamp.',
        why='FIT whole-sensor observations versus seated-crop controls were different camera-view distributions. Add real public FIT framing variation; no user data.',
        constraints=['Original replay preserved','No inference-based targets','No blanket start deletion','No masked hand automatically rest/correct','No C/T3F/web/H14/15/private access','No downloads or Git'],
        supervision='Direct source action labels admitted only after crop evidence audit. Cropped-off shape/position evidence is withheld; original-labelled nonperformance may guide a separate process-role, never certify hidden finger state.',
        raw_video_and_original_results_readonly=True,pins={str(p):sha(p) for p in files},new_training=0,whole_goal_complete=False)
    plan['original_freeze_sha256']=sha(L/'PREP_FROZEN.json')
    plan['previous_execution_freeze_sha256']=sha(L/'PREP_EXECUTION_FREEZE.json')
    plan['correction']='Preserve native PTS in codec time base as well as container. Source average_rate8640/287 differed from actual30fps; first13th-video encode failed axis validation and is retained. First12validated outputs reused. Review sheets resume from existing image.'
    if (L/'PREP_TIMEBASE_FREEZE.json').exists():assert read(L/'PREP_TIMEBASE_FREEZE.json')['pins']==plan['pins']
    else:exclusive(L/'PREP_TIMEBASE_FREEZE.json',plan)
    done=[];sheets={};start=time.perf_counter()
    for r in rows:
        capacity();cid=r['clip_id'];meta=L/'crop_records'/f'{cid}.json'
        if meta.exists():done.append(read(meta));continue
        assert sha(r['source'])==r['source_sha256'] and sha(r['observation'])==r['observation_sha256']
        ob=np.load(r['observation']);xy=ob['xy_px'];usable=ob['usable'];sh=xy[:min(30,len(xy)),[11,12]];ok=usable[:len(sh),[11,12]].all(1)&np.isfinite(sh).all((1,2));assert ok.any()
        center=np.median(sh[ok].mean(1),axis=0);scale=np.median(np.linalg.norm(sh[ok,0]-sh[ok,1],axis=1));bottom=2*(int(math.floor(float(center[1]+scale)))//2)
        dest=L/'videos'/f'{cid}.mp4';dest.parent.mkdir(exist_ok=True);reuse=dest.exists()
        picks=frame_picks(targets[cid]['points'],r['frames']);selected={};stamps=[];origin=None
        with ExitStack() as stack:
            inp=stack.enter_context(av.open(r['source']));out=None if reuse else stack.enter_context(av.open(str(dest),'w'))
            vs=inp.streams.video[0];width=vs.codec_context.width;height=vs.codec_context.height;assert 0<bottom<height
            if out is not None:
                stream=out.add_stream('libx264',rate=30);stream.width=width;stream.height=bottom;stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast'};stream.time_base=Fraction(1,1000000);stream.codec_context.time_base=Fraction(1,1000000)
            for f in inp.decode(vs):
                assert f.pts is not None;raw=f.pts*vs.time_base
                if origin is None:origin=raw
                us=int(round(float(raw-origin)*1e6));frame_id=len(stamps);stamps.append(us);im=f.to_ndarray(format='rgb24')[:bottom]
                if frame_id in picks:selected[frame_id]=im.copy()
                if out is not None:
                    vf=av.VideoFrame.from_ndarray(im,format='rgb24');vf.pts=us;vf.time_base=Fraction(1,1000000)
                    for packet in stream.encode(vf):out.mux(packet)
            if out is not None:
                for packet in stream.encode():out.mux(packet)
        with av.open(str(dest)) as c:
            v=c.streams.video[0];actual=[round(float(f.pts*v.time_base)*1e6) for f in c.decode(v)]
        assert len(actual)==len(stamps)==r['frames'] and max(abs(a-b) for a,b in zip(actual,stamps))<=2
        # This tests only NEW crop truncation of original observed locations.
        # It is not an assertion that a detector supplied perfect hand truth.
        risks=[]
        for p in targets[cid]['points']:
            if not p['mask'] or p['action_family']>=8:continue
            fam=p['action_family'];f=p['frame'];indices=list(range(54,75))
            if fam in [3,4]:indices += [13,15]
            if fam in [5,6,7]:indices += list(range(33,54))
            outside=[i for i in indices if usable[f,i] and np.isfinite(xy[f,i]).all() and xy[f,i,1]>=bottom]
            if outside:risks.append(dict(frame=f,action_family=fam,cropped_source_observed_joints=outside))
        review=L/'review'/f'ID{r["person"]:02}.jpg'
        if r['person'] not in sheets:sheets[r['person']]=Image.open(review).convert('RGB') if review.exists() else Image.new('RGB',(1800,1120),'white')
        sheet=sheets[r['person']];draw=ImageDraw.Draw(sheet);row_index=sorted({x['word_id'] for x in rows}).index(r['word_id'])
        for j,f in enumerate(picks):
            im=Image.fromarray(selected[f]);x0=max(0,int(center[0]-1.3*scale));x1=min(width,int(center[0]+1.3*scale));im=im.crop((x0,0,x1,bottom));im.thumbnail((296,252))
            x=j*300;y=row_index*280;sheet.paste(im,(x+(300-im.width)//2,y+23));draw.text((x+5,y+3),f'{r["word_id"]} ID{r["person"]:02} f{f} / {r["frames"]}',fill='black');draw.line((x,y+277,x+299,y+277),fill='red',width=2)
        review.parent.mkdir(exist_ok=True);sheet.save(review,quality=90)
        rec=dict(clip_id=cid,word_id=r['word_id'],person=r['person'],source=r['source'],source_sha256=r['source_sha256'],observation=r['observation'],observation_sha256=r['observation_sha256'],
            video=str(dest),video_sha256=sha(dest),crop=[0,0,width,bottom],original_extent=[width,height],shoulder_center=center.tolist(),shoulder_width=float(scale),
            frames=len(actual),source_timestamps_us=stamps,output_timestamps_us=actual,source_frame_indices=list(range(len(actual))),review_picks=picks,review_sheet=str(review),
            cropped_action_evidence=risks,new_action_labels=0,new_rest_labels=0)
        exclusive(meta,rec);done.append(rec);write(L/'PREP_PROGRESS.json',dict(at=clock(),status='ENCODING_PUBLIC_FIT_CROPS',pid=os.getpid(),done=len(done),total=36,seconds=time.perf_counter()-start));print('crop',len(done),cid,'bottom',bottom,'risk points',len(risks),flush=True)
    assert len(done)==36
    exclusive(L/'CROP_MANIFEST.json',dict(at=clock(),rows=done,status='PIXELS_READY_PENDING_VISUAL_AND_TARGET_AUDIT',new_training=0,whole_goal_complete=False))
    write(L/'PREP_PROGRESS.json',dict(at=clock(),status='CROPS_COMPLETE',active=None,done=36,total=36,seconds=time.perf_counter()-start))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (L/'FIRST_FAILURE.txt').exists():(L/'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
