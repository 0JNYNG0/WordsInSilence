"""Public demo candidate inputs. Preserve source role and native frame timing."""
import hashlib, io, json, sys, time, zipfile, traceback
from pathlib import Path
import av, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
U=Path(__file__).resolve().parent;O=U.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
from stream_runtime import Engine
sys.path.insert(0,str(O/'word100_rtmw_shared_input_20261108'))
import adapter as A

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
WORDS={'WORD2633':'숫자 2','WORD2635':'숫자 4','WORD2637':'숫자 6','WORD0099':'사과','WORD1290':'감사'}

def main():
 for d in ['media','raw','channels','records','panels']: (U/d).mkdir(exist_ok=True)
 inv=read(O/'demo_scope_portability_review_20261113/INVENTORY.json');j=O/'word100_rtmw_joint_learning_20261108'
 if not (U/'FROZEN.json').exists():
  pins=[j/'tcn_runs/finetune_C_20261013/model.pt',j/'feedback_viewfix/feedback_runs/pose_transfer_42/model.pt',O/'word100_body_boundary_20261017/training_C_20261013_best.pt',O/'word100_frozen_c_temporal_20261019/T3F_20261013_RESULT.json']
  write(U/'FROZEN.json',dict(authorization='2026-11-13 user: proposed demo plan proceed; consider static handshape learning',
   words=WORDS,fit_ids=list(range(1,10)),dev_ids=[11,12,13],forbidden_ids=[10,14,15,16,17,18],
   goals=['source verification','isolated learned static handshape + native-time hold evidence','four new classifier outputs with old100 replay','Unity supported-word bridge smoke test','local dependency audit'],
   digit_label_policy='Only manually reviewed RGB anchors; unknown finger states masked. Word name never supplies visibility.',
   shape_plan=dict(seeds=[42,43,44],steps=600,lr=.001,hidden=32,features='hand-local 2D geometry + masks; no body location/target/word ID',loss='masked finger BCE + shape CE',threshold=.85),
   tcn_plan=dict(seed=42,epochs=80,lr=.003,mode='freeze old backbone and 100 outputs; train only appended four linear outputs with old100 FIT replay',fit_only=True,dev_no_selection=True),
   protection={str(p):sha(p) for p in pins},training_allowed_after_rgb_admission=True,production_replaced=False,
   greeting='Single cached SEN example only; not merge with NIKL 인사; no single-person automatic admission',
   prior_research='C6/G2 regression preserved. This demo candidate does not clear its gate.'))
  write(U/'GRAPH.json',{'A_source_review':'RUNNING','B_input_and_learning':'PENDING_SOURCE_ADMISSION','C_unity_existing_word_bridge':'PENDING','D_portable_demo':'PENDING_GATES','greeting':'MULTISIGNER_RGB_NOT_READY','prior_C7':'PRESERVED_NOT_REPLACED'})
 engine=None;archives={};rows=[];failures=[]
 for code,label in WORDS.items():
  candidates=sorted([r for r in inv['targets'][code] if r['record_id'] in list(range(1,10))+[11,12,13]],key=lambda x:x['record_id'])
  for row in candidates:
   cid=row['clip_id'];rp=U/'records'/f'{cid}.json'
   if rp.exists():rows.append(read(rp));continue
   start=time.perf_counter();vpath=U/'media'/f'{cid}.mp4'
   if not vpath.exists():
    if row['archive'] not in archives:archives[row['archive']]=zipfile.ZipFile(row['archive'])
    vpath.write_bytes(archives[row['archive']].read(row['member']))
   # Validate the original container before inference. Keep failed originals in
   # the planned-input denominator; never relabel or silently replace them.
   try:
    with av.open(str(vpath)) as probe:
     assert len(probe.streams.video)==1
   except (av.error.InvalidDataError, AssertionError) as exc:
    if row['archive'] not in archives:archives[row['archive']]=zipfile.ZipFile(row['archive'])
    source=archives[row['archive']].read(row['member'])
    assert hashlib.sha256(source).hexdigest()==sha(vpath), 'Local/source mismatch'
    failure=dict(clip_id=cid,word=code,person=row['record_id'],role='fit' if row['record_id']<=9 else 'development',source_sha256=sha(vpath),source_bytes=len(source),error=str(exc),admitted=False)
    failures.append(failure);write(U/'INPUT_FAILURES.json',failures)
    print('SOURCE FAILURE',cid,flush=True);continue
   sample_fractions=[.35,.5,.65] if code.startswith('WORD263') else [.1,.25,.4,.55,.7,.85]
   lo,hi=row['metadata']['intervals'][0]
   # Existing gratitude observations can be reused unchanged.
   cached=j/'records'/f'{cid}.json'
   if cached.exists():
    old=read(cached);rawpath=Path(old['raw_path']);assert sha(rawpath)==old['raw_sha256'];z=np.load(rawpath);raw={k:z[k] for k in z.files};reused=True
   else:
    rawpath=U/'raw'/f'{cid}.npz'
    if rawpath.exists():raise RuntimeError('Uncommitted extraction; inspect before resume: '+cid)
    if engine is None:
     engine=Engine();write(U/'ENGINE.json',dict(devices=engine.devices,compile_seconds=engine.compile_seconds,warm_seconds=engine.warm_seconds))
    xy=[];sc=[];ts=[];cost=[]
    with av.open(str(vpath)) as c:
     st=c.streams.video[0];st.codec_context.thread_count=2;base=None
     for frame in c.decode(st):
      if base is None:base=frame.pts
      tus=round(float((frame.pts-base)*st.time_base)*1e6);im=frame.to_ndarray(format='bgr24');q=engine.infer(im)
      xy.append(q['xy']);sc.append(q['score']);ts.append(tus);cost.append(q['compute_ms'])
    raw=dict(xy_px=np.asarray(xy),score=np.asarray(sc),timestamp_us=np.asarray(ts,np.int64),frame_id=np.arange(len(ts),dtype=np.int64),source_width=np.int64(im.shape[1]),source_height=np.int64(im.shape[0]))
    np.savez_compressed(rawpath,**raw);reused=False
   a=A.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
   cp=U/'channels'/f'{cid}.npz';np.savez_compressed(cp,**a)
   anchors=[int(np.argmin(abs(raw['timestamp_us']/1e6-(lo+(hi-lo)*f)))) for f in sample_fractions]
   r=dict(clip_id=cid,word=code,label=label,person=row['record_id'],role='fit' if row['record_id']<=9 else 'development',
    video=str(vpath),source_sha256=sha(vpath),archive=row['archive'],member=row['member'],raw=str(rawpath),raw_sha256=sha(rawpath),
    channels=str(cp),channels_sha256=sha(cp),frames=len(raw['frame_id']),anchors=anchors,source_interval=[lo,hi],
    reused=reused,elapsed_seconds=time.perf_counter()-start,training_admitted=False)
   write(rp,r);rows.append(r);write(U/'PROGRESS.json',dict(done=len(rows),total=60,last=cid,status='PREPARING'))
   print('READY',len(rows),cid,round(r['elapsed_seconds'],2),flush=True)
  # Review panels: three anchor moments per number, six per other sign.
  rr=[r for r in rows if r['word']==code];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
  for offset in range(0,len(rr),3):
   selected=rr[offset:offset+3];cols=len(selected[0]['anchors']);cw=240;ch=290
   board=Image.new('RGB',(cw*cols,40+ch*len(selected)),'white');d=ImageDraw.Draw(board);d.text((10,8),f'{label} / RGB 원본 검수용 (모델 정답·예측 아님)',font=font,fill='#172b49')
   for ri,r in enumerate(selected):
    cap=cv2.VideoCapture(r['video'])
    for ci,f in enumerate(r['anchors']):
     cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
     h,w=im.shape[:2];crop=im[int(h*.10):int(h*.80),int(w*.25):int(w*.72)]
     pim=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));pim.thumbnail((235,250));x=ci*cw;y=40+ri*ch
     board.paste(pim,(x+(cw-pim.width)//2,y));d.text((x+5,y+252),f'ID{r["person"]:02} {r["role"]} f{f}',font=font,fill='#172b49')
    cap.release()
   board.save(U/'panels'/f'{code}_{offset//3}.jpg',quality=94)
 for z in archives.values():z.close()
 write(U/'INPUTS.json',dict(rows=rows,failures=failures,planned_inputs=60,source_review_pending=True,complete=True))
 write(U/'PROGRESS.json',dict(done=len(rows),failed=len(failures),total=60,status='INPUTS_PREPARED_RGB_REVIEW_PENDING'))
 print('INPUTS COMPLETE',len(rows),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
