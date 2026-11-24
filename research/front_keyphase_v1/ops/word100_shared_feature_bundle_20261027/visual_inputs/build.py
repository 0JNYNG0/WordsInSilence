"""24 additional public source clips: bounded RGB export, no model execution."""
from pathlib import Path
import hashlib,importlib.util,json,traceback,zipfile,shutil
import av
import numpy as np
from PIL import Image
U=Path(__file__).resolve().parent;OPS=U.parent.parent
PRIOR=OPS/'word100_source_token_alignment_20261027/visual_inputs'
spec=importlib.util.spec_from_file_location('prior_visual_export',PRIOR/'build.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
sha=prior.sha;read=prior.read
def write(n,d):(U/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def source(row,archives,archive_stats):
    cid=row['clip_id'];candidates=[]
    done=OPS/'word100_face_body_stage_20261027/data'/cid/'DONE.json'
    if done.exists():candidates.append(Path(read(done)['video']))
    candidates.append(Path('W:/WordsInSilence/data/AIHub_103/word100_web_references_20261013')/(cid+'.mp4'))
    for p in candidates:
        if p.exists():
            assert sha(p)==row['source']['sha256'],('cache content mismatch',str(p))
            return p,'existing_cache'
    arc=row['source']['archive'];member=row['source']['member']
    if arc not in archives:
        st=Path(arc).stat();archive_stats[arc]=[st.st_size,st.st_mtime_ns];archives[arc]=zipfile.ZipFile(arc)
    z=archives[arc];info=z.getinfo(member)
    assert info.CRC==row['source']['crc32'] and info.file_size==row['source']['bytes']
    p=U/'videos'/(row['alias']+'.mp4');p.parent.mkdir(exist_ok=True)
    if p.exists():assert sha(p)==row['source']['sha256']
    else:
        with z.open(member) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst)
    assert sha(p)==row['source']['sha256']
    return p,'exact_frozen_archive_member_copied_no_download'

def main():
    if (U/'INDEX.json').exists():raise FileExistsError('Completed exports must be reused')
    frozen_path=U.parent/'FROZEN.json';frozen=read(frozen_path);rows=frozen['rows'];new=[r for r in rows if r['is_new']];old=[r for r in rows if not r['is_new']]
    assert len(rows)==36 and len(new)==24 and len(old)==12 and len({r['clip_id'] for r in rows})==36
    assert len({r['source']['sha256'] for r in rows})==36, 'Duplicate source content'
    assert {r['recorded_person_id'] for r in rows}=={1,2,3}
    index0=read(PRIOR/'INDEX.json');oldby={r['clip_id']:r for r in index0['rows']}
    pins={str(frozen_path):sha(frozen_path),str(PRIOR/'INDEX.json'):sha(PRIOR/'INDEX.json'),str(PRIOR/'build.py'):sha(PRIOR/'build.py'),str(U/'build.py'):sha(U/'build.py')}
    for p,h in frozen['source_pins'].items():assert sha(p)==h;pins[p]=h
    archive_stats={};archives={};outputs={};done=[]
    for row in old:
        r=oldby[row['clip_id']];assert r['source_sha256']==row['source']['sha256'] and r['alias']==row['alias']
        done.append({**r,'is_new':False,'reuse_index':str(PRIOR/'INDEX.json'),'role':row['role']})
    for row in new:
        alias=row['alias'];folder=U/alias;folder.mkdir(exist_ok=True)
        rp=Path(row['record']);assert sha(rp)==row['record_sha256'];record=read(rp)
        assert record['source']['sha256']==row['source']['sha256']
        video,method=source(row,archives,archive_stats)
        op=Path(record['views']['native']['observation']);sidecar=op.with_suffix('.sidecar.json');sc=read(sidecar)
        assert sha(op)==sc['observation_sha256'];assert sc['source_sha256']==row['source']['sha256']
        for p in (video,rp,op,sidecar):pins[str(p)]=sha(p)
        with np.load(op,allow_pickle=False) as z:ts=z['timestamp_us'].copy();fid=z['frame_id'].copy();xy=z['xy_px'].copy();usable=z['usable'].copy()
        assert np.array_equal(fid,np.arange(len(fid)))
        begin=max(0,min(a for a,b in row['intervals'])-.25);end=min(ts[-1]/1e6,max(b for a,b in row['intervals'])+.25)
        targets=np.linspace(begin*1e6,end*1e6,16);ix=np.array([int(np.argmin(np.abs(ts-t))) for t in targets]);assert len(set(ix.tolist()))==16
        selected={};pts=[];first=None;pix={}
        with av.open(str(video)) as c:
            for n,frame in enumerate(c.decode(c.streams.video[0])):
                if first is None:first=frame.pts
                t=int(round(float((frame.pts-first)*frame.time_base)*1e6));pts.append(t)
                if n in ix:
                    rgb=frame.to_ndarray(format='rgb24');pix[n]=hashlib.sha256(rgb.tobytes()).hexdigest();selected[n]=Image.fromarray(rgb)
        assert len(pts)==len(fid) and np.max(np.abs(np.asarray(pts)-ts))<=1
        w,h=selected[int(ix[0])].size;ids=np.r_[np.arange(23),np.arange(33,75)];p=xy[ix][:,ids];ok=usable[ix][:,ids]&np.isfinite(p).all(-1);pp=p[ok];assert len(pp)
        crop=(max(0,int(np.floor(pp[:,0].min()-80))),0,min(w,int(np.ceil(pp[:,0].max()+80))),min(h,int(np.ceil(pp[:,1].max()+100))))
        frames=[];images=[]
        for j,idx in enumerate(ix):
            idx=int(idx);im=selected[idx];p=folder/f'frame_{j+1:02d}_f{idx:04d}.png';im.save(p)
            with Image.open(p) as check:assert hashlib.sha256(np.asarray(check.convert('RGB')).tobytes()).hexdigest()==pix[idx]
            outputs[str(p)]=sha(p);images.append(im);frames.append({'ordinal':j+1,'frame_id':idx,'pts_us':pts[idx],'observation_timestamp_us':int(ts[idx]),'target_us':float(targets[j]),'file':str(p),'rgb_sha256':pix[idx]})
        full=folder/'full_contact.png';detail=folder/'detail_contact.png';prior.sheet(images,frames,alias).save(full);prior.sheet(images,frames,alias,crop).save(detail)
        for p in (full,detail):outputs[str(p)]=sha(p)
        entry={'alias':alias,'clip_id':row['clip_id'],'word_id':row['word_id'],'label':row['label'],'recorded_person_id':row['recorded_person_id'],'role':row['role'],'is_new':True,'source':str(video),'source_method':method,'source_sha256':row['source']['sha256'],'archive_source':row['source'],
          'annotation_intervals':row['intervals'],'export_span_seconds':[begin,end],'full_frame_size':[w,h],'display_only_crop_xyxy':list(crop),'crop_policy':'All selected visible upper-body and hand points union, 80px horizontal/100px bottom margin, top0 preserves head. Display only; no training ROI change.',
          'full_sheet':str(full),'detail_sheet':str(detail),'frames':frames,'decoded_frames':len(pts),'maximum_pts_difference_us':int(np.max(np.abs(np.asarray(pts)-ts)))}
        done.append(entry);write('PROGRESS.json',{'new_completed':sum(r['is_new'] for r in done),'new_total':24,'old_reused':12,'rows':done})
        print(json.dumps({'alias':alias,'completed':sum(r['is_new'] for r in done),'sheet':str(detail)},ensure_ascii=False),flush=True)
    for z in archives.values():z.close()
    write('SOURCE_PINS_BEFORE.json',pins);after={p:sha(p) for p in pins};assert after==pins;write('SOURCE_PINS_AFTER.json',after)
    for arc,st in archive_stats.items():now=Path(arc).stat();assert [now.st_size,now.st_mtime_ns]==st
    write('ARCHIVE_STAT_PINS.json',archive_stats);write('OUTPUT_PINS.json',outputs)
    done.sort(key=lambda r:(r['word_id'],r['recorded_person_id']))
    write('INDEX.json',{'status':'COMPLETE','new_clips':24,'new_raw_rgb_frames':384,'new_contact_sheets':48,'old_reused_clips':12,'old_reextracted':0,'total_clips':36,'rows':done,
       'fixed_selection':str(frozen_path),'no_detection':True,'no_training':True,'no_new_dtw':True,'no_user_mentor_access':True,'labels_absent_from_images':True,
       'sampling_limit':'Sixteen time samples summarize motion; short changes require original continuous video or additional explicitly recorded review.'})
    viewer=[]
    for e in done:
        viewer.append({'alias':e['alias'],'frames':[{'src':Path(f['file']).as_uri(),'frame':f['frame_id'],'time':f['pts_us']/1e6} for f in e['frames']]})
    page='''<!doctype html><meta charset="utf-8"><title>Public frame review</title><style>body{margin:0;background:#111;color:#eee;font:16px sans-serif}header{padding:14px;position:sticky;top:0;background:#222}img{display:block;max-width:100%;margin:auto;cursor:zoom-in}img.zoom{max-width:none}button,select{font:inherit;margin-right:12px}</style><header><select id="clips"></select><button id="prev">Previous</button><button id="next">Next</button><span id="info"></span> | Left/Right: frame; click: native size</header><img id="frame"><script>const DATA=DATA_PLACEHOLDER;let c=0,i=0;const sel=document.querySelector('#clips'),im=document.querySelector('#frame');DATA.forEach((x,j)=>sel.add(new Option(x.alias,j)));function show(){let f=DATA[c].frames[i];im.src=f.src;document.querySelector('#info').textContent=DATA[c].alias+' | '+(i+1)+'/16 | frame '+f.frame+' | '+f.time.toFixed(3)+'s'}function step(d){i=Math.max(0,Math.min(15,i+d));show()}sel.onchange=()=>{c=+sel.value;i=0;show()};document.querySelector('#prev').onclick=()=>step(-1);document.querySelector('#next').onclick=()=>step(1);document.onkeydown=e=>{if(e.key==='ArrowRight')step(1);if(e.key==='ArrowLeft')step(-1)};im.onclick=()=>im.classList.toggle('zoom');show();</script>'''.replace('DATA_PLACEHOLDER',json.dumps(viewer))
    (U/'VIEWER.html').write_text(page,encoding='utf-8')
    write('VERIFY.json',{'status':'PASS','new_clips':24,'old_reused_clips':12,'old_reextracted':0,'unique_source_hashes':36,'raw_png':384,'sheets':48,'source_pins_unchanged':len(pins),'archive_stats_unchanged':len(archive_stats),'pts_within1us':True,'decoded_rgb_equals_saved_png':True,'source_methods':{k:sum(r.get('source_method')==k for r in done) for k in ('existing_cache','exact_frozen_archive_member_copied_no_download')}})

if __name__=='__main__':
    try:main()
    except Exception:
        p=U/'FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
