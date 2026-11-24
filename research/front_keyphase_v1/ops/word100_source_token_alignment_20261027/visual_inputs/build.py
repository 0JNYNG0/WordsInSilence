"""Read-only public frame export for blinded descriptive observation."""
from pathlib import Path
import hashlib,json,traceback
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;OPS=U.parent.parent
OLD=OPS/'word100_face_body_stage_20261027'
WORDS={'WORD0949','WORD1139','WORD1290','WORD2551'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(n,d):(U/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sheet(images,records,alias,crop=None):
    cw,ch=480,350 if crop is not None else 300
    out=Image.new('RGB',(4*cw,4*ch),(245,247,250));draw=ImageDraw.Draw(out)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    for n,(im,r) in enumerate(zip(images,records)):
        image=im.crop(crop) if crop is not None else im.copy();image.thumbnail((cw-8,ch-34),Image.Resampling.LANCZOS)
        x=(n%4)*cw+(cw-image.width)//2;y=(n//4)*ch+30
        out.paste(image,(x,y));draw.text(((n%4)*cw+8,(n//4)*ch+5),f'{alias} | f{r["frame_id"]:04d} | {r["pts_us"]/1e6:.3f}s',font=font,fill=(20,30,50))
    return out

def main():
    if (U/'INDEX.json').exists():raise FileExistsError('Completed visual exports preserved')
    frozen=read(OLD/'FROZEN.json');rows=sorted((r for r in frozen['rows'] if r['word'] in WORDS),key=lambda r:(r['word'],r['signer']))
    assert len(rows)==12 and all(r['signer'] in (1,2,3) and r['role']=='train_candidate' for r in rows)
    pins={str(OLD/'FROZEN.json'):sha(OLD/'FROZEN.json')};done=[];outputs={}
    for i,row in enumerate(rows):
        alias=f'V{i+1:02d}';folder=U/alias;folder.mkdir(exist_ok=True)
        base=OLD/'data'/row['clip_id'];dn=read(base/'DONE.json');video=Path(dn['video'])
        assert sha(video)==row['source']['sha256'],('video sha mismatch',alias)
        op=base/'masked.npz';sc=read(base/'masked.sidecar.json')
        assert sha(op)==sc['observation_sha256'] and sc['source_sha256']==row['source']['sha256']
        for p in (video,op,base/'DONE.json',base/'masked.sidecar.json'):pins[str(p)]=sha(p)
        with np.load(op,allow_pickle=False) as z:ts=z['timestamp_us'].copy();fid=z['frame_id'].copy();xy=z['xy_px'].copy();usable=z['usable'].copy()
        assert np.array_equal(fid,np.arange(len(fid)))
        begin=max(0,min(x[0] for x in row['intervals'])-.25);end=min(ts[-1]/1e6,max(x[1] for x in row['intervals'])+.25)
        targets=np.linspace(begin*1e6,end*1e6,16);indices=np.array([int(np.argmin(np.abs(ts-t))) for t in targets])
        assert len(set(indices.tolist()))==16
        selected={};all_pts=[];first_pts=None;pixel_hashes={}
        with av.open(str(video)) as c:
            stream=c.streams.video[0]
            for n,frame in enumerate(c.decode(stream)):
                if first_pts is None:first_pts=frame.pts
                t=int(round(float((frame.pts-first_pts)*frame.time_base)*1e6));all_pts.append(t)
                if n in indices:
                    rgb=frame.to_ndarray(format='rgb24');pixel_hashes[n]=hashlib.sha256(rgb.tobytes()).hexdigest();selected[n]=Image.fromarray(rgb)
        assert len(all_pts)==len(fid)==dn['frames']
        assert np.max(np.abs(np.array(all_pts)-ts))<=1,('PTS mismatch',alias)
        w,h=selected[int(indices[0])].size
        # A fixed display crop per clip, containing head/face and all observed
        # upper-body/hand positions in the selected window. Top remains zero.
        ids=np.r_[np.arange(23),np.arange(33,75)];p=xy[indices][:,ids];ok=usable[indices][:,ids]&np.isfinite(p).all(-1)
        pp=p[ok];assert len(pp)
        crop=(max(0,int(np.floor(pp[:,0].min()-80))),0,min(w,int(np.ceil(pp[:,0].max()+80))),min(h,int(np.ceil(pp[:,1].max()+100))))
        assert crop[2]>crop[0] and crop[3]>crop[1]
        records=[];images=[]
        for j,idx in enumerate(indices):
            idx=int(idx);im=selected[idx];p=folder/f'frame_{j+1:02d}_f{idx:04d}.png';im.save(p)
            # Verify lossless save, including exact RGB bytes after decoding PNG.
            with Image.open(p) as check:assert hashlib.sha256(np.asarray(check.convert('RGB')).tobytes()).hexdigest()==pixel_hashes[idx]
            outputs[str(p)]=sha(p);images.append(im)
            records.append({'ordinal':j+1,'frame_id':idx,'pts_us':all_pts[idx],'observation_timestamp_us':int(ts[idx]),'target_us':float(targets[j]),'file':str(p),'rgb_sha256':pixel_hashes[idx]})
        full=folder/'full_contact.png';zoom=folder/'detail_contact.png'
        sheet(images,records,alias).save(full);sheet(images,records,alias,crop).save(zoom)
        outputs[str(full)]=sha(full);outputs[str(zoom)]=sha(zoom)
        entry={'alias':alias,'clip_id':row['clip_id'],'word_id':row['word'],'label':row['label'],'recorded_person_id':row['signer'],'source':str(video),'source_sha256':row['source']['sha256'],'annotation_intervals':row['intervals'],'export_span_seconds':[begin,end],
               'full_frame_size':[w,h],'display_only_crop_xyxy':list(crop),'crop_policy':'union of selected upperbody and hand observations + horizontal80px/bottom100px margin; top=0 to preserve full head; no inference/training ROI change',
               'full_sheet':str(full),'detail_sheet':str(zoom),'frames':records,'decoded_frames':len(all_pts),'maximum_pts_difference_us':int(np.max(np.abs(np.array(all_pts)-ts)))}
        done.append(entry);write('PROGRESS.json',{'completed':len(done),'total':12,'rows':done})
        print(json.dumps({'complete':alias,'count':len(done),'full_sheet':str(full),'detail_sheet':str(zoom)},ensure_ascii=False),flush=True)
    write('SOURCE_PINS_BEFORE.json',pins)
    after={p:sha(p) for p in pins};assert pins==after
    write('SOURCE_PINS_AFTER.json',after);write('OUTPUT_PINS.json',outputs)
    write('INDEX.json',{'status':'COMPLETE','selection':'Fixed four source-linked word IDs within existing18 TRAIN pilot, all01/02/03; not selected by result','count':12,'frames_per_clip':16,'raw_rgb_frames':192,'contact_sheets':24,'rows':done,
        'no_detection':True,'no_training':True,'no_new_dtw':True,'no_user_mentor_access':True,'labels_absent_from_images':True,
        'visual_blinding_limit':'Images show neutral V aliases; identity/label map retained separately. Exporter knows labels; observer must avoid INDEX until first observation pass.',
        'sampling_limit':'16 frames summarize motion, do not prove every brief handshape transition or contact. Original clips remain available for targeted review.'})
    viewer=[]
    for e in done:
        viewer.append({'alias':e['alias'],'frames':[{'src':f'{e["alias"]}/{Path(f["file"]).name}','frame':f['frame_id'],'time':f['pts_us']/1e6} for f in e['frames']]})
    page='''<!doctype html><meta charset="utf-8"><title>Public frame review</title><style>body{margin:0;background:#111;color:#eee;font:16px sans-serif}header{padding:14px;position:sticky;top:0;background:#222}img{display:block;max-width:100%;margin:auto;cursor:zoom-in}img.zoom{max-width:none}button,select{font:inherit;margin-right:12px}</style><header><select id="clips"></select><button id="prev">Previous</button><button id="next">Next</button><span id="info"></span> | Left/Right arrows: frame; click image: native size</header><img id="frame"><script>const DATA=DATA_PLACEHOLDER;let c=0,i=0;const sel=document.querySelector('#clips'),im=document.querySelector('#frame');DATA.forEach((x,j)=>sel.add(new Option(x.alias,j)));function show(){let f=DATA[c].frames[i];im.src=f.src;document.querySelector('#info').textContent=DATA[c].alias+' | '+(i+1)+'/16 | frame '+f.frame+' | '+f.time.toFixed(3)+'s'}function step(d){i=Math.max(0,Math.min(15,i+d));show()}sel.onchange=()=>{c=+sel.value;i=0;show()};document.querySelector('#prev').onclick=()=>step(-1);document.querySelector('#next').onclick=()=>step(1);document.onkeydown=e=>{if(e.key==='ArrowRight')step(1);if(e.key==='ArrowLeft')step(-1)};im.onclick=()=>im.classList.toggle('zoom');show();</script>'''.replace('DATA_PLACEHOLDER',json.dumps(viewer))
    (U/'VIEWER.html').write_text(page,encoding='utf-8')
    write('VERIFY.json',{'status':'PASS','videos':12,'frames':192,'sheets':24,'source_pins':len(pins),'source_hashes_unchanged':True,'video_sha_matches_frozen':True,'pts_within_1us':True,'png_rgb_bytes_match_decode':True,'same_agent_verification':True})

if __name__=='__main__':
    try:main()
    except Exception:
        p=U/'FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
