"""Public RGB + pinned saved sensor observations. No new detector, no user data."""
from pathlib import Path
import json,hashlib,zipfile
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;B=U.parent/'word100_body_boundary_20261017';ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
refs=read(ROOT/'research/front_keyphase_v1/config/word100_analysis_references.json')['items']
rows={r['clip_id']:r for r in read(B/'PREP_CONTEXT.json')['ordered_rows'] if r['role']=='train_candidate'}
stats=[s for p in sorted((U/'records').glob('*.json')) for s in read(p)['input_stats'] if s['view']=='sensor']
# Fixed input-only selection; never consult predictions or softmax scores.
risk=[]
for key in ('core_trimmed_frames','longest_no_hand_seconds'):
    chosen=0
    for s in sorted(stats,key=lambda s:(-s[key],s['clip_id'])):
        if s['clip_id'] in {r['clip_id'] for r in risk}:continue
        risk.append({'clip_id':s['clip_id'],'label':s['label'],'reason':key,'value':s[key]});chosen+=1
        if chosen==4:break
spec={'references':[{'clip_id':r['clip_id'],'label':r['label'],'video_path':r['video_path'],'expected_sha256':r['source_sha256']} for r in refs],
    'risk':risk,'selection':'Existing fixed100references;4largest annotation trim and4longest both-hand absence (unique) selected only from input stats, no prediction-based choice.',
    'display':'Full original frame with saved sensor usable joints; cyan lower ROI line. Missing joints are not interpolated. Fixed8frames per reference,32 evenly spaced per risk plus gap endpoints. Display does not alter model input.'}
if (U/'VISUAL_SCOPE.json').exists():assert read(U/'VISUAL_SCOPE.json')==spec
else:write(U/'VISUAL_SCOPE.json',spec)
out=U/'visuals';out.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14)
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
sources={};manifest=[]
def source(cid,path=None,expected=None):
    if cid in sources:return Path(sources[cid]['path'])
    r=rows[cid];rec=read(B/'data/records'/f'{cid}.json');p=Path(path or r.get('retained_video') or 'MISSING');expected=expected or rec['source']['sha256']
    if not p.is_file():
        p=out/f'{cid}.mp4';assert not p.exists()
        with zipfile.ZipFile(rec['source']['archive'],'r') as z:data=z.read(rec['source']['member'])
        assert hashlib.sha256(data).hexdigest()==expected;p.write_bytes(data)
    assert sha(p)==expected;sources[cid]={'path':str(p),'sha256':expected};return p
def strip(cid,label,count,path=None,expected=None,risk=False,render=True):
    p=source(cid,path,expected);rec=read(B/'data/records'/f'{cid}.json');obs=np.load(rec['views']['sensor']['observation'],allow_pickle=False);ch=np.load(rec['views']['sensor']['channel_file'],allow_pickle=False)
    a,b=rows[cid]['intervals'][0];ci=np.flatnonzero((obs['timestamp_us']>=round(a*1e6))&(obs['timestamp_us']<=round(b*1e6)))
    ix=list(ci[np.rint(np.linspace(0,len(ci)-1,count)).astype(int)])
    if risk:
        ss=next(s for s in stats if s['clip_id']==cid)
        ix=sorted(set(ix+[j for run in ss['no_hand_runs']+ss['trimmed_runs'] for j in (run['start_index'],run['end_index'])]))
    lookup={int(v):i for i,v in enumerate(ch['whole_frame_id'])};cap=cv2.VideoCapture(str(p)) if render else None;images=[];times=[]
    if render:assert cap.isOpened()
    for j in ix:
        fid=int(obs['frame_id'][j]);time=float(obs['timestamp_us'][j])/1e6
        if fid in lookup:
            mask=ch['x_whole'][lookup[fid],195:237].reshape(2,21)>0;status=f'L{mask[0].sum()} R{mask[1].sum()}'
        else:status='INPUT TRIM'
        times.append({'frame':fid,'seconds':time,'input':status})
        if not render:continue
        cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,frame=cap.read();assert ok,(cid,fid)
        h,w=frame.shape[:2];pic=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((320,180));d=ImageDraw.Draw(pic);pts=obs['xy_px'][j];valid=obs['usable'][j]
        def point(k):return (float(pts[k,0])*320/w,float(pts[k,1])*180/h)
        for l,r in [(11,12),(11,13),(13,15),(12,14),(14,16)]:
            if valid[l] and valid[r]:d.line([point(l),point(r)],fill='#ffdb58',width=2)
        for start,color in [(33,'#00ff8c'),(54,'#ff7fa8')]:
            for l,r in edges:
                if valid[start+l] and valid[start+r]:d.line([point(start+l),point(start+r)],fill=color,width=1)
            for k in range(start,start+21):
                if valid[k]:x,y=point(k);d.ellipse((x-1,y-1,x+1,y+1),fill=color)
        bottom=rec['calibration']['box_source_pixels'][3]*180/h;d.line([(0,bottom),(320,bottom)],fill='#37e6fa',width=1)
        panel=Image.new('RGB',(320,208),'#fff');panel.paste(pic,(0,0));draw=ImageDraw.Draw(panel);time=float(obs['timestamp_us'][j])/1e6
        if fid in lookup:
            mask=ch['x_whole'][lookup[fid],195:237].reshape(2,21)>0;status=f'L{mask[0].sum()} R{mask[1].sum()}'
        else:status='INPUT TRIM'
        draw.text((5,183),f'{time:.2f}s / f{fid} / {status}',font=small,fill='#c01f27' if 'TRIM' in status or status=='L0 R0' else '#202e43');images.append(panel)
    if cap is not None:cap.release()
    obs.close();ch.close();manifest.append({'clip_id':cid,'label':label,'risk':risk,'frames':times,'source_sha256':sources[cid]['sha256']})
    return images
for block in range(20):
    existing=out/f'reference_{block+1:02}.jpg'
    if existing.exists():
        assert sha(existing)==read(U/'VISUAL_FIRST_ERROR.json')['reference_hashes'][existing.name]
        for r in spec['references'][block*5:(block+1)*5]:strip(r['clip_id'],r['label'],8,r['video_path'],r['expected_sha256'],render=False)
        continue
    sheet=Image.new('RGB',(2560,1190),'#e9edf2');d=ImageDraw.Draw(sheet)
    for rownum,r in enumerate(spec['references'][block*5:(block+1)*5]):
        y=rownum*238;d.text((8,y+4),f"{block*5+rownum+1:03}  {r['label']}  |  {r['clip_id']}  |  공개 학습 원본 + 저장 관절 (검출 재실행 없음)",font=font,fill='#102a40')
        for col,pic in enumerate(strip(r['clip_id'],r['label'],8,r['video_path'],r['expected_sha256'])):sheet.paste(pic,(320*col,y+29))
    sheet.save(out/f'reference_{block+1:02}.jpg',quality=90)
    print(f'references {(block+1)*5}/100',flush=True)
for num,r in enumerate(risk,1):
    pics=strip(r['clip_id'],r['label'],32,risk=True);height=40+208*((len(pics)+7)//8);sheet=Image.new('RGB',(2560,height),'white');d=ImageDraw.Draw(sheet);d.text((5,8),f"{r['label']} | {r['clip_id']} | {r['reason']}={r['value']}",font=font,fill='black')
    for k,pic in enumerate(pics):sheet.paste(pic,((k%8)*320,40+(k//8)*208))
    sheet.save(out/f'risk_{num:02}.jpg',quality=94)
    print(f'risk {num}/8',flush=True)
for v in sources.values():assert sha(v['path'])==v['sha256']
write(U/'VISUAL_MANIFEST.json',{'sources':sources,'samples':manifest,'files':{p.name:sha(p) for p in out.glob('*.jpg')},'scope_sha256':sha(U/'VISUAL_SCOPE.json'),'training':0,'new_detector':0})
