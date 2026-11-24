"""Frozen TRAIN10 geometry contact sheets; original RGB plus legacy joints.

No MediaPipe/classifier. Source ZIP members copied only into this unit after
CRC/size/hash checks. All original bytes preserved for reviewer inspection.
"""
from pathlib import Path
import json, hashlib, zipfile, shutil, sys
import numpy as np
import av
from PIL import Image, ImageDraw, ImageFont
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent; OPS=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
FONT=Path('C:/Windows/Fonts/malgun.ttf')
def font(n):return ImageFont.truetype(str(FONT),n)
def draw_frame(rgb,xy,usable,bottom,title,size=(640,360)):
    im=Image.fromarray(rgb).convert('RGBA'); layer=Image.new('RGBA',im.size,(0,0,0,0));d=ImageDraw.Draw(layer)
    w,h=im.size
    d.rectangle((0,bottom,w,h),fill=(240,30,30,38))
    d.line((0,bottom,w,bottom),fill=(255,220,0,255),width=5)
    # Full-source body geometry and assigned hand markers; not a fresh detector.
    for side,body,base,color in [('L',(11,13,15),33,(60,255,110,255)),('R',(12,14,16),54,(20,215,255,255))]:
        for a,b in zip(body,body[1:]):
            if usable[a] and usable[b]: d.line(tuple(xy[a])+tuple(xy[b]),fill=color,width=4)
        edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
        for a,b in edges:
            a+=base;b+=base
            if usable[a] and usable[b]:d.line(tuple(xy[a])+tuple(xy[b]),fill=color,width=2)
        for j in range(base,base+21):
            if usable[j]:
                x,y=xy[j];c=color if y<bottom and 0<=x<w and 0<=y else (255,90,70,255)
                d.ellipse((x-4,y-4,x+4,y+4),fill=c)
    im=Image.alpha_composite(im,layer).convert('RGB')
    if size: im=im.resize(size,Image.Resampling.LANCZOS)
    d=ImageDraw.Draw(im);d.rectangle((0,0,im.width,30),fill=(18,23,31));d.text((7,3),title,font=font(17),fill='white')
    return im

def process(cid):
    m={r['clip_id']:r for r in read(OPS/'word100_training_20261013/MANIFEST.json') if r['role']=='train_candidate'}
    row=m[cid]; rec=read(Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013/records')/(cid+'.json'))
    source=U/'review_sources'/(cid+'.mp4');source.parent.mkdir(exist_ok=True)
    e=row['video_entries'][0]
    with zipfile.ZipFile(e['archive']) as z:
        info=z.getinfo(e['member']);assert info.CRC==e['crc32'] and info.file_size==e['bytes']
        if not source.exists():
            with z.open(info) as src,source.open('xb') as dst:shutil.copyfileobj(src,dst,1<<20)
    assert source.stat().st_size==e['bytes'] and sha(source)==rec['source_sha256']
    diag={r['clip_id']:r for r in read(U/'design_data_shoulder_window_RESULT.json')['fallback_diagnostics']}[cid]
    bottom=diag['proposed_bottom']
    with np.load(rec['observation'],allow_pickle=False) as z:xy=z['xy_px'];usable=z['usable'];ts=z['timestamp_us']
    a,b=row['intervals'][0];core=np.flatnonzero((ts>=round(a*1e6))&(ts<=round(b*1e6)))
    lost=np.zeros(len(ts),int)
    for start in (33,54):
        p=start+np.array([0,5,9,13,17]);before=usable[:,p].all(1);after=(usable[:,p]&(xy[:,p,1]<bottom)).all(1)
        lost+=before&~after
    worst=int(core[np.argmax(lost[core])]);highest_loss=np.flatnonzero((lost==lost[worst]) & (np.arange(len(ts))>=core[0]) & (np.arange(len(ts))<=core[-1]))
    selected=sorted(set([0,len(ts)-1,*np.linspace(core[0],core[-1],8).round().astype(int).tolist(),worst]))
    images={};decoded=0
    with av.open(str(source)) as container:
        for f in container.decode(video=0):
            if decoded in selected:images[decoded]=f.to_ndarray(format='rgb24')
            decoded+=1
    assert decoded==len(ts) and set(images)==set(selected)
    title=f"{row['canonical_label']} / {cid} / 하단 {bottom}px"
    cols=3;tw,th=640,360;rows=(len(selected)+cols-1)//cols
    sheet=Image.new('RGB',(cols*tw,rows*(th+36)+80),(16,22,32));d=ImageDraw.Draw(sheet)
    d.text((14,8),title,font=font(25),fill='white')
    d.text((14,45),'노랑: 고정 하단 / 붉은 영역·점: 입력 제외 / 초록 L·하늘 R: 기존 전체 영상 관측',font=font(21),fill=(205,214,226))
    for n,i in enumerate(selected):
        x=(n%cols)*tw;y=80+(n//cols)*(th+36)
        tile=draw_frame(images[i],xy[i],usable[i],bottom,f'f{i} / {ts[i]/1e6:.3f}s / 손바닥 손실쪽 {lost[i]}')
        sheet.paste(tile,(x,y));d.text((x+8,y+th+4),'핵심 주석 구간' if i in core else '시작/종료 참고',font=font(18),fill=(220,225,235))
    out=U/f'evidence_{cid}_contacts.png';assert not out.exists();sheet.save(out)
    detail=U/f'evidence_{cid}_loss_frame.png';assert not detail.exists()
    draw_frame(images[worst],xy[worst],usable[worst],bottom,f'{title} / f{worst} {ts[worst]/1e6:.3f}s',size=None).save(detail)
    meta={'clip_id':cid,'label':row['canonical_label'],'role':'train_candidate','source':str(source),'source_sha256':sha(source),
          'archive':e['archive'],'member':e['member'],'CRC32':e['crc32'],'source_bytes':e['bytes'],
          'observation':rec['observation'],'observation_sha256':rec['observation_sha256'],
          'boundary_coefficient_sha256':sha(U/'design_data_COEFFICIENT.json'),'bottom':bottom,'frame_count':decoded,
          'selected_frames':selected,'core_frame_range':[int(core[0]),int(core[-1])],
          'worst_palm5_loss_sample':worst,'maximum_loss_frames':highest_loss.tolist(),
          'images':{str(out):sha(out),str(detail):sha(detail)},
          'limitations':'Sparse frames from original RGB with legacy joints, not new crop detections. Counts are observation geometry, not ground truth. Unsampled transitions/contact timing not reviewed; possible support-hand loss requires human sign review.',
          'new_detector_conditions':0,'training':0}
    with (U/f'evidence_{cid}.json').open('x',encoding='utf-8') as f:json.dump(meta,f,ensure_ascii=False,indent=2)
    return meta

if __name__=='__main__':
    for cid in sys.argv[1:]:
        m=process(cid); print(json.dumps({'clip_id':cid,'images':list(m['images'])},ensure_ascii=False),flush=True)
