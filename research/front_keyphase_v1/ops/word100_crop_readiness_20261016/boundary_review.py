"""Inspect three TRAIN source examples; decode only, no detector/classifier."""
from pathlib import Path
import json,io,zipfile,hashlib,sys
import av,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
manifest={r['clip_id']:r for r in read(U.parent/'word100_training_20261013/MANIFEST.json') if r['role']=='train_candidate'}
v=read(U/'VERIFICATION.json');a={r['clip_id']:r for r in read(U/'RESULT.json')['rows']}
targets=['NIA_SL_WORD1643_REAL01_F','NIA_SL_WORD0668_REAL01_F','NIA_SL_WORD2177_REAL02_F']
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',21);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
canvas=Image.new('RGB',(1500,1060),'white');d=ImageDraw.Draw(canvas);index=[]
for row,cid in enumerate(targets):
    m=manifest[cid];e=m['video_entries'][0];rec=read(Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013/records')/(cid+'.json'))
    flag=next(r for r in v['boundary_flags'] if r['clip_id']==cid);target=flag['middle_frame'];ids=[target-1,target,target+1]
    with zipfile.ZipFile(e['archive']) as z:
        zi=z.getinfo(e['member']);assert zi.CRC==e['crc32'] and zi.file_size==e['bytes'];data=z.read(zi)
    assert hashlib.sha256(data).hexdigest()==rec['source_sha256']
    with np.load(rec['observation']) as z:xy=z['xy_px'];use=z['usable'];t=z['timestamp_us']
    imgs={}
    with av.open(io.BytesIO(data)) as c:
        for i,f in enumerate(c.decode(video=0)):
            if i in ids:imgs[i]=f.to_image()
            if i==max(ids):break
    y=row*350;d.text((10,y+3),f'{m["canonical_label"]} / REAL{m["signer_id"]:02} / 노란선=고정 경계, 점=기존 관절 추정',font=font,fill='black')
    for col,f in enumerate(ids):
        im=imgs[f];dr=ImageDraw.Draw(im);dr.rectangle(a[cid]['box'],outline='yellow',width=6)
        for off,color in [(33,'cyan'),(54,'magenta')]:
            for j in range(off,off+21):
                if use[f,j]:
                    x,yh=xy[f,j];dr.ellipse([x-5,yh-5,x+5,yh+5],fill=color)
        im.thumbnail((490,276));canvas.paste(im,(col*500+5,y+60));d.text((col*500+10,y+34),f'frame {f} / {t[f]/1e6:.3f}s',font=small,fill='black')
    index.append({'clip_id':cid,'label':m['canonical_label'],'frames':ids,'source_sha256_verified':rec['source_sha256'],'longest_core_geometric_gap_s':flag['longest_core_run_seconds']})
canvas.save(U/'PUBLIC_BOUNDARY_REVIEW.jpg',quality=92)
(U/'BOUNDARY_VISUAL_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),'utf-8')
