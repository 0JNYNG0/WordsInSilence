from pathlib import Path
import json, hashlib
import av
from PIL import Image, ImageDraw

OPS=Path(__file__).resolve().parents[2]
UNIT=Path(__file__).resolve().parents[1]
SOURCE=OPS/'word100_core_stage_evidence_20261031'/'media'
records=[]
for alias in ['P036','P048','P066','P072']:
    video=SOURCE/f'{alias}.webm'
    meta=json.loads((SOURCE/f'{alias}.json').read_text(encoding='utf-8'))
    out=UNIT/'evidence'/alias
    out.mkdir(exist_ok=True,parents=True)
    frames=[]
    with av.open(str(video)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            img=frame.to_image()
            img.save(out/f'f{i:04d}.jpg',quality=95)
            frames.append((img,frame.time))
    assert len(frames)==meta['frames']==len(meta['source_pts_seconds'])
    sheets=[]
    for start in range(0,len(frames),24):
        stop=min(start+24,len(frames))
        sheet=Image.new('RGB',(1160,1800),'#eeeeee')
        draw=ImageDraw.Draw(sheet)
        for k,i in enumerate(range(start,stop)):
            x=(k%4)*290; y=(k//4)*300
            tile=frames[i][0].crop((190,0,770,540)).resize((290,270))
            sheet.paste(tile,(x,y+30))
            draw.text((x+4,y+8),f'{alias} f{i:03d} {meta["source_pts_seconds"][i]:.3f}s',fill='black')
        path=out/f'continuous_{start:03d}_{stop-1:03d}.jpg'
        sheet.save(path,quality=95)
        sheets.append({'path':str(path),'frame_ids':list(range(start,stop))})
    records.append({'alias':alias,'source_webm':str(video),'source_webm_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
                    'timing_metadata':str(SOURCE/f'{alias}.json'),'native_frame_count':len(frames),
                    'source_pts_seconds':meta['source_pts_seconds'],'display_pts_seconds':[x[1] for x in frames],
                    'contact_crop_pixels':[190,0,770,540],'contact_tile_pixels':[290,270],
                    'individuals_full_frame':True,'sheets':sheets})
(UNIT/'evidence'/'DEV_CONTACT_MANIFEST.json').write_text(json.dumps({'method':'decode existing RGB display WebM; no model or landmark inference','clips':records},ensure_ascii=False,indent=2),encoding='utf-8')
print('prepared',len(records),'clips',sum(len(r['sheets']) for r in records),'continuous sheets',sum(r['native_frame_count'] for r in records),'frames')
