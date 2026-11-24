"""Enlarge four already reviewed source frames; no extraction/inference or new selection scores."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import av
from PIL import Image,ImageDraw
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('body_reference_detail_drawing',U/'evidence_fallback.py')
draw=importlib.util.module_from_spec(spec);spec.loader.exec_module(draw)
TARGETS=[('NIA_SL_WORD0795_REAL01_F',96),('NIA_SL_WORD1880_REAL01_F',33),
         ('NIA_SL_WORD1936_REAL01_F',61),('NIA_SL_WORD2994_REAL01_F',44)]
rows={r['clip_id']:r for r in draw.read(U/'training_reference_montage_RESULT.json')['rows']}
sheet=Image.new('RGB',(1920,1180),(16,22,32));sd=ImageDraw.Draw(sheet);items=[]
for n,(cid,index) in enumerate(TARGETS):
    r=rows[cid];assert index in r['selected_frames']
    with np.load(r['observation'],allow_pickle=False) as z:
        xy=z['xy_px'];usable=z['usable'];ts=z['timestamp_us']
    with av.open(r['source']) as c:
        for i,f in enumerate(c.decode(video=0)):
            if i==index:rgb=f.to_ndarray(format='rgb24');break
    counts=[int(usable[index,33:54].sum()),int(usable[index,54:75].sum())]
    title=f'{r["label"]} / {cid} / f{index} {ts[index]/1e6:.3f}s / 원래 손점 L{counts[0]} R{counts[1]}'
    tile=draw.draw_frame(rgb,xy[index],usable[index],r['bottom'],title,size=(960,540))
    x=(n%2)*960;y=(n//2)*590;sheet.paste(tile,(x,y))
    sd.text((x+10,y+548),'원본 전체 영상 + 기존 관측. 노란선 위 RGB 손과 점 표시를 대조.',font=draw.font(21),fill='white')
    items.append(dict(clip_id=cid,frame_id=index,time_seconds=float(ts[index])/1e6,
                      source=r['source'],source_sha256=r['source_sha256'],
                      observation=r['observation'],observation_sha256=r['observation_sha256'],
                      native_left_usable_points=counts[0],native_right_usable_points=counts[1],bottom=r['bottom']))
p=U/'training_reference_detail_legacy_missing.jpg';assert not p.exists();sheet.save(p,quality=95,subsampling=0)
with (U/'training_reference_detail_RESULT.json').open('x',encoding='utf-8') as f:
    json.dump(dict(status='DETAIL_IMAGE_PENDING_VIEW',rows=items,image=str(p),sha256=draw.sha(p),
                   detector=0,training=0,development=0),f,ensure_ascii=False,indent=2)
print(json.dumps(items,ensure_ascii=True))
