"""Decode only fixed public event contexts into review sheets. No detectors."""
from pathlib import Path
import json, hashlib, datetime
import cv2
from PIL import Image, ImageDraw
U=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):
    with (U/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    prop=U/'review/EVENT_WINDOW_PROPOSAL.json';d=json.loads(prop.read_text(encoding='utf-8'))
    assert d['selection_policy']['proposed_windows']==2
    assert not (U/'EVENT_EXTRACTION.json').exists()
    write('FROZEN_EVENT_REVIEW.json',dict(proposal_sha256=sha(prop),approved_by='root under user-approved bounded plan',
       at=datetime.datetime.now().astimezone().isoformat(),max_paired_windows=2,exact_context_frames=89,
       labels_for='diagnosis only; no train/selection truth',selection_by='existing missing evidence before scores',
       stage_claim='independent observed flow, never same-kind as unique correspondence'))
    out=[]
    for w in d['windows']:
        for ax in ('query_axis','reference_axis'):
            a=w[ax];vp=Path(a['source_video']['path']);assert sha(vp)==a['source_video']['sha256']
            dest=U/'visuals'/a['alias'];dest.mkdir(parents=True,exist_ok=True)
            needed=set(a['frame_ids']);focals={f['frame_id'] for f in a['focal_frames']}
            cap=cv2.VideoCapture(str(vp));fps=cap.get(cv2.CAP_PROP_FPS); assert cap.isOpened()
            images=[]; index=0
            while index<=max(needed):
                ok,fr=cap.read();assert ok,(a['alias'],index)
                if index in needed:
                    im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
                    original=dest/f'f{index:04d}.jpg';im.save(original,quality=97)
                    tile=im.copy();tile.thumbnail((480,270));canvas=Image.new('RGB',(480,300),'#f2f4f6')
                    canvas.paste(tile,((480-tile.width)//2,30));dr=ImageDraw.Draw(canvas)
                    dr.text((8,8),f'{a["alias"]}  native f{index}   {index/fps:.3f}s'+('  FOCAL' if index in focals else ''),fill='black')
                    images.append(canvas)
                index+=1
            cap.release()
            pages=[]
            for k in range(0,len(images),9):
                subset=images[k:k+9];sheet=Image.new('RGB',(1440,300*((len(subset)+2)//3)),'white')
                for n,im in enumerate(subset):sheet.paste(im,((n%3)*480,(n//3)*300))
                page=dest/f'page{k//9+1:02d}.jpg';sheet.save(page,quality=95);pages.append(str(page))
            out.append(dict(window=w['window_id'],alias=a['alias'],fps=fps,frame_ids=a['frame_ids'],
                 focal_frame_ids=sorted(focals),pages=pages,source_sha256=sha(vp)))
    assert sum(len(x['frame_ids']) for x in out)==89
    write('EVENT_EXTRACTION.json',dict(status='COMPLETE_FIXED_CONTEXT_ONLY',axes=out,decoded_review_frames=89,
          new_detector_runs=0,code_sha256=sha(__file__)))
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
