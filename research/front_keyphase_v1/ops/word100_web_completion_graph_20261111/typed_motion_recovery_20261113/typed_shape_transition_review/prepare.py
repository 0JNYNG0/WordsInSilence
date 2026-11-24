"""Freeze six public-FIT ranges; extract every original RGB frame for review."""
from pathlib import Path
import json,hashlib,math
from datetime import datetime,timezone
import cv2
from PIL import Image,ImageDraw,ImageFont

P=Path(__file__).resolve().parent
H=P.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    assert not(P/'FROZEN.json').exists(),'Do not overwrite frozen review'
    proposal=H/'paired_static_progression_learning/SHAPE_TRANSITION_EVIDENCE_REVIEW.json'
    sources=read(proposal)['proposed_short_contract']['sources']
    assert len(sources)==6 and sum(r['frames'] for r in sources)==206
    assert {r['person'] for r in sources}=={2,3,4,5,6,7}
    for r in sources:assert sha(r['source'])==r['source_sha256']
    contract=dict(at=datetime.now(timezone.utc).isoformat(),status='FROZEN_BEFORE_NEW_RGB_INSPECTION',
        selection='Existing family2 outer onset-4 through family3 outer offset+4, no model-score selection.',
        sources=sources,source_proposal=str(proposal),proposal_sha256=sha(proposal),
        hand_labels=['open','closed','flexing','unknown'],
        body_labels=['wrist_lowering','forearm_raising','hold','transition','unknown'],
        provenance='Direct RGB frame labels and explicitly separated temporal inference only. No dense truth by interpolation. Body and hand certainty are independent.',
        semantic_completion=None,educational_correctness=None,
        no_model_predictions_used=True,no_training=True,no_dev_or_private_or_final8=True,
        crop_body_fraction=[.22,.05,.78,.88],crop_hand_context_fraction=[.32,.12,.61,.55],
        montage='Six consecutive frames per page; full upper-body/arm context and hand-area zoom side-by-side. Original full frames also saved.',
        script_sha256=sha(__file__))
    write(P/'FROZEN.json',contract)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    rows=[]
    for r in sources:
        a,b=r['proposed_inspection_range'];cap=cv2.VideoCapture(r['source']);assert cap.isOpened()
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        dest=P/r['clip_id'];dest.mkdir(exist_ok=True);frames=[];images=[]
        for frameid in range(b+1):
            ok,bgr=cap.read();assert ok,(r['clip_id'],frameid)
            if frameid<a:continue
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);im=Image.fromarray(rgb)
            p=dest/f'frame_{frameid:04}.png';assert not p.exists();im.save(p)
            frames.append(dict(frame=frameid,path=str(p),sha256=sha(p),
                timestamp_seconds=float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000))
            images.append(im)
        cap.release();assert len(frames)==b-a+1
        pages=[]
        w,h=images[0].size
        bodybox=tuple(round(v*d) for v,d in zip(contract['crop_body_fraction'],[w,h,w,h]))
        handbox=tuple(round(v*d) for v,d in zip(contract['crop_hand_context_fraction'],[w,h,w,h]))
        for start in range(0,len(images),6):
            chunk=images[start:start+6];records=frames[start:start+6]
            canvas=Image.new('RGB',(1280,60+370*math.ceil(len(chunk)/2)),'#e4e9ef');draw=ImageDraw.Draw(canvas)
            draw.text((12,12),f"{r['clip_id']} | ALL original frames {records[0]['frame']}..{records[-1]['frame']}",font=font,fill='#172638')
            for j,(im,rec) in enumerate(zip(chunk,records)):
                x=(j%2)*640;y=60+(j//2)*370
                draw.text((x+8,y+4),f"F{rec['frame']}  {rec['timestamp_seconds']:.3f}s | body / hand context",font=small,fill='#172638')
                body=im.crop(bodybox);body.thumbnail((265,330))
                hand=im.crop(handbox);hand.thumbnail((355,330))
                canvas.paste(body,(x+5+(265-body.width)//2,y+32+(330-body.height)//2))
                canvas.paste(hand,(x+275+(355-hand.width)//2,y+32+(330-hand.height)//2))
            p=dest/f'page_{start//6+1:02}.jpg';canvas.save(p,quality=95)
            pages.append(dict(path=str(p),sha256=sha(p),frames=[f['frame'] for f in records],viewed=False))
        rows.append(dict(**r,original_frame_count=n,decoder_fps=fps,
            original_size=[w,h],body_crop=bodybox,hand_context_crop=handbox,
            frame_records=frames,pages=pages,all_frames_extracted=True,visual_review_complete=False))
        print(r['clip_id'],len(frames),len(pages),flush=True)
    write(P/'PREPARED.json',dict(at=datetime.now(timezone.utc).isoformat(),rows=rows,
        frames=sum(len(r['frame_records']) for r in rows),pages=sum(len(r['pages']) for r in rows),
        frozen_sha256=sha(P/'FROZEN.json'),status='EXTRACTED_NOT_VISUALLY_REVIEWED'))

if __name__=='__main__':main()
