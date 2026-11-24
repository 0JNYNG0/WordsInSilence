"""Decode every frame in pre-existing FIT hold ranges. No learned inference."""
from pathlib import Path
import hashlib, json, math
from datetime import datetime, timezone
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCES=[ROOT/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',
         ROOT/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json']
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    assert not (HERE/'PREPARED.json').exists(), 'Do not overwrite a completed review input set'
    rows=[]
    for source in SOURCES:
        for r in read(source)['rows']:
            if r['person'] in [2,3]: rows.append((source,r))
    assert len(rows)==8 and len({r['clip_id'] for _,r in rows})==8
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    out=[]
    for annotation,r in rows:
        p=Path(r['source']); assert sha(p)==r['source_sha256']
        assert 'visual_inputs' in str(p) and 'NIA_SL_WORD' in r['clip_id']
        lo,hi=min(r['hold_sample_frames']),max(r['hold_sample_frames'])
        frames=list(range(lo,hi+1)); cap=cv2.VideoCapture(str(p)); fps=cap.get(cv2.CAP_PROP_FPS)
        assert fps==r['fps']==30.0
        dst=HERE/r['clip_id'];dst.mkdir(parents=True,exist_ok=True)
        records=[];cells=[];crop_box=None
        for idx in range(hi+1):
            ok,bgr=cap.read(); assert ok,(r['clip_id'],idx)
            if idx<lo: continue
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); im=Image.fromarray(rgb)
            raw=dst/f'frame_{idx:04d}.png'; im.save(raw)
            w,h=im.size
            crop_box=[int(w*.22),int(h*.05),int(w*.78),int(h*.88)]
            crop=im.crop(crop_box); crop.thumbnail((350,330))
            cell=Image.new('RGB',(360,366),'white');cell.paste(crop,((360-crop.width)//2,32))
            d=ImageDraw.Draw(cell);d.text((9,4),f'Frame {idx}  {idx/fps:.3f}s',font=font,fill='#182b42')
            cells.append(cell);records.append(dict(frame=idx,time_seconds=idx/fps,path=str(raw),sha256=sha(raw)))
        cap.release()
        assert [x['frame'] for x in records]==frames
        cols=4; nr=math.ceil(len(cells)/cols);sheet=Image.new('RGB',(cols*360,nr*366+52),'#dfe5eb')
        d=ImageDraw.Draw(sheet);d.text((12,12),r['clip_id']+f' | every frame {lo}..{hi} | source RGB only',font=font,fill='#182b42')
        for i,cell in enumerate(cells):sheet.paste(cell,((i%cols)*360,52+(i//cols)*366))
        sp=dst/'all_frames.jpg';sheet.save(sp,quality=95)
        group=max(r['groups'],key=lambda g:g['offset'][1])
        out.append(dict(clip_id=r['clip_id'],person=r['person'],role='fit',word_id=r['clip_id'].split('_')[2],source=str(p),source_sha256=r['source_sha256'],source_annotation=str(annotation),source_annotation_sha256=sha(annotation),fps=fps,frames=frames,original_hold_samples=r['hold_sample_frames'],family=group['family'],source_group=group,frame_records=records,contact_sheet=str(sp),contact_sheet_sha256=sha(sp),contact_sheet_crop_box=crop_box,all_frames_decoded=True,visual_inspection_completed=False,semantic_completion=None))
    result=dict(at=datetime.now(timezone.utc).isoformat(),status='PREPARED_NOT_YET_REVIEWED',clips=len(out),total_frames=sum(len(r['frames']) for r in out),rows=out,policy=['Only public FIT record IDs 02/03.','All original frames in each predeclared terminal hold range decoded.','Full frames stored; sheet applies only a visual-review torso crop.','No model inference, no training, no educational correctness or completion targets.'])
    (HERE/'PREPARED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'clips':len(out),'frames':result['total_frames'],'output':str(HERE/'PREPARED.json')}))

if __name__=='__main__':main()
