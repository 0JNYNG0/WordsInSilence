"""Fixed training-only sample: original ZIP bytes and decoded PTS versus cache.

No landmark extraction, no classifier, no writes to source archives. Contact
sheet is for framing/alignment inspection, not expert validation of sign meaning.
"""
import io
from pathlib import Path
import hashlib
import zipfile
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import U,OUT,read,write,sha,pin_check

def main():
    pin_check();plan=read(U/'MEDIA_AUDIT_PLAN.json')
    assert not (U/'MEDIA_CHECK.json').exists(), 'Reuse completed audit'
    manifest={r['clip_id']:r for r in read(U/'MANIFEST.json')}
    panels=[];checks=[]
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
    for cid in plan['clip_ids']:
        r=manifest[cid];assert r['role']=='train_candidate'
        e=r['video_entries'][0];record=read(OUT/'records'/f'{cid}.json')
        with zipfile.ZipFile(e['archive']) as z:
            b=z.read(e['member'])
        assert len(b)==e['bytes']
        assert hashlib.sha256(b).hexdigest()==record['source_sha256']
        with np.load(record['observation'],allow_pickle=False) as obs:
            t=obs['timestamp_us'];xy=obs['xy_px'];valid=obs['usable']
            mid=(r['intervals'][0][0]+r['intervals'][0][1])/2*1e6
            targets=[0,int(np.argmin(abs(t-mid))),len(t)-1]
            decoded=[];pictures={}
            with av.open(io.BytesIO(b)) as c:
                origin=None
                for i,frame in enumerate(c.decode(video=0)):
                    assert frame.pts is not None and frame.time_base is not None
                    pts=frame.pts*frame.time_base
                    if origin is None:origin=pts
                    decoded.append(round((pts-origin)*1000000))
                    if i in targets:pictures[i]=frame.to_image()
            np.testing.assert_array_equal(np.array(decoded,np.int64),t)
            assert len(pictures)==len(set(targets))
            tiles=[]
            for index,name in zip(targets,['영상 시작','주석 핵심 중간','영상 끝']):
                im=pictures[index];w,h=im.size;s=320/w
                im=im.resize((320,round(h*s)))
                canvas=Image.new('RGB',(320,240),'white');canvas.paste(im,(0,32))
                d=ImageDraw.Draw(canvas);d.text((5,4),f'{name} {t[index]/1e6:.2f}s',font=font,fill='#172b4d')
                for j in range(33,75):
                    if valid[index,j]:
                        x,y=xy[index,j]*s;y+=32
                        d.ellipse((x-2,y-2,x+2,y+2),fill='#ee2449' if j<54 else '#fcbf16')
                tiles.append(canvas)
            row=Image.new('RGB',(960,280),'#f3f5f7');d=ImageDraw.Draw(row)
            d.text((8,5),f"{cid}  |  {r['canonical_label']}  |  학습용 촬영자 {r['signer_id']:02d}",font=font,fill='#172b4d')
            for j,tile in enumerate(tiles):row.paste(tile,(j*320,35))
            panels.append(row)
            checks.append(dict(clip_id=cid,frames=len(t),source_sha256=record['source_sha256'],native_pts_equal=True))
    sheet=Image.new('RGB',(960,280*len(panels)),'white')
    for i,p in enumerate(panels):sheet.paste(p,(0,i*280))
    sheet.save(U/'MEDIA_CONTACT_SHEET.jpg',quality=92)
    write(U/'MEDIA_CHECK.json',dict(status='PASS',scope='Six fixed training clips; source bytes and frame PTS, no sign semantic judgement',
        landmark_extractions=0,classifier_calls=0,clips=checks,contact_sheet=str(U/'MEDIA_CONTACT_SHEET.jpg')))

if __name__=='__main__':main()
