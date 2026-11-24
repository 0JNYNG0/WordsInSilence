"""Matched-coordinate/source RGB diagnosis for fixed FIT scenes; no inference."""
from audit import *
import cv2
from PIL import Image,ImageDraw,ImageFont
R=U/'rgb_review'

def main():
    assert not(R/'MATCHED_COORDINATES.json').exists()
    manifest=read(R/'MANIFEST.json');selection=read(R/'SELECTION.json');a=read(U/'RESULTS.json')
    ready={r['clip_id']:r for r in read(OPS/'word100_shared_motion_learning_20261109/READY.json')['rows']if r['role']=='fit'and r['person']in range(2,8)}
    crop={r['clip_id']:r for p in[old.R/'CROP_INPUTS_COMPLETE.json',old.D/'CROP_INPUTS_COMPLETE.json']for r in read(p)['rows']if r['clip_id']in ready}
    sys.path.insert(0,str(OPS/'word100_observation_guard_20261108'))
    guard=old.load('_fixed_rgb_guard',OPS/'word100_observation_guard_20261108/guard.py');ad=old.load('_fixed_rgb_ad',OPS/'word100_rtmw_shared_input_20261108/adapter.py')
    pin={str(U/'RESULTS.json'):sha(U/'RESULTS.json'),str(R/'MANIFEST.json'):sha(R/'MANIFEST.json'),str(Path(__file__)):sha(Path(__file__))};out=[];sheets=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    for case in manifest['cases']:
        cid=case['clip_id'];sr=ready[cid];cr=crop[cid]
        for p,h in[(sr['channel'],sr['channel_sha256']),(sr['path'],sr['sha256']),(cr['path'],cr['sha256']),(case['source'],case['source_sha256'])]:assert sha(p)==h;pin[p]=h
        ch=np.load(sr['channel']);z=np.load(sr['path']);cz=np.load(cr['path']);ts=z['timestamp_us'];assert np.array_equal(ts,cz['timestamp_us'])
        cl=guard.quarantine(cz['xy_px'],cz['score'],ts,640,480);conv=ad.convert(cl['xy_px'],cz['score'],ts,cz['frame_id'],640,480)
        xx={'full':ch['full'],'actual_crop':conv['tcn280']}
        spans=case['target_spans']
        for span in spans:
            aa,bb=map(int,span)
            for view,x in xx.items():out.append(dict(case_id=case['id'],clip_id=cid,span=[aa,bb],view=view,measurement=components(x,ts,aa,bb)))
        fw,fh=case['source_dimensions'];roi=[int(fw*.32),int(fh*.29),int(fw*.68),int(fh*.63)]
        rgbcap=cv2.VideoCapture(case['source']);assert rgbcap.isOpened();tiles=[]
        l,t,rr,bb=cr['crop'];cp=conv['native67'].copy();cp[:,:,0]=cp[:,:,0]/640*(rr-l)+l;cp[:,:,1]=cp[:,:,1]/480*(bb-t)+t
        fp=ch['native67'];ids=[6,7,4]
        for rg,(aa,bb)in enumerate(case['ranges']):
            for f in range(aa,bb+1):
                rgbcap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=rgbcap.read();assert ok
                im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop(tuple(roi));draw=ImageDraw.Draw(im)
                for pts,col in[(fp[f],'#17456f'),(cp[f],'#d17920')]:
                    p=[(float(pts[k,0]-roi[0]),float(pts[k,1]-roi[1]))for k in ids]
                    draw.line([p[0],p[1]],fill=col,width=4)
                    for (px,py),label in zip(p,['L elbow','L wrist','R wrist']):
                        draw.ellipse((px-5,py-5,px+5,py+5),fill=col)
                im.thumbnail((460,270));tile=Image.new('RGB',(480,305),'white');tile.paste(im,((480-im.width)//2,32));ImageDraw.Draw(tile).text((10,6),f'range{rg+1} frame{f}',font=font,fill='#14243a');tiles.append(tile)
        rgbcap.release();cols=4;nr=(len(tiles)+cols-1)//cols
        sheet=Image.new('RGB',(cols*480,50+nr*305),'#e4e9ee');dd=ImageDraw.Draw(sheet);dd.text((12,8),case['id']+' | blue: full; orange: crop; dots are estimates',font=font,fill='#14243a')
        for k,im in enumerate(tiles):sheet.paste(im,((k%cols)*480,50+(k//cols)*305))
        dest=R/(case['id']+'_coordinate_zoom.jpg');sheet.save(dest,quality=94);sheets.append(dict(id=case['id'],path=str(dest),sha256=sha(dest),roi=roi,ranges=case['ranges']))
    write(R/'MATCHED_COORDINATES.json',dict(at=stamp(),rows=out,sheets=sheets,pins=pin,model_inference=False,scope='Same original source/frame comparison, existing full and crop detector caches only. Overlay estimates do not create landmark truth.'))
    for r in out:print(r['case_id'],r['span'],r['view'],r['measurement'].get('delta'),r['measurement'].get('anchor_total_rotation_degrees'),flush=True)

if __name__=='__main__':main()
