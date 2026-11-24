"""Bounded public-RGB diagnostic sheets; no detector/classifier, no tuning."""
from pathlib import Path
import json,zipfile,hashlib
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;B=U.parents[1]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'VISUAL_INDEX.json').exists()
    rows=read(U/'TRANSITIONS.json');chosen=[]
    # Select after comparison strictly for explaining failures, never changing rules.
    for r in sorted(rows,key=lambda r:(r['candidate']['refs']-r['legacy']['refs'],r['candidate']['clip_id'])):
        x=r['candidate']
        if r['type']!='lost' or x['axis']!='handshape' or x['clip_id'] in chosen:continue
        chosen.append(x['clip_id'])
        if len(chosen)==4:break
    for cid in ['NIA_SL_WORD0500_REAL13_F','NIA_SL_WORD1139_REAL12_F']:
        if cid not in chosen:chosen.append(cid)
    ff=read(U/'FROZEN.json');lookup={r['clip_id']:r for r in ff['inputs']};out=[]
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    (U/'visual_sources').mkdir(exist_ok=True);(U/'visual').mkdir(exist_ok=True)
    for cid in chosen:
        row=lookup[cid];record=read(B/'ops/word100_body_boundary_20261017/data/records'/(cid+'.json'));source=record['source'];vp=U/'visual_sources'/(cid+'.mp4')
        with zipfile.ZipFile(source['archive']) as z:
            data=z.read(source['member']);assert hashlib.sha256(data).hexdigest()==source['sha256'];vp.write_bytes(data)
        obs={v:dict(np.load(row['inputs'][v],allow_pickle=False)) for v in ['legacy','masked','sensor']}
        a=obs['legacy'];sm=obs['sensor'];m=obs['masked'];assert np.array_equal(a['frame_id'],sm['frame_id']) and np.array_equal(a['timestamp_us'],sm['timestamp_us'])
        times=a['timestamp_us']/1e6
        # Core interval if present in original public annotation; no word-specific threshold.
        intervals=row.get('intervals',[]);core=np.ones(len(times),bool)
        if intervals:core=np.logical_or.reduce([(times>=i[0])&(times<=i[1]) for i in intervals])
        changed=core&(((a['usable'][:,33:54].all(1)&(~sm['usable'][:,33:54].all(1)|~m['usable'][:,33:54].all(1))))|((a['usable'][:,54:].all(1)&(~sm['usable'][:,54:].all(1)|~m['usable'][:,54:].all(1)))))
        indices=np.flatnonzero(changed)
        if not len(indices):indices=np.flatnonzero(core)
        picks=np.unique(indices[np.linspace(0,len(indices)-1,min(3,len(indices))).round().astype(int)])
        canvas=Image.new('RGB',(1440,100+len(picks)*305),'#f1f5f9');draw=ImageDraw.Draw(canvas)
        draw.text((20,12),row['canonical_label']+' / '+cid,font=font,fill='#0f172a')
        draw.text((20,45),'공개 DEV 진단 표본 · 노란선=상반신 하단 · 초록=L / 분홍=R · 새 검출 없음',font=font,fill='#334155')
        capture=cv2.VideoCapture(str(vp));saved=[]
        for ridx,t in enumerate(picks):
            capture.set(cv2.CAP_PROP_POS_FRAMES,int(a['frame_id'][t]));ok,fr=capture.read();assert ok
            h,w=fr.shape[:2];base=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
            for col,view in enumerate(['source','legacy','masked','sensor']):
                im=base.copy();dd=ImageDraw.Draw(im);bottom=record['calibration']['box_source_pixels'][3];dd.line((0,bottom,w,bottom),fill='#eab308',width=5)
                counts=[]
                if view!='source':
                    ob=obs[view]
                    for side,sl,color in [('L',slice(33,54),'#00ff6a'),('R',slice(54,75),'#ff48ca')]:
                        pts=ob['xy_px'][t,sl];valid=ob['usable'][t,sl];counts.append(f'{side}:{int(valid.sum())}/21')
                        for p in pts[valid]:dd.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill=color)
                im.thumbnail((350,254));x=10+col*360;y=100+ridx*305
                canvas.paste(im,(x,y));draw.text((x,y+258),f'{view} {times[t]:.2f}s '+','.join(counts),font=font,fill='#0f172a')
            saved.append({'frame_id':int(a['frame_id'][t]),'timestamp_us':int(a['timestamp_us'][t]),'complete_counts':{v:[int(obs[v]['usable'][t,33:54].sum()),int(obs[v]['usable'][t,54:].sum())] for v in obs}})
        capture.release();sheet=U/'visual'/(cid+'.jpg');canvas.save(sheet,quality=88)
        out.append({'clip_id':cid,'label':row['canonical_label'],'source_sha256':source['sha256'],'sheet':str(sheet),'frames':saved,'scope':'posthoc observation diagnosis only, no parameter/model/role changes'})
    (U/'VISUAL_INDEX.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8');print(json.dumps({'sheets':len(out),'frames':sum(len(x['frames']) for x in out)},ensure_ascii=False))
if __name__=='__main__':main()
