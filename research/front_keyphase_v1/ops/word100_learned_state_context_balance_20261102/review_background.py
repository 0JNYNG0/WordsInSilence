"""Sparse source pixels, no model predictions or detector-derived rest labels."""
from pathlib import Path
import sys,av
from PIL import Image,ImageDraw,ImageFont
T=Path(__file__).resolve().parent;O=T.parent;P=O/'word100_learned_guide_feedback_20261102'
sys.path.insert(0,str(O/'word100_learned_motion_invariance_20261102'))
from prepare_real_fit import read,sha,exclusive,clock

def main():
    assert not (T/'BACKGROUND_REVIEW.json').exists()
    rows=read(P/'INPUT_MANIFEST.json')['rows'];assert len(rows)==48
    images={};records=[];sheets=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    for r in rows:
        assert r['person'] in list(range(1,10))+[11,12,13] and sha(r['source'])==r['source_sha256']
        ids=[0,3,r['frames']-4,r['frames']-1];sampled={};saved=[]
        with av.open(r['source']) as c:
            for i,f in enumerate(c.decode(video=0)):
                if i in ids:
                    im=f.to_image();im.thumbnail((1280,720));p=T/'background_frames'/r['clip_id']/f'f{i:04d}.jpg';p.parent.mkdir(parents=True,exist_ok=True);im.save(p,quality=92)
                    small=im.copy();small.thumbnail((510,290));sampled[i]=small;saved.append(dict(frame=i,path=str(p),sha256=sha(p)))
            assert i+1==r['frames']
        images[r['clip_id']]=sampled;records.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],role=r['role'],source=r['source'],source_sha256=r['source_sha256'],samples=saved))
    for word in sorted({r['word_id'] for r in rows}):
        for people in [[1,2,3],[4,5,6],[7,8,9],[11,12,13]]:
            rr=[r for r in records if r['word_id']==word and r['person'] in people];out=Image.new('RGB',(2048,1010),'#edf2f4');draw=ImageDraw.Draw(out)
            draw.text((8,8),f'{word} IDs{people} | full original scene | no candidate prediction',font=font,fill='black')
            for j,r in enumerate(rr):
                for k,z in enumerate(r['samples']):
                    x=k*512;y=38+j*324;draw.text((x+6,y+2),f"ID{r['person']:02} f{z['frame']}",font=font,fill='black');im=images[r['clip_id']][z['frame']];out.paste(im,(x+(512-im.width)//2,y+30))
            dest=T/'background_sheets'/f'{word}_{people[0]:02}_{people[-1]:02}.jpg';dest.parent.mkdir(exist_ok=True);out.save(dest,quality=93);sheets.append(dict(path=str(dest),sha256=sha(dest),word=word,people=people))
    exclusive(T/'BACKGROUND_REVIEW.json',dict(at=clock(),rows=records,sheets=sheets,labels_assigned=False,source_pixels_only=True,independent_expert=False))
    print('Prepared48sourceclips192sparseframes16sheets',flush=True)

if __name__=='__main__':main()
