from pathlib import Path
import sys,numpy as np
from PIL import Image,ImageDraw,ImageFont
E=Path(__file__).resolve().parent;D=E.parent/'word100_action_episode_supervision_20261102'
sys.path.insert(0,str(D));from audit import read,write,exclusive,sha,clock

def main():
    assert not (E/'CONTEXT_REVIEW_MANIFEST.json').exists()
    target={r['clip_id']:r for r in read(D/'ACTION_TARGETS.json')['rows']};manifest=read(D/'REVIEW_MANIFEST.json');rows=[]
    for r in manifest['rows']:
        tr=target[r['clip_id']];action=[p['frame'] for p in tr['points'] if p['action_family']<8];known={p['frame'] for p in tr['points']};lo,hi=min(action),max(action)
        available={p['frame']:p for p in r['frames']};before=[f for f in sorted(available) if f<lo and f not in known];after=[f for f in sorted(available) if f>hi and f not in known]
        selected=before+[lo,hi]+after
        rows.append(dict(clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],before=before,after=after,anchor_frames=[lo,hi],frames=[available[f] for f in selected]))
    exclusive(E/'CONTEXT_REVIEW_INTENT.json',dict(at=clock(),rows=rows,selection='Previously unassigned displayed frames before/after first/last action witnesses; selection is NOT preparation/return truth.',predictions=False,fit_only=True))
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',13);outputs=[];dest=E/'context_sheets';dest.mkdir()
    for word in sorted({r['word_id'] for r in rows}):
        rr=[r for r in rows if r['word_id']==word]
        for start in [0,3,6]:
            group=rr[start:start+3];entries=[]
            for r in group:
                for p in r['frames']:entries.append((r,p))
            cw,ch=240,190;im=Image.new('RGB',(cw*6,45+ch*((len(entries)+5)//6)),'#f3f4f6');draw=ImageDraw.Draw(im)
            draw.text((8,8),f'{word} FIT{start+1}-{start+3} | 앞뒤 후보와 주요행동 경계 표본 | 예측 없음',font=font,fill='#111827')
            for k,(r,p) in enumerate(entries):
                assert sha(p['path'])==p['sha256'];frame=Image.open(p['path']);w,h=frame.size;small=frame.crop((int(w*.15),0,int(w*.85),h));small.thumbnail((232,160))
                x=(k%6)*cw+4;y=45+(k//6)*ch;im.paste(small,(x,y+25));marker='MAIN anchor' if p['frame'] in r['anchor_frames'] else 'candidate'
                draw.text((x,y),f"ID{r['person']:02} f{p['frame']} {marker}",font=font,fill='#111827')
            path=dest/f'{word}_{start+1}_{start+3}.jpg';im.save(path,quality=94);outputs.append(dict(path=str(path),sha256=sha(path),clips=[r['clip_id'] for r in group]))
    exclusive(E/'CONTEXT_REVIEW_MANIFEST.json',dict(at=clock(),rows=rows,sheets=outputs,reviewed=False));print('Prepared',len(outputs),'context sheets; candidate frames',sum(len(r['before'])+len(r['after']) for r in rows))
if __name__=='__main__':main()
