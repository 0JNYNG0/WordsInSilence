"""Public FIT-only entry/exit supervision and contract audit; no DEV fitting."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import json,hashlib
import numpy as np,av
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent;L=O/'word100_seated_fit_transfer_20261102'
D=O/'word100_action_episode_supervision_20261102';E=O/'word100_guide_conditioned_episode_20261102'
DEST=Path('W:/WordsInSilence_derived/word100_action_entry_contract_20261102')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clock():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def save(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    assert not (U/'AUDIT.json').exists()
    files=[D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',L/'CROP_MANIFEST.json',L/'CAUSE_AUDIT.json',U/'audit_entries.py']
    save(U/'FROZEN.json',dict(at=clock(),pins={str(p):sha(p) for p in files},fit_ids=list(range(1,10)),visual_ids=[1,5,9],
        purpose='Check what sparse entry/exit labels actually support. First labelled action is a witness, not an exact onset boundary. Source-only audit before any further candidate training.',
        new_training=0,previous_predictions_rescored=False,old_targets_unchanged=True,whole_goal_complete=False))
    sources={r['clip_id']:r for r in read(L/'CROP_MANIFEST.json')['rows']};ctx={r['clip_id']:r for r in read(E/'CONTEXT_TARGETS.json')['rows']};rows=[];cards={};visual=[]
    for r in read(D/'ACTION_TARGETS.json')['rows']:
        assert r['person'] in range(1,10)
        lookup={p['frame']:p['action_family'] for p in ctx[r['clip_id']]['points'] if p['mask']}
        main=[p for p in r['points'] if p['mask'] and p['action_family']<8];first=min(p['frame'] for p in main);last=max(p['frame'] for p in main)
        assert all(lookup[p['frame']]==p['action_family'] for p in main)
        before=[f for f,v in lookup.items() if f<first and v>=8];after=[f for f,v in lookup.items() if f>last and v>=8]
        a=max(before) if before else None;b=min(after) if after else None
        gaps=dict(before_main_unlabelled_frames=sum(i not in lookup for i in range((a or 0)+1,first)),after_main_unlabelled_frames=sum(i not in lookup for i in range(last+1,b or r['frames'])))
        rows.append(dict(clip_id=r['clip_id'],person=r['person'],word=r['word_id'],first_main_witness=first,last_main_witness=last,last_pre_context=a,next_post_context=b,
            main_points=len(main),near_entry_labels={str(f):lookup.get(f,-1) for f in range(max(0,first-4),min(first+5,r['frames']))},**gaps))
        if r['person'] not in [1,5,9]:continue
        meta=sources[r['clip_id']];assert sha(meta['video'])==meta['video_sha256'];picks={}
        for side,anchor in [('entry',first),('exit',last)]:picks[side]=[max(0,min(r['frames']-1,anchor+d)) for d in [-4,-2,-1,0,1,2,4]]
        wanted=set(picks['entry']+picks['exit']);frames={}
        with av.open(meta['video']) as c:
            for i,f in enumerate(c.decode(video=0)):
                if i in wanted:frames[i]=Image.fromarray(f.to_ndarray(format='rgb24'))
        for side,ids in picks.items():
            key=(r['word_id'],side);canvas=cards.setdefault(key,Image.new('RGB',(1680,720),'white'));draw=ImageDraw.Draw(canvas);y=[1,5,9].index(r['person'])*240
            for j,f in enumerate(ids):
                im=frames[f].copy();im.thumbnail((238,206));x=j*240;canvas.paste(im,(x,y+30))
                label=lookup.get(f,-1);txt='UNLABELLED' if label==-1 else 'ACTION'+str(label) if label<8 else 'REST' if label==8 else 'CONTEXT'
                draw.text((x+3,y+3),f"ID{r['person']:02} frame{f} {txt}",fill='black')
            visual.append(dict(clip_id=r['clip_id'],side=side,frames=ids,labels=[lookup.get(f,-1) for f in ids]))
    DEST.mkdir(parents=True,exist_ok=True)
    paths=[]
    for (word,side),canvas in cards.items():
        p=DEST/f'{word}_{side}.jpg';assert not p.exists();canvas.save(p,quality=91);paths.append(dict(word=word,side=side,image=str(p),sha256=sha(p)))
    save(U/'AUDIT.json',dict(at=clock(),fit_clips=36,rows=rows,visual_windows=visual,cards=paths,main_labels_preserved=sum(r['main_points'] for r in rows),
        no_main_context_label_conflicts=True,exact_dense_boundary_truth_available=False,unlabelled_frames_not_context=True,
        scope='Sparse labels constrain observed configurations and a few process roles. No guarantee that first scored action point equals actual first meaningful frame. Entry and exit can share shape; this does not make either shape observation false.',
        same_codex_review_not_independent_expert=True,new_training=0,whole_goal_complete=False))
    print('36labels audited; main',sum(r['main_points'] for r in rows),'entry gaps',dict(Counter(r['before_main_unlabelled_frames'] for r in rows)),'cards',len(paths))
if __name__=='__main__':main()
