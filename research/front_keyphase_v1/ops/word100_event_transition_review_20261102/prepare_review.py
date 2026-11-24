"""Public FIT-only native RGB review; no predictions or new labels."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib
import av
from PIL import Image, ImageDraw

U=Path(__file__).resolve().parent; O=U.parent
D=O/'word100_action_episode_supervision_20261102'; E=O/'word100_guide_conditioned_episode_20261102'
OUT=Path('W:/WordsInSilence_derived/word100_event_transition_review_20261102')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def main():
    assert not (U/'REVIEW_INPUTS.json').exists()
    old={r['clip_id']:r for r in read(E/'CONTEXT_TARGETS.json')['rows']}
    # Select by recorded ID and source label coverage, never by DEV prediction.
    rows=[r for r in read(D/'ACTION_TARGETS.json')['rows'] if r['person'] in [2,4,8]]
    assert len(rows)==12
    save(U/'FROZEN.json',dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        input_scope='12 public FIT clips, recorded IDs02,04,08; all4supported words. No DEV/H/private data.',
        selection='Existing label gaps at entry, between distinct family witnesses, exit. No model outputs displayed.',
        purpose='Inspect actual unlabeled neighboring native frames for additional physical-event/process supervision. Do not infer boundaries from nearest existing label.',
        annotation_policy='Only directly visually checked points may be added. Unknown remains unknown. No automatic copying/interpolation or linguistic correctness labels.',
        original_source_and_labels_unchanged=True, new_training=0, whole_goal_complete=False,
        pins={str(p):sha(p) for p in [D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',U/'prepare_review.py']}))
    results=[]
    for r in rows:
        source=Path(r['source']);assert sha(source)==r['source_sha256']
        labels={p['frame']:p['action_family'] for p in old[r['clip_id']]['points'] if p['mask']}
        main=sorted((p['frame'],p['action_family']) for p in r['points'] if p['mask'] and p['action_family']<8)
        first,last=main[0][0],main[-1][0]
        pre=max((f for f,y in labels.items() if f<first and y>=8),default=max(0,first-8))
        post=min((f for f,y in labels.items() if f>last and y>=8),default=min(r['frames']-1,last+8))
        windows=[dict(name='entry',lo=max(pre,first-12),hi=min(first+2,r['frames']-1))]
        for (a,ga),(b,gb) in zip(main,main[1:]):
            if ga!=gb:windows.append(dict(name=f'family{ga}_to_{gb}',lo=max(0,a-2),hi=min(b+2,r['frames']-1)))
        windows.append(dict(name='exit',lo=max(0,last-2),hi=min(post,last+12)))
        wanted={i for w in windows for i in range(w['lo'],w['hi']+1)};frames={};times={}
        with av.open(str(source)) as c:
            origin=None
            for i,f in enumerate(c.decode(video=0)):
                if origin is None:origin=float(f.time)
                if i in wanted:frames[i]=Image.fromarray(f.to_ndarray(format='rgb24'));times[i]=float(f.time)-origin
        assert set(frames)==wanted
        panels=[]
        for w in windows:
            ids=list(range(w['lo'],w['hi']+1))
            for page in range((len(ids)+11)//12):
                selected=ids[page*12:(page+1)*12];canvas=Image.new('RGB',(1440,((len(selected)+3)//4)*360),'white');draw=ImageDraw.Draw(canvas)
                for j,i in enumerate(selected):
                    im=frames[i];width,height=im.size
                    # Fixed central display enlargement, independent of detections.
                    # Original frame remains available; no inference uses this crop.
                    box=(int(width*.23),0,int(width*.77),height)
                    shown=im.crop(box);shown.thumbnail((356,330))
                    x=(j%4)*360;y=(j//4)*360;canvas.paste(shown,(x,y+27))
                    lab=labels.get(i,-1);name='UNLABELLED' if lab<0 else 'REST' if lab==8 else 'CONTEXT' if lab==9 else f'ACTION{lab}'
                    draw.text((x+4,y+4),f'ID{r["person"]:02} f{i} {times[i]:.3f}s {name}',fill='black')
                p=OUT/r['clip_id']/f'{w["name"]}_{page}.jpg';p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();canvas.save(p,quality=94)
                panels.append(dict(name=w['name'],image=str(p),sha256=sha(p),frames=selected,times=[times[i] for i in selected],
                                   existing_labels=[labels.get(i,-1) for i in selected],display_crop_fraction=[.23,0,.77,1],
                                   full_original_source=str(source)))
        results.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],source=str(source),source_sha256=r['source_sha256'],panels=panels))
    save(U/'REVIEW_INPUTS.json',dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),rows=results,
         clips=len(results),panels=sum(len(r['panels']) for r in results),original_frame_indices=True,
         sampling='Every native frame within the declared window; adjacent panel cells are adjacent frames.',
         new_labels=0,new_training=0,whole_goal_complete=False))
    print('Prepared',len(results),'clips;',sum(len(r['panels']) for r in results),'native-frame panels')

if __name__=='__main__':main()
