"""Native-frame public FIT process review. No detectors, model or predictions."""
from pathlib import Path
import json,hashlib,os,traceback
from datetime import datetime,timezone
import av
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent
D=O/'word100_action_episode_supervision_20261102';M=O/'word100_learned_motion_order_20261102'
OUT=Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def now():return datetime.now(timezone.utc).isoformat()
def main():
    assert not (U/'PROCESS_REVIEW_MANIFEST.json').exists()
    assert not (U/'PROCESS_REVIEW_INTENT.json').exists(), 'Inspect any existing preparation before resuming'
    action=read(D/'ACTION_TARGETS.json');meta={r['clip_id']:r for r in action['rows']}
    motion=read(M/'MOTION_TARGETS.json');plans=[]
    for r in motion['rows']:
        if r['role']!='fit' or r['kind']!='directed':continue
        q=meta[r['clip_id']];assert 1<=q['person']<=9 and r['source_sha256']==q['source_sha256']
        positive=[i for i in range(5) if r['mask'][i] and r['y'][i]>0];assert len(positive)==1
        lo,hi=r['frames'];bounds=[max(0,lo-12),min(q['frames']-1,hi+12)]
        plans.append(dict(id=r['id'],clip_id=r['clip_id'],word_id=r['word_id'],person=q['person'],
            label=q['label'],source=q['source'],source_sha256=q['source_sha256'],source_frames=q['frames'],
            physical_query=motion['targets'][positive[0]],target_index=positive[0],existing_motion_interval=r['frames'],
            display_bounds=bounds,frame_selection='Every native frame in previous FIT motion interval plus12-frame context each side; no predictions/DEV used.',
            previous_interval_is_completion_truth=False))
    assert len(plans)==45 and len({r['clip_id'] for r in plans})==36
    sources=[U/'SCOPE.json',D/'ACTION_TARGETS.json',M/'MOTION_TARGETS.json',U/'PROCESS_LABEL_PROTOCOL.json']
    intent=dict(status='FROZEN_FIT_ONLY_REVIEW_SELECTION',at=now(),plans=plans,
        source_pins={str(p):sha(p) for p in sources},reviewer='Codex visual physical evidence; not independent KSL expert',
        previous_root_preview='Three old ID01 sheets (0739,1133,2523) viewed to understand scope; no new labels assigned from them.',
        new_model_inference=0,training=0,predictions_shown=False)
    write(U/'PROCESS_REVIEW_INTENT.json',intent)
    (OUT/'frames').mkdir(parents=True,exist_ok=False);(OUT/'sheets').mkdir()
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
    title=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    records={};clips=sorted({p['clip_id'] for p in plans})
    for ix,cid in enumerate(clips):
        query=[p for p in plans if p['clip_id']==cid];q=query[0];assert sha(q['source'])==q['source_sha256']
        wanted={i for p in query for i in range(p['display_bounds'][0],p['display_bounds'][1]+1)};found={};origin=None;previous=None
        with av.open(q['source']) as c:
            s=c.streams.video[0]
            for i,f in enumerate(c.decode(video=0)):
                assert f.pts is not None
                if origin is None:origin=f.pts
                us=int(round(float((f.pts-origin)*s.time_base)*1e6))
                assert previous is None or us>previous;previous=us
                if i not in wanted:continue
                image=f.to_image();image.thumbnail((1280,720))
                dest=OUT/'frames'/f'{cid}_f{i:04d}.jpg';image.save(dest,quality=93)
                found[i]=dict(frame=i,timestamp_us=us,path=str(dest),sha256=sha(dest))
            assert i+1==q['source_frames']
        assert set(found)==wanted
        records[cid]=found
        write(U/'PROCESS_REVIEW_PROGRESS.json',dict(at=now(),status='EXTRACTING_PUBLIC_RGB_ONLY',pid=os.getpid(),clips_done=ix+1,clips_total=36))
        print('review RGB',ix+1,cid,len(found),flush=True)
    rows=[]
    for p in plans:
        frames=[records[p['clip_id']][i] for i in range(p['display_bounds'][0],p['display_bounds'][1]+1)]
        pages=[];cw,ch=300,225;cols=5;per_page=25
        for j in range(0,len(frames),per_page):
            part=frames[j:j+per_page];sheet=Image.new('RGB',(cols*cw,80+ch*((len(part)+cols-1)//cols)),'#f4f5f7');dr=ImageDraw.Draw(sheet)
            dr.text((10,6),f"{p['label']} | ID{p['person']:02d} | {p['physical_query']} | page{j//per_page+1}",font=title,fill='#111827')
            dr.text((10,37),'Every original frame / no predictions / prior interval is not full-action truth',font=font,fill='#111827')
            for k,r in enumerate(part):
                im=Image.open(r['path']);w,h=im.size
                small=im.crop((int(w*.15),0,int(w*.85),h));small.thumbnail((cw-8,ch-28))
                x=k%cols*cw+4;y=80+k//cols*ch
                sheet.paste(small,(x,y+25));dr.text((x,y),f"f{r['frame']} | {r['timestamp_us']/1e6:.3f}s",font=font,fill='#111827')
            dest=OUT/'sheets'/f"{p['id']}_p{j//per_page+1}.jpg";sheet.save(dest,quality=94)
            pages.append(dict(path=str(dest),sha256=sha(dest),frames=[r['frame'] for r in part]))
        rows.append(dict(**p,frames=frames,pages=pages,reviewed=False))
    manifest=dict(at=now(),status='PREPARED_NOT_VISUALLY_LABELLED',rows=rows,clips=36,queries=45,
        unique_frames=sum(len(v) for v in records.values()),pages=sum(len(r['pages']) for r in rows),
        intent_sha256=sha(U/'PROCESS_REVIEW_INTENT.json'),new_inference=0,new_training=0,whole_goal_complete=False)
    write(U/'PROCESS_REVIEW_MANIFEST.json',manifest)
    write(U/'PROCESS_REVIEW_PROGRESS.json',dict(at=now(),status='PREPARED_NOT_VISUALLY_LABELLED',active=None,clips_done=36,queries=45,pages=manifest['pages']))
    print('Prepared',manifest['unique_frames'],'unique native frames and',manifest['pages'],'pages',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'PREPARATION_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
