"""Public IDs11-13 process labels BEFORE candidate prediction; no model calls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,traceback
import av
from PIL import Image, ImageDraw, ImageFont

U=Path(__file__).resolve().parent; O=U.parent
M=O/'word100_learned_motion_order_20261102'
P=O/'word100_learned_guide_feedback_20261102'
OUT=Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102/development_review')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def now():return datetime.now(timezone.utc).isoformat()

def main():
    assert not (U/'DEV_PROCESS_REVIEW_MANIFEST.json').exists()
    assert not (U/'DEV_PROCESS_REVIEW_INTENT.json').exists(), 'Existing prep must be inspected, not overwritten'
    motion=read(M/'MOTION_TARGETS.json')
    source={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows'] if r['role']=='development'}
    plans=[]
    for r in motion['rows']:
        if r['role']!='development' or r['kind']!='directed':continue
        q=source[r['clip_id']];assert q['person'] in [11,12,13] and r['source_sha256']==q['source_sha256']
        positive=[i for i in range(5) if r['mask'][i] and r['y'][i]>0];assert len(positive)==1
        lo,hi=r['frames']
        plans.append(dict(id=r['id'],clip_id=r['clip_id'],word_id=r['word_id'],person=q['person'],role='development',
            label=q['label'],source=q['source'],source_sha256=q['source_sha256'],source_frames=q['frames'],
            physical_query=motion['targets'][positive[0]],target_index=positive[0],existing_motion_interval=r['frames'],
            display_bounds=[max(0,lo-12),min(q['frames']-1,hi+12)],previous_interval_is_completion_truth=False))
    assert len(plans)==15 and len({r['clip_id'] for r in plans})==12
    pins={str(p):sha(p) for p in [Path(__file__).resolve(),M/'MOTION_TARGETS.json',P/'INPUT_MANIFEST.json',U/'PROCESS_LABEL_PROTOCOL.json']}
    write(U/'DEV_PROCESS_REVIEW_INTENT.json',dict(at=now(),plans=plans,pins=pins,
        purpose='Same physical-process rubric as FIT, prior to new candidate predictions; repeated development, not independent final test.',
        no_new_predictions=True,no_training=True,previous_interval_is_completion_truth=False))
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'frames').mkdir();(OUT/'sheets').mkdir()
    found={}
    for cid in sorted({r['clip_id'] for r in plans}):
        qs=[q for q in plans if q['clip_id']==cid];q=qs[0];assert sha(q['source'])==q['source_sha256']
        wanted={f for q in qs for f in range(q['display_bounds'][0],q['display_bounds'][1]+1)};found[cid]={}
        with av.open(q['source']) as c:
            s=c.streams.video[0];origin=None;previous=None
            for i,f in enumerate(c.decode(video=0)):
                assert f.pts is not None
                if origin is None:origin=f.pts
                us=round(float((f.pts-origin)*s.time_base)*1e6)
                assert previous is None or us>previous;previous=us
                if i not in wanted:continue
                im=f.to_image();im.thumbnail((1280,720));dest=OUT/'frames'/f'{cid}_f{i:04d}.jpg';im.save(dest,quality=93)
                found[cid][i]=dict(frame=i,timestamp_us=us,path=str(dest),sha256=sha(dest))
            assert i+1==q['source_frames']
        assert set(found[cid])==wanted
        print('DEV RGB preparation',cid,len(wanted),flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
    rows=[]
    for q in plans:
        frames=[found[q['clip_id']][f] for f in range(q['display_bounds'][0],q['display_bounds'][1]+1)]
        pages=[]
        for start in range(0,len(frames),25):
            part=frames[start:start+25];cw,ch=300,200;sheet=Image.new('RGB',(1500,65+ch*((len(part)+4)//5)),'#f4f5f7');dr=ImageDraw.Draw(sheet)
            dr.text((10,5),q['id'],font=font,fill='#111827')
            dr.text((10,30),'Every native frame / full image / no predictions / context is not completion truth',font=font,fill='#111827')
            for k,r in enumerate(part):
                im=Image.open(r['path']);im.thumbnail((292,168));x=k%5*cw+4;y=65+k//5*ch
                sheet.paste(im,(x,y+25));dr.text((x,y),f"f{r['frame']} | {r['timestamp_us']/1e6:.3f}s",font=font,fill='#111827')
            dest=OUT/'sheets'/f"{q['id']}_p{start//25+1}.jpg";sheet.save(dest,quality=94)
            pages.append(dict(path=str(dest),sha256=sha(dest),frames=[r['frame'] for r in part]))
        rows.append(dict(**q,frames=frames,pages=pages,reviewed=False))
    for p,h in pins.items():assert sha(p)==h
    write(U/'DEV_PROCESS_REVIEW_MANIFEST.json',dict(at=now(),status='PREPARED_NOT_LABELLED',rows=rows,clips=12,queries=15,
        unique_frames=sum(len(v) for v in found.values()),pages=sum(len(r['pages']) for r in rows),
        intent_sha256=sha(U/'DEV_PROCESS_REVIEW_INTENT.json'),new_inference=0,new_training=0,
        repeated_development=True,independent_final_test=False))
    print('DEV15 review prepared; no new targets or model inference',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'DEV_REVIEW_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
