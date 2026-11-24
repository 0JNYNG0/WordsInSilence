"""Read only cached public descriptions and source-selected FIT RGB, never poses."""
from pathlib import Path
import json,hashlib,re,zipfile,datetime
import cv2
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent.parent;D=U/'narrow_search'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not (D/'SEARCH_AND_FRAMES.json').exists();D.mkdir(exist_ok=True)
    audit=O/'word100_linguistic_feedback_pipeline_20261027/source_audit'
    entries=read(audit/'SOURCE_ENTRY_DETAILS.json')['entries']
    ready=read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']
    words={r['word_id'] for r in ready if r['role']=='fit'}
    strict=re.compile(r'손끝.{0,10}모[으은아]|손가락.{0,10}모[으은아]|오므|엄지.{0,30}맞')
    broader=re.compile(r'모[으은아았]|오므|맞대|붙|[15]지')
    hits=[]
    for r in entries:
        text=r.get('description_text') or ''
        if r['query_word_id'] not in words or not broader.search(text):continue
        decision='not_a_complete_finger_bundle_narrow_candidate'
        if r['query_word_id']=='WORD2523' and strict.search(text):decision='existing_Egypt_draft_not_new_expression'
        elif r['query_word_id']=='WORD2551' and strict.search(text):decision='new_record_RGB_review_candidate'
        elif r['query_word_id'] in ['WORD2309','WORD0949']:decision='index_thumb_contact_not_same_four_finger_bundle_construct'
        hits.append(dict(word_id=r['query_word_id'],label=r['label'],entry_id=r['entry_id'],description=text,url=r['url'],strict_hit=bool(strict.search(text)),variant_verified=r['variant_verified'],decision=decision))
    sources={r['clip_id']:r for r in read(O/'word100_rtmw_joint_learning_20261108/SOURCES.json') if r['role']=='fit'}
    choices={4:[50,58,66,74,82,90,98,106,114,122],7:[24,30,36,42,48,54,60,66,72,78]}
    frames=[];videos=[]
    for person,fids in choices.items():
        cid=f'NIA_SL_WORD2551_REAL{person:02d}_F';s=sources[cid]
        assert s['role']=='fit' and s['person']==person
        src=O/'word100_stage_feedback_cycle_20261031/videos/P078.mp4' if person==4 else D/(cid+'.mp4')
        if not src.exists():
            # Only one selected public member is read; no archive-wide extraction.
            with zipfile.ZipFile(s['archive']) as z:
                raw=z.read(s['member'])
            assert hashlib.sha256(raw).hexdigest()==s['source_sha256'];src.write_bytes(raw)
        assert sha(src)==s['source_sha256']
        cap=cv2.VideoCapture(str(src));assert cap.isOpened()
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        sheet=Image.new('RGB',(1400,5*355),'white');draw=ImageDraw.Draw(sheet)
        for n,f in enumerate(fids):
            assert f<count;cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
            im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));assert im.size==(1920,1080)
            p=D/f'{cid}_f{f:04d}.png';im.save(p)
            box=(580,0,1280,700);crop=im.crop(box);cp=D/f'{cid}_f{f:04d}_crop.png';crop.save(cp)
            # Sheet reduces images for navigation; full native crop remains available.
            thumb=crop.resize((350,350));x=n%2*700;y=n//2*355;sheet.paste(thumb,(x,y));draw.text((x+355,y+10),f'REAL{person:02d} frame {f}\n{f/fps:.3f}s\nNative crop separate',fill='black')
            frames.append(dict(clip_id=cid,word_id='WORD2551',record_id=person,role='fit',frame_id=f,seconds=f/fps,
                source=str(src),source_sha256=s['source_sha256'],rgb=str(p),rgb_sha256=sha(p),crop=str(cp),crop_sha256=sha(cp),crop_xyxy=list(box)))
        cap.release();sp=D/f'REAL{person:02d}_overview.png';sheet.save(sp)
        videos.append(dict(clip_id=cid,source=str(src),sha256=s['source_sha256'],frame_count=count,fps=fps,overview=str(sp)))
    rawfiles=[p for p in (audit/'public_raw').iterdir() if p.suffix in ['.html','.json']]
    rawhits=[dict(path=str(p),sha256=sha(p)) for p in rawfiles if strict.search(p.read_text(encoding='utf-8-sig',errors='replace'))]
    data=dict(status='SOURCE_DIRECTED_RGB_SEARCH_NO_COORDINATES',at=datetime.datetime.now().astimezone().isoformat(),
        current_fit_words=len(words),cache=dict(detail_entries=len(entries),detail_query_ids=len({r['query_word_id'] for r in entries}),
          current100_entries_with_description=sum(r['query_word_id'] in words and bool(r.get('description_text')) for r in entries),
          current100_words_with_description=len({r['query_word_id'] for r in entries if r['query_word_id'] in words and r.get('description_text')}),
          raw_cache_files_scanned=len(rawfiles),strict_raw_hits=rawhits),
        source_pins=[dict(path=str(audit/'SOURCE_ENTRY_DETAILS.json'),sha256=sha(audit/'SOURCE_ENTRY_DETAILS.json'))],candidates=hits,
        selection='The only newly reviewable complete-finger closing description beyond already-audited Egypt is WORD2551. Inspect new FIT records04/07; do not force missing second expression.',
        selected_expression_count=1,selected_record_ids=[4,7],videos=videos,frames=frames,
        private_access=False,dev_loaded=False,pose_arrays_loaded=False,model_scores_read=False,training=False,new_download=False,
        errors_preserved=[dict(error='TypeError when regex received description_text=null',resolution='Keep missing descriptions as missing; use empty string for search only.'),
          dict(error='Scoped filename listing encountered unrelated inaccessible stgcn temporary folder',resolution='Did not retry or bypass; further content search restricted to named public caches.'),
          dict(error='Guessed FROZEN/FRAMES metadata filenames were absent',resolution='Used actual source ledger SOURCES.json and prepared P078.json; absent files not invented.')],
        limitations=['Source variant match remains unverified until RGB inspected.','Cached descriptions do not cover all 100 expressions. No absence-of-feature conclusion about all videos.','This is not independent human/KSL expert annotation.'])
    write(D/'SEARCH_AND_FRAMES.json',data);print(json.dumps(dict(cache=data['cache'],videos=videos),ensure_ascii=False))

if __name__=='__main__':main()
