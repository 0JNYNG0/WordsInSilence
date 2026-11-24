"""Finite saved-export verification; never reextracts existing images."""
from pathlib import Path
from collections import Counter
import json,hashlib
from PIL import Image
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=read(U/'INDEX.json');pins=read(U/'OUTPUT_PINS.json');new=[r for r in d['rows'] if r['is_new']];old=[r for r in d['rows'] if not r['is_new']]
assert len(new)==24 and len(old)==12 and len(pins)==432
assert len({r['clip_id'] for r in d['rows']})==36 and len({r['source_sha256'] for r in d['rows']})==36
for p,h in pins.items():
    assert sha(p)==h
    with Image.open(p) as im:im.verify()
for row in new:
    frames=row['frames'];assert len(frames)==16 and len({f['frame_id'] for f in frames})==16
    assert all(a['pts_us']<b['pts_us'] for a,b in zip(frames,frames[1:]))
    assert row['display_only_crop_xyxy'][1]==0
    for f in frames:
        assert abs(f['pts_us']-f['observation_timestamp_us'])<=1
        assert abs(f['pts_us']-f['target_us'])<=20000, 'Nearest frame selection discrepancy >20ms'
        with Image.open(f['file']) as im:assert list(im.size)==row['full_frame_size']
before=read(U/'SOURCE_PINS_BEFORE.json');after=read(U/'SOURCE_PINS_AFTER.json');assert before==after
for p,h in before.items():assert sha(p)==h
for p,stat in read(U/'ARCHIVE_STAT_PINS.json').items():
    st=Path(p).stat();assert [st.st_size,st.st_mtime_ns]==stat
out={'status':'PASS','new_clips':24,'old_reused':12,'old_reextracted':0,'new_images':432,'source_pins':len(before),
     'image_bytes':sum(Path(p).stat().st_size for p in pins),'new_video_copies':len(list((U/'videos').glob('*.mp4'))) if (U/'videos').exists() else 0,
     'new_video_copy_bytes':sum(p.stat().st_size for p in (U/'videos').glob('*.mp4')) if (U/'videos').exists() else 0,
     'checks':['36 distinct clip and content IDs','new432 PNG hash/decode verified','24 clips each16 unique monotonic frame times','native frame dimensions retained','source actual PTS matches observation within1us','nearest time target within20ms','all sources unchanged','archives read-only stats unchanged'],
     'visual_spotchecks':['X01/detail_contact.png'],'visual_notes':'Full head, active hands and time labels visible; not a sign meaning/correctness verification.',
     'same_agent_separate_arithmetic':True,'independent_expert_truth':False}
(U/'VERIFY_SAVED.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))
