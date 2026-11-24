"""Verify saved pixels and neutral mappings; add requested image_key schema alias."""
from pathlib import Path
import hashlib,json,random
from PIL import Image
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
indexp=U/'INDEX.json';before_index=sha(indexp);d=read(indexp)
needs_alias=any('image_key' not in frame for case in d['rows'] for frame in case['frames'])
for case in d['rows']:
    for frame in case['frames']:frame['image_key']=frame['neutral_image_id']
if needs_alias:
    indexp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (U/'SCHEMA_ALIAS_UPDATE.json').write_text(json.dumps({'purpose':'Parent requested image_key for neutral_image_id; no frame/label/time or source change','before_sha256':before_index,'after_sha256':sha(indexp),'fields_added':234},indent=2)+'\n',encoding='utf-8')
assert len(d['rows'])==18
pins=read(U/'OUTPUT_PINS.json');assert len(pins)==540
for p,h in pins.items():
    assert sha(p)==h
    with Image.open(p) as image:image.verify()
for case in d['rows']:
    frames=case['frames'];assert len(frames)==13
    assert [x['offset_frames'] for x in frames]==list(range(-6,7))
    assert len({x['image_key'] for x in frames})==13 and {x['image_key'] for x in frames}=={f'K{i:02d}' for i in range(1,14)}
    assert [x['frame_id'] for x in frames]==list(range(case['center_frame_id']-6,case['center_frame_id']+7))
    assert all(a['actual_pts_us']<b['actual_pts_us'] for a,b in zip(frames,frames[1:]))
    for frame in frames:
        assert abs(frame['actual_pts_us']-frame['observation_timestamp_us'])<=1
        with Image.open(frame['full_file']) as image:assert hashlib.sha256(np.asarray(image).tobytes()).hexdigest()==frame['rgb_sha256']
    rng=random.Random(int(case['seed_sha256'][:16],16));codes=list(range(1,14));rng.shuffle(codes);order=list(range(13));rng.shuffle(order)
    assert case['ordered_image_ids']==[f'K{i:02d}' for i in codes]
    assert case['shuffled_image_ids']==[f'K{codes[i]:02d}' for i in order]
    center=next(f for f in frames if f['offset_frames']==0)
    assert case['center_image_id']==center['image_key'] and sha(case['center_only'])==sha(center['focus_file'])
before=read(U/'SOURCE_PINS_BEFORE.json');after=read(U/'SOURCE_PINS_AFTER.json');assert before==after
for p,h in before.items():assert sha(p)==h
result={'status':'PASS','windows':18,'full_png':234,'focus_png':234,'all_images':540,'pins_unchanged':len(before),
 'unique_clip_frame_pairs':len({(c['clip_id'],f['frame_id']) for c in d['rows'] for f in c['frames']}),
 'image_bytes':sum(Path(p).stat().st_size for p in pins),'neutral_order_reproduced_from_seed':True,'center_only_exact_focus_copy':True,
 'all_frame_offsets_exact':True,'png_rgb_hashes_match':True,'image_key_alias_added':needs_alias,
 'visual_spotcheck':['T01/shuffled_contact.png'],'visual_check_scope':'Hand/arm detail and neutral K labels legible; no sign/feature truth judgment.',
 'same_agent_saved_artifact_verification':True,'independent_human_ground_truth':False}
(U/'VERIFY_SAVED.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
