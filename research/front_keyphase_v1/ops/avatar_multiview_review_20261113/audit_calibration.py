"""Inspect existing public REAL02 keypoint ZIP; three frames x five views only."""
from pathlib import Path
import json,struct,re,zlib,hashlib
HERE=Path(__file__).resolve().parent
PLAN=HERE.parent/'word_keypoint_archive_proposal_20260922/INSPECTION_PLAN.json'
p=json.loads(PLAN.read_text(encoding='utf-8-sig')); arc=Path(p['archive'])
assert arc.stat().st_size==p['archive_bytes']
pattern=re.compile(rb'NIA_SL_WORD1567_REAL02_([FDULR])_(\d+)_keypoints\.json$')
selected=[]; directory_hash=hashlib.sha256()
with arc.open('rb') as f:
    f.seek(p['directory_offset'])
    for _ in range(p['directory_entries']):
        head=f.read(46); assert head[:4]==b'PK\x01\x02'
        nl,el,cl=struct.unpack_from('<3H',head,28); name=f.read(nl); extra=f.read(el); comment=f.read(cl)
        directory_hash.update(head+name+extra+comment)
        m=pattern.search(name)
        if not m or int(m[2]) not in [30,60,90]: continue
        flags,method=struct.unpack_from('<2H',head,8);crc,cs,us=struct.unpack_from('<3L',head,16);off=struct.unpack_from('<L',head,42)[0]
        j=0
        while j+4<=len(extra):
            tag,size=struct.unpack_from('<2H',extra,j);e=extra[j+4:j+4+size];j+=4+size
            if tag==1:
                q=0
                if us==0xffffffff:us=struct.unpack_from('<Q',e,q)[0];q+=8
                if cs==0xffffffff:cs=struct.unpack_from('<Q',e,q)[0];q+=8
                if off==0xffffffff:off=struct.unpack_from('<Q',e,q)[0]
        selected.append(dict(name=name.decode(),view=m[1].decode(),frame=int(m[2]),offset=off,crc=crc,compressed=cs,uncompressed=us,method=method))
    assert f.tell()==p['directory_offset']+p['directory_bytes']
expected=json.loads((PLAN.parent/'INSPECTION_RESULT.json').read_text(encoding='utf-8-sig'))['directory_sha256']
assert directory_hash.hexdigest()==expected
out={'scope':'Existing public REAL02 WORD1567; selected keypoint frame30/60/90 for all5views; not FIT01 video camera parameters','archive':str(arc),'directory_sha256':expected,'samples':[]}
with arc.open('rb') as f:
    for m in selected:
        f.seek(m['offset']);a=struct.unpack('<IHHHHHIIIHH',f.read(30));assert a[0]==0x04034b50
        name=f.read(a[-2]).decode();f.read(a[-1]);assert name==m['name']
        raw=zlib.decompress(f.read(m['compressed']),-15);assert len(raw)==m['uncompressed'] and (zlib.crc32(raw)&0xffffffff)==m['crc']
        payload=json.loads(raw)
        out['samples'].append(dict(member=m['name'],view=m['view'],frame=m['frame'],sha256=hashlib.sha256(raw).hexdigest(),payload=payload))
(HERE/'CALIBRATION_SAMPLES.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for x in out['samples']:
    q=x['payload'];people=q.get('people',[]);person=people[0] if isinstance(people,list) and people else people
    print(x['view'],x['frame'],'rootkeys',list(q.keys()),'peoplekeys',list(person.keys()) if isinstance(person,dict) else type(person),'camparam',json.dumps(q.get('camparam',person.get('camparam') if isinstance(person,dict) else None))[:900])
print('samples',len(out['samples']))
