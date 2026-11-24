from common_split import *
import zipfile,collections
assert not(U/'FROZEN.json').exists()
targets={r['clip_id']:r for r in read(U/'INVENTORY.json')['candidates']};assert all(not r['seen_fine']for r in targets.values());found={};archives=sorted(Path('W:/WordsInSilence/data/AIHub_103/videos').rglob('*.zip'));assert archives
for ap in archives:
 with zipfile.ZipFile(ap)as z:
  for info in z.infolist():
   cid=Path(info.filename).stem
   if cid in targets:
    assert cid not in found;found[cid]=dict(archive=str(ap),member=info.filename,crc32=f'{info.CRC:08x}',size=info.file_size)
assert set(found)==set(targets)
(U/'media').mkdir(exist_ok=True);rows=[]
for cid,meta in targets.items():
 f=found[cid]
 with zipfile.ZipFile(f['archive'])as z:b=z.read(f['member'])
 path=U/'media'/f'{cid}.mp4';path.write_bytes(b)
 rows.append(meta|dict(path=str(path),sha256=sha(path),role='diagnostic'if meta['person']==10 else'confirmation_reserved',source_archive=f))
write(U/'FROZEN.json',dict(sources=rows,model_pins=read(Q/'FROZEN.json')['model_pins'],selection='Fixed6words x diagnostic10/reserve16; no finger model output used. Source RGB labels include quality positive support check BEFORE candidate training.'))
print('public originals copied',len(rows),'bytes',sum(r['source_archive']['size']for r in rows))
