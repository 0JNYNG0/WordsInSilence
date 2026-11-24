from common import *
import zipfile,re,collections
assert not (F/'PUBLIC_ARCHIVE_INVENTORY.json').exists()
base=Path('W:/WordsInSilence/data/AIHub_103/videos');archives=sorted(base.rglob('*real_word_video.zip'));rows=[];pattern=re.compile(r'NIA_SL_(WORD0739|WORD1133|WORD1567|WORD2523)_REAL(\d+)_F\.mp4$')
for index,p in enumerate(archives):
 with zipfile.ZipFile(p) as z:
  for i in z.infolist():
   m=pattern.search(i.filename)
   if m:rows.append(dict(archive=str(p),member=i.filename,clip_id=Path(i.filename).stem,word=m[1],person=int(m[2]),bytes=i.file_size,crc=i.CRC))
 if index%5==0:event('archive_metadata',done=index+1,total=len(archives))
write(F/'PUBLIC_ARCHIVE_INVENTORY.json',dict(archives=[str(p) for p in archives],rows=rows,read_only=True,no_video_inference=True))
print('INVENTORY',len(rows),'IDS',dict(collections.Counter(r['person'] for r in rows)))
