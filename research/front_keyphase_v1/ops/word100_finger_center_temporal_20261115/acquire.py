from common import *
import zipfile,cv2
assert not(U/'SOURCES.json').exists()
targets=[]
for w in ['WORD2551','WORD2748']:
 for p in [2,5,7]:targets.append(dict(word=w,person=p,role='fit',hand='R'))
for w,people in [('WORD2551',[10,16]),('WORD2748',[11,12]),('WORD1290',[14,15])]:
 for role,p in zip(['diagnostic','confirmation'],people):targets.append(dict(word=w,person=p,role=role,hand='R'))
for r in targets:r['clip_id']=f'NIA_SL_{r["word"]}_REAL{r["person"]:02d}_F'
unused=read(U/'EXPOSURE_AUDIT.json')['unused_clip_combinations']
assert all(r['role']=='fit'or r['person']in unused[r['word']]for r in targets)
write(U/'SELECTED.json',dict(rows=targets,selection='Pre-prediction fixed: two varying-posture words plus both-hand parallel-open control; all eligible listed performers fixed, no best-result selection',reserve_policy='Seal model before confirmation predictions; original RGB labels audited before training'))
needed={r['clip_id'][:-1]+v for r in targets for v in (['F','L','R']if r['word']!='WORD1290'else['F'])};found={}
for ap in sorted(Path('W:/WordsInSilence/data/AIHub_103/videos').rglob('*.zip')):
 with zipfile.ZipFile(ap)as z:
  for i in z.infolist():
   cid=Path(i.filename).stem
   if cid in needed:found[cid]=(ap,i.filename,i.CRC)
assert set(found)==needed,needed-set(found)
(U/'media').mkdir();rows=[]
for r in targets:
 views={}
 for v in (['F','L','R']if r['word']!='WORD1290'else['F']):
  cid=r['clip_id'][:-1]+v;ap,member,crc=found[cid]
  with zipfile.ZipFile(ap)as z:b=z.read(member)
  p=U/'media'/f'{cid}.mp4';p.write_bytes(b);cap=cv2.VideoCapture(str(p));frames=int(cap.get(7));fps=cap.get(5);cap.release();views[v]=dict(path=str(p),sha256=sha(p),archive=str(ap),member=member,crc=crc,frames=frames,fps=fps)
 rec=J/'records'/f'{r["clip_id"]}.json';j=read(rec)if rec.exists()else{}
 rows.append(r|dict(path=views['F']['path'],views=views,raw_path=j.get('raw_path'),channel_path=j.get('channel_path')))
write(U/'SOURCES.json',dict(rows=rows,multiview='Public same-take view identities and neighboring actions checked; equal fps/index not calibrated synchronization. Runtime remains frontal-only.'))
print([(r['clip_id'],r['role'],r['views']['F']['frames'],bool(r['raw_path']))for r in rows])
