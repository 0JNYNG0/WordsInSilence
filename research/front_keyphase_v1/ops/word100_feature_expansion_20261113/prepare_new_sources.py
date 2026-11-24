from common import *
import zipfile, av

def main():
 assert not (U/'SOURCES_NEW.json').exists()
 rows=[json.loads(s) for s in (O/'word_acquisition_inventory_20261011/FRONT_MANIFEST.jsonl').read_text('utf-8-sig').splitlines()]
 codes={'WORD2632':'숫자1','WORD2634':'숫자3','WORD2636':'숫자5'}
 wanted={r['clip_id']:r for r in rows if r['word_code'] in codes and r['filename_signer_id'] in list(range(1,10))+[11,12,13]}
 archives=read(O/'demo_scope_portability_review_20261113/INVENTORY.json')['archives'];found={}
 # Index only ZIP member names; source video payloads limited to approved records.
 for ar in archives:
  with zipfile.ZipFile(ar['path']) as z:
   for name in z.namelist():
    cid=Path(name).stem
    if cid in wanted:
     assert cid not in found,cid;found[cid]=dict(archive=ar['path'],member=name)
 assert len(found)==len(wanted)==36
 out=[];fail=[];media=U/'media';media.mkdir(exist_ok=True)
 for cid,m in sorted(wanted.items()):
  p=media/(cid+'.mp4');src=found[cid]
  if not p.exists():
   with zipfile.ZipFile(src['archive']) as z:p.write_bytes(z.read(src['member']))
  try:
   with av.open(str(p)) as c:assert len(c.streams.video)==1
  except Exception as exc:fail.append(dict(clip_id=cid,error=str(exc),path=str(p),sha256=sha(p)));continue
  row=dict(clip_id=cid,word=m['word_code'],label=codes[m['word_code']],person=m['filename_signer_id'],role='fit' if m['filename_signer_id']<=9 else 'development',source_sha256=sha(p),video=str(p),source_interval=m['intervals'][0],**src)
  out.append(row);write(U/'SOURCE_PROGRESS.json',dict(done=len(out),failed=len(fail),planned=36,last=cid))
 write(U/'SOURCES_NEW.json',dict(rows=out,failures=fail,planned=36,selection_reason='Existing source previews establish index-only/three-extended/thumb-only configurations missing in2/4/6; further per-finger admission requires independent RGB annotation. No DEV predictions used.'))
 event('new_public_sources',ready=len(out),failed=len(fail))
if __name__=='__main__':main()
