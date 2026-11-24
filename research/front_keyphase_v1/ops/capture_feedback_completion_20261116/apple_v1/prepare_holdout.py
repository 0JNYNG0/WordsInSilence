from apple_io import *
import zipfile,importlib.util
def main():
 dest=U/'holdout';dest.mkdir(exist_ok=True);(dest/'media').mkdir(exist_ok=True);assert not(dest/'FROZEN.json').exists()
 inv=read(O/'demo_scope_portability_review_20261113/INVENTORY.json')['targets']['WORD0099'];sources=[]
 for person in[16,17]:
  r=next(x for x in inv if x['record_id']==person);p=dest/'media'/f'{r["clip_id"]}.mp4'
  with zipfile.ZipFile(r['archive'])as z:
   info=z.getinfo(r['member']);assert f'{info.CRC:08x}'==r['crc32'];data=z.read(r['member'])
  assert len(data)==r['bytes'];p.write_bytes(data)
  sources.append(dict(clip_id=r['clip_id'],word='WORD0099',person=person,role='candidate_holdout',path=str(p),sha256=sha(p),archive=r['archive'],member=r['member'],source_interval=r['metadata']['intervals'][0]))
 write(dest/'FROZEN.json',dict(sources=sources,model_pins=read(O/'word100_public_g4_20261114/FROZEN.json')['model_pins'],candidate={str(p):sha(p)for p in list(U.glob('*.py'))+list((U/'runs').glob('*.pt'))},
  gate='Inspect native RGB first. No confirmed wrong claim on full normal; no completion for before-descent, constant, absent or reversed-order controls. Report abstentions.',
  scope='These two public people excluded from this candidate FIT/DEV selection. No project-wide never-used claim. No camera or personal data.'))
 sp=importlib.util.spec_from_file_location('apple_held_extract',O/'word100_public_g4_20261114/prepare.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.U=dest;m.main()
if __name__=='__main__':main()
