from prepare import *
import zipfile,importlib.util

def main():
 V=U/'local_shape/fresh';V.mkdir(exist_ok=True);(V/'media').mkdir(exist_ok=True);assert not(V/'FROZEN.json').exists()
 inventory=read(O/'demo_scope_portability_review_20261113/INVENTORY.json')['targets']['WORD1290'];rows=[]
 for person in [16,17]:
  r=next(v for v in inventory if v['record_id']==person);path=V/'media'/f'{r["clip_id"]}.mp4'
  with zipfile.ZipFile(r['archive'])as z:
   info=z.getinfo(r['member']);assert f'{info.CRC:08x}'==r['crc32'];b=z.read(r['member'])
  assert len(b)==r['bytes'];path.write_bytes(b)
  rows.append(dict(clip_id=r['clip_id'],person=person,role='confirmation',path=str(path),sha256=sha(path),archive=r['archive'],member=r['member'],source_interval=r['metadata']['intervals'][0]))
 prior=read(O/'word100_public_g4_20261114/FROZEN.json')
 write(V/'FROZEN.json',dict(sources=rows,model_pins=prior['model_pins'],candidate=sha(U/'local_shape/CONSENSUS_FROZEN.json'),scope='Public people16/17, excluded from current model fit and selection; source labels before candidate inference. No private footage.'))
 spec=importlib.util.spec_from_file_location('basic_fresh_extract',O/'word100_public_g4_20261114/prepare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.U=V;m.main()
 print('FRESH PUBLIC SOURCES READY')
if __name__=='__main__':main()
