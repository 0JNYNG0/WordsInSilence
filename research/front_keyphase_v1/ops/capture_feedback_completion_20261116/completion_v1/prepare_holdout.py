from train_shape import *
import zipfile,importlib.util
def main():
 dest=V/'holdout';dest.mkdir(exist_ok=True);(dest/'media').mkdir(exist_ok=True);assert not(dest/'FROZEN.json').exists()
 inv=read(O/'demo_scope_portability_review_20261113/INVENTORY.json')['targets']['WORD1290']
 r=next(v for v in inv if v['record_id']==10);p=dest/'media'/f'{r["clip_id"]}.mp4'
 with zipfile.ZipFile(r['archive'])as z:
  info=z.getinfo(r['member']);assert f'{info.CRC:08x}'==r['crc32'];data=z.read(r['member'])
 assert len(data)==r['bytes'];p.write_bytes(data)
 row=dict(clip_id=r['clip_id'],word='WORD1290',person=10,role='candidate_holdout',path=str(p),sha256=sha(p),archive=r['archive'],member=r['member'],source_interval=r['metadata']['intervals'][0])
 pins=read(O/'word100_public_g4_20261114/FROZEN.json')['model_pins']
 write(dest/'FROZEN.json',dict(sources=[row],model_pins=pins,
  candidate={str(p):sha(p)for p in[V/'visible_lesson.py',V/'target_region.py',V/'view_conditioned_shape.py']},
  scope='Public person10 absent from this FIT/DEV/model selection. Source truth review before candidate prediction. No project-wide never-used claim.',
  criteria='All six visible normal requirements supported; never treat missing/one arrival/static as lesson complete; physical3D contact not claimed.'))
 spec=importlib.util.spec_from_file_location('completion_holdout_extract',O/'word100_public_g4_20261114/prepare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.U=dest;m.main()
if __name__=='__main__':main()
