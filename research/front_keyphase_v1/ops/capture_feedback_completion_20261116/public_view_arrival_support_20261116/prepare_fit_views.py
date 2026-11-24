from arrival_candidate import *
import importlib.util

def main():
 s=importlib.util.spec_from_file_location('extract_public_camera',D/'prepare_confirmation.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.D=K
 for person in[2,5]:
  name=f'fit_{person:02d}'
  m.main(person,name,'arrival_candidate.py')
  out=K/name;q=read(out/'FROZEN.json');q.update(purpose='Public FIT person camera augmentation, not holdout performance.',training=True,split='FIT; person2/5 already in original nine FIT people');write(out/'FROZEN.json',q)
  q=read(out/'INPUTS.json')
  for r in q['rows']:r.update(training=True,candidate_selection=True,role='fit',person=person)
  write(out/'INPUTS.json',q)
if __name__=='__main__':main()
