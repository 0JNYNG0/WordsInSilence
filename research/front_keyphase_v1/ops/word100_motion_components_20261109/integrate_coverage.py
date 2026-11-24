from common import *
from component_coverage import apply_coverage
import shutil,copy

def attach(result,word):
 q=copy.deepcopy(result)
 for rule in read(U/'COVERAGE_CONTRACTS.json')['contracts']:
  if rule['word']!=word:continue
  for card in q['cards']:
   if card['id']==rule['card']:card.update({k:rule[k] for k in ['coverage_requirements','coverage_message']})
 return apply_coverage(q)

def main():
 assert read(U/'PUBLIC_DECISION.json')['coverage']['deploy_research'] and not (U/'INTEGRATION.json').exists()
 for f in ['consumer.py','CARDS.json','EXAMPLES.json']:
  assert not (U/'before'/f).exists();shutil.copyfile(D/f,U/'before'/f)
 for f in ['consumer.py','CARDS.json']:assert sha(D/f)==read(U/'FROZEN.json')['pins'][str(D/f)]
 cards=read(D/'CARDS.json')
 for rule in read(U/'COVERAGE_CONTRACTS.json')['contracts']:
  target=[c for c in cards['words'][rule['word']] if c['id']==rule['card']];assert len(target)==1
  target[0].update({k:rule[k] for k in ['coverage_requirements','coverage_message']})
 write(D/'CARDS.json',cards)
 p=D/'consumer.py';s=p.read_text('utf-8');assert s.count('  e=refine(e,p[0].numpy(),x,ts,fid,phase)')==1
 s=s.replace('from positive_evidence import refine','from positive_evidence import refine\nsys.path.insert(0,str(O/\'word100_motion_components_20261109\'))\nfrom component_coverage import apply_coverage')
 s=s.replace('  e=refine(e,p[0].numpy(),x,ts,fid,phase)','  e=refine(e,p[0].numpy(),x,ts,fid,phase)\n  e=apply_coverage(e)')
 p.write_text(s,encoding='utf-8');compile(s,str(p),'exec')
 examples=read(D/'EXAMPLES.json');changed=[];new=[]
 for ex in examples:
  old=read(D/ex['url'].lstrip('/'));updated=attach(old,ex['word']);assert updated['tcn_top3']==old['tcn_top3'];assert updated['path']==old['path']
  folder=D/'examples'/('components_'+ex['id']);write(folder/'result.json',updated)
  new.append(dict(ex,url='/examples/'+folder.name+'/result.json'))
  changed.append(dict(id=ex['id'],before=[c.get('display_status') for c in old['cards']],after=[c.get('display_status') for c in updated['cards']],tcn_unchanged=True,dtw_unchanged=True))
 write(D/'EXAMPLES.json',new)
 write(U/'INTEGRATION.json',dict(status='CODE_AND_SAVED_EXAMPLES_UPDATED_PENDING_SERVER_RESTART',examples=changed,model_changed=False,detector_calls=0,private_used=False,code_sha256=sha(D/'consumer.py'),cards_sha256=sha(D/'CARDS.json')))
 print(json.dumps(changed,ensure_ascii=False))
if __name__=='__main__':main()
