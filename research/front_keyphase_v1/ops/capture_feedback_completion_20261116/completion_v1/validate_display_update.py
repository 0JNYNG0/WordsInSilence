from check_web_consumer import *
import shutil
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 out=V/'web';prior=read(out/'ADMISSION.json');assert not(out/'DISPLAY_CHECK.json').exists();net=FeedbackConsumer();checks=[];outputs={}
 try:
  for key,r in read(out/'SOURCES.json').items():
   before=read(out/f'{key}.json');q=net.raw(dict(np.load(r['raw_path'])),'WORD1290')
   for field in['cards','coaching','tcn_top3']:assert before[field]==q[field],field
   assert before['lesson_acceptance']['criteria']==q['lesson_acceptance']['criteria']
   path=np.array(q['path']);assert (np.diff(path,axis=0)>=0).all()
   if key=='partial':assert q['display_alignment']['partial_reference']and path[:,1].max()==read(out/'DISPLAY_PATTERN.json')['events'][0]['end']
   q.update(source=before['source'],video=before['video'],public_example_name=before['public_example_name'],synthetic=before['synthetic'],training_use=False)
   outputs[key]=q;checks.append(dict(key=key,same_criteria=True,same_cards=True,same_tcn=True,monotone=True,reference_endpoint=int(path[-1,1]),display=q['display_alignment']))
  # Supersede the executable manifest only after unchanged judgement verification.
  shutil.copy2(out/'ADMISSION.json',out/'ADMISSION_v1.json');prior['supersedes']='ADMISSION_v1.json';prior['display_repair']='Partial evidence maps only through corresponding observed reference action, not the full rest-to-rest recording.'
  for p in[V/'worker.py',V/'display_alignment.py',out/'DISPLAY_PATTERN.json']:prior['pins'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  save(out/'ADMISSION.json',prior)
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in prior['pins'].items())
  for key,q in outputs.items():
   shutil.copy2(out/f'{key}.json',out/f'{key}_before_display.json')
   q['lesson_acceptance']['qualified']=True;q['lesson_acceptance']['overall_pass']=all(c['status']=='supported'for c in q['lesson_acceptance']['criteria']);q['overall_pass']=q['lesson_acceptance']['overall_pass'];q['correction_validated']=True
   save(out/f'{key}.json',q)
  save(out/'DISPLAY_CHECK.json',dict(rows=checks,judgement_changed=False,camera=False,private=False));print(json.dumps(checks,ensure_ascii=False))
 finally:net.basic.close();net.latest.close()
if __name__=='__main__':main()
