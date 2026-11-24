from bilateral_core import *

def main():
 assert read(U/'BROWSER_VERIFIED.json')['status']=='PASS'
 assert read(U/'VERIFICATION.json')['status']=='PASS'
 final=read(U/'FINAL_RESULT.json');assert not any(g['pass'] for g in final['gates'].values())
 changes=[]
 for name in ['index.html','app.js','style.css']:
  p=D/name;candidate=U/'web'/name;before=sha(U/'before_web'/name)
  assert sha(p)==before,'Concurrent UI change; do not overwrite'
  assert sha(candidate)==read(U/'WEB_CANDIDATE.json')['files'][str(candidate)]
  p.write_bytes(candidate.read_bytes());assert sha(p)==sha(candidate)
  changes.append(dict(path=str(p),before=before,after=sha(p)))
 for name in ['consumer.py','server.py','CARDS.json']:assert sha(D/name)==sha(U/'before_web'/name)
 write(U/'DEPLOYMENT.json',dict(status='DISPLAY_ONLY_APPLIED',changes=changes,research_web_port=8814,model_candidate_adopted=False,reason='New candidate failed normal completion/abstention regression; UI uses existing evidence only',camera=False,user_recordings_read=False,TCN_DTW_weights_unchanged=True,browser_checks=read(U/'BROWSER_VERIFIED.json')['count']))
 event('display_applied',files=len(changes),model_candidate_adopted=False)
if __name__=='__main__':main()
