from pathlib import Path
import sys,json,time,importlib.util
import numpy as np
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;N=G/'frontal_scope_recovery_20261117';E=G/'public_frontal_evidence_repair_20261117'
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'));import server
sys.path.insert(0,str(D));from stable_runtime import FeedbackConsumer,CaptureService
from source_policy import require_front
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def close(net):
 for name in ['apple','basic','latest']:getattr(net,name).close()
def main():
 net=FeedbackConsumer();rows=[]
 try:
  for r in read(N/'INPUTS.json')['rows']:
   require_front(r);a=read(E/'web'/('normal_'+r['id']+'.json'));b=net.raw(dict(np.load(r['raw'])),r['word']);aa={c['id']:c['status']for c in a['lesson_acceptance']['criteria']};bb={c['id']:c['status']for c in b['lesson_acceptance']['criteria']}
   row=dict(id=r['id'],before=aa,after=bb,passed=aa==bb and a['tcn_top3']==b['tcn_top3'] and b['compute_seconds']<1.5,same_tcn=a['tcn_top3']==b['tcn_top3'],before_seconds=a['compute_seconds'],after_seconds=b['compute_seconds'])
   rows.append(row);write(D/'web'/('normal_'+r['id']+'.json'),b);print('RUNTIME',r['id'],row['passed'],round(row['after_seconds'],3),flush=True)
  write(D/'web/CONSUMER_CHECK.json',dict(rows=rows,passed=all(r['passed']for r in rows),private_data=False))
 finally:close(net)
 spec=importlib.util.spec_from_file_location('new_card_regression',N/'check_cards.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 original=m.read
 m.read=lambda p:original(N/'INPUTS.json'if Path(p)==D/'INPUTS.json'else p)
 m.N=D;m.FeedbackConsumer=FeedbackConsumer;m.main()
if __name__=='__main__':main()
