from pathlib import Path
import sys,json,time,importlib.util,hashlib
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;N=G/'frontal_scope_recovery_20261117'
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'));import server
sys.path.insert(0,str(D));from frontal_repair_runtime import FeedbackConsumer,CaptureService
sys.path.insert(0,str(N));from source_policy import require_front
import numpy as np
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):Path(p).parent.mkdir(exist_ok=True,parents=True);Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
def close(net):
 for name in ['apple','basic','latest']:getattr(net,name).close()
def main():
 net=FeedbackConsumer();net.apple_admitted=net.basic_admitted=True;rows=[]
 try:
  for r in read(N/'INPUTS.json')['rows']:
   require_front(r);a=read(N/'web'/('normal_'+r['id']+'.json'));z=dict(np.load(r['raw']));b=net.raw(z,r['word']);aa={c['id']:c['status']for c in a['lesson_acceptance']['criteria']};bb={c['id']:c['status']for c in b['lesson_acceptance']['criteria']}
   row=dict(id=r['id'],before=aa,after=bb,passed=aa==bb and a['tcn_top3']==b['tcn_top3'] and b['compute_seconds']<max(1.5,a['compute_seconds']+.5),before_seconds=a['compute_seconds'],after_seconds=b['compute_seconds'],same_tcn=a['tcn_top3']==b['tcn_top3'])
   rows.append(row);write(D/'web'/('normal_'+r['id']+'.json'),b);print('RUNTIME',r['id'],row['passed'],round(row['after_seconds'],3),flush=True)
  write(D/'web/CONSUMER_CHECK.json',dict(rows=rows,passed=all(r['passed']for r in rows),private_data=False,baseline_scope='Prior admitted original output, same public inputs; timings are checks, not repeated benchmark'))
 finally:close(net)
 # Reuse frozen card/quest controls with new consumer and a separate output path.
 spec=importlib.util.spec_from_file_location('front_repair_card_tests',N/'check_cards.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 original_read=m.read
 def mapped_read(p):return original_read(N/'INPUTS.json' if Path(p)==D/'INPUTS.json' else p)
 m.N=D;m.FeedbackConsumer=FeedbackConsumer;m.read=mapped_read;m.main()
if __name__=='__main__':main()
