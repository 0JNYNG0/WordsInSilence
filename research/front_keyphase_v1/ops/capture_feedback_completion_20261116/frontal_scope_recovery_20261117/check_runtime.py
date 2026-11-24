from pathlib import Path
import sys,json,time,hashlib,importlib.util
N=Path(__file__).resolve().parent;G=N.parent;O=G.parent
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
import server
sys.path.insert(0,str(N))
from frontal_web_adapter import FeedbackConsumer,CaptureService
sys.path.insert(0,str(G/'public_cycle_observability_20261116'))
from cycle_web_adapter import FeedbackConsumer as PriorConsumer
from source_policy import require_front
import numpy as np
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
def close(net):
 for name in ['apple','basic','latest']:getattr(net,name).close()
def main():
 recheck='--recheck'in sys.argv
 prior_rows={r['id']:r for r in read(N/'web/CONSUMER_CHECK.json')['rows']}if recheck else {}
 old=None if recheck else PriorConsumer();new=FeedbackConsumer();rows=[]
 # Isolated verification admission; live server remains the previous revision.
 new.basic_admitted=new.apple_admitted=True
 try:
  for r in read(N/'INPUTS.json')['rows']:
   require_front(r);z=dict(np.load(r['raw']));a=read(N/'web'/('normal_'+r['id']+'.json'))if recheck else old.raw(z,r['word']);b=new.raw(z,r['word']);aa=prior_rows[r['id']]['before']if recheck else {c['id']:c['status']for c in a['lesson_acceptance']['criteria']};bb={c['id']:c['status']for c in b['lesson_acceptance']['criteria']}
   if recheck:a['compute_seconds']=prior_rows[r['id']]['before_seconds']
   row=dict(id=r['id'],word=r['word'],before=aa,after=bb,same_tcn=a['tcn_top3']==b['tcn_top3'],before_seconds=a['compute_seconds'],after_seconds=b['compute_seconds'],lost=[k for k,v in aa.items()if v=='supported'and bb[k]!='supported'],new_error=[k for k,v in bb.items()if v=='difference'and aa[k]!='difference'],path_monotonic=all(p[0]<=q[0]and p[1]<=q[1]for p,q in zip(b['path'],b['path'][1:])),unknown_not_wrong=all(c['display_status']not in ['difference','wrong']for c in b['cards']if c['status']=='unknown'))
   row['passed']=row['same_tcn']and not row['lost']and not row['new_error']and row['path_monotonic']and row['unknown_not_wrong']and b['compute_seconds']<max(1.5,a['compute_seconds']+.5)
   rows.append(row);print(r['id'],row['passed'],round(row['before_seconds'],3),round(row['after_seconds'],3),flush=True)
   write(N/'web'/('normal_'+r['id']+'.json'),b)
  write(N/'web/CONSUMER_CHECK.json',dict(rows=rows,passed=all(r['passed']for r in rows),physical_camera=False,private_data=False,isolated_admission=True,prior_measured_baseline_reused=recheck))
 finally:
  if old is not None:close(old)
  close(new)
if __name__=='__main__':main()
