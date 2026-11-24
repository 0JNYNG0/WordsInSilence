from check_runtime import *
from frontal_web_adapter import FeedbackConsumer as Baseline
import statistics

def main():
 old=Baseline();new=FeedbackConsumer();new.apple_admitted=new.basic_admitted=True;rows=[]
 try:
  inputs=[r for r in read(N/'INPUTS.json')['rows'] if r['person']in[1,13]]
  for r in inputs:
   require_front(r);z=dict(np.load(r['raw']));a=old.raw(z,r['word']);b=new.raw(z,r['word'])
   rows.append(dict(id=r['id'],before_seconds=a['compute_seconds'],after_seconds=b['compute_seconds'],same_tcn=a['tcn_top3']==b['tcn_top3'],same_criteria={c['id']:c['status']for c in a['lesson_acceptance']['criteria']}=={c['id']:c['status']for c in b['lesson_acceptance']['criteria']},passed=b['compute_seconds']<max(1.5,a['compute_seconds']+.5)))
 finally:close(old);close(new)
 q=dict(rows=rows,passed=all(r['passed']and r['same_tcn']and r['same_criteria']for r in rows),before_median=statistics.median(r['before_seconds']for r in rows),after_median=statistics.median(r['after_seconds']for r in rows),scope='One paired pass per public clip in the same process/session; inference only, not webcam/browser or a repeated statistical speed benchmark',private_data=False)
 write(D/'web/LATENCY_CHECK.json',q);print(json.dumps(q,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
