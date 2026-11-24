from pathlib import Path
import sys,json,time,numpy as np
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;A=G/'apple_v1';H=G/'public_shape_arrival_repair_20261116';V=G/'completion_v1'
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
import server
sys.path.insert(0,str(D))
from form_web_adapter import FeedbackConsumer,PriorConsumer
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def close(net):
 for name in['apple','basic','latest']:getattr(net,name).close()
def main():
 old,new=PriorConsumer(),FeedbackConsumer();assert new.form_admitted and new.visibility_admitted and new.repair_admitted and new.apple_admitted;rows=[]
 try:
  selected=[dict(id=r['clip_id'],word='WORD0099',raw=r.get('raw',r.get('raw_path')))for r in read(A/'INPUTS.json')['rows']+read(A/'holdout/EXTRACTED.json')['rows']if r['person']>=10]
  selected +=[dict(id=r['id'],word='WORD1290',raw=r['raw'])for r in read(H/'PUBLIC_INPUTS.json')['rows']if r['person']in[1,7,11,13,18]and r['id'].endswith('_F')]
  for r in selected:
   z=dict(np.load(r['raw']));a=old.raw(z,r['word']);b=new.raw(z,r['word']);aa={c['id']:c['status']for c in a['lesson_acceptance']['criteria']};bb={c['id']:c['status']for c in b['lesson_acceptance']['criteria']};row=dict(**r,before=aa,after=bb,before_seconds=a['compute_seconds'],after_seconds=b['compute_seconds'],same_tcn=a['tcn_top3']==b['tcn_top3'],same_path=a['path']==b['path'],no_lost_supported=all(bb[k]=='supported'for k,v in aa.items()if v=='supported'),no_new_difference=not any(v=='difference'and aa[k]!='difference'for k,v in bb.items()),module=b.get('hand_form_module'));rows.append(row);print(r['id'],row['same_tcn'],row['same_path'],round(row['after_seconds'],3),flush=True)
  passed=all(r['same_tcn']and r['same_path']and r['no_lost_supported']and r['no_new_difference']and r['after_seconds']<max(1.5,r['before_seconds']+.5)for r in rows);write(D/'web/CONSUMER_CHECK.json',dict(rows=rows,passed=passed,physical_camera=False,private_data=False));assert passed
 finally:close(old);close(new)
if __name__=='__main__':main()
