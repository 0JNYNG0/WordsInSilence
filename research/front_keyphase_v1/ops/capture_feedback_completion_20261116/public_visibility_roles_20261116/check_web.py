from pathlib import Path
import sys,json,time,numpy as np
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;A=G/'apple_v1';H=G/'public_shape_arrival_repair_20261116';V=G/'completion_v1'
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
import server  # idle constructor; no camera, socket or prepare call
sys.path.insert(0,str(D))
from visibility_web_adapter import FeedbackConsumer,PriorConsumer
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def close(net):
 for name in ['apple','basic','latest']:getattr(net,name).close()

def main():
 old,new=PriorConsumer(),FeedbackConsumer();assert new.visibility_admitted and new.repair_admitted and new.apple_admitted
 rows=[]
 try:
  selected=[dict(id=r['clip_id'],word='WORD0099',raw=r.get('raw',r.get('raw_path')))for r in read(A/'INPUTS.json')['rows']+read(A/'holdout/EXTRACTED.json')['rows']if r['person']>=10]
  selected += [dict(id=r['id'],word='WORD1290',raw=r['raw'])for r in read(H/'PUBLIC_INPUTS.json')['rows']if r['person']in[1,7,11,13,18]and r['id'].endswith('_F')]
  selected += [dict(id=r['id'],word='WORD1290',raw=r['raw'])for r in read(H/'public_views/INPUTS.json')['rows']if r['id']=='NIA_SL_WORD1290_REAL13_R']
  for r in selected:
   raw=dict(np.load(r['raw']));a=old.raw(raw,r['word']);b=new.raw(raw,r['word']);bs={c['id']:c['status']for c in a['lesson_acceptance']['criteria']};ns={c['id']:c['status']for c in b['lesson_acceptance']['criteria']}
   row=dict(**r,before=bs,after=ns,before_seconds=a['compute_seconds'],after_seconds=b['compute_seconds'],same_tcn=a['tcn_top3']==b['tcn_top3'],same_path=a['path']==b['path'],no_lost_supported=all(ns[k]=='supported' for k,v in bs.items()if v=='supported'),module=b.get('visibility_module'))
   rows.append(row);print(r['id'],row['same_tcn'],row['same_path'],round(row['after_seconds'],3),flush=True)
  passed=all(x['same_tcn']and x['same_path']and x['no_lost_supported']and x['after_seconds']<max(1.5,x['before_seconds']+.5)for x in rows)
  write(D/'web/CONSUMER_CHECK.json',dict(rows=rows,passed=passed,physical_camera=False,private_data=False));assert passed
 finally:close(old);close(new)
if __name__=='__main__':main()
