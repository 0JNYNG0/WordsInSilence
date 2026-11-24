"""Old/new real web consumers, explicit public inputs, no camera."""
from pathlib import Path
import sys,json,time,hashlib,numpy as np
H=Path(__file__).resolve().parent;G=H.parent;O=G.parent;V=G/'completion_v1';A=G/'apple_v1'
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
import server  # Constructor is idle, does not open the camera or listen.
from repair_web_adapter import FeedbackConsumer,PriorConsumer
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def close(net):
 for name in['apple','basic','latest']:getattr(net,name).close()
def main():
 out=H/'web';out.mkdir(exist_ok=True);assert not(out/'CONSUMER_CHECK.json').exists()
 a=PriorConsumer();b=FeedbackConsumer();assert b.repair_admitted and b.basic_admitted and b.apple_admitted;rows=[]
 try:
  selected=[dict(word='WORD0099',id=r['clip_id'],raw=r.get('raw',r.get('raw_path')))for r in read(A/'INPUTS.json')['rows']+read(A/'holdout/EXTRACTED.json')['rows']if r['person']>=10]
  selected += [dict(word='WORD1290',id=r['id'],raw=r['raw'])for r in read(H/'PUBLIC_INPUTS.json')['rows']if r['person']in[1,13,18]and r['id'].endswith('_F')]
  selected += [dict(word='WORD1290',id=r['id'],raw=r['raw'])for r in read(H/'public_views/INPUTS.json')['rows']if r['id']in['NIA_SL_WORD1290_REAL07_L','NIA_SL_WORD1290_REAL11_R']]
  for r in selected:
   raw=dict(np.load(r['raw']));x=a.raw(raw,r['word']);y=b.raw(raw,r['word']);write(out/(r['id']+'.json'),dict(before=x,after=y))
   row=dict(**r,same_cards=x['cards']==y['cards'],same_coaching=x['coaching']==y['coaching'],same_path=x['path']==y['path'],same_tcn=x['tcn_top3']==y['tcn_top3'],before_seconds=x['compute_seconds'],after_seconds=y['compute_seconds'],before_pass=x['overall_pass'],after_pass=y['overall_pass'],before={c['id']:c['status']for c in x['lesson_acceptance']['criteria']},after={c['id']:c['status']for c in y['lesson_acceptance']['criteria']})
   rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
  write(out/'CONSUMER_CHECK.json',dict(rows=rows,private_data=False,physical_camera=False))
 finally:close(a);close(b)
if __name__=='__main__':main()
