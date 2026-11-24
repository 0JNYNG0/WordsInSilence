"""Frozen candidate through actual quarantine and web worker. Public inputs only."""
from pathlib import Path
import sys,json,time
import numpy as np
P=Path(__file__).resolve().parent;O=P.parents[1];V=P.parent/'completion_v1'
sys.path[:0]=[str(O/'word100_action_feedback_demo_20261108'),str(O/'web_tutorial_live_integration_20261113')]
import capture
sys.path[:0]=[str(O/'word100_action_linkage_repair_20261113'),str(O/'word100_actionable_feedback_20261113')]
import runtime_adapter
sys.path[:0]=[str(V),str(O/'word100_live_g6_20261114')]
from apple_web_adapter import FeedbackConsumer,PreviousConsumer
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 out=P/'web';assert not(out/'CONSUMER_CHECK.json').exists();net=FeedbackConsumer();rows=[];reg=[]
 try:
  sources=[r for r in read(P/'INPUTS.json')['rows']if r['person']>=11]+read(P/'holdout/EXTRACTED.json')['rows']
  for r in sources:
   q=net.raw(dict(np.load(r.get('raw',r.get('raw_path')))),'WORD0099');save(out/f'public_{r["person"]}.json',q)
   rows.append(dict(person=r['person'],supported=all(c['status']=='supported'for c in q['lesson_acceptance']['criteria']),criteria={c['id']:c['status']for c in q['lesson_acceptance']['criteria']},rank=q['tcn_top3'][0],seconds=q['compute_seconds']))
  for word in ['WORD0739','WORD1133','WORD2523','WORD2683']:
   p=O/'word100_action_feedback_demo_20261108/parallel'/word/'rtmw/observations.npz'
   if not p.exists():continue
   raw=dict(np.load(p));before=PreviousConsumer.raw(net,raw,word);after=net.raw(raw,word)
   reg.append(dict(word=word,same_evidence=all(before[k]==after[k]for k in ['cards','coaching','path']),top1_before=before['tcn_top3'][0],top1_after=after['tcn_top3'][0]))
  for r in read(V/'web/SOURCES.json').values():
   if r.get('word','WORD1290')!='WORD1290' or not r.get('raw_path'):continue
   raw=dict(np.load(r['raw_path']));before=PreviousConsumer.raw(net,raw,'WORD1290');after=net.raw(raw,'WORD1290')
   reg.append(dict(word='WORD1290',same_evidence=all(before[k]==after[k]for k in ['cards','coaching','path']),top1_before=before['tcn_top3'][0],top1_after=after['tcn_top3'][0]));break
  result=dict(rows=rows,regression=reg,camera_used=False,user_data=False,admission=net.apple_admitted)
  save(out/'CONSUMER_CHECK.json',result);print(json.dumps(result,ensure_ascii=False))
 finally:net.apple.close();net.basic.close();net.latest.close()
if __name__=='__main__':main()
