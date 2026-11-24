"""Deny Python-visible external-disk reads; not a claim of physical disconnection."""
import sys,os,json
from pathlib import Path
denied=[];opened=set()
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=os.fsdecode(args[0]);opened.add(p)
  if p.replace('/','\\').upper().startswith('W:'):
   denied.append(p);raise PermissionError('External HDD access denied by portability probe')
sys.addaudithook(audit)
from check_consumer import *
from stream_runtime import Engine
def main():
 net=FeedbackConsumer();engine=Engine();rows=[]
 try:
  for word,root in [('WORD0099',P),('WORD1290',V)]:
   s=read(root/'web/SOURCES.json')['normal'];q=net.raw(dict(np.load(s['raw_path'])),word)
   rows.append(dict(word=word,pass_result=q['overall_pass'],top1=q['tcn_top3'][0]['word'],reference_video_local=Path(read(root/'web/REFERENCE.json')['local_video']).is_file()))
  result=dict(rows=rows,denied_external_reads=denied,python_file_count=len(opened),rtmw_device=engine.devices,all_models_and_references_local=all(r['pass_result']and r['reference_video_local']for r in rows),scope='Python audit hook and configured native model paths; HDD physically connected. Native filesystem calls and USB-disconnected physical webcam not completely traced.',physical_camera=False)
  assert not denied and result['all_models_and_references_local'];save(P/'web/PORTABILITY_CHECK.json',result);print(json.dumps(result,ensure_ascii=False))
 finally:net.apple.close();net.basic.close();net.latest.close()
if __name__=='__main__':main()
