from check_consumer import *
import copy,av
from feedback_acceptance import feedback_gate
import contract
def selection(raw,ids):
 n=len(raw['frame_id']);return {k:(v[ids].copy()if v.ndim and len(v)==n else v.copy())for k,v in raw.items()}
def export(video,ids,dest):
 with av.open(video)as f:frames=[fr.to_ndarray(format='bgr24')for fr in f.decode(video=0)]
 with av.open(str(dest),'w')as f:
  stream=f.add_stream('libx264',rate=30);stream.width=frames[0].shape[1];stream.height=frames[0].shape[0];stream.pix_fmt='yuv420p';stream.options={'crf':'24','preset':'fast'}
  for i in ids:
   fr=av.VideoFrame.from_ndarray(frames[i],format='bgr24')
   for packet in stream.encode(fr):f.mux(packet)
  for packet in stream.encode():f.mux(packet)
def main():
 out=P/'web';assert not(out/'EXAMPLES_CHECK.json').exists();net=FeedbackConsumer();assert net.apple_admitted
 held=read(P/'holdout/EXTRACTED.json')['rows'][0];dev=next(r for r in read(P/'INPUTS.json')['rows']if r['person']==11);truth=next(r for r in read(P/'SOURCE_TRUTH.json')['rows']if r['person']==11)
 sources={};checks=[]
 try:
  for key,label,r,ids in [('normal','사과 · 공개 정상 동작',held,None),('partial','사과 · 첫 동작 뒤 자른 시험',held,np.arange(54)),('reverse','사과 · 첫 동작을 거꾸로 재생한 시험',dev,'reverse')]:
   rawpath=r.get('raw',r.get('raw_path'));video=r.get('video',r.get('path'));raw=dict(np.load(rawpath))
   if isinstance(ids,str):
    ids=np.arange(len(raw['frame_id']));a=max(0,truth['mouth_sweep'][0]-10);b=truth['wipe_sweeps'][0][0]-5;ids[a:b+1]=ids[a:b+1][::-1]
   if ids is not None:
    z=selection(raw,ids);z['timestamp_us']=np.arange(len(ids),dtype=np.int64)*33333;z['frame_id']=np.arange(len(ids));raw=z
    rawpath=str(out/f'{key}_raw.npz');np.savez_compressed(rawpath,**raw);dest=out/f'{key}.mp4';export(video,ids,dest);video=str(dest)
   q=net.raw(raw,'WORD0099');q.update(source='public_integration_replay',video=f'/apple-media/{key}.mp4',public_example_name=label,synthetic=key!='normal',training_use=False)
   save(out/f'{key}.json',q);sources[key]=dict(raw_path=rawpath,video=video,label=label)
   checks.append(dict(key=key,accepted=feedback_gate(q,'WORD0099')['accepted'],difference=any(c['display_status']=='difference'for c in q['cards']),summary=q['coaching']['summary'],criteria={c['id']:c['status']for c in q['lesson_acceptance']['criteria']},seconds=q['compute_seconds']))
  assert checks[0]['accepted']and not checks[1]['accepted']and not checks[2]['accepted']and checks[2]['difference']
  # A missing requirement is never allowed to silently make the lesson easier.
  q=read(out/'normal.json');bad=copy.deepcopy(q);bad['lesson_acceptance']['criteria']=bad['lesson_acceptance']['criteria'][:-1];bad['lesson_acceptance']['required_ids']=[c['id']for c in bad['lesson_acceptance']['criteria']];assert not feedback_gate(bad,'WORD0099')['accepted']
  # A public replay cannot advance the live quest. A contract fixture may test
  # the application state machine, but is not reported as an actual user test.
  contract.SUPPORTED.update(WORD0099='사과',WORD1290='감사');quest=contract.QuestProgress(steps=['WORD0099','WORD1290'])
  def attempt(i,target):return dict(id=str(i),mode='quest',target=target,generation=quest.generation)
  assert not quest.consume(attempt(1,'WORD0099'),q)['advanced']
  user_fixture=copy.deepcopy(q);user_fixture['source']='user_inference_only';user_fixture['stop_reason']='arms_returned';assert quest.consume(attempt(2,'WORD0099'),user_fixture)['advanced'];assert not quest.consume(attempt(2,'WORD0099'),user_fixture)['advanced']
  thanks=read(V/'web/normal.json');thanks['source']='user_inference_only';thanks['stop_reason']='arms_returned';assert quest.consume(attempt(3,'WORD1290'),thanks)['completed'];assert quest.completions==1
  old='WORD1567';raw=dict(np.load(O/'word100_action_feedback_demo_20261108/parallel'/old/'rtmw/observations.npz'));a=PreviousConsumer.raw(net,raw,old);b=net.raw(raw,old);assert all(a[k]==b[k]for k in['cards','coaching','path']);assert a['tcn_top3'][0]['word']==b['tcn_top3'][0]['word']
  save(out/'SOURCES.json',sources);save(out/'EXAMPLES_CHECK.json',dict(rows=checks,missing_requirement_rejected=True,quest_state_fixture=True,public_does_not_advance=True,additional_old_word_regression=old,actual_webcam=False));print(json.dumps(checks,ensure_ascii=False))
 finally:net.apple.close();net.basic.close();net.latest.close()
if __name__=='__main__':main()
