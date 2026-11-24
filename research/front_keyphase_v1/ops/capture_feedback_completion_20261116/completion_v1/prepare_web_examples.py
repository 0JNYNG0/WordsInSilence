from check_web_consumer import *
import cv2
from g6_runtime_adapter import FeedbackConsumer as PreviousConsumer
from feedback_acceptance import feedback_gate
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 out=V/'web';assert not(out/'EXAMPLES_CHECK.json').exists();net=FeedbackConsumer();assert net.basic_admitted
 old=O/'word100_action_feedback_demo_20261108';ref=read(V/'web/REFERENCE.json');normal=read(O/'basic_demo_feedback_20261115/source_corrected/CONFIRMATION_INPUTS.json')['rows'][0]
 sources={};rows=[]
 try:
  for key,label,rawpath,video in[
   ('normal','감사 · 공개 정상 동작',normal['raw'],normal['video']),
   ('partial','감사 · 첫 동작 뒤 잘라낸 시험',normal['raw'],normal['video']),
   ('other_word','다른 공개 단어 · 감사의 손모양과 비교',str(old/'parallel/WORD1567/rtmw/observations.npz'),str(old/'parallel/WORD1567/video.mp4'))]:
   raw=dict(np.load(rawpath))
   if key=='partial':
    end=55
    for k in['xy_px','score','timestamp_us','frame_id']:raw[k]=raw[k][:end]
    dst=out/'partial.mp4';cap=cv2.VideoCapture(video);writer=None
    for i in range(end):
     ok,im=cap.read();assert ok
     if writer is None:writer=cv2.VideoWriter(str(dst),cv2.VideoWriter_fourcc(*'mp4v'),30,(im.shape[1],im.shape[0]))
     writer.write(im)
    writer.release();cap.release();video=str(dst);rawpath=str(out/'partial_raw.npz');np.savez_compressed(rawpath,**raw)
   q=net.raw(raw,'WORD1290');q.update(source='public_integration_replay',video=f'/basic-media/{key}.mp4',public_example_name=label,synthetic=key=='partial',training_use=False)
   save(out/f'{key}.json',q);sources[key]=dict(raw_path=rawpath,video=video,label=label)
   accepted=feedback_gate(q,'WORD1290')['accepted'];difference=any(c['display_status']=='difference'for c in q['cards'])
   rows.append(dict(key=key,accepted=accepted,difference=difference,summary=q['coaching']['summary'],rank=q['tcn_top3'][0]['label'],seconds=q['compute_seconds']))
  assert rows[0]['accepted']and not rows[1]['accepted']and not rows[2]['accepted']
  # Same unmodified previous consumer method must give identical old-word findings.
  regressions=[]
  for word in['WORD0739','WORD1133','WORD1567','WORD2523']:
   raw=dict(np.load(old/'parallel'/word/'rtmw/observations.npz'));a=net.raw(raw,word);b=PreviousConsumer.raw(net,raw,word)
   keys=['tcn_top3','cards','coaching','path'];checks={k:a[k]==b[k]for k in keys};assert all(checks.values());regressions.append(dict(word=word,checks=checks))
  save(out/'SOURCES.json',sources);save(out/'EXAMPLES_CHECK.json',dict(rows=rows,old_four_regression=regressions,camera=False,private=False))
  print(json.dumps(rows,ensure_ascii=False));print('OLD_FOUR_IDENTICAL',len(regressions))
 finally:net.basic.close();net.latest.close()
if __name__=='__main__':main()
