from batch_observer import *
import hashlib,shutil,av
def main():
 web=U/'web';web.mkdir(exist_ok=True);assert not(web/'EXAMPLES_CHECK.json').exists()
 ref=read(LIVE/'source/REFERENCE.json');raw=dict(np.load(LIVE/'source/reference.npz'));a=ActionObserver();ch=a.adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
 np.savez_compressed(web/'reference_features.npz',features=ch['tcn280'])
 write(web/'REFERENCE.json',dict(video='/hello-media/reference.mp4',local_video=ref['source'],clip_id='NIA_SL_SEN0354_REAL01_F',sha256=ref['sha256'],width=ref['width'],height=ref['height'],timestamp_us=raw['timestamp_us'].tolist(),xy=np.nan_to_num(raw['xy_px'],nan=-1).tolist(),action_span=[36,78],clauses=dict(sweep=[36,45],fists=[51,78],whole=[36,78]),events=[dict(region=1,start=36,end=45),dict(region=2,start=51,end=78)],purpose='Display-only single public example, not individual comparison during inference'))
 # Imported adapter setup is identical to the current capture server.
 sys.path.insert(0,str(U.parent/'apple_v1'));from check_consumer import FeedbackConsumer as Old
 from hello_web_adapter import FeedbackConsumer
 net=FeedbackConsumer();rows=[];sources={}
 try:
  for key,ids in [('normal',np.arange(len(raw['frame_id']))),('partial',np.arange(49))]:
   z={k:(v[ids]if v.ndim and len(v)==len(raw['frame_id'])else v)for k,v in raw.items()};p=web/f'{key}.npz';np.savez_compressed(p,**z);video=ref['source']
   label='안녕하세요 · 공개 정상 두 단계'if key=='normal'else'안녕하세요 · 첫 동작 뒤 자른 시험'
   if key=='partial':
    with av.open(video)as f:frames=[fr for fr in f.decode(video=0)][:49]
    video=str(web/'partial.mp4')
    with av.open(video,'w')as f:
     s=f.add_stream('libx264',rate=30);s.width=ref['width'];s.height=ref['height'];s.pix_fmt='yuv420p';s.options={'crf':'24','preset':'fast'}
     for fr in frames:
      fr.pts=None
      for pkt in s.encode(fr):f.mux(pkt)
     for pkt in s.encode():f.mux(pkt)
   q=net.raw(z,'SEN0354');q.update(source='public_integration_replay',video=f'/hello-media/{key}.mp4',public_example_name=label,synthetic=key=='partial',training_use=False);write(web/f'{key}.json',q)
   sources[key]=dict(raw_path=str(p),video=video,label=label);rows.append(dict(key=key,exercise_complete=q['exercise_complete'],game_pass=q['overall_pass'],tcn=q['tcn_top3'],seconds=q['compute_seconds']))
  assert rows[0]['exercise_complete']and not rows[1]['exercise_complete']and not any(r['game_pass']or r['tcn']for r in rows)
  write(web/'SOURCES.json',sources);write(web/'EXAMPLES_CHECK.json',dict(rows=rows,physical_camera=False,one_public_person=True));print(rows)
 finally:net.hello.close();net.apple.close();net.basic.close();net.latest.close()
if __name__=='__main__':main()
