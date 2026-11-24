"""Record16 actually inspected dense source sheets and preserve coarse predecessors."""
from finalize_rgb_review import D,read,sha,write
REFINED={
 '0739_05':[([54,56],[62,64]),([68,70],[78,82])],
 '1133_05':[([46,48],[58,62])],
 '1567_05':[([44,46],[52,54]),([54,56],[62,64]),([64,66],[72,74]),([74,76],[82,86])],
 '2523_05':[([62,64],[68,72])],
 '0739_06':[([70,72],[78,82]),([84,86],[92,96])],
 '1133_06':[([42,44],[50,54])],
 '1567_06':[([52,54],[58,60]),([60,62],[66,68]),([68,70],[76,80])],
 '2523_06':[([68,70],[74,78])],
 '0739_08':[([60,62],[68,72]),([74,76],[80,84])],
 '1133_08':[([70,72],[88,92])],
 '1567_08':[([60,62],[68,70]),([70,72],[78,80]),([80,82],[90,92]),([92,94],[102,106])],
 '2523_08':[([54,56],[60,64])],
 '0739_09':[([52,54],[60,64]),([64,66],[72,76])],
 '1133_09':[([46,48],[56,60])],
 '1567_09':[([56,58],[66,68]),([68,70],[74,76]),([76,78],[82,86])],
 '2523_09':[([66,68],[72,76])]}
def main():
 out=D/'SOURCE_EPISODES_DENSE.json';assert not out.exists();source=read(D/'SOURCE_EPISODES.json');dense={r['clip_id']:r for r in read(D/'DENSE_REMAINING_QUEUE.json')['rows']};changes=[]
 for r in source['rows']:
  if r['clip_id'] not in dense:continue
  d=dense[r['clip_id']];assert sha(d['path'])==d['sha256'];r['sheets'].append(dict(path=d['path'],sha256=d['sha256'],frames=d['frames'],viewed=True));key=r['word_id'][4:]+'_'+str(r['person']).zfill(2)
  assert len(r['events'])==len(REFINED[key]);seen=sorted(set(f for s in r['sheets'] for f in s['frames']))
  for e,(on,off) in zip(r['events'],REFINED[key]):
   changes.append(dict(clip=r['clip_id'],event=e['event_id'],coarse=[e['onset'],e['offset']],dense=[on,off]));e.update(onset=on,offset=off,progression_witness_frames=[f for f in d['frames'] if on[1]<f<off[0]])
  for g in r['groups']:
   es=[e for e in r['events'] if e['event_id'] in g['event_ids']];g['onset']=es[0]['onset'];g['offset']=es[-1]['offset']
  r['hold_sample_frames']=[f for f in seen if any(a<=f<=b for a,b in r['holds'])]
 source.update(status='24_PUBLIC_FIT_DENSE_RGB_REVIEWED',predecessor=dict(path=str(D/'SOURCE_EPISODES.json'),sha256=sha(D/'SOURCE_EPISODES.json')),dense_refinements=changes,training_allowed=False)
 source['counts']['dense_sheets']=24
 source['policy'].append('Dense review performed before any new stage training or probe scores. Coarse versions preserved; still sampled frames and uncertainty bands, not exact every-frame truth.')
 write(out,source);print('Recorded16 dense reviews;24 clips/30 occurrence groups retained. No model scores read.')
if __name__=='__main__':main()
