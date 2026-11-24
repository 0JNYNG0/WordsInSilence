from visible_lesson import *
def main():
 torch.set_num_threads(2);out=V/'web';out.mkdir(exist_ok=True);assert not(out/'REFERENCE.json').exists()
 r=next(r for r in read(B/'INPUTS.json')['rows']if r['person']==1);raw=dict(np.load(r['raw']));ch=adapter.load(r['raw']);f,_,_=Features()(ch)
 np.savez_compressed(out/'reference_features.npz',features=f)
 write(out/'REFERENCE.json',dict(video='/basic-media/reference.mp4',local_video=r['video'],sha256=r['source_sha256'],clip_id=r['clip_id'],
  width=int(raw['source_width']),height=int(raw['source_height']),timestamp_us=raw['timestamp_us'].tolist(),xy=np.nan_to_num(raw['xy_px'],nan=-1).tolist(),
  action_span=[47,64],purpose='One display example only. Native reference timeline used only for replay, not for judging completion.'))
 print('REFERENCE_READY')
if __name__=='__main__':main()
