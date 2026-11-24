from verified_candidate import *
def main():
 torch.set_num_threads(2);dest=U/'web';dest.mkdir(exist_ok=True);assert not(dest/'REFERENCE.json').exists();r=next(r for r in read(U/'INPUTS.json')['rows']if r['person']==1);raw=dict(np.load(r['raw']));net=VerifiedCandidate();q=net.analyze(raw)
 np.savez_compressed(dest/'reference_features.npz',features=net.last_features)
 write(dest/'REFERENCE.json',dict(video='/apple-media/reference.mp4',local_video=r['video'],clip_id=r['clip_id'],sha256=r['source_sha256'],width=int(raw['source_width']),height=int(raw['source_height']),timestamp_us=raw['timestamp_us'].tolist(),xy=np.nan_to_num(raw['xy_px'],nan=-1).tolist(),action_span=[43,82],clauses=dict(mouth=[43,55],wipe=[65,82],whole=[43,82]),events=q['events'],purpose='One example for playback; no reference video reads or per-person comparison in judgement.'))
 print('APPLE_REFERENCE_READY')
if __name__=='__main__':main()
