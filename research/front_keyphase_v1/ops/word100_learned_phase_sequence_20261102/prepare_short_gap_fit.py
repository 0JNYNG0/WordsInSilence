"""FIT sensor-only gaps: same RGB and native time, no invented coordinates."""
import sys,numpy as np
from phase_model import *

def main():
    assert not (S/'SHORT_GAP_FIT_MANIFEST.json').exists()
    sys.path.insert(0,str(P));from engine import LearnedGuideEngine
    engine=LearnedGuideEngine();records=[]
    targets={r['clip_id']:r for r in read(S/'PHYSICAL_STATE_TARGETS.json')['rows'] if r['role']=='fit'}
    for r in read(N/'INPUT_MANIFEST.json')['rows']:
        if r['kind']!='original':continue
        assert 1<=r['person']<=9 and sha(r['input'])==r['analysis_sha256'];a=dict(np.load(r['input']))
        known=[p['frame'] for p in targets[r['clip_id']]['points'] if p['mask']];center=known[len(known)//2];ix=np.arange(center-1,center+2)
        b={k:v.copy() for k,v in a.items()};b['x'][ix]=0;b['mask'][ix]=False
        assert np.array_equal(a['rgb'],b['rgb']) and np.array_equal(a['prelinear'],b['prelinear'])
        pred=engine.infer(b,r['word_id']);dest=S/'short_gap_fit'/f"{r['clip_id']}.npz";dest.parent.mkdir(exist_ok=True);assert not dest.exists()
        np.savez_compressed(dest,**{k:v for k,v in pred.items() if isinstance(v,np.ndarray)},frame_id=a['frame_id'],timestamp_us=a['timestamp_us'],image_available=a['image_available'])
        records.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],role='fit',kind='pose_only_3',input=str(dest),analysis_sha256=sha(dest),source_input=r['input'],source_input_sha256=r['analysis_sha256'],source_frame_indices=r['source_frame_indices'],gap_frames=ix.tolist(),rgb_preserved=True))
    exclusive(S/'SHORT_GAP_FIT_MANIFEST.json',dict(rows=records,no_new_pixel_extraction=True,old_observers_frozen=True))
    print('Prepared36FIT short-gap observation latents; no training/development inference')

if __name__=='__main__':main()
