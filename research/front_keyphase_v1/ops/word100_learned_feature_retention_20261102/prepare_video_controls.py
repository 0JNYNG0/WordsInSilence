"""Create only local public-derived controls, with exact source-frame ledger."""
from pathlib import Path
from fractions import Fraction
import hashlib,json
import av,numpy as np
from audit import U,O,read,save
CUTS={'WORD0739':[36,62],'WORD1133':[60,75],'WORD1567':[46,73],'WORD2523':[43,59]}
NOTES={'WORD0739':'open-hand face approach removed; retained clip starts right closed-hand phase',
 'WORD1133':'main palm travel along opposite arm removed; arrival/hold retained',
 'WORD1567':'earlier alternating sequence removed; repeated-pattern count is NOT a certified error',
 'WORD2523':'hands-together triangular configuration removed; later separated hands retained'}
def main():
 assert not (U/'VIDEO_CONTROLS.json').exists();rows=[]
 for word,(lo,hi) in CUTS.items():
  source=O/'word100_seated_web_framing_20261102/public_crops'/f'NIA_SL_{word}_REAL12_F.mp4'
  with av.open(str(source)) as c:frames=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
  n=len(frames);cases=[('reencoded_normal',np.arange(n),None),('speed_0.8',np.minimum(np.arange(0,n,.8).astype(int),n-1),None),
   ('speed_1.25',np.minimum(np.arange(0,n,1.25).astype(int),n-1),None),('trim_initial_rest',np.arange(30,n),None),
   ('action_cut',np.array([i for i in range(n) if not lo<=i<=hi]),None),('main_blackout',np.arange(n),(lo,hi))]
  for kind,indices,black in cases:
   dest=U/'video_controls'/f'{word}_{kind}.mp4';dest.parent.mkdir(exist_ok=True);assert not dest.exists()
   with av.open(str(dest),'w') as out:
    s=out.add_stream('libx264',rate=30);s.width=frames[0].shape[1];s.height=frames[0].shape[0];s.pix_fmt='yuv420p';s.options={'crf':'18','preset':'fast'}
    for i,source_i in enumerate(indices):
     im=frames[int(source_i)]
     if black and black[0]<=source_i<=black[1]:im=np.zeros_like(im)
     f=av.VideoFrame.from_ndarray(im,format='rgb24');f.pts=i;f.time_base=Fraction(1,30)
     for packet in s.encode(f):out.mux(packet)
    for packet in s.encode():out.mux(packet)
   rows.append(dict(word_id=word,kind=kind,path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_frame_indices=indices.tolist(),output_fps=30,blackout_source_frames=black,
    removed_source_frames=[lo,hi] if kind=='action_cut' else None,physical_edit_note=NOTES[word] if kind=='action_cut' else kind,
    independent_linguistic_error=False,person_id=12,role='development_controls_not_training'))
   print(word,kind,len(indices),flush=True)
 save('VIDEO_CONTROLS.json',dict(rows=rows,review='Same Codex reviewed existing source contact sheets before edited-video model results; not independent human gold',
  source_contacts='word100_learned_guide_feedback_20261102/review_contacts/*REAL12_F.jpg',
  expectations={'normal_speed_rest':'should not introduce confident important-error warnings',
   'action_cut':'physical difference or insufficient comparison; not an automatically certified wrong sign',
   'main_blackout':'unobservable time retained, never reported fully verified'},
  no_original_modification=True,no_private_inputs=True))
if __name__=='__main__':main()
