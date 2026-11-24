from prepare import *
import av
def main():
 v=U/'source_corrected';(v/'confirmation').mkdir(exist_ok=True)
 d=read(v/'DEVELOPMENT.json');summary={}
 for kind in ['frame_mlp','context_gru']:
  rr=[r for r in d['rows']if r['split']=='development'and r['kind']==kind]
  summary[kind]={k:sum(r[k]for r in rr)for k in ['anchors','correct','wrong','unknown']}
 decision=dict(chosen='frame_mlp',reason='Same corrected held-person anchor accuracy32/35; fewer definite mistakes (1 vs2), no extra recurrent layer needed. Frozen100 Conv+GRU still supplies temporal input.',
  summary=summary,qualification='candidate only; repeat observation must pass held-person probes; full-sign success still unqualified',
  confirmation_people=[14,15],exposure='Both clips used in earlier finger research. Not globally unseen: withheld from THIS phase-head training and selection only.',
  source_truth_sha=sha(v/'SOURCE_TRUTH.json'),models={str(v/'runs'/f'frame_mlp_{s}.pt'):sha(v/'runs'/f'frame_mlp_{s}.pt')for s in [42,43,44]},
  expected='two distinct approaches supported on normal, zero on constant/missing, <=1 on one-approach prefix, timestamps invariant. No full-lesson promotion from these alone.')
 assert not (v/'CONFIRMATION_FROZEN.json').exists();write(v/'CONFIRMATION_FROZEN.json',decision)
 inventory=read(O/'demo_scope_portability_review_20261113/INVENTORY.json')['targets']['WORD1290'];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18);rows=[]
 for pid in [14,15]:
  p=O/'word100_finger_center_temporal_20261115/raw'/f'NIA_SL_WORD1290_REAL{pid}_F.json';r=read(p)
  for path,key in [('path','sha256'),('raw_path','raw_sha256'),('channel_path','channel_sha256')]:assert sha(r[path])==r[key]
  i=next(t for t in inventory if t['record_id']==pid);lo,hi=i['metadata']['intervals'][0]
  frames=list(range(max(0,round(lo*30)-4),min(r['frames']-1,round(hi*30)+12),3));pics={}
  with av.open(r['path'])as c:
   for f,fr in enumerate(c.decode(video=0)):
    if f in frames:
     rgb=fr.to_ndarray(format='rgb24');h,w=rgb.shape[:2];pics[f]=Image.fromarray(rgb[int(h*.22):int(h*.62),int(w*.26):int(w*.72)])
  for page in range((len(frames)+11)//12):
   ff=frames[page*12:page*12+12];board=Image.new('RGB',(1600,900),'white');draw=ImageDraw.Draw(board)
   for k,f in enumerate(ff):
    im=pics[f];im.thumbnail((397,263));x=k%4*400;y=k//4*300;board.paste(im,(x,y));draw.text((x+5,y+267),f'P{pid} native f{f}',font=font,fill='#152c46')
   board.save(v/'confirmation'/f'{pid}_{page}.jpg',quality=95)
  rows.append(dict(clip_id=r['clip_id'],word='WORD1290',person=pid,video=r['path'],source_sha256=r['sha256'],raw=r['raw_path'],raw_sha256=r['raw_sha256'],channels=r['channel_path'],channels_sha256=r['channel_sha256'],frames=r['frames'],source_interval=[lo,hi]))
 write(v/'CONFIRMATION_INPUTS.json',{'rows':rows,'predictions_opened':False});print('CONFIRMATION SOURCES READY')
if __name__=='__main__':main()
