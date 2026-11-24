"""Additional arithmetic and public RGB comparisons; no model execution."""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
U=Path(__file__).resolve().parent; O=U.parent; B=O/'word100_body_boundary_20261017'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 p=U/n; assert not p.exists(); p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
inp=read(U/'INPUT_ANALYSIS.json'); head=read(U/'HEAD_DECOMPOSITION.json'); scope=read(U/'SCOPE.json')
ctx=read(B/'training_CONTEXT.json'); rows={r['clip_id']:r for r in ctx['ordered_rows']}
timing=[]
for cid in sorted({x['clip_id'] for x in inp['rows'].values()}):
 z={v:load(rows[cid]['views'][v]['channel_file']) for v in ['masked','sensor']}
 uu={}; ww={}
 for v,a in z.items():
  dt=a['x_whole'][:,279].astype(np.float64); elapsed=np.cumsum(dt); u=(elapsed-elapsed[0])/(elapsed[-1]-elapsed[0]); uu[v]=u
  ww[v]=np.stack([np.maximum(1-2*u,0),1-np.abs(2*u-1),np.maximum(2*u-1,0)],axis=1)
 common,im,ise=np.intersect1d(z['masked']['whole_frame_id'],z['sensor']['whole_frame_id'],return_indices=True)
 assert len(common)>0
 d=np.abs(uu['masked'][im]-uu['sensor'][ise]); dw=np.abs(ww['masked'][im]-ww['sensor'][ise])
 timing.append({'clip_id':cid,'label':inp['rows'][cid+'/sensor']['label'],'shared_frames':len(common),'max_relative_time_shift':float(d.max()),'mean_relative_time_shift':float(d.mean()),'max_soft_weight_change':float(dw.max()),'frame_ranges':{v:[int(z[v]['whole_frame_id'][0]),int(z[v]['whole_frame_id'][-1])] for v in z}})
save('TIME_POSITION_AUDIT.json',{'status':'DESCRIPTIVE_NO_RETUNING','rows':timing,'meaning':'Compare existing views at the same original frame. This is an arithmetic difference, not a causal classification ablation; no input or score was changed.'})

# Stored results, independently aggregated without imported experiment functions.
stats={}; changes={}
for view in ['sensor','masked']:
 stats[view]={}; changes[view]=[]
 for name,folder,prefix in [('C',B,'training_C'),('T3',O/'word100_temporal_phase_training_20261019','T3')]:
  counts=[]; reentry=[]
  for seed in [20261013,20261014,20261015]:
   result=read(folder/f'{prefix}_{seed}_RESULT.json'); pp=result['predictions'][view]
   assert len(pp)==300
   correct=[int(np.argmax(p['logits']))==p['label_index'] for p in pp] if 'label_index' in pp[0] else [int(np.argmax(p['logits']))==rows[p['clip_id']]['label_index'] for p in pp]
   assert all(bool(c)==p['correct'] for c,p in zip(correct,pp))
   counts.append(sum(correct)); reentry.append(sum(c for c,p in zip(correct,pp) if ctx['missingness_groups'][view][p['clip_id']]=='internal_reentry'))
  stats[view][name]={'counts':counts,'mean_percent':sum(counts)/9,'reentry_correct':reentry,'reentry_n_per_seed':sum(g=='internal_reentry' for g in ctx['missingness_groups'][view].values())}
 for r in head['rows']:
  assert abs(r['global_margin']+sum(r['phase_margin_by_early_middle_late'])-r['stored_total_margin'])<2e-4
  if r['view']==view and r['group']=='internal_reentry' and r['stored_C_correct']!=r['stored_T3_correct']:changes[view].append(r)
pins={**inp['pins'],**head['pins']}
for p,h in pins.items(): assert sha(p)==h,p
assert all(r['role'] in ('train_candidate','development_only') for r in inp['rows'].values())
assert len(head['rows'])==522 and len({r['clip_id'] for r in head['rows']})==87
assert all(rows[r['clip_id']]['role']=='development_only' for r in head['rows'])
save('ARITHMETIC_VERIFICATION.json',{'status':'PASS','operator':'same Codex, separate arithmetic; not an independent human/agent','checked_source_pins':len(pins),'decomposition_rows':522,'selected_DEV_clips':87,'public_input_clips':132,'stats':stats,'reentry_changes':changes,'max_saved_logit_reconstruction_error':head['max_saved_logit_reconstruction_error'],'all_decomposition_margin_sums_match':True,'no_training':True,'user_mentor_data_accessed':False})

font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20); small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
manifest=read(U/'VISUAL_MANIFEST.json'); video={x['clip_id']:x for x in manifest['sheets']}
targets={'NIA_SL_WORD1675_REAL13_F':[134,139,145,146],'NIA_SL_WORD1545_REAL12_F':[42,55,70,80],'NIA_SL_WORD2768_REAL12_F':[86,95,105,106]}
extra=[]
for cid,frames in targets.items():
 rec=read(B/'data/records'/f'{cid}.json'); obs={v:load(rec['views'][v]['observation']) for v in ['masked','sensor']}
 path=Path(video[cid]['video']); assert sha(path)==rec['source']['sha256']
 cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS); sheet=Image.new('RGB',(1600,660),'white');draw=ImageDraw.Draw(sheet)
 draw.text((10,8),video[cid]['word']+' · 같은 원본 장면에서 두 저장 관측 비교',font=font,fill='black')
 draw.text((10,39),'위: 전체 화면 검출 후 범위 제한(masked) / 아래: 화면 제한 후 검출(sensor)',font=small,fill='black')
 used=[]
 for k,f in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read(); assert ok
  for vi,v in enumerate(['masked','sensor']):
   im=img.copy();j=int(np.argmin(abs(obs[v]['frame_id']-f)));assert obs[v]['frame_id'][j]==f
   xy=obs[v]['xy_px'][j];valid=obs[v]['usable'][j]
   for st,col in [(33,(40,255,40)),(54,(240,40,240))]:
    for i in range(st,st+21):
     if valid[i]:cv2.circle(im,tuple(np.rint(xy[i]).astype(int)),5,col,-1)
   bottom=rec['calibration']['bottom'];cv2.line(im,(0,bottom),(im.shape[1]-1,bottom),(0,220,255),3)
   pim=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));pim.thumbnail((400,225));sheet.paste(pim,(k*400,80+vi*280))
   draw.text((k*400+4,310+vi*280),f'{v} f{f} {f/fps:.2f}s L{valid[33:54].sum()} R{valid[54:75].sum()}',font=small,fill='black')
   used.append({'view':v,'frame':f,'left_points':int(valid[33:54].sum()),'right_points':int(valid[54:75].sum())})
 cap.release();p=U/'visual'/f'{cid}_paired.jpg';sheet.save(p,quality=92);extra.append({'clip_id':cid,'sheet':str(p),'sha256':sha(p),'frames':used})
save('PAIRED_VISUAL_MANIFEST.json',{'public_videos':3,'unique_source_frames':12,'panels':24,'sheets':extra,'new_detector_inference':0,'new_model_inference':0})

plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for ax,v,title in zip(axs,['sensor','masked'],['화면 제한 후 검출','전체 화면 검출 후 범위 제한']):
 arr=[r for r in inp['rows'].values() if r['label']=='상처' and r['view']==v and r['role']=='train_candidate']
 for j,side in enumerate(['오른손','왼손']):
  ix=1-j;values=[r['wrist_motion'][ix]['relative_time_motion_center']*100 for r in arr]
  ax.scatter(values,[j]*len(values),color='#4b7baa',alpha=.75,s=42,label='학습 9개 영상' if j==0 else None)
  x=inp['rows']['NIA_SL_WORD1519_REAL12_F/'+v]['wrist_motion'][ix]['relative_time_motion_center']*100
  ax.scatter([x],[j],marker='D',color='#d14f40',s=90,label='문제 영상: 촬영자 12' if j==0 else None)
  ax.text(x,j+.12,f'{x:.1f}',ha='center',color='#ab352a')
 ax.set(xlim=(0,100),ylim=(-.45,1.5),yticks=[0,1],yticklabels=['오른손','왼손'],xlabel='유효한 손목 이동이 몰린 상대 시간 (시작 0 → 끝 100)',title=title)
 ax.grid(axis='x',alpha=.2);ax.legend(loc='upper left',fontsize=9)
fig.suptitle('상처: 같은 단어라도 좌우 움직임이 집중되는 시점이 다름',fontsize=16)
fig.savefig(U/'timing_plot.png',dpi=160);plt.close(fig)
print(json.dumps({'stats':stats,'source_pins':len(pins),'focus_time_shift':[x for x in timing if x['clip_id'] in scope['focus_cases']]},ensure_ascii=False))
