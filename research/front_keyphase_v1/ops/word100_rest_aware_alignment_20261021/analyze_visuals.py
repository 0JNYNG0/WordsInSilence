"""Read-only outcome diagnosis and raw-source correspondence review sheets."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import study as d
U=d.U
f=d.read(U/'FROZEN.json');res=d.read(U/'RESULT.json')
rows={r['clip_id']:r for r in f['rows']}
samples={cid:d.sample(r) for cid,r in rows.items()}
summaries={};changed=[];selections=[];panels=[]
(U/'visuals').mkdir()
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14)
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

def add_selection(p,i,reason,side=0):
    key=(p['query'],p['reference'],int(i))
    if not any((a['query'],a['reference'],a['index'])==key for a in selections):
        selections.append(dict(query=p['query'],reference=p['reference'],index=int(i),reason=reason,side=side))

for label in ('수어','후원','멀다'):
    pp=[p for p in res['pairs'] if p['label']==label]
    summaries[label]=dict(pairs=len(pp),changed_paths=sum(p['path_changed'] for p in pp),
       insufficient_alignment=[sum(p[k]<.35 for p in pp) for k in ('old_support','support_only','new_support')],
       active_to_rest={c:sum(p[c+'_active_to_rest_s'] for p in pp) for c in ('baseline','rest_alignment')},
       rest_to_active={c:sum(p[c+'_rest_to_active_s'] for p in pp) for c in ('baseline','rest_alignment')})

for p in res['pairs']:
    if not p['path_changed']:continue
    pid=p['query']+'__'+p['reference'];q=samples[p['query']];r=samples[p['reference']]
    old=d.load(U/'pairs'/('baseline__'+pid+'.npz'));new=d.load(U/'pairs'/('rest_alignment__'+pid+'.npz'))
    qo,ro=old['path'].T;qn,rn=new['path'].T;tq=q['timestamp_us']/1e6;tr=r['timestamp_us']/1e6
    oldmap=np.array([np.median(tr[ro[qo==i]]) for i in range(len(tq))])
    newmap=np.array([np.median(tr[rn[qn==i]]) for i in range(len(tq))])
    newflags={k:new[k]&~old[k] for k in ('active_to_rest','rest_to_active')}
    counts={}
    for k,flags in newflags.items():
        counts[k]=int(flags.sum())
        for h in (0,1):
            ids=np.flatnonzero(flags[:,h])
            for seg in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
                if len(seg):
                    for i in (seg[0],seg[len(seg)//2],seg[-1]):add_selection(p,i,'new '+k,h)
    i=int(np.argmax(abs(newmap-oldmap)));add_selection(p,i,'largest correspondence shift')
    oj=d.read(U/'pairs'/('baseline__'+pid+'.json'));nj=d.read(U/'pairs'/('rest_alignment__'+pid+'.json'))
    changed.append(dict(query=p['query'],reference=p['reference'],maximum_shift_s=float(abs(newmap-oldmap).max()),
      mean_shift_s=float(abs(newmap-oldmap).mean()),new_conflict_frames=counts,
      max_refs_per_query_before=oj['max_reference_rows_per_query'],max_refs_per_query_after=nj['max_reference_rows_per_query']))
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(tq[qo],tr[ro],label='기존 연결',color='#246ccc',linewidth=2)
    ax.plot(tq[qn],tr[rn],label='손내림 반영 연결',color='#d65544',linestyle='--')
    sch=d.load(d.R/'schedules'/(pid+'.npz'))
    for ids in np.split(np.flatnonzero(sch['rest'][:,0]),np.flatnonzero(np.diff(np.flatnonzero(sch['rest'][:,0]))>1)+1):
        if len(ids):ax.axhspan(tr[ids[0]]-.0167,tr[ids[-1]]+.0167,color='#e7bd36',alpha=.25)
    ax.set(xlabel='비교 영상 시간(초)',ylabel='연결된 참고 영상 시간(초)',title=p['query'].split('_')[-2]+' → '+p['reference'].split('_')[-2]+' / 후원')
    ax.legend();fig.tight_layout();fig.savefig(U/'visuals'/(pid+'_path.png'),dpi=130);plt.close(fig)

# Positive nonchanged controls: closest two observed palms in core, not a claim of anatomical truth.
for label in ('수어','멀다'):
    p=next(p for p in res['pairs'] if p['label']==label)
    q=samples[p['query']];obs=d.load(rows[p['query']]['observation'])
    core=np.flatnonzero(q['core']);dist=[]
    for i in core:
        a=obs['xy_px'][i,33:54];b=obs['xy_px'][i,54:75]
        va=obs['usable'][i,33:54];vb=obs['usable'][i,54:75]
        dist.append(np.linalg.norm(a[va].mean(0)-b[vb].mean(0)) if va.any() and vb.any() else np.inf)
    i=core[int(np.argmin(dist))] if np.isfinite(dist).any() else core[len(core)//2]
    add_selection(p,i,'unchanged normal/crossing control')

def raw_image(cid,i):
    row=rows[cid];q=samples[cid];frame=int(q['frame_id'][i]);path=Path(row['video']);assert path.is_file()
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();cap.release();assert ok,(cid,frame)
    return Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((480,270)),frame

for num,s in enumerate(selections):
    cid,rid,i=s['query'],s['reference'],s['index'];pid=cid+'__'+rid;q=samples[cid];r=samples[rid]
    old=d.load(U/'pairs'/('baseline__'+pid+'.npz'));new=d.load(U/'pairs'/('rest_alignment__'+pid+'.npz'))
    oo=old['path'][old['path'][:,0]==i,1];nn=new['path'][new['path'][:,0]==i,1]
    jo=int(oo[len(oo)//2]);jn=int(nn[len(nn)//2])
    if s['reason'].startswith('new active_to_rest'):
        sch=d.load(d.R/'schedules'/(pid+'.npz'));hit=nn[sch['rest'][nn,s['side']]]
        if len(hit):jn=int(hit[len(hit)//2])
    card=Image.new('RGB',(1440,350),'#102030');dr=ImageDraw.Draw(card)
    for col,(cc,ii,title) in enumerate([(cid,i,'비교 장면'),(rid,jo,'기존에 연결한 참고 장면'),(rid,jn,'새로 연결한 참고 장면')]):
        im,frame=raw_image(cc,ii);card.paste(im,(col*480,28));dr.text((col*480+8,4),title,font=font,fill='white')
        dr.text((col*480+8,301),cc.split('_')[-2]+f' f{frame} / {samples[cc]["timestamp_us"][ii]/1e6:.3f}초',font=font,fill='white')
    dr.text((8,328),f'{num+1:02d} {rows[cid]["label"]} · {s["reason"]} · side {s["side"]}',font=small,fill='#bdd2ea')
    s.update(old_reference_index=jo,new_reference_index=jn,
      query_frame=int(q['frame_id'][i]),old_reference_frame=int(r['frame_id'][jo]),new_reference_frame=int(r['frame_id'][jn]),
      query_time_s=float(q['timestamp_us'][i]/1e6),old_reference_time_s=float(r['timestamp_us'][jo]/1e6),new_reference_time_s=float(r['timestamp_us'][jn]/1e6))
    panels.append(card)
pages=[]
for k in range(0,len(panels),3):
    sheet=Image.new('RGB',(1440,350*len(panels[k:k+3])),'#102030')
    for j,im in enumerate(panels[k:k+3]):sheet.paste(im,(0,j*350))
    path=U/'visuals'/f'raw_{k//3+1:02d}.jpg';sheet.save(path,quality=94);pages.append(str(path))

transitions={}
for a,b in [('baseline','support_only'),('support_only','rest_alignment'),('baseline','rest_alignment')]:
    c=Counter();delta=[]
    for cid in rows:
        aa=d.read(U/'queries'/(a+'__'+cid+'.json'))['rows'];bb=d.read(U/'queries'/(b+'__'+cid+'.json'))['rows']
        for x,y in zip(aa,bb):
            c[x['status']+' -> '+y['status']]+=1
            if x['compared_median_deg'] is not None and y['compared_median_deg'] is not None:
                delta.append(abs(y['compared_median_deg']-x['compared_median_deg']))
    transitions[a+'__'+b]=dict(statuses=dict(c),changed_medians=sum(v>1e-9 for v in delta),max_median_change_deg=max(delta,default=0))
diagnosis=dict(words=summaries,changed_pairs=changed,transitions=transitions,
  new_active_to_rest_handframes=sum(x['new_conflict_frames']['active_to_rest'] for x in changed),
  units='Conflict seconds are query-time cells summed across repeated reference pairs, not unique video duration; phase witnesses are geometry-derived, not sign semantic ground truth.',
  no_new_training=True)
d.write(U/'DIAGNOSIS.json',diagnosis)
d.write(U/'VISUAL_MANIFEST.json',dict(panels=selections,pages=pages,policy='All newly-added conflict runs endpoints/middle and largest shift each changed pair; unchanged two-word controls. Raw full original RGB, no overlays or image generation.'))
print('DIAGNOSIS',json.dumps(diagnosis,ensure_ascii=False));print('VISUALS',len(panels),len(pages))
