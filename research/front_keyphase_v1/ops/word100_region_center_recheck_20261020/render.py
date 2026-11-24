"""Render only saved public observations and candidate diagnostics."""
import json,subprocess,importlib.util
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
U=Path(__file__).resolve().parent;V=U/'revised';P=U.parent/'word100_image_tracking_pair_20261020'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
spec=importlib.util.spec_from_file_location('old_visual_only',P/'render.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
BLUE=(209,104,8);PINK=(138,61,187);ORANGE=(0,150,240);GREEN=(100,175,20);PURPLE=(205,60,170)
def overlay(image,e,obs,mode='natural'):
    copied=dict(e,estimate=[None,None]);out=old.draw(image,copied,obs,e['index'])
    o=e[mode];fac=e['factor']
    oldest=e['old_estimate'] if mode=='natural' else e['old_test_estimate']
    for s in (0,1):
        show=mode!='natural' or e['direct'][s] is None
        if show and oldest[s] is not None:
            p=tuple((np.array(oldest[s])*fac).astype(int));cv2.drawMarker(out,p,ORANGE,cv2.MARKER_TILTED_CROSS,13,2)
        region=o['region'][s]
        if show and region is not None:
            cv2.polylines(out,[np.array(region['hull'],np.int32)],True,PURPLE,1,cv2.LINE_AA)
        if show and o['center_estimate'][s] is not None:
            p=tuple(np.array(o['center_estimate'][s]).astype(int));cv2.circle(out,p,7,GREEN,2)
    return out
def graphs(row,x,obs):
    ts=np.array([e['time'] for e in x['frames']]);a=obs['xy_px'];m=obs['usable']
    sm=m[:,11]&m[:,12];origin=np.median((a[sm,11]+a[sm,12])/2,0);scale=np.median(np.linalg.norm(a[sm,11]-a[sm,12],axis=1))
    fig,axes=plt.subplots(3,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[2,2,1]},layout='constrained')
    for s,color in [(0,'#0868d1'),(1,'#bb3d8a')]:
        d=np.array([e['direct'][s] if e['direct'][s] is not None else [np.nan]*2 for e in x['frames']])
        oldp=np.array([e['old_estimate'][s] if e['direct'][s] is None and e['old_estimate'][s] is not None else [np.nan]*2 for e in x['frames']])
        newp=np.array([np.array(e['natural']['center_estimate'][s])/e['factor'] if e['direct'][s] is None and e['natural']['center_estimate'][s] is not None else [np.nan]*2 for e in x['frames']])
        for k in (0,1):
            axes[k].plot(ts,(d[:,k]-origin[k])/scale,color=color,label=('왼손' if s==0 else '오른손')+' 관측',lw=1.6)
            axes[k].scatter(ts,(oldp[:,k]-origin[k])/scale,marker='x',s=38,color='#c58211',label='이전 추정' if s==0 else None,zorder=4)
            axes[k].scatter(ts,(newp[:,k]-origin[k])/scale,marker='o',s=52,facecolors='none',edgecolors='#258c47',label='새 중심 추정' if s==0 else None,zorder=5)
        for state,col in [('observed',color),('estimated','#258c47'),('region_only','#ae4cb5'),('missing','#e2e6eb')]:
            mask=np.array([e['natural']['source'][s]==state for e in x['frames']])
            axes[2].scatter(ts[mask],np.full(mask.sum(),1-s),color=col,marker='s',s=18)
    for ax in axes[:2]:
        for a,b in row['intervals']:ax.axvspan(a,b,color='#d5eada',alpha=.4)
        ax.grid(alpha=.2);ax.legend(loc='upper right',ncol=4,fontsize=8)
    axes[0].set_ylabel('가로 / 어깨너비');axes[1].set_ylabel('세로 / 어깨너비\n아래로 갈수록 +')
    axes[2].set_yticks([0,1],['오른손','왼손']);axes[2].set_ylim(-.5,1.5);axes[2].set_xlabel('영상 시간(초)')
    axes[0].set_title(f"{row['label']} · 촬영자{row['signer_id']:02d} | 보라=영역만, 회색=중심 미확인",loc='left')
    axes[2].set_xlim(ts[0],ts[-1]);fig.savefig(U/'plots'/f"{row['clip_id']}.png",dpi=135);plt.close(fig)
def select_targets():
    targets={}
    def add(cid,s,f,why,mode='natural'):
        key=(cid,s,f,mode)
        if key not in targets:targets[key]=dict(clip_id=cid,side=s,index=f,mode=mode,reasons=[])
        targets[key]['reasons'].append(why)
    for a in read(V/'NATURAL.json'):add(a['clip_id'],a['side'],a['index'],'new_missing_position_'+a['state'])
    for a in read(P/'EPISODES.json'):add(a['clip_id'],a['side'],a['start'],'old_natural_support')
    sy=read(V/'SYNTHETIC.json')
    for a in sy:
        if a['old_error'] is not None and a['old_error']>.1:add(a['clip_id'],a['side'],a['index'],'old_high_proxy_error','test')
        if a['new_error'] is not None and a['new_error']>.1:add(a['clip_id'],a['side'],a['index'],'new_high_proxy_error','test')
    both=[a for a in sy if a['old_error'] is not None and a['new_error'] is not None]
    for a in sorted(both,key=lambda a:a['new_error']-a['old_error'],reverse=True)[:3]:add(a['clip_id'],a['side'],a['index'],'largest_regression','test')
    for a in sorted(both,key=lambda a:a['old_error']-a['new_error'],reverse=True)[:3]:add(a['clip_id'],a['side'],a['index'],'largest_improvement','test')
    rr=read(V/'RECHECKS.json')
    for kind in ['reappearance','continuous']:
        pool=[a for a in rr if a['event']==kind and a['old_estimate_discrepancy'] is not None]
        for a in sorted(pool,key=lambda a:-a['old_estimate_discrepancy'])[:2]:add(a['clip_id'],a['side'],a['index'],kind+'_recheck')
    out=[dict(v,id=i) for i,v in enumerate(targets.values())]
    (U/'REVIEW_TARGETS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    return out
def contacts(rows,targets):
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    for start in range(0,len(targets),3):
        subset=targets[start:start+3];canvas=Image.new('RGB',(1500,320*len(subset)),'#f4f7fa');d=ImageDraw.Draw(canvas)
        for ri,t in enumerate(subset):
            row=rows[t['clip_id']];x=read(V/'records'/f"{t['clip_id']}.json");obs=load(row['views']['sensor']['observation'])
            cap=cv2.VideoCapture(row['video']);e=x['frames'][t['index']];s=t['side'];mode=t['mode'];y=ri*320
            d.text((10,y+6),f"#{t['id']} {row['label']} / {row['signer_id']:02d} / {'왼손' if s==0 else '오른손'} / {e['time']:.3f}s / {mode}",font=font,fill='#18334a')
            im=old.fullframe(cap,e['frame_id'],row['box'][2],row['box'][3])
            full=overlay(im,e,obs,mode);panel=Image.fromarray(cv2.cvtColor(full,cv2.COLOR_BGR2RGB));panel.thumbnail((570,265));canvas.paste(panel,(0,y+42))
            o=e[mode];center=e['direct'][s] or e['old_estimate'][s]
            if center is None and o['region'][s] is not None:center=(np.array(o['region'][s]['center'])/e['factor']).tolist()
            assert center is not None
            cx,cy=np.array(center).astype(int);rad=180
            for ci,offset in enumerate([-1,0,1]):
                g=min(len(x['frames'])-1,max(0,t['index']+offset));q=x['frames'][g];qo=q[mode];fac=q['factor']
                image=old.fullframe(cap,q['frame_id'],row['box'][2],row['box'][3])
                x0=max(0,cx-rad);y0=max(0,cy-rad);x1=min(image.shape[1],cx+rad);y1=min(image.shape[0],cy+rad)
                crop=image[y0:y1,x0:x1].copy()
                base=q['old_estimate'][s] if mode=='natural' else q['old_test_estimate'][s]
                if base is not None:cv2.drawMarker(crop,tuple((np.array(base)-[x0,y0]).astype(int)),ORANGE,cv2.MARKER_TILTED_CROSS,18,3)
                region=qo['region'][s]
                if region is not None:cv2.polylines(crop,[((np.array(region['hull'])/fac)-[x0,y0]).astype(np.int32)],True,PURPLE,2)
                if qo['center_estimate'][s] is not None:cv2.circle(crop,tuple((np.array(qo['center_estimate'][s])/fac-[x0,y0]).astype(int)),11,GREEN,3)
                if q['direct'][s] is not None:cv2.circle(crop,tuple((np.array(q['direct'][s])-[x0,y0]).astype(int)),6,BLUE if s==0 else PINK,-1)
                panel=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));panel.thumbnail((300,245));px=580+ci*305
                canvas.paste(panel,(px,y+65));d.text((px,y+40),f"{q['time']:.3f}s · {qo['source'][s]}",font=font,fill='#18334a')
            cap.release()
        canvas.save(U/'evidence'/f"review_{start//3+1}.png")
def main():
    for name in ['plots','videos','evidence']:(U/name).mkdir(exist_ok=True)
    rows=read(V/'FROZEN.json')['rows'];contacts({r['clip_id']:r for r in rows},select_targets())
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    for i,row in enumerate(rows):
        x=read(V/'records'/f"{row['clip_id']}.json");obs=load(row['views']['sensor']['observation']);graphs(row,x,obs)
        height=round(row['box'][3]*640/row['box'][2]);even=height+height%2
        fps=1/np.median(np.diff([e['time'] for e in x['frames']]))
        proc=subprocess.Popen([ff,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'640x{even}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(U/'videos'/f"{row['clip_id']}.mp4")],stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
        cap=cv2.VideoCapture(row['video']);f=0
        for e in x['frames']:
            while f<=e['frame_id']:ok,im=cap.read();assert ok;f+=1
            out=overlay(im[:row['box'][3],:row['box'][2]],e,obs)
            if even!=height:out=cv2.copyMakeBorder(out,0,1,0,0,cv2.BORDER_CONSTANT)
            proc.stdin.write(out.tobytes())
        cap.release();proc.stdin.close();err=proc.stderr.read().decode();assert proc.wait()==0,err
        print('RENDER',i+1,row['clip_id'],flush=True)
if __name__=='__main__':cv2.setNumThreads(1);main()
