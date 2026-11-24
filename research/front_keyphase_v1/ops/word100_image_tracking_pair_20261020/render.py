"""Local-only scientific figures and annotated public videos."""
import json,subprocess,html
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
fontpath='C:/Windows/Fonts/malgun.ttf'
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
COLORS=['#0868d1','#bb3d8a'];BGR=[(209,104,8),(138,61,187)]
PALM=[0,5,9,13,17]
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
       (9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def fullframe(cap,fid,w,h):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(fid));ok,im=cap.read();assert ok
    return im[:h,:w].copy()
def draw(im,e,obs,f):
    w=im.shape[1];scale=640/w;out=cv2.resize(im,(640,int(round(im.shape[0]*scale))))
    xy=obs['xy_px'][f]*scale;m=obs['usable'][f]
    for a,b in [(11,12),(11,13),(13,15),(12,14),(14,16)]:
        if m[a] and m[b]:cv2.line(out,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),(110,190,110),1,cv2.LINE_AA)
    for s,off in enumerate([33,54]):
        for a,b in EDGES:
            if m[off+a] and m[off+b]:
                cv2.line(out,tuple(xy[off+a].astype(int)),tuple(xy[off+b].astype(int)),BGR[s],1,cv2.LINE_AA)
        if e['direct'][s] is not None:
            p=tuple((np.array(e['direct'][s])*scale).astype(int));cv2.circle(out,p,4,BGR[s],-1)
        if e['estimate'][s] is not None and e['direct'][s] is None:
            p=tuple((np.array(e['estimate'][s])*scale).astype(int));cv2.circle(out,p,11,(0,170,255),2)
            cv2.putText(out,'L?' if s==0 else 'R?',(p[0]+13,p[1]),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,170,255),1,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(640,27),(22,30,40),-1)
    cv2.putText(out,f"{e['time']:.2f}s  frame {e['frame_id']}   {'ANNOTATED ACTION' if e['core'] else 'BEFORE / AFTER'}",
                (9,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
    return out
def plot(row,x,obs):
    t=np.array([e['time'] for e in x['frames']]);N=len(t)
    p=obs['xy_px'];m=obs['usable']
    shoulders=m[:,11]&m[:,12]
    origin=np.nanmedian((p[shoulders,11]+p[shoulders,12])/2,axis=0)
    scale=float(np.nanmedian(np.linalg.norm(p[shoulders,11]-p[shoulders,12],axis=1)))
    fig,axes=plt.subplots(3,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[2,2,.9]},layout='constrained')
    for s in (0,1):
        direct=np.array([e['direct'][s] if e['direct'][s] is not None else [np.nan,np.nan] for e in x['frames']])
        extra=np.array([e['estimate'][s] if e['direct'][s] is None and e['estimate'][s] is not None else [np.nan,np.nan] for e in x['frames']])
        d=(direct-origin)/scale;q=(extra-origin)/scale
        for k in (0,1):
            axes[k].plot(t,d[:,k],color=COLORS[s],lw=1.8,label=('왼손' if s==0 else '오른손')+' 관측')
            axes[k].scatter(t,q[:,k],marker='D',s=25,facecolors='none',edgecolors='#dc8a00',zorder=5,label='영상 추정' if s==0 else None)
        states=np.array([2 if e['direct'][s] is not None else 1 if e['estimate'][s] is not None else 0 for e in x['frames']])
        for code,color in [(0,'#e4e8ee'),(1,'#e7a52b'),(2,COLORS[s])]:
            axes[2].scatter(t[states==code],np.full((states==code).sum(),1-s),marker='s',s=18,color=color)
    for ax in axes[:2]:
        for a,b in row['intervals']:ax.axvspan(a,b,color='#d6eadc',alpha=.35)
        ax.grid(alpha=.22);ax.legend(loc='upper right',fontsize=9,ncol=3)
    axes[0].set_ylabel('가로 위치 / 어깨 너비');axes[1].set_ylabel('세로 위치 / 어깨 너비\n아래쪽으로 갈수록 +')
    axes[2].set_yticks([0,1],['오른손','왼손']);axes[2].set_ylim(-.5,1.5);axes[2].set_xlabel('영상 시간(초)')
    axes[0].set_title(f"{row['label']} · 공개 촬영자 {row['signer_id']:02d} | 실선=관측, ◇=영상 추정, 회색=비교 위치 없음",loc='left',fontsize=12)
    axes[2].set_xlim(t[0],t[-1]);fig.savefig(U/'plots'/f"{row['clip_id']}.png",dpi=130);plt.close(fig)
def videos_and_plots():
    c=read(U/'FROZEN.json');ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for ri,row in enumerate(c['rows']):
        cid=row['clip_id'];x=read(U/'records'/f'{cid}.json');obs=load(row['views']['sensor']['observation'])
        z=load(row['views']['sensor']['raw']);plot(row,x,obs)
        path=U/'videos'/f'{cid}.mp4'
        h=int(round(int(z['height'][0])*640/int(z['width'][0])));evenh=h+(h%2)
        fps=1/np.median(np.diff([e['time'] for e in x['frames']]))
        cmd=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'640x{evenh}',
             '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
        cap=cv2.VideoCapture(row['video']);frame=0
        for e in x['frames']:
            while frame<=e['frame_id']:
                ok,im=cap.read();assert ok;frame+=1
            f=e['index'];out=draw(im[:int(z['height'][f]),:int(z['width'][f])],e,obs,f)
            if h!=evenh:out=cv2.copyMakeBorder(out,0,1,0,0,cv2.BORDER_CONSTANT)
            proc.stdin.write(out.tobytes())
        cap.release();proc.stdin.close();err=proc.stderr.read().decode();assert proc.wait()==0,err
        print('RENDER',ri+1,cid,flush=True)
def contact_sheets():
    rows={r['clip_id']:r for r in read(U/'FROZEN.json')['rows']}
    targets=[]
    for ep in read(U/'EPISODES.json'):targets.append(dict(**ep,kind='natural'))
    bad=[a for a in read(U/'SYNTHETIC.json') if a['error_shoulder'] is not None and a['error_shoulder']>.1]
    for i,a in enumerate(bad):targets.append(dict(id=9+i,clip_id=a['clip_id'],word=rows[a['clip_id']]['label'],side=a['side'],
        start=a['index'],end=a['index'],kind='synthetic',error_shoulder=a['error_shoulder']))
    font=ImageFont.truetype(fontpath,21);small=ImageFont.truetype(fontpath,16)
    for group in range((len(targets)+2)//3):
        canvas=Image.new('RGB',(1600,960),'#f5f7fa');d=ImageDraw.Draw(canvas)
        for ri,target in enumerate(targets[group*3:group*3+3]):
            row=rows[target['clip_id']];x=read(U/'records'/f"{row['clip_id']}.json")
            raw=load(row['views']['sensor']['raw']);obs=load(row['views']['sensor']['observation']);cap=cv2.VideoCapture(row['video'])
            f=target['start'];e=x['frames'][f];side=target['side'];key='estimate' if target['kind']=='natural' else 'test_estimate'
            center=e[key][side];y=ri*320
            d.text((12,y+7),f"#{target['id']} {target['word']} / {row['signer_id']:02d} / {'왼손' if side==0 else '오른손'} / {e['time']:.3f}s / {target['kind']}",font=font,fill='#172b42')
            im=fullframe(cap,e['frame_id'],int(raw['width'][f]),int(raw['height'][f]))
            annotated=draw(im,e,obs,f);cv2.circle(annotated,tuple((np.array(center)*640/im.shape[1]).astype(int)),13,(0,170,255),2)
            panel=Image.fromarray(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB));panel.thumbnail((640,260))
            canvas.paste(panel,(0,y+46))
            for ci,delta in enumerate([-1,0,1]):
                fi=max(0,min(len(x['frames'])-1,f+delta));entry=x['frames'][fi]
                img=fullframe(cap,entry['frame_id'],int(raw['width'][fi]),int(raw['height'][fi]))
                # Fixed crop around the target location, just a labelled magnifier.
                cx,cy=np.array(center).astype(int);rad=170
                x0=max(0,cx-rad);y0=max(0,cy-rad);x1=min(img.shape[1],cx+rad);y1=min(img.shape[0],cy+rad)
                roi=img[y0:y1,x0:x1].copy();point=entry[key][side]
                if point is not None:cv2.circle(roi,tuple((np.array(point)-[x0,y0]).astype(int)),10,(0,170,255),3)
                if entry['direct'][side] is not None:cv2.circle(roi,tuple((np.array(entry['direct'][side])-[x0,y0]).astype(int)),7,BGR[side],-1)
                panel=Image.fromarray(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB));panel.thumbnail((310,245))
                px=650+ci*315;canvas.paste(panel,(px,y+65))
                d.text((px,y+43),f"{'이전' if delta<0 else '현재' if delta==0 else '다음'} {entry['time']:.3f}s 확대",font=small,fill='#172b42')
            cap.release()
        canvas.save(U/'evidence'/f'review_{group+1}.png')
    (U/'REVIEW_TARGETS.json').write_text(json.dumps(targets,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
if __name__=='__main__':
    cv2.setNumThreads(1)
    contact_sheets() if len(__import__('sys').argv)>1 else videos_and_plots()
