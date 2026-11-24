from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2,av
U=Path(__file__).resolve().parent
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
public={}
with av.open(str(U/'video_only/NIA_SL_WORD1567_REAL02_F.mp4'))as c:
    for f,im in enumerate(c.decode(video=0)):
        if f in [64,96]:public[f]=im.to_ndarray(format='rgb24')
fig,axes=plt.subplots(2,3,figsize=(12,8.2),facecolor='white')
titles=['실제 공개 영상','첫 시도 · 손이 너무 높음','수정 · 몸통 기준으로 연결']
for r,f in enumerate([64,96]):
    im=public[f];h,w=im.shape[:2];im=im[:1000,w//2-500:w//2+500]
    images=[im]
    for d in ['public_approximate_preview','public_approximate_preview_body_axis']:
        a=cv2.cvtColor(cv2.imread(str(U/d/('front_source%03d.png'%f))),cv2.COLOR_BGR2RGB);images.append(a[110:690,70:650])
    for col,im in enumerate(images):
        ax=axes[r,col];ax.imshow(im);ax.axis('off')
        if r==0:ax.set_title(titles[col],fontsize=16,color='#1b304b',pad=12)
        if col==0:ax.text(.02,.03,'같은 순간 · %.2f초'%(f/30),transform=ax.transAxes,color='white',fontsize=12,bbox=dict(facecolor='#1b304b',alpha=.85,pad=4,edgecolor='none'))
fig.suptitle('동작을 옮긴 뒤, 위치가 달라진 원인까지 확인',fontsize=22,color='#1b304b',x=.05,ha='left',y=.99)
fig.text(.05,.02,'몸통축 오류는 수정했습니다. 캐릭터 체형에 따른 얼굴·가슴 위치와 손 접촉은 아직 추가 확인이 필요합니다.',fontsize=12,color='#42566f')
fig.subplots_adjust(left=.035,right=.98,bottom=.07,top=.89,wspace=.035,hspace=.04)
fig.savefig(U/'BODY_AXIS_BEFORE_AFTER_KO.png',dpi=170);plt.close(fig)
print('COMPARISON_READY')
