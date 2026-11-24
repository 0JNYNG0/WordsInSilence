from pathlib import Path
import json,hashlib,av,cv2,numpy as np
from fractions import Fraction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;U=P.parent;D=P/'MANNEQUIN_PUBLIC_PREVIEW'
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',24)
source=[]
with av.open(str(U/'video_only/NIA_SL_WORD1567_REAL02_F.mp4'))as c:
    for f in c.decode(video=0):source.append(f.to_ndarray(format='rgb24'))
assert len(source)==127
def encode(path,images,size):
    with av.open(str(path),'w')as c:
        s=c.add_stream('libx264',rate=30);s.width=size[0];s.height=size[1];s.pix_fmt='yuv420p';s.options={'crf':'20','preset':'fast'}
        for n,im in enumerate(images):
            f=av.VideoFrame.from_ndarray(im,format='rgb24');f.pts=n;f.time_base=Fraction(1,30)
            for packet in s.encode(f):c.mux(packet)
        for packet in s.encode():c.mux(packet)
    with av.open(str(path))as c:
        f=list(c.decode(video=0));assert len(f)==127;assert np.max(np.abs(np.array([float(a.pts*a.time_base)for a in f])-np.arange(127)/30))<1e-6
avatar=[];paired=[]
for n in range(127):
    im=cv2.cvtColor(cv2.imread(str(D/'frames'/('frame%03d.png'%n))),cv2.COLOR_BGR2RGB);avatar.append(im)
    ref=source[n];h,w=ref.shape[:2];ref=cv2.resize(ref[:,(w-h)//2:(w+h)//2],(512,512))
    canvas=Image.new('RGB',(1024,564),'white');canvas.paste(Image.fromarray(ref),(0,52));canvas.paste(Image.fromarray(im),(512,52));draw=ImageDraw.Draw(canvas)
    draw.text((20,10),'실제 공개 영상',font=font,fill='#1b304b');draw.text((532,10),'사람 비율 확인용 마네킹',font=font,fill='#1b304b');paired.append(np.array(canvas))
encode(D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.mp4',avatar,(512,512));encode(P/'SOURCE_VS_MANNEQUIN_30FPS.mp4',paired,(1024,564))
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(2,3,figsize=(12,8.3),facecolor='white')
titles=['공개 참고 영상','기존 캐릭터','사람 비율 시험 마네킹']
for r,f in enumerate([64,96]):
    im=source[f];h,w=im.shape[:2];images=[im[:1000,w//2-500:w//2+500]]
    for folder in [U/'public_approximate_preview_body_axis',D]:
        im=cv2.cvtColor(cv2.imread(str(folder/('front_source%03d.png'%f))),cv2.COLOR_BGR2RGB);images.append(im[110:690,70:650])
    for col,im in enumerate(images):
        ax=axes[r,col];ax.imshow(im);ax.axis('off')
        if r==0:ax.set_title(titles[col],fontsize=16,color='#1b304b',pad=12)
        if col==0:ax.text(.02,.03,'같은 순간 · %.2f초'%(f/30),transform=ax.transAxes,color='white',fontsize=12,bbox=dict(facecolor='#1b304b',alpha=.85,pad=4,edgecolor='none'))
fig.suptitle('사람과 비슷한 비율, 잘 보이는 팔과 손가락',fontsize=22,color='#1b304b',x=.045,ha='left',y=.99)
fig.text(.045,.022,'같은 공개 동작을 적용했습니다. 마네킹은 체형·관절 확인용 예시이며 얼굴 부위 대응과 접촉은 아직 검증 중입니다.',fontsize=11.5,color='#42566f')
fig.subplots_adjust(left=.03,right=.98,bottom=.07,top=.89,wspace=.035,hspace=.04);fig.savefig(P/'SOURCE_STYLIZED_MANNEQUIN_KO.png',dpi=170);plt.close(fig)
assets=[P/'SOURCE_STYLIZED_MANNEQUIN_KO.png',P/'SOURCE_VS_MANNEQUIN_30FPS.mp4',D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.mp4',D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.blend',D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.fbx',P/'PROPORTION_MANNEQUIN_STATIC.blend']
read=lambda p:json.loads(p.read_text())
out=dict(status='PROPORTION_EXAMPLE_COMPLETE_NOT_TEACHING_READY',source='Existing REAL02 WORD1567 estimated XYZ, unchanged',source_frames=127,fps=30,new_pose_extraction=False,new_download=False,private_recording_used=False,
         local_inventory=[dict(asset='Existing NPC Ch29_nonPBR',fingers='30physical +10terminal',decision='Usable rig, stylized head/body proportions limit bodypart correspondence'),
                          dict(asset='Existing Meshy player static geometry',bones=24,finger_bones=0,decision='Not a fine-handshape candidate without additional rigging',source=read(P/'MESHY_STATIC_INSPECTION.json')['source'],sha256=read(P/'MESHY_STATIC_INSPECTION.json')['sha256'],license='Local project asset; terms not established here'),
                          dict(asset='Locally created proportion-test mannequin',decision='Illustrative fallback, not production art or expert-validated avatar',design=read(P/'MANNEQUIN_DESIGN.json'))],
         verification=dict(fbx=read(D/'FBX_TRANSPORT_VERIFIED.json'),bake={k:v for k,v in read(D/'BAKE_RECORD.json').items()if k!='frame_records'},mp4_frame_count=127,mp4_native_timestamps=True,
             hand_mapping='New rest palm basis and source palm basis recomputed; no NPC synthetic flexion sign reused',
             visual_checks='Front and oblique64/96 thumb/palm and arm alternation inspected by child/root; full anatomical contact/collision accuracy not established'),
         changes='Designed human-like torso/shoulder/neck/arm/palm/finger proportions, not an isolated shrink of original head; connected mannequin shoulder geometry before final example',
         limitations=['Camera3D still approximate','Not a learned animation generator','Face-bodypart anchoring still differs at some moments','Not all thumb poses/collisions/contact frames independently validated','Primitive rigid pieces show seams; not final game character'],
         hdd_needed_for_playback=False,original_assets_modified=False,
         artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in assets])
(P/'EVIDENCE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8')
text='''사람 비율 아바타 예시 준비 완료

같은 실제 공개 ‘조상’ 동작127프레임을 기존 캐릭터와 사람 비율 시험 마네킹에 적용했다. 새 관절 검출이나 모델 다운로드는 하지 않았다.

기존 자료 확인
- 기존 NPC: 손가락 뼈가 있어 동작을 옮길 수 있지만 큰 머리/짧은 몸통 비율 때문에 손과 얼굴의 위치 관계가 달라진다.
- 프로젝트의 다른 Meshy 플레이어: 정적 메시만 읽어 확인했다. 뼈24개, 손가락 뼈0개여서 지금 그대로는 자세한 손모양을 표시하기 어렵다. 원래 동작 파일의 애니메이션은 불러오지 않았다. 파일명만으로 배포 권리를 단정하지 않았다.
- 따라서 새 모델을 내려받는 대신, 어깨·목·몸통·팔·손 크기를 함께 지정한 간단한 사람 비율 마네킹을 로컬에서 만들었다. 완성 게임 캐릭터가 아니라 비율과 관절 가동을 보여주는 예시다.

결과
양손의 엄지가 펴지고 나머지 손가락이 접히는 모습, 양팔이 번갈아 올라가는 흐름이 더 잘 보인다. 새 리그의 손바닥 기준을 다시 계산해 연결했고, 합성 시험의 굽힘 부호를 그대로 사용하지 않았다.
127프레임30fps 영상과 FBX 재가져오기 검사를 통과했다. 원본과 마네킹을 한 영상에 나란히 넣어 비교할 수 있다.

남은 점
체형이 더 비슷해도 얼굴의 이마·코 등 정확한 신체 부위 대응이 자동으로 해결되는 것은 아니다. 일부 순간에는 원본과 손 높이/얼굴 관계가 다르다. 관절 추정, 카메라 투영, 손가락 크기 및 신체부위 기준의 리타게팅을 더 확인해야 한다. 아직 교육용 정답 동작으로 쓰는 단계는 아니다. 자기충돌과 접촉을 모든 프레임에서 검증한 것도 아니다.

확인 파일
SOURCE_STYLIZED_MANNEQUIN_KO.png: 공개 원본 / 기존 캐릭터 / 사람 비율 예시, 같은64·96프레임
SOURCE_VS_MANNEQUIN_30FPS.mp4: 실제 공개 영상과 마네킹의 전체 동작 비교
MANNEQUIN_PUBLIC_PREVIEW/APPROXIMATE_PUBLIC_WORD1567_REAL02.mp4: 마네킹 단독 재생
같은 폴더.blend/.fbx: 저장된 실제 캐릭터 동작

위 재생 파일은 C:에 있으며 HDD 없이 볼 수 있다. 기존 Assets·운영 웹·C·T3-F는 변경하지 않았다. 개인 촬영은 사용하지 않았다. 다음에는 사람형 캐릭터를 고를 때 머리 크기뿐 아니라 팔·손 비율, 독립 손가락 뼈, 손목 회전, 얼굴·가슴 기준 위치가 읽히는지를 함께 확인하면 된다.
'''
(P/'RESULT_KO.txt').write_text(text,'utf-8');print('EXAMPLES_PACKAGED',len(assets))
