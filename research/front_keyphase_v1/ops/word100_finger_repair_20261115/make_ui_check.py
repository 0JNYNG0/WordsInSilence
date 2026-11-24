from common import *
import shutil,re,subprocess,imageio_ffmpeg,ast,importlib.util
W=O/'web_tutorial_live_integration_20261113';Q=U/'ui_check';Q.mkdir(exist_ok=True)
from projection_extension import project_fingers
spec=importlib.util.spec_from_file_location('display_contract',O/'word100_visual_feedback_g5_20261114/adapter.py');display=importlib.util.module_from_spec(spec);spec.loader.exec_module(display)
GUIDES=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words'];CARDS=read(O/'word100_action_feedback_demo_20261108/CARDS.json')['words']
project=lambda row,guides,frames:project_fingers(display.project,row,guides,frames)
# Execute the production pure presentation function without initializing camera/model modules.
tree=ast.parse((O/'word100_live_g6_20261114/g6_runtime_adapter.py').read_text('utf-8'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='presentation')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'g6_presentation_ui_probe','exec'))
rows=read(U/'REPLAY_RESULTS.json')['rows'];r=next(r for r in rows if r['details']and r['kind']=='right_closed_hand_main')
saved=read(r['output']);z=np.load(saved['source']);x=z['x'];ts=z['ts'];n=len(ts)
cards,coaching,projected=presentation(saved['after'],'WORD1133',ts)
# Renderer-only fixture uses actual saved model evidence; no invented predictions.
# Show synthetic input coordinates, not unchanged RGB that would contradict them.
p=x[:,:102].reshape(-1,51,2);valid=x[:,186:237]>.5
lo=np.min(p[valid],0);hi=np.max(p[valid],0);s=min(580/max(.1,hi[0]-lo[0]),340/max(.1,hi[1]-lo[1]));coords=(p-(lo+hi)/2)*s+np.array([320,190])
xy=np.full((n,133,2),-1.,dtype=float);mapping=[0,5,6,7,8,9,10,11,12]+list(range(91,133))
for j,k in enumerate(mapping):xy[:,k]=np.where(valid[:,j,None],coords[:,j],-1)
duration=float(ts[-1])/1e6+.2
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','lavfi','-i','color=c=0x121a28:s=640x380:r=30','-t',str(duration),'-c:v','libx264','-pix_fmt','yuv420p',str(Q/'canvas.mp4')],check=True)
data=dict(label='잘하다',source='public_synthetic_joint_control',synthetic=True,public_example_name='공개 관절 변형 대조 · 화면 연결 점검',video='canvas.mp4',width=640,height=380,timestamp_us=ts.tolist(),query_xy=xy.tolist(),path=[[i,i]for i in range(n)],cards=cards,coaching=coaching,feedback_version='G6-finger-20261115',training_use=False)
ref=dict(video='canvas.mp4',width=640,height=380,timestamp_us=ts.tolist(),xy=xy.tolist())
write(Q/'fixture.json',dict(data=data,reference=ref))
normal=read(U/'PUBLIC_WEB_RESULT.json');normal['video']='http://127.0.0.1:8821'+normal['video'];normal['public_example_name']='공개 정상 동작 · 최신 웹 실제 분석'
refnormal=read(O/'word100_action_feedback_demo_20261108/REFERENCES.json')['WORD1133'];refnormal['video']='http://127.0.0.1:8821/media/WORD1133.mp4'
write(Q/'normal.json',dict(data=normal,reference=refnormal))
for name in ['feedback.js','live.css']:shutil.copy2(W/'site'/name,Q/name)
dialog=re.search(r'<dialog id="feedback-dialog">.*?</dialog>',(W/'site/index.html').read_text('utf-8'),re.S).group()
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>공개 분석 표시 확인</title><link rel="icon" href="data:,"><link rel="stylesheet" href="live.css"><style>body{background:#eef2f6;color:#233148;font:16px sans-serif;padding:30px}button{padding:12px}</style><h1>공개 분석 표시 확인</h1><p>카메라 사용 없음 · 학습 없음</p><button id="normal">정상 공개 영상</button> <button id="fingers">손가락 강조 표시 점검</button>'''+dialog+'''<script type="module">import {openFeedback} from './feedback.js';async function show(file,fixture){const p=await(await fetch(file)).json();await openFeedback(p.data,p.reference);if(fixture){document.querySelector('.comparison figure:first-child figcaption').textContent='관절 대조 입력 · 표시 확인용 동일 시간축';document.getElementById('query-caption').textContent='합성 관절 입력 · 실제 촬영 오류가 아님';document.getElementById('feedback-source').textContent='공개 관절 변형 시험의 실제 저장 판정 · 영상 대신 관절만 표시 · DTW 검증용 화면 아님';}}document.getElementById('normal').onclick=()=>show('normal.json',false);document.getElementById('fingers').onclick=()=>show('fixture.json',true);</script></html>'''
(Q/'index.html').write_text(html,'utf-8')
write(U/'UI_FIXTURE_MANIFEST.json',dict(source=r['output'],source_sha256=sha(r['output']),files={str(p):sha(p)for p in Q.iterdir()if p.is_file()},purpose='Renderer integration only, real saved evidence and synthetic input coordinates; identical visual time mapping is NOT a DTW test',camera=False))
print(str(Q))
