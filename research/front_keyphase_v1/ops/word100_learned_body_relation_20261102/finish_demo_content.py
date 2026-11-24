from pathlib import Path
import json,numpy as np
from relation_model import V,read
from train_relations import save
U=V.parent/'word100_learned_feature_retention_20261102'
js=(V/'demo.js').read_text(encoding='utf8')
js=js.replace('const featNames={',"const featNames={IMAGE_ESTIMATE_NEEDS_CONFIRMATION:'영상·앞뒤 흐름으로 추정 · 확인 필요',")
js=js.replace("(f.observed_mean>=.5?'있음':'없음')","(f.observed_mean===null?'확인 불가':f.observed_mean>=.5?'있음':'없음')")
js+="""\n$('publicCut').onclick=async()=>{try{let w=$('word').value;let r=await(await fetch('/api/control/'+w)).json();if(r.error)throw Error(r.error);clearStream();$('query').src='/media/control/'+w;$('queryLabel').textContent='공개 영상의 주요 구간을 잘라낸 시험';render(r,'동작 구간 삭제 시험 · 실제 학습자의 오답 영상은 아닙니다. 어느 동작이 빠졌는지 확인하는 시험입니다.');}catch(e){$('progress').textContent=e.message;}};\n"""
(V/'demo.js').write_text(js,encoding='utf8')
html=(V/'demo.html').read_text(encoding='utf8').replace('<button id="public">공개 영상의 시험 결과 보기</button>','<button id="public">정상 공개 영상 시험</button><button id="publicCut">주요 구간을 뺀 공개 시험</button>')
(V/'demo.html').write_text(html,encoding='utf8')
for word in ['WORD0739','WORD1133','WORD1567','WORD2523']:
 out=read(V/'control_feedback'/f'rgb_geometry_42_{word}_action_cut.json');folder=U/'actual_controls'/f'{word}_action_cut';a=np.load(folder/'observations.npz');side=read(folder/'observations.sidecar.json')
 xy=a['xy_px'].astype(float);xy[~a['usable']]=np.nan
 out['overlay']=dict(width=side['width'],height=side['height'],times=(a['timestamp_us']/1e6).tolist(),points=[[[float(v) if np.isfinite(v) else None for v in pt] for pt in frame] for frame in xy])
 out['latency']['total_analysis_seconds']=out['latency']['total_seconds'];out['training']=False
 save(V/'demo_controls'/word/'RESULT.json',out)
