from pathlib import Path
import hashlib,json
from relation_model import V,read
from train_relations import save
P=V.parent/'word100_learned_guide_feedback_20261102';U=V.parent/'word100_learned_feature_retention_20261102'
def write(n,s):
 with (V/n).open('x',encoding='utf8') as f:f.write(s)
def main():
 live=(U/'retained_live.py').read_text(encoding='utf8')
 live=live.replace('from audit import U,O,P,read,save','from relation_runtime import V as U,O,P,read,RelationsEngine')
 live=live.replace('from retention_engine import RetentionEngine','from engine import WORDS\nwrite=shared.write')
 live=live.replace('@lru_cache(maxsize=1)\ndef engine():return RetentionEngine(42)','@lru_cache(maxsize=2)\ndef engine(condition):return RelationsEngine(condition)')
 live=live.replace('def analyze(source,directory,word):','def analyze(source,directory,word,condition="rgb_geometry",progress=None):')
 live=live.replace('e=engine();p=e.infer','e=engine(condition);p=e.infer')
 live=live.replace("result.update(latency=timing,device=device,frames=len(ids),source_sha256=shared.sha(source),new_training=0)",'''timing['total_analysis_seconds']=timing['total_seconds'];timing['relation_head_seconds']=p['relation_head_seconds']
  result.update(latency=timing,device=device,frames=len(ids),source_sha256=shared.sha(source),new_training=0,training=False,
    actual_sensor_extent=[0,0,w,h],rest_hands_required=False)
  xy=ob.xy_px.astype(float);xy[~ob.usable]=np.nan
  result['overlay']=dict(width=w,height=h,times=(ts/1e6).tolist(),points=[[[float(v) if np.isfinite(v) else None for v in pt] for pt in frame] for frame in xy])''')
 live=live.replace('def warmup():shared.warmup();engine()','def warmup():\n result=shared.warmup()\n for c in ["rgb_geometry","geometry_only"]:engine(c)\n return result')
 write('demo_adapter.py',live)
 server=(P/'run_demo.py').read_text(encoding='utf8').replace('from live_adapter import','from demo_adapter import').replace('PORT=8805','PORT=8806')
 server=server.replace("manifest=read(U/'INPUT_MANIFEST.json')['rows']","manifest=read(U.parent/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json')['rows']")
 server=server.replace("U/'parallel_public'/w/'RESULT.json'","U/'demo_public'/w/'RESULT.json'")
 server=server.replace("('projection_finetune','frozen_projection')","('rgb_geometry','geometry_only')")
 write('run_demo.py',server)
 html=(P/'demo.html').read_text(encoding='utf8')
 html=html.replace('배운 동작으로 내 영상을 확인해요','손모양과 신체 관계를 함께 확인해요')
 html=html.replace('손모양과 동작 흐름을 확인합니다.','손모양·얼굴과 손·양손 관계 및 동작 흐름을 확인합니다.')
 html=html.replace('두 조건 모두 새 피드백 신경망을 학습했습니다. 미세조정 조건은 SignRep의 마지막 변환층 일부도 함께 학습했습니다. 미세조정의 추가 성능 향상은 아직 확인되지 않았습니다. C·T3-F는 변경하지 않았습니다.','기존 손모양과 학습된 시간 흐름은 보존하고, 신체 관계를 읽는 작은 판정부를 추가했습니다. 아래 선택은 이 관계 판정부가 영상 특징도 함께 사용할지 비교합니다. C·T3-F와 이전 데모는 그대로입니다.')
 html=html.replace('<option value="projection_finetune">부분 미세조정 후보 · 고정 seed 42</option><option value="frozen_projection">SignRep 고정 + 새 판정부 · 고정 seed 42</option>','<option value="rgb_geometry">신체 관계: 영상 + 관절</option><option value="geometry_only">신체 관계: 관절만 · 비교용</option>')
 write('demo.html',html)
 js=(P/'demo.js').read_text(encoding='utf8')
 js=js.replace("const statusNames={","const statusNames={IMPORTANT_RELATION_UNCONFIRMED:'손모양 외에 신체 관계 확인 필요',")
 js=js.replace("if(s.source_phrases?.length)","""for(let t of s.relations||[]){let row=el('div',undefined,'feature relation');row.append(el('div',t.name),el('span',({POSITIVE_RELATION_OBSERVED:'이 관계가 보임',EXPECTED_RELATION_NOT_OBSERVED:'이 관계를 확인하지 못함',RELATION_UNCERTAIN:'근거가 부족해 보류'})[t.status]));card.append(row,el('p',t.message,'small'));}if(s.source_phrases?.length)""")
 js=js.replace('저장된 공개 영상 시험 · 미세조정 후보','저장된 공개 영상 시험 · 손모양 보존 + 신체 관계 후보')
 js=js.replace('영상과 관절 근거가 없어 비교하지 못했습니다.','영상의 관측 근거가 부족해 비교하지 못했습니다.')
 write('demo.js',js)
 # Saved public examples: zero new extraction. Use exact original edited-control inputs/overlay.
 for word in ['WORD0739','WORD1133','WORD1567','WORD2523']:
  out=read(V/'control_feedback'/f'rgb_geometry_42_{word}_reencoded_normal.json')
  import numpy as np
  folder=U/'actual_controls'/f'{word}_reencoded_normal';a=np.load(folder/'observations.npz');side=read(folder/'observations.sidecar.json')
  xy=a['xy_px'].astype(float);xy[~a['usable']]=np.nan
  out['overlay']=dict(width=side['width'],height=side['height'],times=(a['timestamp_us']/1e6).tolist(),points=[[[float(v) if np.isfinite(v) else None for v in pt] for pt in frame] for frame in xy])
  out['latency']['total_analysis_seconds']=out['latency']['total_seconds'];out['training']=False
  save(V/'demo_public'/word/'RESULT.json',out)
 save(V/'DEMO_PREPARED.json',dict(port=8806,existing8805_preserved=True,old_files_read_only=True,
  code_sha256={n:hashlib.sha256((V/n).read_bytes()).hexdigest() for n in ['demo_adapter.py','run_demo.py','demo.js','demo.html','relation_runtime.py']}))
 print('Research demo8806 prepared, not yet started or validated')
if __name__=='__main__':main()
