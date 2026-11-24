from pathlib import Path
import json,hashlib,ast
from datetime import datetime,timezone
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 base=read(O/'word100_stage_feedback_cycle_20261031/INPUTS.json')['rows']
 rows={r['clip_id']:dict(r) for r in base}
 extra=read(O/'word100_small_multipeople_angle_20261021/FROZEN.json')['rows']
 for r in extra:
  if r['clip_id'] in rows:continue
  paths=[O/'word100_small_multipeople_angle_20261021/videos'/f"{r['clip_id']}.mp4",Path('W:/WordsInSilence/data/AIHub_103/word100_web_references_20261013')/f"{r['clip_id']}.mp4"]
  source=next((p for p in paths if p.is_file()),None)
  if source:rows[r['clip_id']]=dict(clip_id=r['clip_id'],word_id=r['word_code'],label=r['label'],person=r['signer_id'],source=str(source),source_sha256=r['source']['sha256'],intervals=r['intervals'])
 for r in read(O/'word100_yolo_five_scene_view_20261025/VISUAL_MANIFEST.json'):
  if r['clip_id'] not in rows:rows[r['clip_id']]=dict(clip_id=r['clip_id'],word_id=r['clip_id'].split('_')[2],label=r['label'],person=int(r['clip_id'].split('_')[3][4:]),source=r['source_video'],source_sha256=r['source_video_sha256'],intervals=None)
 selection={'WORD0739':[1,2,6],'WORD1133':[1,2,6],'WORD1567':[1,2,6],'WORD2523':[1,2,6],'WORD0003':[1,2,6],'WORD0949':[1,2,6],'WORD0185':[1,5,6],'WORD0004':[2,5,6]}
 selected=[]
 for word,ids in selection.items():
  for person in ids:
   cid=f'NIA_SL_{word}_REAL{person:02d}_F';r=rows[cid];p=Path(r['source']);assert p.is_file(),p;assert sha(p)==r['source_sha256'],p
   selected.append({k:r[k] for k in ['clip_id','word_id','label','person','source','source_sha256','intervals']})
 assert len({r['source_sha256'] for r in selected})==24
 code=O/'word100_integrated_test_20261104/parallel_capture.py';tree=ast.parse(code.read_text(encoding='utf-8-sig'))
 pins=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets))
 for rel,h in pins.items():assert sha(R/rel)==h,rel
 protected={str(R/k):v for k,v in pins.items()};protected[str(code)]=sha(code)
 for name in ['learner_adapter.py','integrated_adapter.py','local_runtime.py','run_demo.py','demo.js']:
  p=O/'word100_integrated_test_20261104'/name
  if p.exists():protected[str(p)]=sha(p)
 models={str(p.relative_to(U)):sha(p) for p in (U/'models').rglob('*.onnx')}
 plan=dict(at=datetime.now(timezone.utc).isoformat(),authorization='2026-11-08 user explicitly requested RTMW-l download and same public-video comparison; no training/user data/production replacement',rows=selected,selection_reason='4 supported feedback expressions + crossing/repetition and previously difficult hand/face-region examples; fixed public development diagnostics, not fresh unbiased holdout',conditions=['current_mediapipe_full_capture','RTMW-l-384x288_OpenVINO_GPU'],candidate=dict(pose='RTMW-l 2D distilled x-to-l, official 70.1 AP row',detector='YOLOX-m HumanArt person boxes only, not YOLO hand model',person_policy='largest detected person box; no box means unavailable, never whole-frame hallucination fallback',detection_interval=1,input_color='YOLOX BGR; RTMW RGB per official export Normalize.to_rgb=true',input_dimensions='Verify ONNX graph 384x288; exported json contains stale 256x192 entries; do not use those to reshape',smoothing=False,frame_skip=False,flip_test=False,display_threshold=0.3,sensitivity_thresholds=[0.2,0.3,0.5]),metrics=['fixed reviewed RGB centers: shape/identity/wrist qualitative labels','native-frame output coverage (NOT accuracy or visibility)','body-relative normalized hand second differences (motion/noise proxy, not independent error truth)','crossing/reappearance visual continuous windows','sequential per-video wall time, detector compute, cold compile separate; CPU secondary'],speed_repeats=dict(clips=['NIA_SL_WORD0003_REAL01_F','NIA_SL_WORD0739_REAL01_F','NIA_SL_WORD1133_REAL01_F'],repetitions=3,order='alternate MP then RTMW; RTMW GPU and CPU separately; no scoring changes'),labels='Existing source-image annotations where available plus same-Codex visual review; no independent human ground truth claim. Hidden fingers not labelled correct from model agreement.',protected=protected,models=models,source_pins={str(O/'word100_stage_feedback_cycle_20261031/INPUTS.json'):sha(O/'word100_stage_feedback_cycle_20261031/INPUTS.json')},training=0,user_mentor_used=False)
 with (U/'FROZEN.json').open('x',encoding='utf-8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
 print('FROZEN',len(selected),'videos',len(models),'model files')
if __name__=='__main__':main()
