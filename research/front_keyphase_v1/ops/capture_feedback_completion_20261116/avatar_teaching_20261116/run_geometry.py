from pathlib import Path
import subprocess,sys,json,time
U=Path(__file__).resolve().parent
for clip in ['NIA_SL_WORD0099_REAL01','NIA_SL_WORD1290_REAL01']:
 D=U/clip
 for script,artifact in [('video_epipolar_audit.py','EPIPOLAR_AUDIT.json'),('estimate_video_cameras.py','ESTIMATED_CAMERAS.json'),('check_video_geometry.py','GEOMETRY_DIAGNOSTIC.json')]:
  if (D/'video_only'/artifact).exists():continue
  if not (D/'video_only/EXTRACTION_COMPLETE.json').exists():raise RuntimeError('Extraction not completed: '+clip)
  print('RUN',clip,script,flush=True);t=time.perf_counter()
  with (D/(script+'.log')).open('w',encoding='utf-8')as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(D/script)],stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError('See '+str(D/(script+'.log')))
  print('DONE',clip,script,round(time.perf_counter()-t,2),flush=True)
 print('GEOMETRY_COMPLETE',clip,flush=True)
