"""Continue frozen public-reference checks after existing image process; no duplicate extraction."""
import prepare as R
import subprocess,sys,time,os,traceback
from pathlib import Path
STEPS=[('verify_images.py',[], 'VERIFIED_IMAGES.json'),('merge.py',['run'],'PREPARED.json'),('merge.py',['verify'],'INPUTS_VERIFIED.json'),('range.py',['freeze'],'RANGE_FROZEN.json'),('range.py',['run'],'RANGE_DONE.json'),('verify_range.py',[],'RANGE_VERIFIED.json'),('controls.py',['freeze'],'CONTROLS_FROZEN.json'),('controls.py',['run'],'CONTROLS_DONE.json'),('verify_controls.py',[],'CONTROLS_VERIFIED.json')]

def run():
 assert not (R.U/'PIPELINE_RUNNING.json').exists();assert R.read(R.U/'ALIGNMENT_VERIFIED.json')['counts']['pairs']==7184
 scripts=['prepare.py','verify_images.py','merge.py','range.py','verify_range.py','controls.py','verify_controls.py',Path(__file__).name];pins={str(R.U/x):R.sha(R.U/x) for x in scripts};pins.update({str(R.U/x):R.sha(R.U/x) for x in ['FROZEN.json','ALIGNMENT_VERIFIED.json','RANGE_PROTOCOL.json','CONTROLS_PROTOCOL.json']})
 R.write(R.U/'PIPELINE_FROZEN.json',dict(at=R.now(),pins=pins,steps=STEPS,wait_for='Existing recheck.py session79424 only; never launch another recheck',limits='PublicTRAIN899, frozen structural range and control research. No model training/classifier inference/YOLO/camera/download/operating or private access. Fail stops preserved first error, never relax range or discard failed cases.'))
 R.write(R.U/'PIPELINE_RUNNING.json',dict(at=R.now(),pid=os.getpid()));(R.U/'pipeline_logs').mkdir();(R.U/'pipeline_records').mkdir()
 def status(stage):
  p=R.U/'PIPELINE_STATUS.json';tmp=R.U/'PIPELINE_STATUS.tmp';tmp.write_text(__import__('json').dumps(dict(at=R.now(),stage=stage),ensure_ascii=False),encoding='utf8');tmp.replace(p)
 status('WAITING_EXISTING_IMAGE_PROCESS');print('WAIT_EXISTING_IMAGES599',flush=True)
 while (R.U/'RUNNING.json').exists() or not(R.U/'DONE.json').exists():
  assert not (R.U/'FIRST_ERROR.txt').exists(),'Original image error; preserve and investigate'
  time.sleep(10)
 assert R.read(R.U/'DONE.json')['clips']==599;R.check(pins)
 for i,(script,args,output) in enumerate(STEPS):
  R.check(pins);record=R.U/'pipeline_records'/f'{i:02}.json'
  assert not record.exists() and not(R.U/output).exists(),'Stage unexpectedly already done; inspect and reuse, never rerun'
  status(f'RUNNING_{i:02}_{script}_{args}');print('PIPELINE_STAGE',i,script,args,flush=True);log=R.U/'pipeline_logs'/f'{i:02}.txt';start=time.perf_counter()
  with log.open('x',encoding='utf8') as fp:result=subprocess.run([sys.executable,'-B','-X','utf8',str(R.U/script),*args],stdout=fp,stderr=subprocess.STDOUT,cwd=R.ROOT)
  R.write(record,dict(at=R.now(),script=script,args=args,exit_code=result.returncode,seconds=time.perf_counter()-start,log=str(log),output=output,output_sha256=R.sha(R.U/output) if (R.U/output).exists() else None))
  assert result.returncode==0 and (R.U/output).exists(),f'Stage {i} failed; inspect {log}'
 R.check(pins);R.check(R.read(R.U/'FROZEN.json')['protected']);R.write(R.U/'PIPELINE_DONE.json',dict(at=R.now(),stages=len(STEPS),status='COMPUTED_AND_ARITHMETIC_VERIFIED_PENDING_SCIENTIFIC_REVIEW_REPORT',goal_complete=False));status('AWAITING_SCIENTIFIC_RESULT_REVIEW');(R.U/'PIPELINE_RUNNING.json').unlink();print('PIPELINE_COMPLETE_PENDING_RESULT_REVIEW',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException:
  p=R.U/'PIPELINE_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
