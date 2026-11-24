"""Same frozen training; bounded retries for Windows status-file replacement only."""
import time
import common
import train

original=common.write
def status_retry(path,value,new=False):
 if common.Path(path).resolve()!=common.U/'STATUS.json':return original(path,value,new)
 for attempt in range(20):
  try:return original(path,value,new)
  except PermissionError:
   if attempt==19:raise
   time.sleep(.05*(attempt+1))

if __name__=='__main__':
 recovery=common.read(common.U/'TRAIN_STATUS_RECOVERY.json')
 common.check(recovery['pins']);assert common.sha(__file__)==recovery['wrapper_sha256']
 common.write=status_retry
 train.main()
