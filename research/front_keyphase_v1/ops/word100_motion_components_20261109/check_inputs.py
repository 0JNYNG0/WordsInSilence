from common import *
from motion import *
from train_motion import physical

def mirror(x):
 z=x.copy();p=z[:,:102].reshape(-1,51,2);p[:,:,0]*=-1
 swap=np.array([0,2,1,4,3,6,5,8,7]+list(range(30,51))+list(range(9,30)))
 z[:,:102]=p[:,swap].reshape(-1,102);z[:,186:237]=z[:,186:237][:,swap]
 return z

def main():
 checks=0;rows=read(U/'READY.json')['rows'];cfg=read(U/'FROZEN.json')
 assert sha(U/'READY.json')==cfg['data_sha256'];checks+=1
 assert {r['person'] for r in rows if r['role']=='fit'}==set(range(1,10));checks+=1
 assert {r['person'] for r in rows if r['role']=='development'}=={11,12,13};checks+=1
 assert len({r['clip_id'] for r in rows})==48;checks+=1
 for r in rows:
  assert sha(r['path'])==r['sha256'];checks+=1
  z=np.load(r['path']);x=physical(z['x_full']);ts=z['timestamp_us'];assert np.diff(ts).min()>0;checks+=1
  for p in r['pairs']:
   args=(ts,p['a'],p['b'],KINDS[p['axis']]);f,ok,_=features(x,*args)
   assert np.isfinite(f).all() and f.shape==(58,);checks+=1
   translated=x.copy();pts=translated[:,:102].reshape(-1,51,2);pts[:]=pts*1.7+np.array([2.4,-.8]);f2,ok2,_=features(translated,*args)
   assert ok==ok2 and np.allclose(f,f2,atol=2e-4);checks+=1
   fm,om,_=features(mirror(x),*args,moving='L',anchor='R');assert ok==om and np.allclose(f,fm,atol=2e-4);checks+=1
   for speed in [.75,1.25]:
    fs,os,_=features(x,ts*speed,*args[1:]);assert ok==os and np.allclose(f,fs,atol=2e-5);checks+=1
   blank=x.copy();blank[:,186:237]=0;fb,ob,_=features(blank,*args);assert not ob and not fb.any();checks+=1
 # Synthetic generic arm-direction sanity: same handshape, hand advances along forearm.
 x=np.zeros((15,280),np.float32);x[:,186:237]=1;p=x[:,:102].reshape(15,51,2)
 p[:,1]=[0,0];p[:,2]=[1,0];p[:,3]=[0,1];p[:,5]=[1,1]
 for k in [30,35,39,43,47]:p[:,k,0]=np.linspace(0,1,15);p[:,k,1]=1
 f,ok,info=features(x,np.arange(15)*33333,0,14,'along_other_forearm');assert ok and info['signed_change']>.9;checks+=1
 _,ok,back=features(x,np.arange(15)*33333,0,14,'along_other_forearm',reverse=True);assert ok and back['signed_change']<-.9;checks+=1
 write(U/'INPUT_CHECK.json',dict(passed=True,checks=checks,scope='split, source hashes, shape, body translation/scale, mirrored side binding, speed-time invariance, missing evidence, signed motion',code={f:sha(U/f) for f in ['motion.py','train_motion.py','check_inputs.py']}));print(checks)
if __name__=='__main__':main()
