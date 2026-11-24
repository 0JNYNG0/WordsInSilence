"""One RTMW observation, two candidate consumers. No image/model/camera access.

Finite in-frame points are available estimates, NOT proven visible/correct points.
Shape-compatible with C's 280 channels; detector-compatible weights are not claimed.
"""
from pathlib import Path
import sys
import numpy as np
U=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(U.parents[2]/'sign_model_pilot_v1/src'))
from pilot.data import build_channels,PilotDataError
RTMW_BODY9=(0,5,6,7,8,9,10,11,12)
BODY25_SLOTS=(0,5,2,6,3,7,4,12,9)
SCHEMA='rtmw133_shared_2d_full_timeline_v1'

def convert(xy,score,ts,fid,width,height):
 xy=np.asarray(xy,np.float32);score=np.asarray(score,np.float32);ts=np.asarray(ts,np.int64);fid=np.asarray(fid,np.int64)
 n=len(ts)
 if not n or xy.shape!=(n,133,2) or score.shape!=(n,133) or fid.shape!=(n,):raise ValueError('RTMW133 shape')
 if not np.all(np.diff(ts)>0) or not np.all(np.diff(fid)>0):raise ValueError('Nonmonotonic input')
 if width<=0 or height<=0:raise ValueError('Image dimensions')
 available=np.isfinite(xy).all(2)&(xy[:,:,0]>=0)&(xy[:,:,0]<width)&(xy[:,:,1]>=0)&(xy[:,:,1]<height)
 native=np.zeros((n,67,2),np.float32);present=np.zeros((n,67),bool)
 for dst,src in zip(BODY25_SLOTS,RTMW_BODY9):native[:,dst]=xy[:,src];present[:,dst]=available[:,src]
 native[:,25:46]=xy[:,91:112];present[:,25:46]=available[:,91:112]
 native[:,46:67]=xy[:,112:133];present[:,46:67]=available[:,112:133]
 native[~present]=0
 channels=build_channels(native,present,ts) # Raises on absent/degenerate body scale, never invents it.
 shoulders=available[:,5]&available[:,6];origin=(xy[:,5]+xy[:,6])/2
 relmask=available&shoulders[:,None];rel=np.zeros_like(xy)
 rawrel=(xy-origin[:,None,:])/channels['body_scale'];rel[relmask]=rawrel[relmask]
 # Known geometric failure=0; in-frame estimates remain unknown=-1, never verified=1.
 visibility_evidence=np.where(available,-1,0).astype(np.int8)
 return dict(raw_xy_px=xy.copy(),raw_score=score.copy(),timestamp_us=ts.copy(),frame_id=fid.copy(),
  point_available=available,visibility_evidence=visibility_evidence,body_relative_xy=rel,body_relative_available=relmask,
  hand_local_xy=channels['x'][:,102:186].reshape(n,2,21,2),hand_local_available=channels['hand_valid'].astype(bool).reshape(n,2,21),
  tcn280=channels['x'],native67= native,native_available=present,source_width=np.int64(width),source_height=np.int64(height),schema=np.array(SCHEMA))

def load(path):
 with np.load(path,allow_pickle=False) as z:return convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']))
