"""Missing-input and target-independence checks; no public-score rerun."""
from common import *
import numpy as np,torch
from runtime_candidate import Candidate

def main():
 assert not (U/'RUNTIME_CHECK.json').exists()
 shell=Candidate.__new__(Candidate);ts=np.arange(6,dtype=np.int64)*33333;fid=np.arange(6)
 raw=dict(xy_px=np.full((6,133,2),np.nan,np.float32),score=np.zeros((6,133),np.float32),timestamp_us=ts,frame_id=fid,source_width=640,source_height=480)
 missing=shell.from_raw(raw,'WORD0739')
 assert missing['tcn_top3']==[] and missing['normal_or_wrong'] is None and missing['frame_id']==fid.tolist()
 assert shell.from_channels(np.zeros((6,280),np.float32),ts,fid,'WORD0739')['status']=='INSUFFICIENT_COORDINATE_INPUT'
 # Mock only the TCN call: an unsupported target must not enter the classifier.
 calls=[]
 class FakeTCN:
  def __call__(self,x,p):calls.append((x.clone(),p.clone()));return torch.arange(100,dtype=torch.float32)[None]
 shell.tcn=FakeTCN();shell.f=read(U/'FROZEN.json');x=np.zeros((6,280),np.float32);x[:,237:279]=1
 a=shell.from_channels(x,ts,fid,'unsupported_a');b=shell.from_channels(x,ts,fid,'unsupported_b')
 assert a['tcn_top3']==b['tcn_top3'] and torch.equal(calls[0][0],calls[1][0])
 assert a['feedback_status']=='UNSUPPORTED_WORD' and a['normal_or_wrong'] is None
 write(U/'RUNTIME_CHECK.json',dict(status='PASS',tests=['missing body produces insufficient input','missing hands produce no forced word','unsupported target cannot change classifier input/output'],
   real_model_inference=0,private_inputs=0,pins={str(U/n):sha(U/n) for n in ['runtime_candidate.py','check_runtime.py']}))
 event('runtime_contract_checked')
if __name__=='__main__':main()
