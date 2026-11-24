"""Admit only the diagnosed one-hand role repair; preserve proven pair head."""
from verified_candidate import *
class RequiredRegions:
 def __init__(self):self.mouth=Ensemble('mouth_role');self.previous=Ensemble('region_local')
 def predict(self,x,confidence):
  a,_=self.mouth.predict(x[:,:2],confidence);old,_=self.previous.predict(x,confidence)
  a[~x[:,4].astype(bool)]=-1;b=np.where((old==2)&x[:,5].astype(bool),1,0)
  s=np.where(a==1,1,np.where(b==1,2,0));s[(a==1)&(b==1)]=-1
  return s,None
class RequiredRoleCandidate(VerifiedCandidate):
 def __init__(self):super().__init__();self.region=RequiredRegions()
