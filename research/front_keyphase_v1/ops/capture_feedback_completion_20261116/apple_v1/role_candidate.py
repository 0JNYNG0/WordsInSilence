from verified_candidate import *
class RoleRegions:
 def __init__(self):self.mouth=Ensemble('mouth_role');self.pair=Ensemble('pair_role')
 def predict(self,x,confidence):
  a,_=self.mouth.predict(x[:,:2],confidence);b,_=self.pair.predict(x[:,2:4],confidence);a[~x[:,4].astype(bool)]=-1;b[~x[:,5].astype(bool)]=-1
  s=np.where(a==1,1,np.where(b==1,2,0));s[(a==1)&(b==1)]=-1
  return s,None
class RoleCandidate(VerifiedCandidate):
 def __init__(self):super().__init__();self.region=RoleRegions()
