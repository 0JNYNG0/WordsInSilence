"""Conservative projected mutual hand overlap, NOT visibility reconstruction."""
import numpy as np,cv2
def hand_occlusion(ch,hand):
 p=ch['body_relative_xy'];av=ch['point_available'];a,b=(91,112)if hand=='L'else(112,91);flags=np.zeros(len(p),bool)
 for i in range(len(p)):
  if not(av[i,a:a+21].all()and av[i,b:b+21].all()):flags[i]=True;continue
  hull=cv2.convexHull(p[i,b:b+21].astype('float32'));ids=np.array([6,7,8,10,11,12,14,15,16,18,19,20])+a
  # Positive mask means potentially covered: do not label a hidden hand open
  # merely because its predicted fingers follow the other visible hand.
  inside=[cv2.pointPolygonTest(hull,tuple(map(float,q)),False)>=0 for q in p[i,ids]]
  flags[i]=sum(inside)>=6
 return flags
