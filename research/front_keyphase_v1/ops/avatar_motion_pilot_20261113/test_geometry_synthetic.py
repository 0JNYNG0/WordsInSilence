"""Artificial geometry only; this is not a real clip or source calibration."""
from pathlib import Path
import json
import numpy as np
import cv2
from geometry_core import camera,project,triangulate,undistort_pixels,hand_basis

U=Path(__file__).resolve().parent
rng=np.random.default_rng(20261113)
points=rng.uniform([-.3,-.3,2],[.3,.3,3],size=(42,3))
K=np.array([[1400.,.3,960.],[0,1410,540],[0,0,1.]])
d=np.array([-.03,.015,.0001,-.0002,.001])
cams=[]
for cx in [-.4,-.2,0,.2,.4]:
    M=np.column_stack([np.eye(3),[-cx,0,0]])
    record={k:{'data':' '.join(map(str,v.ravel()))}for k,v in [('Intrinsics',K),('CameraMatrix',M),('Distortion',d)]}
    cams.append(camera(record))
uv=[]
for c in cams:
    image,valid=project(points,*c,coordinate_mode='world');assert valid.all();uv.append(image)
recovered=triangulate(uv,cams)
error=np.linalg.norm(recovered-points,axis=-1)
assert error.max()<1e-8
# Check zero-skew projection against a distinct implementation; own path retains skew.
k0=K.copy();k0[0,1]=0
ours,_=project(points,k0,cams[0][1],d,coordinate_mode='world')
opencv,_=cv2.projectPoints(points,np.zeros(3),cams[0][1][:,3],k0,d)
independent_error=float(np.max(abs(ours-opencv[:,0])))
assert independent_error<1e-9
rejects=[]
for name,call in [
    ('unknown_frame_convention',lambda:project(points,*cams[0],coordinate_mode='guess')),
    ('one_camera',lambda:triangulate(uv[:1],cams[:1])),
    ('degenerate_palm',lambda:hand_basis(np.zeros((21,3))))]:
    try:call()
    except ValueError:rejects.append(name)
assert len(rejects)==3
behind=points.copy();behind[0,2]=-1
_,v=project(behind,*cams[2],coordinate_mode='world');assert not v[0]
out=dict(status='PASS_SYNTHETIC_GEOMETRY_ONLY',real_source_processed=False,synthetic_points=42,synthetic_cameras=5,
         source_camera_parameters_used=False,new_models_used=False,max_3d_roundtrip_error=float(error.max()),
         independent_opencv_projection_difference_px=independent_error,rejections=rejects,
         skew_supported=True,distortion='Brown5',note='No real sign reprojection or animation success is implied.')
(U/'SYNTHETIC_GEOMETRY_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps(out,ensure_ascii=False))
