import bpy,json
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;P=U.parent.parent/'avatar_motion_pilot_20261113'
bpy.ops.wm.open_mainfile(filepath=str(P/'static_character.blend'),load_ui=False,use_scripts=False)
rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');mesh=next(o for o in bpy.data.objects if o.type=='MESH')
pts=np.array([rig.matrix_world.inverted()@mesh.matrix_world@v.co for v in mesh.data.vertices]);rows=[]
for y in np.arange(135.,144.1,.5):
 q=pts[(abs(pts[:,0])<1)&(abs(pts[:,1]-y)<.4)]
 rows.append(dict(y=y,front_max_z=float(q[:,2].max())if len(q)else None,n=len(q)))
(U/'RIG_FACE_PROFILE.json').write_text(json.dumps(rows,indent=2),'utf-8')
print(rows)
