"""Render exact127-source-frame approximate animation at native30fps."""
import bpy,json,sys
from pathlib import Path
U=Path(__file__).resolve().parent;D=U/('public_approximate_preview_body_axis'if '--body-axis'in sys.argv else'public_approximate_preview');P=D/'frames';P.mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.blend'),load_ui=False,use_scripts=False)
s=bpy.context.scene;s.render.resolution_x=512;s.render.resolution_y=512;s.render.resolution_percentage=100
for f in range(127):
    p=P/('frame%03d.png'%f)
    if p.exists():continue
    s.frame_set(f+1);s.render.filepath=str(p);bpy.ops.render.render(write_still=True)
(D/'RENDER_COMPLETE.json').write_text(json.dumps(dict(source_frames=127,rendered_frames=127,fps=30,source_to_blender_frame_offset=1,meaning='Approximate video-derived animation, not validated groundtruth'),indent=2)+'\n')
