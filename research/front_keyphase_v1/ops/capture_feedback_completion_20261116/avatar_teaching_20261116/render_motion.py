from pathlib import Path
import bpy,sys,json
U=Path(__file__).resolve().parent
args=sys.argv[sys.argv.index('--')+1:];clip,revision=args[:2];D=U/clip/revision
bpy.ops.wm.open_mainfile(filepath=str(D/'motion.blend'),load_ui=False,use_scripts=False)
s=bpy.context.scene;s.render.resolution_x=512;s.render.resolution_y=576;s.render.resolution_percentage=100
F=D/'frames';F.mkdir(exist_ok=True)
for f in range(s.frame_start,s.frame_end+1):
 p=F/f'{f-1:04d}.png'
 if p.exists():continue
 s.frame_set(f);s.render.filepath=str(p);bpy.ops.render.render(write_still=True)
print('RENDER_COMPLETE',clip,revision,flush=True)
