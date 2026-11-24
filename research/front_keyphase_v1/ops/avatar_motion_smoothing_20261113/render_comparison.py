import bpy,json,sys,time,traceback
from pathlib import Path
U=Path(__file__).resolve().parent
label=sys.argv[sys.argv.index('--')+1];assert label in ['baseline','candidate']
out=U/(label+'_frames60');out.mkdir(exist_ok=True)
status=U/('RENDER_'+label.upper()+'.json');start=time.time()
def write(obj):status.write_text(json.dumps(obj,indent=2)+'\n','utf-8')
try:
    bpy.ops.wm.open_mainfile(filepath=str(U/(label.upper()+'_ORIGINAL_MATERIAL.blend')),load_ui=False,use_scripts=False)
    sc=bpy.context.scene;sc.render.threads_mode='FIXED';sc.render.threads=2
    sc.render.resolution_x=512;sc.render.resolution_y=512;sc.render.resolution_percentage=100
    sc.display.shading.studiolight_rotate_z=0
    write({'status':'RUNNING','frames':0,'output_fps':60,'source_fps':30,'source_frames':127})
    for i in range(254):
        sf=min(i/2,126);bf=sf+1;sc.frame_set(int(bf),subframe=bf%1)
        sc.render.filepath=str(out/('frame%03d.png'%i));bpy.ops.render.render(write_still=True)
        if i%30==0:write({'status':'RUNNING','frames':i+1,'seconds':time.time()-start,'output_fps':60,'source_fps':30})
    # Same camera and material; actual native source64 and96 at larger scale.
    sc.render.resolution_x=sc.render.resolution_y=720
    for sf in [32,51,64,88,96,126]:
        sc.frame_set(sf+1);sc.render.filepath=str(U/(label+'_source%03d.png'%sf));bpy.ops.render.render(write_still=True)
    write({'status':'COMPLETE','frames':254,'output_fps':60,'source_fps':30,'source_frames':127,'seconds':time.time()-start,'duration':127/30,'last_half_frame':'last native pose held','changing_source_timestamps':False})
except Exception as e:
    write({'status':'ERROR','error':repr(e),'traceback':traceback.format_exc()});raise
