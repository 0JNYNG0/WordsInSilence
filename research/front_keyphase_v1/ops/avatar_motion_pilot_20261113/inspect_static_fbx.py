"""Read only the static, non-recording character asset; explicitly disable animation import."""
import bpy,json,hashlib,time
from pathlib import Path
U=Path(__file__).resolve().parent
SOURCE=Path('C:/MyUnityProjects/WordsInSilence/Assets/GameAssets/NPC_01/Ch29_nonPBR.fbx')
sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
pin=dict(source=str(SOURCE),sha256=sha,bytes=SOURCE.stat().st_size,use_anim=False,
         source_class='Existing static NPC character FBX, not an animation or personal recording')
(U/'STATIC_FBX_PIN.json').write_text(json.dumps(pin,indent=2)+'\n','utf-8')
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
try:
    bpy.ops.import_scene.fbx(filepath=str(SOURCE),use_anim=False,use_image_search=False)
    assert len(bpy.data.actions)==0,'Static asset unexpectedly contains imported actions'
    rigs=[]
    for ob in bpy.data.objects:
        if ob.type!='ARMATURE':continue
        rigs.append(dict(name=ob.name,bone_count=len(ob.data.bones),matrix_world=[list(x)for x in ob.matrix_world],
          bones=[dict(name=b.name,parent=b.parent.name if b.parent else None,head=list(b.head_local),tail=list(b.tail_local),length=b.length)for b in ob.data.bones]))
    meshes=[]
    for ob in bpy.data.objects:
        if ob.type!='MESH':continue
        meshes.append(dict(name=ob.name,vertices=len(ob.data.vertices),polygons=len(ob.data.polygons),
            group_names=[g.name for g in ob.vertex_groups],
            modifiers=[dict(type=m.type,object=m.object.name if getattr(m,'object',None)else None)for m in ob.modifiers],
            dimensions=list(ob.dimensions),matrix_world=[list(x)for x in ob.matrix_world]))
    result=dict(status='STATIC_FBX_IMPORT_VERIFIED',pin=pin,animation_imported=False,actions=len(bpy.data.actions),
         rigs=rigs,meshes=meshes,source_unchanged=hashlib.sha256(SOURCE.read_bytes()).hexdigest()==sha)
    (U/'STATIC_FBX_INSPECTION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    bpy.ops.wm.save_as_mainfile(filepath=str(U/'static_character.blend'))
    print('STATIC_CHARACTER_READY',len(rigs),len(meshes),len(bpy.data.actions))
except Exception as e:
    path=U/'FIRST_STATIC_FBX_ERROR.json'
    if not path.exists():path.write_text(json.dumps(dict(error=repr(e),time=time.time()),indent=2)+'\n','utf-8')
    raise
