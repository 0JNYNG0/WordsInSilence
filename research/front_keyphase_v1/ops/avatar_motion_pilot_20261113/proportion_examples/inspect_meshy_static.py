import bpy,json,hashlib
from pathlib import Path
from mathutils import Vector
U=Path(__file__).resolve().parent
P=Path('C:/MyUnityProjects/WordsInSilence/Assets/GameAssets/Player_model_animation_meshy/Meshy_AI_OriginalVer_Player_biped_Animation_Walking_withSkin.fbx')
before=hashlib.sha256(P.read_bytes()).hexdigest()
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.fbx(filepath=str(P),use_anim=False,use_image_search=False)
assert len(bpy.data.actions)==0
rigs=[];meshes=[]
for o in bpy.data.objects:
    if o.type=='ARMATURE':rigs.append(dict(name=o.name,bones=[dict(name=b.name,parent=b.parent.name if b.parent else None,head=list(b.head_local),tail=list(b.tail_local))for b in o.data.bones],matrix_world=[list(r)for r in o.matrix_world]))
    if o.type=='MESH':meshes.append(dict(name=o.name,vertices=len(o.data.vertices),groups=[g.name for g in o.vertex_groups],modifiers=[m.type for m in o.modifiers]))
meshobs=[o for o in bpy.data.objects if o.type=='MESH'];points=[o.matrix_world@Vector(p)for o in meshobs for p in o.bound_box];mins=Vector([min(p[i]for p in points)for i in range(3)]);maxs=Vector([max(p[i]for p in points)for i in range(3)]);center=(mins+maxs)/2
mat=bpy.data.materials.new('Static inspection');mat.diffuse_color=(.48,.57,.67,1)
for o in meshobs:
    o.data.materials.clear();o.data.materials.append(mat)
    for p in o.data.polygons:p.material_index=0
scene=bpy.context.scene;scene.render.engine='BLENDER_WORKBENCH';scene.display.shading.light='STUDIO';scene.display.shading.color_type='MATERIAL';scene.display.shading.show_cavity=True
cd=bpy.data.cameras.new('Static camera');cam=bpy.data.objects.new(cd.name,cd);scene.collection.objects.link(cam);scene.camera=cam;cd.type='ORTHO';cd.ortho_scale=max(maxs-mins)*1.2
cam.location=center+Vector((0,-max(maxs-mins)*3,0));cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.resolution_x=600;scene.render.resolution_y=700;scene.render.resolution_percentage=100;scene.render.filepath=str(U/'MESHY_STATIC.png');bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(U/'MESHY_STATIC.blend'))
out=dict(source=str(P),sha256=before,source_unchanged=hashlib.sha256(P.read_bytes()).hexdigest()==before,source_animation_loaded=False,actions=0,rigs=rigs,meshes=meshes,world_bounds=[list(mins),list(maxs)],license='Local project asset; rights/terms not established by this inspection')
(U/'MESHY_STATIC_INSPECTION.json').write_text(json.dumps(out,indent=2)+'\n');print('MESHY_STATIC_READY',len(rigs),len(meshes))
