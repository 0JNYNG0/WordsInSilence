"""Local-only derivatives. Original Unity assets and prior frozen results are read-only."""
from pathlib import Path
import bpy, json, hashlib
from mathutils import Vector

U = Path(__file__).resolve().parent
R = U.parents[3]
D = U / 'site' / 'assets'
D.mkdir(parents=True, exist_ok=True)
report = {}

def describe():
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    points = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    return dict(objects=[dict(name=o.name,type=o.type) for o in bpy.context.scene.objects],
                bounds=[[min(p[k] for p in points) for k in range(3)],
                        [max(p[k] for p in points) for k in range(3)]],
                triangles=sum(len(o.data.polygons) for o in meshes),
                images=[dict(name=i.name,size=list(i.size),packed=bool(i.packed_file)) for i in bpy.data.images])

def small_textures():
    for im in bpy.data.images:
        if im.type != 'IMAGE' or im.size[0] == 0: continue
        w,h=im.size
        if max(w,h)>1024:
            f=1024/max(w,h)
            im.scale(round(w*f),round(h*f))
            im.pack()

def export(name, animated):
    bpy.ops.object.select_all(action='DESELECT')
    for o in bpy.context.scene.objects:
        if o.type in {'MESH','ARMATURE'}: o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(D/(name+'.glb')),export_format='GLB',
        use_selection=True,export_animations=animated,export_frame_range=True,
        export_animation_mode='SCENE',export_force_sampling=True,
        export_cameras=False,export_lights=False,export_yup=True,export_all_influences=True,
        export_image_format='AUTO',export_materials='EXPORT')
    report[name]['bytes']=(D/(name+'.glb')).stat().st_size
    report[name]['sha256']=hashlib.sha256((D/(name+'.glb')).read_bytes()).hexdigest()

# FBX animation intentionally not imported. Robot idle is a scene transform, not a sign gesture.
bpy.ops.wm.read_factory_settings(use_empty=True)
source=R/'Assets/GameAssets/petrobot_model_animation/MOA_robot.fbx'
bpy.ops.import_scene.fbx(filepath=str(source), use_anim=False)
texture=R/'Assets/GameAssets/petrobot_model_animation/Meshy_AI_Cosmic_Companion_0807084014_texture.png'
im=bpy.data.images.load(str(texture),check_existing=True)
for o in bpy.context.scene.objects:
    if o.type!='MESH':continue
    mat=bpy.data.materials.new('MOA_original_color')
    mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Roughness'].default_value=0.65
    tx=mat.node_tree.nodes.new('ShaderNodeTexImage');tx.image=im
    mat.node_tree.links.new(tx.outputs['Color'],bsdf.inputs['Base Color'])
    o.data.materials.clear();o.data.materials.append(mat)
report['robot']=describe();report['robot']['source']=str(source)
small_textures();export('robot',False)

# This .blend contains ONLY the prior public-derived Ch29 trial action.
source=U.parent/'avatar_motion_smoothing_20261113/CANDIDATE_ORIGINAL_MATERIAL.blend'
bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
bpy.context.scene.frame_set(1)
report['npc']=describe();report['npc']['source']=str(source)
small_textures();export('npc',True)
(U/'ASSET_EXPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('EXPORT_COMPLETE',json.dumps({k:{'bounds':v['bounds'],'bytes':v['bytes']} for k,v in report.items()}))
