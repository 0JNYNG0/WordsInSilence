"""Blender static datablock inspection. Does not load scene, objects or actions."""
import bpy
from pathlib import Path
import json,hashlib

U=Path(__file__).resolve().parent
SOURCE=Path('C:/Users/local-user/OneDrive/바탕 화면/test_mixamo_girl.blend')
before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
with bpy.data.libraries.load(str(SOURCE),link=False) as (available,selected):
    names=dict(armatures=list(available.armatures),meshes=list(available.meshes))
    selected.armatures=list(available.armatures)
    selected.meshes=list(available.meshes)
rigs=[]
for rig in selected.armatures:
    bones=[]
    for b in rig.bones:
        bones.append(dict(name=b.name,parent=b.parent.name if b.parent else None,head=list(b.head_local),
                          tail=list(b.tail_local),length=b.length,matrix_local=[list(row)for row in b.matrix_local],
                          use_deform=b.use_deform))
    hand={side:{finger:[b.name for b in rig.bones if side+'Hand'+finger in b.name]for finger in ['Thumb','Index','Middle','Ring','Pinky']}for side in ['Left','Right']}
    rigs.append(dict(name=rig.name,bone_count=len(bones),bones=bones,finger_chains=hand))
meshes=[dict(name=m.name,vertices=len(m.vertices),polygons=len(m.polygons),has_deform_layer=any(len(v.groups)>0 for v in m.vertices),
             dimensions_local=[max(v.co[k]for v in m.vertices)-min(v.co[k]for v in m.vertices)for k in range(3)]if len(m.vertices)else[])for m in selected.meshes]
out=dict(status='STATIC_DATABLOCK_INSPECTION_COMPLETE',source=str(SOURCE),source_sha256=before,
         blender_version=bpy.app.version_string,loaded_types=['Armature rest data','Mesh rest geometry'],
         scene_loaded=False,objects_loaded=False,actions_loaded=False,animation_evaluated=False,rigs=rigs,meshes=meshes,
         bone_mapping_limit='Data-only load deliberately omits source objects/vertex-group names, actions and private-derived animation. Rest skeleton is inspected; binding and skin deformation not yet validated.',
         source_unchanged=hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before)
(U/'RIG_INSPECTION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8')
print('STATIC_RIG_RESULT',json.dumps(dict(blender=out['blender_version'],rigs=[dict(name=r['name'],bones=r['bone_count'],fingers=r['finger_chains'])for r in rigs],meshes=meshes,actions_loaded=False),ensure_ascii=False))
