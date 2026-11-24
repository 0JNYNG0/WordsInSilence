"""Locally generated proportion-test mannequin, not a downloaded or clinically validated human model."""
import bpy,json,math
from pathlib import Path
from mathutils import Vector,Matrix
P=Path(__file__).resolve().parent
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
arm=bpy.data.armatures.new('Proportion test skeleton');rig=bpy.data.objects.new('Armature',arm);bpy.context.collection.objects.link(rig);bpy.context.view_layer.objects.active=rig;rig.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bones={}
def bone(name,a,b,parent=None):
    x=arm.edit_bones.new('mixamorig:'+name);x.head=a;x.tail=b
    if parent:x.parent=arm.edit_bones['mixamorig:'+parent]
    bones[name]=(Vector(a),Vector(b));return x
bone('Hips',(0,95,0),(0,105,0));bone('Spine',(0,105,0),(0,120,0),'Hips');bone('Spine1',(0,120,0),(0,135,0),'Spine');bone('Spine2',(0,135,0),(0,145,0),'Spine1');bone('Neck',(0,145,0),(0,152,0),'Spine2');bone('Head',(0,152,0),(0,174,0),'Neck')
for side,sign in [('Left',1),('Right',-1)]:
    bone(side+'Shoulder',(0,145,0),(sign*20,145,0),'Spine2');bone(side+'Arm',(sign*20,145,0),(sign*50,145,0),side+'Shoulder');bone(side+'ForeArm',(sign*50,145,0),(sign*77,145,0),side+'Arm');bone(side+'Hand',(sign*77,145,0),(sign*85,145,0),side+'ForeArm')
    for finger,z,lens in [('Index',2.6,[3.8,2.5,2.0]),('Middle',.6,[4.1,2.8,2.2]),('Ring',-1.3,[3.8,2.5,2.1]),('Pinky',-3.0,[3.,2.,1.7])]:
        x=84.;parent=side+'Hand'
        for k,length in enumerate(lens+[1.],1):
            name=side+'Hand'+finger+str(k);bone(name,(sign*x,145,z),(sign*(x+length),145,z),parent);x+=length;parent=name
    thumb=[(79.,144.5,3.8),(81.7,144.3,5.8),(84.2,144.2,7.4),(86.4,144.2,8.3),(87.4,144.2,8.7)]
    parent=side+'Hand'
    for k in range(1,5):
        a,b=thumb[k-1],thumb[k];name=side+'HandThumb'+str(k);bone(name,(sign*a[0],a[1],a[2]),(sign*b[0],b[1],b[2]),parent);parent=name
    bone(side+'UpLeg',(sign*9,95,0),(sign*9,52,0),'Hips');bone(side+'Leg',(sign*9,52,0),(sign*9,9,0),side+'UpLeg');bone(side+'Foot',(sign*9,9,0),(sign*9,5,11),side+'Leg');bone(side+'ToeBase',(sign*9,5,11),(sign*9,5,19),side+'Foot')
bpy.ops.object.mode_set(mode='OBJECT')
rig.matrix_world=Matrix.Rotation(math.pi/2,4,'X')@Matrix.Scale(.01,4)
def material(name,col):
    m=bpy.data.materials.new(name);m.diffuse_color=(*col,1);return m
skin=material('Warm ivory hands and face',(.68,.72,.75));suit=material('Navy torso',(.10,.18,.30));joint=material('Joint dark',(.19,.26,.36));face=material('Face landmark accents',(.09,.15,.24))
parts=[]
def bind(ob,name,mat):
    ob.data.materials.clear();ob.data.materials.append(mat);g=ob.vertex_groups.new(name='mixamorig:'+name);g.add(list(range(len(ob.data.vertices))),1.,'REPLACE');parts.append(ob)
    for p in ob.data.polygons:p.use_smooth=True
def ellipsoid(center,scale,name,mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12,location=center);o=bpy.context.object;o.scale=scale;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);bind(o,name,mat);return o
def cylinder(a,b,r,name,mat):
    a,b=Vector(a),Vector(b);d=b-a;bpy.ops.mesh.primitive_cylinder_add(vertices=16,radius=r,depth=d.length,location=(a+b)/2);o=bpy.context.object;o.rotation_euler=d.to_track_quat('Z','Y').to_euler();bpy.ops.object.transform_apply(location=False,rotation=True,scale=False);bind(o,name,mat)
ellipsoid((0,121,0),(17,25,9.5),'Spine1',suit);ellipsoid((0,96,0),(15,12,10),'Hips',suit)
cylinder((0,143,0),(0,152,0),4.2,'Neck',skin);ellipsoid((0,162,0),(8.4,12.,8.6),'Head',skin)
# Explicit simple facial landmarks for reading the form, not extracted face groundtruth.
ellipsoid((0,161,8.5),(1.7,2.,2.1),'Head',skin)
for x in [-3.4,3.4]:ellipsoid((x,165,7.7),(1.4,.65,.65),'Head',face)
ellipsoid((0,156.5,7.7),(2.,.45,.5),'Head',face)
for side,sign in [('Left',1),('Right',-1)]:
    cylinder((0,140,0),(sign*20,145,0),5.5,'Spine2',suit);ellipsoid((sign*20,145,0),(5.3,)*3,'Spine2',joint)
    for suffix,radius in [('Arm',4.6),('ForeArm',3.5),('UpLeg',6.3),('Leg',4.8)]:
        a,b=bones[side+suffix];cylinder(a,b,radius,side+suffix,suit);ellipsoid(a,(radius,radius,radius),side+suffix,joint);ellipsoid(b,(radius*.9,)*3,side+suffix,joint)
    ellipsoid((sign*9,5,8),(6,5,13),side+'Foot',joint)
    ellipsoid((sign*81,145,0),(4.9,1.7,3.8),side+'Hand',skin)
    for finger in ['Thumb','Index','Middle','Ring','Pinky']:
        for k in range(1,4):
            name=side+'Hand'+finger+str(k);a,b=bones[name];radius=.77 if finger!='Pinky'else.65
            cylinder(a,b,radius,name,skin);ellipsoid(a,(radius,)*3,name,skin);ellipsoid(b,(radius,)*3,name,skin)
# Join all rigid weighted pieces in armature-local coordinates, then use the standard skin modifier.
bpy.ops.object.select_all(action='DESELECT')
for o in parts:o.select_set(True)
bpy.context.view_layer.objects.active=parts[0];bpy.ops.object.join();mesh=bpy.context.object;mesh.name='ProportionMannequin'
bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);mesh.matrix_world=rig.matrix_world.copy();mod=mesh.modifiers.new('Skeleton skin','ARMATURE');mod.object=rig
bpy.ops.wm.save_as_mainfile(filepath=str(P/'PROPORTION_MANNEQUIN_STATIC.blend'))
out=dict(status='LOCAL_PROPORTION_TEST_ASSET_CREATED',production_character=False,downloaded=False,source='Code-generated primitives and named skeleton; no external character geometry reused',
        head_height_cm=24,approx_height_cm=174,shoulder_width_cm=40,upper_arm_cm=30,forearm_cm=27,palm_length_cm=8,finger_chains_per_hand=5,physical_finger_bones_per_hand=15,
        purpose='Human-like proportion and readable hands comparison, not individual body reconstruction or measured anthropometry',
        joints_fixed_length=True,face_markers='Simple designed eyes/nose/mouth, not detected source face landmarks',license_note='Created locally from primitive geometry; no third-party model license claim',vertices=len(mesh.data.vertices),bone_count=len(arm.bones))
(P/'MANNEQUIN_DESIGN.json').write_text(json.dumps(out,indent=2)+'\n');print('PROPORTION_MANNEQUIN_CREATED',out['bone_count'],out['vertices'])
