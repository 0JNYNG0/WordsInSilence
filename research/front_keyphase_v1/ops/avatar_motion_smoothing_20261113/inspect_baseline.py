import bpy,json
from pathlib import Path
U=Path(__file__).resolve().parent
P=U.parent/'avatar_motion_pilot_20261113'
out={}
for label,file in [('static',P/'static_character.blend'),('baseline',P/'public_approximate_preview_body_axis/APPROXIMATE_PUBLIC_WORD1567_REAL02.blend')]:
    bpy.ops.wm.open_mainfile(filepath=str(file),load_ui=False,use_scripts=False)
    rig=next(o for o in bpy.data.objects if o.type=='ARMATURE')
    meshes=[o for o in bpy.data.objects if o.type=='MESH']
    out[label]={'meshes':[{'name':o.name,'materials':[m.name if m else None for m in o.data.materials],'polygon_material_ids':sorted(set(p.material_index for p in o.data.polygons))}for o in meshes],
        'materials':[{'name':m.name,'diffuse':list(m.diffuse_color),'images':[{'image':n.image.name,'filepath':n.image.filepath,'packed':bool(n.image.packed_file),'size':list(n.image.size)}for n in m.node_tree.nodes if n.type=='TEX_IMAGE' and n.image]}for m in bpy.data.materials if m.use_nodes],
        'actions':[{'name':a.name,'legacy':a.is_action_legacy,'fcurves':len(a.fcurves)if a.is_action_legacy else None,'slots':[s.identifier for s in a.slots],'layers':len(a.layers)}for a in bpy.data.actions]}
    if label=='baseline':
        a=rig.animation_data.action
        for l in a.layers:
            for s in l.strips:
                for c in s.channelbags:
                    out[label].setdefault('channelbags',[]).append({'slot':c.slot_handle,'curves':len(c.fcurves),'sample':[(f.data_path,f.array_index,len(f.keyframe_points),f.keyframe_points[0].interpolation)for f in list(c.fcurves)[:5]]})
(U/'BASELINE_INSPECTION.json').write_text(json.dumps(out,indent=2),'utf-8')
print('BASELINE_INSPECTION_COMPLETE')
