from pathlib import Path
P=Path(__file__).resolve().parent;U=P.parent
s=(U/'build_public_approximate_preview.py').read_text()
s=s.replace("U=Path(__file__).resolve().parent;sys.path.insert(0,str(U));V=U/'video_only';BODY_AXIS='--body-axis' in sys.argv","P=Path(__file__).resolve().parent;U=P.parent;sys.path.insert(0,str(U));V=U/'video_only';BODY_AXIS=True")
s=s.replace("D=U/('public_approximate_preview_body_axis'if BODY_AXIS else'public_approximate_preview');D.mkdir(exist_ok=True)","D=P/'MANNEQUIN_PUBLIC_PREVIEW';D.mkdir(exist_ok=True)")
s=s.replace("U/'static_character.blend'","P/'PROPORTION_MANNEQUIN_STATIC.blend'")
s=s.replace("    mat=bpy.data.materials.new('Approximate character silver');mat.diffuse_color=(.48,.57,.67,1);mesh.data.materials.clear();mesh.data.materials.append(mat)\n    for p in mesh.data.polygons:p.material_index=0","    # Preserve explicit face/hand and torso materials for readability.")
s=s.replace("'Existing static NPC rig, no source/private motion imported'","'Locally generated proportion-test mannequin; no external/private animation'")
(P/'retarget_mannequin.py').write_text(s)
(P/'FIRST_PREPARATION_ERROR.json').write_text('{"error":"Initial inline PowerShell/Python preparation command had quoting syntax error; no script or asset changed. Recovered with separate Python file.","source_mutation":false}\n')
print('ISOLATED_RETARGET_PREPARED')
