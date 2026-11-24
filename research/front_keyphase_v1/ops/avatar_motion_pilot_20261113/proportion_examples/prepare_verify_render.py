from pathlib import Path
P=Path(__file__).resolve().parent;U=P.parent
for src,dst in [('render_public_sequence.py','render_mannequin.py'),('verify_public_fbx.py','verify_mannequin_fbx.py')]:
    s=(U/src).read_text()
    s=s.replace("U=Path(__file__).resolve().parent;D=U/('public_approximate_preview_body_axis'if '--body-axis'in sys.argv else'public_approximate_preview')","U=Path(__file__).resolve().parent;D=U/'MANNEQUIN_PUBLIC_PREVIEW'")
    (P/dst).write_text(s)
print('PREPARED_RENDER_VERIFY')
