from common import *
lp=U/'shape_recovery/EXTRA_LABELS.json'
panels=sorted((U/'shape_recovery/panels').glob('*_geometry.jpg'))
write(U/'ROOT_LABEL_REVIEW.json',dict(status='PASS_RESEARCH_LABELS',labels_sha256=sha(lp),reviewer='root direct RGB and RTMW panels; separate Codex pass, not independent human/expert truth',all_12_geometry_panels_reviewed=True,reviewed_frames=36,panels={str(p):sha(p) for p in panels},accepted_scope='visible E/B/F fine states only; no semantic permission or completion truth',limitations=['Thumb B/F not supported','Left middle/ring/little B concentrated in one expression','RGB perspective is not exact joint-angle truth','Known historical development cannot become fresh evaluation'],no_extension_of_egypt_boundary=True))
