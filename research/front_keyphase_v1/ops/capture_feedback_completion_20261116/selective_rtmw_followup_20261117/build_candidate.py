from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent;P=D.parent/'selective_rtmw_retry_20261117'
s=(P/'selective_retry.py').read_text(encoding='utf-8')
s=s.replace("VERSION = 'selective_front_rtmw_retry_v1'","VERSION = 'selective_front_rtmw_retry_v2'")
s=s.replace('stream=container.streams.video[0]\n            start=min(wanted)',"stream=container.streams.video[0]\n            # Bounded parallel decoding keeps exact PTS/pixels and all requested frames.\n            stream.thread_type='AUTO'\n            stream.codec_context.thread_count=2\n            start=min(wanted)")
s=s.replace("images=reader.read(candidate['frames'],deadline-.4)","read_started=time.perf_counter()\n        images=reader.read(candidate['frames'],deadline-.4)\n        meta['decode_seconds']=time.perf_counter()-read_started")
s=s.replace("im=images[i]; hand_outputs=[]","im=images[i]; hand_outputs=[]\n            rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)")
s=s.replace('engine.pose(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),bboxes=[bb])','engine.pose(rgb,bboxes=[bb])')
(D/'selective_retry.py').write_text(s,encoding='utf-8')
for name in ['probe.py','retry_runtime.py','serve.py','check_capture.py']:
    (D/name).write_bytes((P/name).read_bytes())
v=(P/'verify_public.py').read_text(encoding='utf-8')
v=v.replace("pins=read(P/'web/ADMISSION.json')['pins']","pins=read(G/'selective_rtmw_retry_20261117/web/ADMISSION.json')['pins']")
v=v.replace("[11,12,16,17]","[3,4,11,12,16,17]")
v=v.replace("natural_sources=[r['id']for r in sources if r['person']==11]","natural_sources=[],extra_control_sources=[r['id']for r in sources if r['person']in[3,4]]")
start=v.index('        # Natural detection at webcam-sized resolution:')
end=v.index('        hashes=all(',start)
v=v[:start]+"        # Natural extracts remain previous evidence; only decoder scheduling changed.\n        assert read(D/'DECODE_PROFILE.json')['pixels_equal']\n"+v[end:]
v=v.replace('Known public F sources, controlled observation losses and 4 natural re-extractions; not physical webcam or general accuracy','Known public F sources; controlled observation losses, added persons3/4 conditions; exact decoded-pixel preservation. Not natural webcam accuracy')
(D/'verify_public.py').write_text(v,encoding='utf-8')
g=(P/'check_guards.py').read_text(encoding='utf-8').replace("read(D/'probe2/","read(G/'selective_rtmw_retry_20261117/probe2/")
(D/'check_guards.py').write_text(g,encoding='utf-8')
note=dict(changes=['Use2 bounded decoder workers with frame threading','Convert each selected BGR image to RGB once and reuse identical RGB for both crops','Record decoder time'],unchanged=['RTMW weights and resolutions','Every selection/rejection threshold','6-frame/300ms/16-call/1.2-second budgets','Both-crop and bracket checks, all21 hand points','Preserve existing criteria/error locations/TCN','No training or interpolation'],diagnosis='Overlap blocks4 controlled cases; inconsistent middle fingertip (joint12) in apple person16 blocks2. Keep those unresolved. Decoder work before reaching selected frames consumes part of same deadline.',scope='Public candidates frozen and qualified before optional private inference-only retrospective. Private recordings never affect code/thresholds/admission.',private_recheck_requested=True,implementation='Codex local, no independent expert review')
(D/'ENGINEERING_NOTE.json').write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf-8')
print('CANDIDATE',hashlib.sha256((D/'selective_retry.py').read_bytes()).hexdigest())
