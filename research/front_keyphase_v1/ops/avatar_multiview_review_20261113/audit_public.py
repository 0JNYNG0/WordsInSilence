"""Read-only public FIT01 five-view inspection; never loads private recordings."""
from pathlib import Path
import json, zipfile, hashlib, re
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DATA = Path('W:/WordsInSilence/data/AIHub_103')
ARCHIVE = DATA / 'videos/1.Training/[원천]01_real_word_video.zip'
LABELS = DATA / 'annotations/1.Training/[라벨]01_real_word_morpheme.zip'
WORD_IDS = ['1133', '1567']
ALLOWED = re.compile(r'NIA_SL_WORD(1133|1567)_REAL01_[FDULR]\.mp4$')
out = {'scope':'public FIT record01 only; no private sources; two words; no model inference', 'source_archive':str(ARCHIVE), 'source_labels':str(LABELS), 'clips':[], 'labels':[], 'archive_audit':{}}
video_dir = HERE/'public_fit01_samples'
video_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(ARCHIVE) as z:
    out['archive_audit']['members_total'] = len(z.infolist())
    out['archive_audit']['non_video_member_names'] = [x.filename for x in z.infolist() if not x.is_dir() and not x.filename.lower().endswith('.mp4')]
    out['archive_audit']['calibration_named_members'] = [x.filename for x in z.infolist() if re.search(r'calib|intrinsic|extrinsic|camera|\.(yaml|yml|xml)$',x.filename,re.I)]
    selected = [x for x in z.infolist() if ALLOWED.search(x.filename)]
    assert len(selected) <= 10
    for entry in selected:
        name = Path(entry.filename).name
        assert ALLOWED.fullmatch(name), name
        dest = video_dir/name
        payload = z.read(entry)
        digest = hashlib.sha256(payload).hexdigest()
        if not dest.exists(): dest.write_bytes(payload)
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
        cap = cv2.VideoCapture(str(dest))
        rec = {'member':entry.filename,'bytes':entry.file_size,'crc32':f'{entry.CRC:08x}','sha256':digest,'fps_reported':cap.get(cv2.CAP_PROP_FPS),'frames_reported':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),'sample_indices':[], 'sample_read_ok':[]}
        # Full timestamp scan uses decoder only, no landmark/model estimation.
        times=[]; thumbs=[]; j=0
        targets={0,30,45,60,75,90,105,120}
        while True:
            ok, frame=cap.read()
            if not ok: break
            times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
            if j in targets:
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                im=Image.fromarray(rgb); im.thumbnail((400,300))
                p=HERE/f'{Path(name).stem}_frame{j:03d}.jpg'; im.save(p,quality=91)
                rec['sample_indices'].append(j); rec['sample_read_ok'].append(True)
            j+=1
        cap.release()
        rec.update(decoded_frames=j,first_pos_msec=times[0] if times else None,last_pos_msec=times[-1] if times else None,timestamps_monotone=all(a<b for a,b in zip(times,times[1:])),timestamp_step_min_ms=min((b-a for a,b in zip(times,times[1:])),default=None),timestamp_step_max_ms=max((b-a for a,b in zip(times,times[1:])),default=None),pos_msec=times)
        out['clips'].append(rec)

with zipfile.ZipFile(LABELS) as z:
    out['label_archive_member_count'] = len(z.infolist())
    out['label_calibration_named_members'] = [x.filename for x in z.infolist() if re.search(r'calib|intrinsic|extrinsic|camera|\.(yaml|yml|xml)$',x.filename,re.I)]
    for x in z.infolist():
        if re.search(r'NIA_SL_WORD(1133|1567)_REAL01_[FDULR]',x.filename) and x.filename.endswith('.json'):
            record=json.loads(z.read(x))
            out['labels'].append({'member':x.filename, 'payload':record})

font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
for w in WORD_IDS:
    # Same frame number is a visual audit probe, not a claim of exact simultaneity.
    canvas=Image.new('RGB',(1550,1130),'white'); d=ImageDraw.Draw(canvas)
    d.text((15,10),f'공개 WORD{w} / REAL01 · 5뷰 동일 인덱스 비교 (정밀 동기화 미검증)',fill='#172b49',font=font)
    for ri,k in enumerate([30,60,90,120]):
        for ci,v in enumerate('FLRUD'):
            p=HERE/f'NIA_SL_WORD{w}_REAL01_{v}_frame{k:03d}.jpg'
            x=ci*310+10; y=ri*265+45
            d.text((x,y),f'{v} · 프레임 {k}',fill='#172b49',font=font)
            if p.exists():
                im=Image.open(p); im.thumbnail((300,225)); canvas.paste(im,(x,y+30))
    canvas.save(HERE/f'WORD{w}_five_view_contact_sheet.jpg',quality=91)
(HERE/'PUBLIC_MEDIA_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'clips':[{k:c[k] for k in ('member','fps_reported','frames_reported','decoded_frames','last_pos_msec','timestamp_step_min_ms','timestamp_step_max_ms')} for c in out['clips']],'labels':len(out['labels']),'archive_audit':out['archive_audit']},ensure_ascii=False,indent=2))
