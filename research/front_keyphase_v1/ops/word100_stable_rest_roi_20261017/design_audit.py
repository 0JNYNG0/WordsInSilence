"""Independent visual evidence only. No detector/model/candidate execution."""
from pathlib import Path
import json, sys, hashlib
from datetime import datetime, timezone, timedelta
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
U = Path(__file__).resolve().parent
D = U/'design_images'
D.mkdir(exist_ok=True)
PLAN = json.loads((U/'PLAN.json').read_text(encoding='utf-8'))
FONT = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 19)
SMALL = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 16)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frame(row, idx):
    cap = cv2.VideoCapture(row['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, bgr = cap.read()
    cap.release()
    if not ok: raise ValueError((row['clip_id'], idx))
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def panel(row, idx, title, obs=None):
    im = frame(row, idx)
    draw = ImageDraw.Draw(im)
    old = row['wrist_box'][3]; new = row['stable_box'][3]
    draw.line((0,old,im.width,old),fill=(255,140,0),width=3)
    draw.line((0,new,im.width,new),fill=(0,210,140),width=3)
    data = np.load(row['cached']['native']['observation']) if obs is None else obs
    for pi in [11,12,13,14,33,54]:
        if data['observed'][idx,pi] and np.isfinite(data['xy_px'][idx,pi]).all():
            x,y = data['xy_px'][idx,pi]
            draw.ellipse((x-6,y-6,x+6,y+6),outline=(255,30,220),width=3)
    im = ImageOps.contain(im,(420,360))
    out = Image.new('RGB',(420,400),'white');out.paste(im,((420-im.width)//2,40+(360-im.height)//2))
    ImageDraw.Draw(out).text((6,8),f'{title} / frame {idx}',font=SMALL,fill='black')
    return out

def calibration_images():
    rows=[]
    for page in range(5):
        sheet=Image.new('RGB',(1680,1760),'white');sd=ImageDraw.Draw(sheet)
        sd.text((10,8),'Calibration source samples | orange=previous bottom, green=stable bottom, pink=observed anchors',font=SMALL,fill='black')
        for j,row in enumerate(PLAN['items'][page*4:(page+1)*4]):
            ids=row['calibration']['selected_frame_indices']; yy=35+j*430
            sd.text((8,yy),f"{row['label']} {row['clip_id']} | selected {ids[0]}..{ids[-1]}, bottom {row['wrist_box'][3]} -> {row['stable_box'][3]}",font=FONT,fill='black')
            pts=[0,ids[0],ids[len(ids)//2],ids[-1]]
            for col,(i,title) in enumerate(zip(pts,['source first','selected start','selected middle','selected end'])):
                sheet.paste(panel(row,i,title),(col*420,yy+30))
            z=np.load(row['cached']['native']['observation'])
            xy=z['xy_px'][ids]; selected=xy[:,[11,12,33,54]]
            width=float(np.median(np.linalg.norm(xy[:,11]-xy[:,12],axis=1)))
            dist=np.linalg.norm(selected-np.median(selected,axis=0),axis=2)
            rows.append(dict(clip_id=row['clip_id'],label=row['label'],selected_frame_indices=ids,
                sample_frames=pts,selected_duration_us=row['calibration']['duration_us'],
                previous_bottom=row['wrist_box'][3],new_bottom=row['stable_box'][3],
                independent_shoulder_width=width,independent_max_anchor_displacement=float(dist.max()),
                independent_max_anchor_displacement_ratio=float(dist.max()/width),
                source_sha256_matches=sha(row['video_path'])==row['source_sha256'],
                contact_sheet=f'design_images/calibration_samples_{page+1}.png'))
        sheet.save(D/f'calibration_samples_{page+1}.png')
    row=next(x for x in PLAN['items'] if x['clip_id']=='NIA_SL_WORD1133_REAL01_F')
    detail=Image.new('RGB',(2100,1240),'white');dd=ImageDraw.Draw(detail)
    dd.text((10,8),'잘하다 source frames 0..12 | selected stable window 3..12 | orange previous / green stable bottom',font=FONT,fill='black')
    for i in range(13):detail.paste(panel(row,i,'selected' if i>=3 else 'before selection'),((i%5)*420,40+(i//5)*400))
    detail.save(D/'settling_word1133_all0_12.png')
    (D/'calibration_evidence.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(stage='calibration_images',rows=len(rows),images=6),ensure_ascii=False))

def compare_panel(row,idx,obs,title,box):
    im=frame(row,idx); draw=ImageDraw.Draw(im)
    if box:
        draw.rectangle((0,box[3],im.width,im.height),fill=(80,80,80))
        draw.line((0,box[3],im.width,box[3]),fill=(0,220,140),width=4)
    counts=[]
    for side,sl,color in [('L',slice(33,54),(255,70,70)),('R',slice(54,75),(30,255,255))]:
        mask=obs['usable'][idx,sl];counts.append(f'{side}:{int(mask.sum())}')
        for pi in range(sl.start,sl.stop):
            if obs['observed'][idx,pi] and np.isfinite(obs['xy_px'][idx,pi]).all():
                x,y=obs['xy_px'][idx,pi]
                draw.ellipse((x-6,y-6,x+6,y+6),outline=color,width=3)
    im=ImageOps.contain(im,(560,460))
    out=Image.new('RGB',(560,510),'white');out.paste(im,((560-im.width)//2,50+(460-im.height)//2))
    ImageDraw.Draw(out).text((8,5),title+' | '+','.join(counts),font=SMALL,fill='black')
    return out

def event_images():
    events=[]; endpoint=[]; prior=U.with_name('word100_wrist_roi_pair_20261017')
    for row in PLAN['items']:
        cid=row['clip_id']; cid_events=[]
        for mode in ['full','sensor']:
            a=np.load(prior/'outputs'/cid/f'wrist_{mode}'/'model_input.npz')
            b=np.load(U/'outputs'/cid/f'stable_{mode}_raw'/'model_input.npz')
            g=np.load(U/'outputs'/cid/f'stable_{mode}_guarded'/'model_input.npz')
            for side,sl in [('left',slice(33,54)),('right',slice(54,75))]:
                am=a['usable'][:,sl].all(1);bm=b['usable'][:,sl].all(1)
                for kind,mask in [('all21_loss',am&~bm),('all21_gain',~am&bm)]:
                    for fi in np.flatnonzero(mask):
                        cid_events.append(dict(clip_id=cid,label=row['label'],source_mode=mode,kind=kind,side=side,frame_index=int(fi),
                            old_point_ids=np.flatnonzero(a['observed'][fi,sl]).tolist(),new_point_ids=np.flatnonzero(b['observed'][fi,sl]).tolist()))
                removed=b['observed'][:,sl]&~g['observed'][:,sl]
                trace=read(U/'outputs'/cid/f'stable_{mode}_guarded'/'guardtrace.json')
                for fi in np.flatnonzero(removed.any(1)):
                    cid_events.append(dict(clip_id=cid,label=row['label'],source_mode=mode,kind='guard_suppressed',side=side,frame_index=int(fi),
                        removed_local_indices=np.flatnonzero(removed[fi]).tolist(),trace=trace['frames'][fi][side]))
                # Endpoint inventory independently read from saved observations.
                for label,idxs in [('first5',range(5)),('last5',range(len(g['frame_id'])-5,len(g['frame_id']))),
                                  ('last_half_second',np.flatnonzero(g['timestamp_us'][-1]-g['timestamp_us']<=500000))]:
                    for fi in idxs:
                        if g['observed'][fi,sl].any():
                            endpoint.append(dict(clip_id=cid,label=row['label'],source_mode=mode,window=label,side=side,frame_index=int(fi),
                                local_point_ids=np.flatnonzero(g['observed'][fi,sl]).tolist(),trace=trace['frames'][fi][side]))
            for event in [e for e in cid_events if e['source_mode']==mode]:
                fi=event['frame_index']; kind=event['kind'];before=b if kind=='guard_suppressed' else a
                after=g if kind=='guard_suppressed' else b
                image=Image.new('RGB',(1680,560),'white');draw=ImageDraw.Draw(image)
                draw.text((10,8),f"{row['label']} {cid} frame{fi} {mode} {kind} {event['side']} | source, before, after",font=FONT,fill='black')
                native=np.load(row['cached']['native']['observation'])
                for col,(o,title,box) in enumerate([(native,'source full observation',None),
                    (before,'before',row['stable_box'] if kind=='guard_suppressed' else row['wrist_box']),
                    (after,'after',row['stable_box'])]):
                    image.paste(compare_panel(row,fi,o,title,box),(col*560,50))
                name=f'{cid}_{mode}_{kind}_{event["side"]}_{fi}.png';image.save(D/name)
                event['image']='design_images/'+name
        events.extend(cid_events)
    (D/'event_evidence.json').write_text(json.dumps(dict(events=events,endpoint_residuals=endpoint),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(events=events,endpoint_residuals=endpoint),ensure_ascii=False))

def endpoint_images():
    row=next(r for r in PLAN['items'] if 'WORD1844_' in r['clip_id'])
    trace=read(U/'outputs'/row['clip_id']/'stable_full_guarded/guardtrace.json')
    out=Image.new('RGB',(1680,2040),'white');draw=ImageDraw.Draw(out)
    draw.text((8,8),'Hongseong last 0.5s: suppressed only93,94,95; others include movement reset / dwell / outside-boundary gaps',font=SMALL,fill='black')
    for n,i in enumerate(range(90,106)):
        out.paste(panel(row,i,trace['frames'][i]['left']['reason']),((n%4)*420,40+(n//4)*400))
    injury=next(r for r in PLAN['items'] if 'WORD1519_' in r['clip_id'])
    for n,i in enumerate([92,94,99,107]):out.paste(panel(injury,i,'상처 residual f92' if i==92 else '상처 endpoint sample'),(n*420,1640))
    out.save(D/'rest_endpoint_source_review.png')
    print('endpoint image generated')

def finalize():
    calibrations=read(D/'calibration_evidence.json')
    evidence=read(D/'event_evidence.json')
    captures={r['clip_id']:r for r in read(U/'CAPTURE_RESULT.json')['rows']}
    for r in calibrations:
        r['visual_review']='Selected start/middle/end samples show both arms down, with no visible gesture initiation in those samples.'
        r['visual_exhaustive_selected_window']=False
        if 'WORD1133_' in r['clip_id']:
            r['visual_review']='Frames0..12 reviewed exhaustively. Frames0..2 show initial lowering/settling. Selected3..12 have arms down but slight continuing settling early in the selected window; bounded stability tolerance, not proof of complete physical stillness. No sign onset visible in0..12.'
            r['visual_exhaustive_selected_window']=True
    for e in evidence['events']:
        if e['kind']=='guard_suppressed':
            e['visual_review']='Both arms visibly down at rest. Only a boundary wrist point was removed; no finger points existed in the masked input.'
        elif e['kind']=='all21_loss' and e['source_mode']=='sensor':
            z=np.load(captures[e['clip_id']]['raw']);i=e['frame_index']
            e['new_raw_candidate_count']=int(z['raw_count'][i])
            e['new_raw_candidate_labels']=z['raw_label_name'][i,:int(z['raw_count'][i])].tolist()
            e['visual_review']='Missing hand is visibly well above new bottom during a gesture. Loss already exists in new raw detector output; neither geometric boundary removal nor temporal guard caused this all21 loss. Observation quality/recognition accuracy not established.'
        elif e['kind']=='all21_loss':
            e['visual_review']='Left hand is low while right hand is raised. New bottom915 versus916 excludes exactly local hand point8 (index fingertip), making21->20. Fixed-mask geometric loss, not temporal guard or new detector loss.'
        elif e['source_mode']=='sensor' and 'WORD0003_' in e['clip_id']:
            e['visual_review']='Left hand visibly inside both crops; new detection includes the hand that was absent before. Adjacent frame35 loses it, so gain34/36 is mixed intermittent observation, not complete tracking resolution.'
        else:
            e['visual_review']='Lower visible left hand gains included points at expanded boundary. Frame43 is during gesture, frame86 is lowering toward rest. These are availability changes, not recognition gains or expert phase labels.'
    findings=[
      dict(id='CALIBRATION_SELECTION',detail='20/20 calibrate and sampled selected poses are arms down.19 select0..9; 잘하다 selects3..12 with remaining small settling. The fixed motion tolerance admits maximum0.044249S displacement in that window; accepted stability is operational, not literal stillness.',evidence='design_images/settling_word1133_all0_12.png'),
      dict(id='STARTUP_RESIDUE_INCREASED',detail='잘하다 lower boundary877->892 increases first5 native hand point-frames5->12. Guard removes none: frames0/1 contain observed finger bases, and brief wrist-only frames2/3 have insufficient dwell. Same sensor first5 remains3 points. Calibration validity does not imply complete exclusion of pre-calibration settling.'),
      dict(id='ALL_ACTUAL_SUPPRESSIONS_REVIEWED',detail='Only native/full 홍성 anatomical-left hand wrist at frames93,94,95 is suppressed,3 point-frames total. All3 source frames visibly rest. Sensor guard removes0. All finger/palm-bearing hands remain unaffected by the guard.'),
      dict(id='ENDPOINT_REST_REMAINS',detail='홍성 last0.5s90..105 reviewed exhaustively; all show arms down. Guard suppresses93..95, resets at96 due to measured movement, then99..101 wrist lies outside ROI, resets again, leaving102..105 dwell-pending. Last5 residual4points; improvement5->4 is boundary change, not guard (the guard removes no last5 points). 상처 frame92 still retains1 wrist point with movement reset and visibly rests.'),
      dict(id='GEOMETRIC_ACTIVE_POINT_LOSS',detail='상처 frame79 full mask loses left index tip because calibrated bottom is1px higher. Hand21->20. Previous source monotonicity guarantee no longer applies because some new bounds shrink1..2px.'),
      dict(id='ACTUAL_CROP_MIXED_DETECTION',detail='All4 sensor all21 losses reviewed: 수어35, 잘하다47, 별35, 상처38. Hands visibly inside new crop; new raw candidate counts are1,1,0,0 respectively. No guard involvement. Three sensor gains at수어34/36 and잘하다86 do not establish uniformly improved detection.'),
      dict(id='GUARD_SAFETY_LIMIT',detail='Observed actual suppressions concern rest only, but only3 frame-sides were suppressed. This is not a general proof of preserving all low gestures; a sufficiently stationary wrist-only low gesture at the same anchor can satisfy the rule. No expert semantic segment labels were produced.'),
      dict(id='OFFLINE_VS_WEBCAM',detail='Calibration uses full-source initial wrist pixels, including pixels later outside the fixed ROI. Cropped-runtime guard uses visible evidence, but a physical webcam with hidden resting wrists cannot acquire the same initial anchors. Startup lookahead and readiness handling were not tested with a camera.'),
      dict(id='NO_ADOPTION_CLAIM',detail='The single frozen candidate produces mixed changes and incomplete idle exclusion. Preserve this diagnostic result; do not recommend guard adoption or infer recognition improvement/training readiness. Do not retune this20-case result.')]
    out=dict(task_id='word100_stable_rest_roi_20261017',reviewer='Codex independent subagent stable_roi_design',
      updated_at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),status='VISUAL_DESIGN_AUDIT_COMPLETE_MIXED_NO_ADOPTION_RECOMMENDED',
      scope='Same20 public TRAIN engineering clips; no user/dev/test/camera/training/model inference. Read-only source and frozen results.',
      plan_sha256=sha(U/'PLAN.json'),scope_sha256=sha(U/'SCOPE.json'),comparison_sha256=sha(U/'COMPARISON.json'),
      methods=dict(visual='20 clips source-first/selected-start/middle/end80panels; 잘하다0..12 all13; all3 suppression frames; all5 all21 losses and5 gains; 홍성90..105 all16 and상처92/94/99/107.',
        coordinate='Direct independent npz masks for event enumeration, source video hash checks, manual source image review. Candidate code/detector not executed.',
        display='All panels preserve full original image extent and aspect ratio via letterboxing. No display zoom/crop. Compared-input panels gray the excluded bottom; source panel shows the entire original.',
        limitation='19 calibration windows are sampled visually, not reviewed at every frame. Stability arithmetic is not semantic rest ground truth. Panel RGB outside ROI is grayed solely for presentation.'),
      summary=dict(calibrations=20,calibration_source_hash_matches=sum(r['source_sha256_matches'] for r in calibrations),selected_pose_samples_arms_down=20,
        literal_complete_stillness_guaranteed=False,actual_guard_removed_point_frames=3,all_actual_guard_suppressions_visually_rest=True,
        sensor_guard_removed_point_frames=0,all21_loss_events_reviewed=5,all21_gain_events_reviewed=5,images_reviewed=20,
        recognition_claim=False,adoption_recommended=False),findings=findings,calibration_rows=calibrations,events=evidence['events'],endpoint_residuals=evidence['endpoint_residuals'],
      image_sha256={str(p.relative_to(U)).replace('\\','/'):sha(p) for p in sorted(D.glob('*.png'))},
      executed=dict(new_detector=0,training=0,classifier_inference=0,production_changes=0,git_mutations=0,root_document_changes=0))
    (U/'DESIGN_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(out['summary'],ensure_ascii=False))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='events': event_images()
    elif len(sys.argv)>1 and sys.argv[1]=='endpoints': endpoint_images()
    elif len(sys.argv)>1 and sys.argv[1]=='finalize': finalize()
    else: calibration_images()
