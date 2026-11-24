"""Freeze direct RGB thumb-aperture drafts before any new coordinate inspection."""
from pathlib import Path
import json, hashlib, datetime, shutil
import cv2
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw

U=Path(__file__).resolve().parent
O=U.parent.parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    target=U/'EXPANDED_RGB_LABELS.json'
    assert not target.exists(), 'RGB label freeze is immutable.'
    out=U/'expanded_rgb'; out.mkdir(exist_ok=True)
    bundles={
      'X':O/'word100_shared_feature_bundle_20261027/visual_inputs/INDEX.json',
      'V':O/'word100_source_token_alignment_20261027/visual_inputs/INDEX.json'}
    idx={r['alias']:r for p in bundles.values() for r in read(p)['rows']}
    ready={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']}
    # Fixed pre-existing sampled ordinals, selected before any coordinate/model values.
    choices=[('X01',8),('X02',8),('X03',8),('X16',8),('X17',8),('X18',8),
             ('V10',5),('V10',11),('V11',5),('V11',11),('V12',5),('V12',11)]
    frames=[]; rows=[]; rgb_verification=[]; verified_sources={}
    for alias,ordinal in choices:
        r=idx[alias]; cid=r['clip_id']; meta=ready[cid]
        assert meta['role']=='fit' and meta['person'] in [1,2,3]
        assert r['source_sha256']==meta['source_sha256']
        frame=next(f for f in r['frames'] if f['ordinal']==ordinal)
        src=Path(frame['file'])
        # An inherited PNG checksum differs although its decoded pixels may still be
        # exact. Never bless that mismatch by replacing the old provenance record:
        # verify the actual public video SHA and exact frame pixels independently.
        source=Path(r['source'])
        if str(source) not in verified_sources:
            verified_sources[str(source)]=sha(source)
        assert verified_sources[str(source)]==r['source_sha256']
        cap=cv2.VideoCapture(str(source)); cap.set(cv2.CAP_PROP_POS_FRAMES,frame['frame_id'])
        ok,decoded=cap.read(); cap.release(); actual=cv2.imread(str(src))
        assert ok and decoded.shape==actual.shape and np.array_equal(decoded,actual), 'Original source-frame pixel mismatch'
        rgb_verification.append(dict(alias=alias,frame_id=frame['frame_id'],source=str(source),source_sha256=r['source_sha256'],
            index_png_sha256=frame['rgb_sha256'],actual_png_sha256=sha(src),file_hash_matches_index=sha(src)==frame['rgb_sha256'],
            exact_original_frame_pixels=True,max_pixel_difference=0))
        stem=f"{r['word_id']}_REAL{meta['person']:02d}_f{frame['frame_id']:04d}"
        copied=out/(stem+'_full.png'); shutil.copyfile(src,copied)
        im=Image.open(copied); assert im.size==(1920,1080)
        # Manual fixed screen crop only, never detector-derived cropping or enhancement.
        box=(640,400,1300,700) if alias.startswith('X') else (640,0,1230,1080)
        crop=out/(stem+'_crop.png'); im.crop(box).save(crop)
        f=dict(alias=alias,ordinal=ordinal,clip_id=cid,word_id=r['word_id'],record_id=meta['person'],
               frame_id=frame['frame_id'],timestamp_us=frame['pts_us'],source_sha256=r['source_sha256'],
               current_role='fit',historical_role=r.get('role'),original_rgb=str(src),original_rgb_sha256=sha(src),
               rgb_copy=str(copied),rgb_copy_sha256=sha(copied),crop=str(crop),crop_sha256=sha(crop),crop_xyxy=list(box),
               provenance_index=str(bundles[alias[0]]),provenance_index_sha256=sha(bundles[alias[0]]))
        frames.append(f)
        for hand in ['R','L']:
            state='unknown'; applicability='unknown'; observed=[]; reasons=[]
            if r['word_id']=='WORD0003':
                applicability='not_applicable'
                observed=['index extended separately from curled remaining fingers']
                reasons=['index_only_configuration_not_four_finger_bundle','thumb relation to complete finger bundle is not the same construct as the Egypt aperture draft']
            elif r['word_id']=='WORD1133':
                if hand=='R':
                    applicability='applicable'; state='wide_projected_aperture'
                    observed=['extended finger bundle viewed approximately edge-on','thumb projects away from the finger bundle with a visible triangular gap']
                    reasons=['exact fingertip contact and out-of-plane angle remain unobserved']
                else:
                    applicability='not_applicable'
                    observed=['four fingers visibly curled into a fist','thumb projects beside the curled fingers']
                    reasons=['fist_is_not_narrow_extended_finger_bundle','do not treat a small thumb-index distance as a fingertip-gathering positive']
            else:
                if hand=='L':
                    observed=['lowered non-signing hand near lower body','relaxed partially side-on hand']
                    reasons=['inactive_hand','complete fingertip and thumb relationship not sufficiently distinct']
                elif ordinal==5 and meta['person'] in [1,2]:
                    applicability='not_applicable'
                    observed=['curled fingers in front of forehead','back of hand faces camera']
                    reasons=['curled_finger_configuration','thumb tip hidden on palm side; cannot label narrow from missing silhouette']
                else:
                    observed=['four fingers are visibly extended/spread','back of hand faces camera']
                    reasons=['thumb_occluded_on_palm_side','four_open_does_not_prove_wide_thumb_aperture']
            rows.append(dict(label_id=f"{stem}_{hand}",clip_id=cid,word_id=r['word_id'],record_id=meta['person'],hand=hand,
                frame_id=frame['frame_id'],timestamp_us=frame['pts_us'],source_sha256=r['source_sha256'],
                rgb_file=str(copied),rgb_sha256=sha(copied),crop_file=str(crop),
                observed_state=state,applicability=applicability,visible_features=observed,unknown_or_exclusion_reasons=reasons,
                observation_kind='direct_single_Codex_RGB_draft',context_inference_used_for_state=False,
                exact_fingertip_contact=None,all_fingers_gathered_semantic=None,
                four_open_existing_head_label=None,four_closed_existing_head_label=None,
                coordinates_seen_before_label=False,model_score_seen=False))
    assert len(rows)==24 and len(frames)==12 and len({r['clip_id'] for r in rows})==9
    result=dict(status='FROZEN_RGB_DRAFTS_BEFORE_COORDINATE_REVIEW',created_at=datetime.datetime.now().astimezone().isoformat(),
        author='Codex graph_label_audit',independent_human_or_KSL_expert_ground_truth=False,
        construct='visible projected thumb-to-extended-four-finger-bundle aperture; not fist/open class, contact, or educational correctness',
        selection_policy='Three additional expressions (0003,1133,2551), recorded IDs01/02/03, fixed existing frame slots8 or5/11. No coordinate or model value used to choose or label frames. RGB known context and nonblind expression selection remain limitations.',
        limit='24 handframes maximum; unknowns retained in denominator. Two hands and two moments from one clip are correlated, not independent examples.',
        counts=dict(handframes=len(rows),frames=len(frames),clips=len({r['clip_id'] for r in rows}),expressions=3,recorded_ids=[1,2,3],
                    state=dict(Counter(r['observed_state'] for r in rows)),applicability=dict(Counter(r['applicability'] for r in rows))),
        findings=['Additional clearly wide drafts exist for right flat hand of WORD1133 in all three recorded IDs.',
          'No additional clear narrow extended-finger-bundle positive was established in these fixed images.',
          'Index-only and fist configurations must not become narrow/gathered labels.',
          'Four extended fingers in WORD2551 do not reveal the thumb hidden on the palm side.'],
        downstream_recommendation='Keep as small FIT observability/applicability audit. Do not train or deploy a common aperture head from an Egypt-only narrow class; add independently visible narrow poses/rotations in other expressions first.',
        initial_read_errors=[{'path_attempted':str(O/'word100_source_token_alignment_20261027/FROZEN.json'),'error':'FileNotFoundError','resolution':'Existing visual_inputs/INDEX.json provides source/frame metadata; no source modified.'},
          {'error':'AssertionError at original PNG checksum verification','first_alias':'X01','first_frame_id':46,
           'index_sha256':'8f415b854780e38a1b1050cb4c629fd98974d4f54dbb561aa6af857fb60d1971','actual_sha256':'61a7dbffd81fe4354ecd2a4889f244f45402e8928a2dd910b66accb5bc8b8517',
           'resolution':'Source video SHA verified, exact decoded original-frame pixels verified for every selected PNG. Old index not rewritten. The reason for binary PNG change is unknown.'}],
        source_rgb_verification=rgb_verification,
        private_access=False,dev_loaded=False,training_ran=False,model_inference_ran=False,coordinates_loaded=False,
        existing_labels_modified=False,frames=frames,rows=rows)
    write(target,result)
    write(U/'EXPANDED_RGB_FREEZE.json',dict(label_file=str(target),sha256=sha(target),frozen_before_coordinate_read=True,
        label_counts=result['counts'],code_sha256=sha(__file__)))
    print(json.dumps(result['counts'],ensure_ascii=False))

if __name__=='__main__': main()
