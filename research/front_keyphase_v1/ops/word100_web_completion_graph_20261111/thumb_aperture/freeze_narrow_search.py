from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent;D=U/'narrow_search'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    target=U/'NARROW_SEARCH_RGB_LABELS.json';assert not target.exists()
    manifest=D/'SEARCH_AND_FRAMES.json';search=json.loads(manifest.read_text(encoding='utf-8'))
    selected={(4,66):'curled',(4,74):'spread',(4,82):'curled',(7,36):'curled',(7,48):'curled',(7,54):'spreading'}
    rows=[]
    for f in search['frames']:
        kind=selected.get((f['record_id'],f['frame_id']))
        if kind is None:continue
        assert sha(f['rgb'])==f['rgb_sha256'] and sha(f['crop'])==f['crop_sha256']
        rows.append(dict(label_id=f"{f['clip_id']}_f{f['frame_id']:04d}_R",clip_id=f['clip_id'],word_id=f['word_id'],record_id=f['record_id'],hand='R',frame_id=f['frame_id'],seconds=f['seconds'],
          source=f['source'],source_sha256=f['source_sha256'],rgb=f['rgb'],rgb_sha256=f['rgb_sha256'],crop=f['crop'],crop_sha256=f['crop_sha256'],
          observed_state='unknown',applicability='not_applicable' if kind=='curled' else 'unknown',
          direct_visible_four_finger_configuration=kind,
          direct_visible_description='Four fingers curl toward the palm; dorsal hand/knuckles face the camera.' if kind=='curled' else 'Four fingers visibly extend/separate; dorsal hand faces the camera.',
          unknown_reasons=['Thumb tip and its relation to all four fingertips are hidden on the palm side.','Curled fingers are not the same target as bringing the thumb toward an extended four-finger bundle.'] if kind=='curled' else ['Thumb tip remains hidden; extended four fingers do not establish wide or narrow thumb aperture.'],
          exact_contact=None,all_fingertips_gathered=None,existing_four_open_label=None,existing_four_closed_label=None,
          label_origin='single_Codex_direct_RGB_draft',context_inference_promoted_to_label=False,coordinates_seen=False,model_score_seen=False))
    assert len(rows)==6
    result=dict(status='NO_ADDITIONAL_NARROW_TRUTH_SECURED_IN_SCOPED_CACHE_SEARCH',at=datetime.datetime.now().astimezone().isoformat(),
        search_manifest=str(manifest),search_manifest_sha256=sha(manifest),source_cache_coverage=search['cache'],
        inspected=dict(new_expression_candidates=1,public_fit_clips=2,recorded_ids=[4,7],coarse_rgb_frames_viewed=20,native_detail_frames_viewed=6,hand_drafts=6,wide=0,narrow=0,unknown=6),
        rows=rows,
        conclusion_ko='기존 설명 캐시에서 이집트 외에는 별만 전체 손가락을 오므린다는 후보로 찾았다. 새 공개 FIT04/07에서 손가락을 구부렸다 펴는 변화는 보였지만 엄지끝은 손바닥 쪽에 가려졌다. 주먹·검지형과 다른 엄지-편 손가락 묶음 좁힘 정답은 추가 확보하지 못했다.',
        decisions=[
          'Do not relabel star fist/curled frames as narrow extended-bundle aperture.',
          'Do not use WORD2309 or WORD0949 index-thumb touching descriptions as all-fingertip gathering truth.',
          'Do not learn a universal aperture cutoff from Egypt-only narrow positives.',
          'Keep the common aperture measurement experimental and distinguish applicability/visibility from numerical computability.',
          'A later expansion needs directly visible narrow poses in other expressions/recorded IDs; current cached descriptions cover only26 of100 words, so this is unavailability in the audited scope, not proof that no such examples exist.'],
        training_run=False,pose_arrays_read=False,model_scores_read=False,dev_used=False,private_used=False,new_download=False,
        public_archive_read='Exactly one existing FIT07 public video member extracted; FIT04 cached original reused, both source SHA verified.',
        existing_labels_or_models_changed=False,independent_human_or_KSL_expert_truth=False)
    write(target,result)
    write(U/'NARROW_SEARCH_FREEZE.json',dict(path=str(target),sha256=sha(target),source_manifest_sha256=sha(manifest),code_sha256=sha(__file__),no_coordinate_read=True))
    print(json.dumps(result['inspected'],ensure_ascii=False))
if __name__=='__main__':main()
