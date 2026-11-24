"""Metadata-only next-input routing while semantic feedback is pending."""
from evaluate import H,L,read,sha,write
from datetime import datetime,timezone,timedelta
from pathlib import Path

def main():
    source=L/'evidence/greeting_candidate_source_20260918';meta=read(source/'split_metadata.json')
    out=L/'ops/greeting_semantic_gate_20260922';out.mkdir(exist_ok=False)
    selected=[r for r in meta['rows'] if 'SEN0354' in r['clip_id'] and r['signer_id'] in list(range(1,10))+[11,12,13]]
    assert len(selected)==12
    assert [r['signer_id'] for r in selected if r['split_role']=='train_candidate']==list(range(1,10))
    assert [r['signer_id'] for r in selected if r['split_role']=='development_only']==[11,12,13]
    bindings=[]
    for row in selected:
        assert row['raw_label_expected']=='안녕하세요' and not row['training_approved'] and row['semantic_target_id'] is None
        idx=L/'ops/native_context_readiness_20260921/new_frame_indexes'/('keypoint_index_%02d.json'%row['signer_id'])
        current=read(idx);older=Path(current['archive_stat']['path']);prior=read(older)
        members=[m for m in prior['members'] if m['clip_id']==row['clip_id']]
        arc=Path(current['archive']);st=arc.stat()
        bindings.append(dict(clip_id=row['clip_id'],signer_id=row['signer_id'],role=row['split_role'],
          label_member=row['label_member'],label_member_sha256=row['label_member_sha256'],annotations=row['annotations'],
          archive=str(arc),archive_bytes=st.st_size,archive_mtime_ns=st.st_mtime_ns,
          existing_index=str(older),existing_index_sha256=sha(older),indexed_target_frames=len(members),
          target_members=members,source_label='안녕하세요',candidate_target_id='INSA',semantic_target_id=None,training_approved=False))
    videos=[dict(role='existing_reference',path=str(L.parents[1]/'Tools/videos/인사.mp4')),
            dict(role='AIHub_training_signer01_review_only',path=str(L/'runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4'),annotation_start_s=1.259,annotation_end_s=2.590)]
    for v in videos:v['sha256']=sha(v['path'])
    record=dict(status='WAITING_FOR_SEMANTIC_JUDGMENT_NO_CLASS_MERGE_OR_TRAINING',at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
       question='May the existing 인사 reference and the AIHub 안녕하세요 expression be treated as the same valid semantic class despite form differences?',
       response=None,request_sent=True,source_pins={str(source/n):sha(source/n) for n in ['split_metadata.json','visual_triage.json','decode_and_display.json','acquisition.json']},
       videos=videos,rows=bindings,metadata_counts=dict(train_candidates=9,development=3,reserved_selected=0),
       indexed_clips=sum(b['indexed_target_frames']>0 for b in bindings),
       actual_source_archive_payload_read=False,training_runs=0,inference_runs=0,source_registration=False,
       excluded_other_source='SEN0254 반갑다 is not included in this question or proposal.',
       policy='Do not infer semantic consent from elapsed time or generic continuation. Positive response permits next source/input review, not automatic training or final generalization claims. Negative/uncertain response keeps sources separate. Reuse existing decode and metadata checks.',
       next_if_confirmed='Resolve source frames from already owned archives and prepare native train01-09/dev11-13 inputs using existing verified adapters; preserve original labels and phrase/form provenance. Freeze one data-addition comparison only after preparation checks.',
       next_if_not_confirmed='Keep target mapping null; no merging or pressure to approve. Review another authorized source or prerequisite independently.')
    write(out/'PENDING_SEMANTIC_REVIEW.json',record)
    print(dict(status=record['status'],train_candidates=9,development_candidates=3,indexed_clips=record['indexed_clips'],archive_payload_read=False))

if __name__=='__main__':main()
