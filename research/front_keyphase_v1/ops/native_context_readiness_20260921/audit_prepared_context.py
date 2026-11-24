"""Verify assembly, metadata eligibility and split preservation without rereading ZIP payloads."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib
import json
import math
import numpy as np

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
eligibility=read(HERE/'eligibility.json'); source_docs=read(HERE/'source_annotations.json')
prepared=read(HERE/'coordinates.json'); plan=read(HERE/'coordinate_plan.json')
assert sha(HERE/'source_annotations.json')==eligibility['source_annotations_sha256']
assert prepared['status']=='PREPARED_PENDING_INDEPENDENT_AUDIT' and not prepared['failures']
assert sha(HERE/'coordinate_plan.json')==prepared['plan_sha256']
independent_eligible={}
for row in eligibility['rows']:
    doc=source_docs[row['clip_id']]['document']; annotations=doc['data']
    possible=len(annotations)==1 and len(annotations[0].get('attributes',[]))==1 and len(row['source_samples'])==1
    if possible:
        s=row['source_samples'][0]; a=annotations[0]
        possible=(s['annotation_index']==0 and a['attributes'][0]['name']==s['raw_label']
                  and a['start']==s['start_s'] and a['end']==s['end_s']
                  and 0<=a['start']<a['end']<=doc['metaData']['duration'])
    assert possible==(row['status']=='ELIGIBLE_BY_SOURCE_ANNOTATION')
    if possible:independent_eligible[row['clip_id']]=row
assert len(independent_eligible)==298
assert len(prepared['rows'])==298 and {r['clip_id'] for r in prepared['rows']}==set(independent_eligible)
index_cache={}; total=new=middle=0; roles=Counter(); signers=defaultdict(set); labels=defaultdict(set)
no_context=[]; first_indices=Counter(); timestamp_end_discrepancy=[]; outputs=[]
for row in prepared['rows']:
    assert row['status']=='PREPARED' and sha(row['npz'])==row['npz_sha256']
    source=independent_eligible[row['clip_id']]; signer=row['signer_id']; role=row['split_role']
    assert signer==source['signer_id'] and role==source['split_role']
    assert (role=='train' and signer in range(1,10)) or (role=='validation' and signer in (11,12,13))
    roles[role]+=1;signers[role].add(signer);labels[role].add(row['reference_label'])
    index_path=row['full_index_source']
    if index_path not in index_cache:
        info=read(index_path); byclip=defaultdict(list)
        for member in info['members']:byclip[member['clip_id']].append(member)
        index_cache[index_path]=byclip
    indexed=sorted(int(m['name'].rsplit('_F_',1)[1].split('_')[0]) for m in index_cache[index_path][row['clip_id']])
    assert len(indexed)==len(set(indexed))
    assert sha(row['original_annotation_npz'])==row['original_annotation_sha256']
    with np.load(row['npz'],allow_pickle=False) as full, np.load(row['original_annotation_npz'],allow_pickle=False) as old:
        frames=full['frame_index']; fi=frames.tolist();n=len(fi)
        assert fi==indexed and fi==list(range(fi[0],fi[-1]+1))
        assert n==row['frames'] and not row['gaps']
        old_idx=old['frame_index']; mapping={v:i for i,v in enumerate(fi)}
        keep=np.array([mapping[int(v)] for v in old_idx],dtype=int)
        np.testing.assert_array_equal(full['annotation_source_rows'],keep)
        for key in ('xy_native','confidence_native','coordinate_present','timestamp_nominal_us'):
            np.testing.assert_array_equal(full[key][keep],old[key])
        assert len(old_idx)==row['reused_annotation_frames']==row['annotation_frames']
        assert n-len(old_idx)==row['new_crc_checked_frames']
        before=sum(i<int(old_idx[0]) for i in fi);after=sum(i>int(old_idx[-1]) for i in fi)
        assert before==row['before_annotation_frames'] and after==row['after_annotation_frames']
        assert before+len(old_idx)+after==n
        if before==0 or after==0:no_context.append(row['clip_id'])
        expected_ts=np.array([round(i/30*1e6) for i in fi],dtype=np.int64)
        np.testing.assert_array_equal(full['timestamp_nominal_us'],expected_ts)
        present=np.isfinite(full['xy_native']).all(-1)&np.isfinite(full['confidence_native'])&np.any(full['xy_native']!=0,axis=-1)
        np.testing.assert_array_equal(full['coordinate_present'],present)
        assert full['x'].shape==(n,280) and np.isfinite(full['x']).all()
        assert full['x'][0,-1]==0
        np.testing.assert_allclose(full['x'][1:,-1],np.diff(expected_ts)/1e6,rtol=0,atol=1e-8)
        total+=n;middle+=len(old_idx);new+=n-len(old_idx);first_indices[fi[0]]+=1
        timestamp_end_discrepancy.append(fi[-1]/30-source['duration_s'])
    outputs.append(Path(row['npz']))
assert not(signers['train']&signers['validation'])
assert len(labels['train'])==len(labels['validation'])==16 and labels['train']==labels['validation']
assert roles=={'train':224,'validation':74}
assert total==38365 and middle==10702 and new==27663
assert sum(r['new_crc_checked_frames'] for r in prepared['archives'])==new
assert sum(r['reused_annotation_frames'] for r in prepared['archives'])==middle
assert sum(r['old_index_clips_reused'] for r in prepared['archives'])==108
assert sum(r['new_index_clips'] for r in prepared['archives'])==190
assert all(r['archive_unchanged'] for r in prepared['archives'])
for path,digest in plan['source_pins'].items():assert sha(path)==digest
result={'status':'PASS_PREPARATION_AND_PROVENANCE_NOT_MODEL_PERFORMANCE',
        'independently_recounted_source_documents':len(eligibility['rows']),'prepared_clips':298,
        'by_role':dict(roles),'labels_each_role':16,'signers':{k:sorted(v) for k,v in signers.items()},
        'total_frames':total,'reused_annotation_frames_exact':middle,'new_source_frames':new,
        'clips_lacking_either_side_context':no_context,'first_available_frame_indices':dict(first_indices),
        'last_index_seconds_minus_label_duration_range':[min(timestamp_end_discrepancy),max(timestamp_end_discrepancy)],
        'original_caches_and_readers_unchanged':True,'frame_gaps':0,'npz_bytes':sum(p.stat().st_size for p in outputs),
        'new_frame_crc_verification':'Existing pinned reader verified each of27663 new members; reused code proof, no duplicate ZIP payload reads in this audit',
        'readiness':'Native context inputs ready for a fixed baseline; exact video alignment, annotation completeness, MediaPipe transfer and learned improvement remain unproven',
        'coordinate_result_sha256':sha(HERE/'coordinates.json'),'audit_script_sha256':sha(__file__),
        'classifier_inference':0,'training':0,'reserved_arrays_read':0,'participant_arrays_read':0}
with (HERE/'INDEPENDENT_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:
    json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False))
