"""Locate exact source JSON members; semantic registration and payload reads remain pending."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,importlib.util,json,sys,time

H=Path(__file__).resolve().parent;L=H.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    pending=read(H/'PENDING_SEMANTIC_REVIEW.json');proof=read(L/'ops/native_context_readiness_20260921/coordinate_plan.json')
    reader_path=next(Path(p) for p in proof['source_pins'] if p.endswith('zip_probe.py'))
    assert sha(reader_path)==proof['source_pins'][str(reader_path)]
    spec=importlib.util.spec_from_file_location('existing_zip_directory_reader',reader_path)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    rows=pending['rows'];assert [r['signer_id'] for r in rows]==list(range(1,10))+[11,12,13]
    assert all(not r['training_approved'] and r['semantic_target_id'] is None for r in rows)
    out=H/'source_frame_indexes';out.mkdir(exist_ok=False)
    plan=dict(at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
       scope='Read only central-directory metadata of12 explicitly listed existing sentence archives; exactSEN0354 F IDs only. No frame JSON bodies,video,model,user or reserved payload.',
       source_rows=[{k:r[k] for k in ['clip_id','signer_id','role','archive','archive_bytes','archive_mtime_ns']} for r in rows],
       pending_semantic_review_sha256=sha(H/'PENDING_SEMANTIC_REVIEW.json'),reader_path=str(reader_path),reader_sha256=sha(reader_path),
       operational_order_clarification='File location does not depend on semantic equivalence. This bounded metadata-only step can precede the answer; target mapping,input conversion and training remain pending.',
       new_semantic_decision=False,new_training=False,payload_reads_allowed=False)
    write(H/'SOURCE_LOCATION_PLAN.json',plan)
    results=[];started=time.monotonic()
    for row in rows:
        archive=Path(row['archive']);st=archive.stat()
        assert (st.st_size,st.st_mtime_ns)==(row['archive_bytes'],row['archive_mtime_ns'])
        members,stats=reader.selected_directory(archive,{row['clip_id']})
        assert all(m['clip_id']==row['clip_id'] for m in members)
        frames=sorted(int(m['name'].rsplit('_F_',1)[1].split('_')[0]) for m in members)
        assert len(frames)==len(set(frames))
        path=out/(row['clip_id']+'.json')
        record=dict(archive=str(archive),clip_id=row['clip_id'],signer_id=row['signer_id'],split_role=row['role'],
             archive_bytes=st.st_size,archive_mtime_ns=st.st_mtime_ns,members=members,stats=stats,
             status='MEMBERS_LOCATED_PAYLOAD_UNVERIFIED' if frames else 'NOT_FOUND_IN_THIS_ARCHIVE',
             semantic_target_id=None,training_approved=False)
        write(path,record)
        assert (archive.stat().st_size,archive.stat().st_mtime_ns)==(st.st_size,st.st_mtime_ns)
        results.append(dict(clip_id=row['clip_id'],signer_id=row['signer_id'],split_role=row['role'],status=record['status'],
             frames=len(frames),first_frame=frames[0] if frames else None,last_frame=frames[-1] if frames else None,
             frame_indices_contiguous=bool(frames and frames==list(range(frames[0],frames[-1]+1))),
             compressed_bytes=sum(m['compressed'] for m in members),uncompressed_bytes=sum(m['uncompressed'] for m in members),
             index=str(path),index_sha256=sha(path),directory_entries_scanned=stats['entries_scanned'],directory_bytes=stats['directory_bytes']))
        print(json.dumps(dict(signer=row['signer_id'],frames=len(frames),payload_read=False)),flush=True)
    result=dict(status='SOURCE_DIRECTORY_LOCATION_COMPLETE_NOT_INPUT_PREPARATION',at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
       plan_sha256=sha(H/'SOURCE_LOCATION_PLAN.json'),rows=results,total_clips_located=sum(r['frames']>0 for r in results),
       total_frame_members=sum(r['frames'] for r in results),target_compressed_bytes=sum(r['compressed_bytes'] for r in results),
       target_uncompressed_bytes=sum(r['uncompressed_bytes'] for r in results),directory_bytes_read=sum(r['directory_bytes'] for r in results),
       seconds=time.monotonic()-started,semantic_response=None,source_frame_payloads_read=0,training_runs=0,encoder_inference=0,
       user_payloads_read=0,reserved_payloads_read=0,new_download_bytes=0,source_registered=False,
       limits=['Directory member existence and sizes only; CRC/decode/coordinate usability/temporal mapping not newly validated','Same-class semantic question is unanswered; no approval inferred','Archive number is not substituted for member signer identity'])
    write(H/'SOURCE_LOCATION_RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','limits']}),flush=True)

if __name__=='__main__':main()
