"""Independent metadata/header binding; never read compressed keypoint bodies."""
from locate_source_frames import H,read,sha,write
from pathlib import Path
import re,struct

def main():
    r=read(H/'SOURCE_LOCATION_RESULT.json');p=read(H/'SOURCE_LOCATION_PLAN.json');pending=read(H/'PENDING_SEMANTIC_REVIEW.json')
    checks={}
    def check(n,ok):checks[n]=bool(ok);assert checks[n],n
    check('plan_hash',r['plan_sha256']==sha(H/'SOURCE_LOCATION_PLAN.json'))
    check('pending_review_unchanged',sha(H/'PENDING_SEMANTIC_REVIEW.json')==p['pending_semantic_review_sha256'] and pending['response'] is None)
    check('roles_exact',[(x['signer_id'],x['split_role']) for x in r['rows']]==[(i,'train_candidate') for i in range(1,10)]+[(i,'development_only') for i in [11,12,13]])
    pattern=re.compile(r'^(\d{2})/(NIA_SL_SEN0354_REAL(\d{2})_F)/(NIA_SL_SEN0354_REAL\d{2}_F)_(\d+)_keypoints\.json$')
    total=packed=raw=header_bytes=0
    for row in r['rows']:
        check('index_hash_%02d'%row['signer_id'],sha(row['index'])==row['index_sha256'])
        index=read(row['index']);archive=Path(index['archive']);stat=archive.stat()
        check('archive_stat_%02d'%row['signer_id'],(stat.st_size,stat.st_mtime_ns)==(index['archive_bytes'],index['archive_mtime_ns']))
        frames=[];offsets=set();ok=True
        with archive.open('rb') as f:
            for m in index['members']:
                match=pattern.match(m['name']);assert match is not None
                a,ident,person,repeat,fi=match.groups();frames.append(int(fi))
                ok=ok and ident==repeat==m['clip_id']==row['clip_id'] and int(a)==int(person)==row['signer_id']
                ok=ok and m['local_offset'] not in offsets;offsets.add(m['local_offset'])
                f.seek(m['local_offset']);header=f.read(30);assert header[:4]==b'PK\x03\x04'
                flags,method=struct.unpack_from('<2H',header,6);nl,el=struct.unpack_from('<2H',header,26)
                name=f.read(nl);header_bytes+=30+nl
                expected_name=name.decode('utf-8' if flags&0x800 else 'cp437')
                ok=ok and expected_name==m['name'] and method==m['method'] and flags==m['flags'] and not(flags&1)
                ok=ok and m['local_offset']+30+nl+el+m['compressed']<=stat.st_size
                total+=1;packed+=m['compressed'];raw+=m['uncompressed']
        check('local_header_identity_%02d'%row['signer_id'],ok)
        check('frame_sequence_%02d'%row['signer_id'],sorted(frames)==list(range(row['frames'])) and row['first_frame']==0 and row['last_frame']==row['frames']-1 and row['frame_indices_contiguous'])
        check('target_not_registered_%02d'%row['signer_id'],index['semantic_target_id'] is None and not index['training_approved'])
    check('aggregate_counts',total==r['total_frame_members']==1405 and packed==r['target_compressed_bytes'] and raw==r['target_uncompressed_bytes'] and r['total_clips_located']==12)
    check('payload_and_training_absent',r['source_frame_payloads_read']==r['training_runs']==r['encoder_inference']==r['user_payloads_read']==r['reserved_payloads_read']==r['new_download_bytes']==0)
    result=dict(status='PASS_SOURCE_LOCATIONS_ONLY',checks=checks,count=len(checks),bound_local_headers=total,local_header_bytes_read=header_bytes,
       compressed_frame_bytes_read=0,source_crc_checked=False,coordinate_decode_checked=False,input_ready=False,semantic_response=None,
       result_sha256=sha(H/'SOURCE_LOCATION_RESULT.json'),audit_script_sha256=sha(__file__))
    write(H/'SOURCE_LOCATION_AUDIT.json',result)
    print({k:v for k,v in result.items() if k!='checks'})

if __name__=='__main__':main()
