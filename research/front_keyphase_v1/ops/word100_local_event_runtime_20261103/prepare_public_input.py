"""Copy one exact public archive member into this unit; retain source bytes."""
from pathlib import Path
import json,hashlib,zipfile,zlib,time,traceback
import av
U=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    plan=read(U/'PUBLIC_INPUT_PLAN.json');src=plan['source'];selected=plan['selection']
    assert selected['registered_id']==10 and selected['word_id']=='WORD1133'
    assert selected['training_allowed'] is False and selected['tuning_allowed'] is False
    destination=Path(src['destination']).resolve()
    assert destination==U/'public_input/NIA_SL_WORD1133_REAL10_F.mp4'
    assert not destination.exists() and not (U/'PUBLIC_INPUT_FROZEN.json').exists()
    archive=Path(src['archive']);before=archive.stat()
    assert before.st_size==src['archive_bytes'] and before.st_mtime_ns==src['archive_mtime_ns']
    for key in ['inventory','catalog','front_manifest']:
        item=plan['public_provenance'][key];assert sha(item['path'])==item['sha256']
    destination.parent.mkdir(exist_ok=True)
    crc=0;count=0
    with zipfile.ZipFile(archive,'r') as z:
        member=z.getinfo(src['member'])
        assert member.file_size==src['member_bytes'] and member.CRC==src['member_crc32']
        assert not (member.flag_bits & 1) and not member.is_dir()
        # Explicit destination rather than extract(member) prevents archive
        # path traversal and copies no unrelated member.
        with z.open(member,'r') as source, destination.open('xb') as out:
            for block in iter(lambda:source.read(1048576),b''):
                count+=len(block);crc=zlib.crc32(block,crc);out.write(block)
    assert count==src['member_bytes'] and (crc & 0xffffffff)==src['member_crc32']
    after=archive.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    ids=[];pts=[];origin=None;last=-1;dropped=0
    with av.open(str(destination)) as container:
        stream=container.streams.video[0]
        width,height=stream.codec_context.width,stream.codec_context.height
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.pts is None:continue
                if origin is None:origin=frame.pts
                timestamp=round(float((frame.pts-origin)*stream.time_base)*1e6)
                if timestamp<=last:dropped+=1;continue
                ids.append(len(ids)+dropped);pts.append(timestamp);last=timestamp
    assert len(ids)>=16 and pts[-1]<=20_000_000
    record=dict(status='ONE_PUBLIC_MEMBER_COPIED_AND_NATIVE_TIMELINE_FROZEN',
        source=str(destination),source_sha256=sha(destination),source_role='public_runtime_inference',
        role='new_exposed_development_integration_not_accuracy_test',
        word_id=selected['word_id'],clip_id=selected['clip_id'],registered_id=selected['registered_id'],
        training=False,tuning=False,candidate_selection=False,future_fresh_holdout_claim=False,
        archive=str(archive),member=src['member'],member_bytes=count,member_crc32=crc & 0xffffffff,
        archive_stat_unchanged=True,archive_full_hash_claimed=False,
        selection_plan_sha256=sha(U/'PUBLIC_INPUT_PLAN.json'),frame_id=ids,timestamp_us=pts,
        width=width,height=height,native_frames=len(ids),source_frame_drop_count=dropped,
        physical_occurrence_gold=None,whole_clause_truth=None,educational_error=None,
        model_forwards=0,training_steps=0,source_decode_purpose='native timestamp preparation only')
    with (U/'PUBLIC_INPUT_FROZEN.json').open('x',encoding='utf-8') as f:
        json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:record[k] for k in ['status','source_sha256','native_frames','registered_id']},ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception:
        with (U/('INPUT_PREPARATION_FAILURE_'+str(time.time_ns())+'.txt')).open('x',encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise
