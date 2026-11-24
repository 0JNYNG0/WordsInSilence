"""Retrospective inference ONLY after public release admission; never candidate selection."""
from probe import *
import hashlib

def states(q):return {c['id']:c['status'] for c in q['lesson_acceptance']['criteria']}
def digest(p):return __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()
def main():
    admission=read(D/'web/ADMISSION.json');assert admission['admitted']
    assert all(digest(f)==h for f,h in admission['pins'].items())
    frozen=digest(D/'web/ADMISSION.json')
    source=N/'seated_user_check/DISCOVERY.json'
    rows=read(source)['rows'];dest=D/'private_diagnostic_only';dest.mkdir(exist_ok=True)
    write(dest/'CONTRACT.json',dict(scope='Already admitted public candidate; user-requested retrospective inference only',public_admission_sha256=frozen,training=False,tuning=False,candidate_selection=False,threshold_choice=False,source_manifest=str(source),no_web_result_overwrite=True))
    torch.set_num_threads(2);cv2.setNumThreads(2);net=FeedbackConsumer();engine=Engine();answers=[]
    try:
        for i,r in enumerate(rows,1):
            folder=Path(r['directory']);saved=read(folder/'inference_result.json')['result']
            word='WORD0099' if i<=4 else 'WORD1290'
            # Assert via the recorded payload instead of relying on presentation order.
            assert word in json.dumps(saved,ensure_ascii=False)
            files=[folder/'recorded.mp4',folder/'rtmw/observations.npz',folder/'inference_result.json']
            if not all(f.is_file() for f in files):
                answers.append(dict(attempt=i,word=word,available=False));continue
            hashes={str(f):digest(f) for f in files}
            raw=dict(np.load(files[1]));before=net.raw(raw,word)
            after,fixed=retry(raw,before,VideoFrames(files[0],raw),engine,net.raw,word)
            preserved=all(v=='unknown' or states(after).get(k)==v for k,v in states(before).items())and before['tcn_top3']==after['tcn_top3']
            assert preserved and all(digest(f)==h for f,h in hashes.items())
            row=dict(attempt=i,word=word,available=True,original_recorded=states(saved),before_current_retry=states(before),after_current_retry=states(after),original_overall_pass=saved['overall_pass'],before_overall_pass=before['overall_pass'],after_overall_pass=after['overall_pass'],retry=after['selective_retry'],known_judgments_preserved=preserved,source_hashes=hashes,original_stop_reason=saved.get('stop_reason'),original_timestamps_preserved=True)
            answers.append(row);write(dest/f'attempt_{i}.json',row)
            print('PRIVATE_DIAGNOSTIC',i,word,before['overall_pass'],after['overall_pass'],row['retry']['reason'],row['retry']['pose_calls'],flush=True)
        assert digest(D/'web/ADMISSION.json')==frozen and all(digest(f)==h for f,h in admission['pins'].items())
        write(dest/'RESULT.json',dict(rows=answers,public_admission_unchanged=True,training_use=False,tuning_use=False,candidate_selection_use=False,source='Saved private recordings, no new camera use',scope='Inference diagnosis only; original vs current includes previous public code repairs. Retry benefit is current-before vs current-after only. No general accuracy or live latency claim.'))
    finally:close(net)
if __name__=='__main__':main()
