from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    old=json.loads((O/'word100_rtmw_stream_reuse_20261108/FROZEN.json').read_text(encoding='utf-8'))
    ids=['NIA_SL_WORD0739_REAL02_F','NIA_SL_WORD1567_REAL02_F','NIA_SL_WORD0003_REAL02_F']
    rows=[next(r for r in old['rows'] if r['clip_id']==cid) for cid in ids]
    pins=dict(old['protected'])
    for n in ['stream_runtime.py','stream_http.py','VERIFICATION.json','HTTP_VERIFICATION.json']:
        p=O/'word100_rtmw_stream_reuse_20261108'/n;pins[str(p)]=sha(p)
    for n in ['word100_candidate.py','word100_t3f.py']:
        p=O.parent/'src/wis_lab/service'/n;pins[str(p)]=sha(p)
    for p,h in pins.items():assert sha(p)==h,p
    for r in rows:assert sha(r['source'])==r['source_sha256']
    obj=dict(rows=rows,protected=pins,training=0,camera=False,recording_pts_tolerance_us=2000,
        input_ready_budget_ms=1000,scope='actual Chromium public file captureStream, raw RGBA transfer, simultaneous RTMW+MP, generated-track MediaRecorder, existing frozen TCN',
        limits='public browser source is not a hardware webcam; not real incorrect-sign/feedback accuracy evaluation',
        reference_sources=['https://developer.chrome.com/docs/capabilities/web-apis/mediastreamtrack-insertable-media-processing'])
    with (U/'FROZEN.json').open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    print('frozen 3 public browser captures')
if __name__=='__main__':main()
