from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent;B=U.parent/'word100_multipeople_live_feedback_20261101'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
assert read(U/'ENCODER_GATE.json')['status']=='PASS'
assert read(U/'review/BENCHMARK_VERIFICATION.json').get('status','').startswith('PASS')
text=(B/'check_live_browser.py').read_text(encoding='utf8')
text=text.replace("BASE='http://127.0.0.1:8802'","BASE='http://127.0.0.1:8803'")
text=text.replace("PREFIX='GUARD_' if GUARD else ''","PREFIX='CANDIDATE_'")
text=text.replace("word,person=('WORD2523',13) if GUARD else ('WORD0739',12)","word,person=('WORD1133',12)")
text=text.replace("read(U/'PUBLIC_INPUTS.json')","read(U.parent/'word100_multipeople_live_feedback_20261101/PUBLIC_INPUTS.json')")
text=text.replace("if GUARD:assert result['evidence_guard']['applied'] is True","assert result['evidence_guard']['applied'] is True\n  assert result.get('detail_enrichment') is not None\n  assert js(\"document.body.textContent.includes('2D')\")")
with (U/'check_candidate_browser.py').open('x',encoding='utf8') as f:f.write(text)
row=next(r for r in read(B/'PUBLIC_INPUTS.json')['rows'] if r['word_id']=='WORD1133' and r['person']==12)
with (U/'INTEGRATION_FROZEN.json').open('x',encoding='utf8') as f:json.dump(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),public_source=row,scope='one fresh public upload for separate8803 integration; resume saved job if UI issue; no webcam or user files',original8802='unchanged',script_sha256=hashlib.sha256(text.encode()).hexdigest(),acceptance='four words9references, fresh protected predictions, old guard, additive trace metadata visible, videos/sync/peak/safe source SHA. No performance comparison claim to previous DIFFERENT video.'),f,ensure_ascii=False,indent=2)
print('prepared fixed public integration')
