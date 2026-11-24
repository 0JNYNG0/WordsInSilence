from pathlib import Path
import json,hashlib,datetime,numpy as np
T=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def main():
 assert (T/'VERIFIED.json').exists();rows=[r for r in read(T/'DONE.json')['rows'] if r['changed']];count=0
 for r in rows:
  with np.load(r['candidate_output']) as z,np.load(r['original_range']) as b:
   for mode in ('point','sequence'):
    np.testing.assert_array_equal(z[mode+'_status'],b[mode+'_status']);np.testing.assert_array_equal(z[mode+'_evidence'],b[mode+'_evidence']);np.testing.assert_allclose(z[mode+'_delta'],b[mode+'_delta'],atol=2e-6,rtol=2e-6,equal_nan=True)
   np.testing.assert_array_equal(z['domain'],b['domain']);count+=1
 assert count==799
 (T/'BASELINE_EQUIVALENCE.json').write_text(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),pairs=count,status='PASS_EXACT_SLOW_CONTROL_STATUS_EQUALS_UNMODIFIED_ORIGINAL',comparison='All candidate slow-control statuses/support exactly equal unmodified originals, differences equal within2e-6. Input source mapping used here only as evaluation provenance; candidate does not read it. Losses versus old re-aligned speed controls are reported separately, not counted as improvement.',verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf8');print('SLOW_BASELINE_EQUIVALENCE799_PASS')
if __name__=='__main__':main()
