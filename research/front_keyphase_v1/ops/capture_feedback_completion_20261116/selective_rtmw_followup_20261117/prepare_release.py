from pathlib import Path
import json
D=Path(__file__).resolve().parent;P=D.parent/'selective_rtmw_retry_20261117'
s=(P/'admit.py').read_text(encoding='utf-8').replace("P=G/'public_frontal_evidence_repair_20261117'","P=G/'selective_rtmw_retry_20261117'")
s=s.replace("assert v['recovered_cases']>0", "assert v['recovered_cases']>0\n    old=read(P/'PUBLIC_VALIDATION.json');oldmap={(r['id'],r['kind']):r for r in old['controls']}\n    pairs=[(oldmap[r['id'],r['kind']],r) for r in v['controls'] if r['kind']in['missing_hand','missing_one_finger']and(r['id'],r['kind'])in oldmap]\n    assert len(pairs)==16 and sum(b['unknown_after']>0 for a,b in pairs)<sum(a['unknown_after']>0 for a,b in pairs)\n    assert read(D/'DECODE_PROFILE.json')['pixels_equal']")
(D/'admit.py').write_text(s,encoding='utf-8')
s=(P/'check_http.py').read_text(encoding='utf-8').replace("assert status['state']=='ready'and", "assert status['selective_retry_version']=='selective_front_rtmw_retry_v2'\n    assert status['state']=='ready'and")
(D/'check_http.py').write_text(s,encoding='utf-8')
