from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent/'selective_rtmw_anchor_repair_20261117'
s=(R/'deploy.ps1').read_text('utf-8')
s=s.replace("selective_rtmw_anchor_repair_20261117'", "seated_stop_short_cycle_20261117'")
s=s.replace('selective_rtmw_followup_20261117\\serve.py','selective_rtmw_anchor_repair_20261117\\serve.py')
(D/'deploy.ps1').write_text(s,encoding='utf-8')
s=(R/'check_http.py').read_text('utf-8')
s=s.replace("assert status['selective_retry_version']=='selective_front_rtmw_retry_v3'", "assert status['selective_retry_version']=='selective_front_rtmw_retry_v3'\n    assert status['stop_short_cycle_active'] and status['stop_short_cycle_version']=='seated_rest_dual_window_v1'")
s=s.replace("passed=result['overall_pass']and", "passed=result['stop_short_cycle_version']=='seated_rest_dual_window_v1'and result['overall_pass']and")
(D/'check_http.py').write_text(s,encoding='utf-8')
