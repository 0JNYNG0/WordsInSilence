"""Narrow branch around old scoring; preserve the old camera/persistence lifecycle."""
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'src/wis_lab/service/app.py'
s=p.read_text('utf-8')
a=s.index("    # this take's query preparation,")
b=s.index('    # Stamped HERE,',a)
old=s[a:b]
new='''    if (t.get('research_model') or {}).get('model_key') == word100.KEY:
        # Explicit new vocabulary: never compare WORD ids against the unrelated old18 templates.
        result = word100.result(obs, t)
        result.update(result_id="res_" + uuid.uuid4().hex[:12], trial_id=t['trial_id'],
                      session_id=t['session_id'], observation_sha256=t['observation_sha256'],
                      contract_id=st.contract['contract_id'], extractor_id=obs.sidecar.get('extractor_id'),
                      origin='camera', diagnostic_capture=capture, end_reason=meta['end_reason'])
    else:
'''+''.join('    '+line if line.strip() else line for line in old.splitlines(keepends=True))
s=s[:a]+new+s[b:]
p.write_text(s,'utf-8')
