from pathlib import Path
V=Path(__file__).resolve().parent
s=(V.parent/'word100_learned_guide_feedback_20261102/check_demo_browser.py').read_text(encoding='utf8').replace('http://127.0.0.1:8805','http://127.0.0.1:8806')
s=s.replace("checks.append(word+' public result, decoded both videos, full skeleton, timed replay')", "checks.append(word+' public result, decoded both videos, full skeleton, timed replay'); assert js('result.relation_head===\"public_supervised_goal_blind_3_relations\"')")
with (V/'check_demo_browser.py').open('x',encoding='utf8') as f:f.write(s)
