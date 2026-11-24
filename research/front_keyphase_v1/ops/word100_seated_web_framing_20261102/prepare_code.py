"""Copy frozen old adapter/UI; only sensor-boundary and honest input messages change."""
from pathlib import Path
import json,hashlib,ast
U=Path(__file__).resolve().parent
B=U.parent/'word100_multipeople_live_feedback_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
text=(B/'adapter.py').read_text(encoding='utf-8-sig')
old="boundary = _module('_live_body_boundary', OPS/'word100_body_boundary_20261017/boundary.py')"
new="boundary = _module('_live_sensor_boundary', UNIT/'sensor_boundary.py')"
assert text.count(old)==1
text=text.replace(old,new)
old="calibration = boundary.calibrate(ob, width, height, read(OPS/'word100_body_boundary_20261017/design_data_COEFFICIENT.json'))"
new="calibration = boundary.calibrate(ob, width, height)"
assert text.count(old)==1
text=text.replace(old,new)
old="        feat = observer.extract_features(masked)"
new=old+"\n        if not feat['mask'].any():\n            unsupported.append('NO_SUPPORTED_VISIBLE_FEATURES')"
assert text.count(old)==1
text=text.replace(old,new)
text=text.replace('Uncalibrated/too-short inputs return explicit unsupported status, not guesses.',
                  'Empty visible-feature/too-short inputs return unsupported; initial rest hands are not required.')
with (U/'adapter.py').open('x',encoding='utf8') as f:f.write(text)
asts=[ast.parse((p/'adapter.py').read_text(encoding='utf-8-sig')) for p in [B,U]]
funcs=[{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))} for t in asts]
changed=[k for k in funcs[0] if funcs[0][k]!=funcs[1][k]]
assert changed==['analyze_video'],changed
for name in ['index.html','app.js']:
    t=(B/name).read_text(encoding='utf-8-sig')
    if name=='index.html':
        t=t.replace('손과 상체가 화면 안에 들어오도록 촬영해 주세요.', '앉아서 촬영해도 됩니다. 준비할 때 손은 화면 밖에 있어도 됩니다. 주요 동작 중에는 필요한 손·팔과 어깨가 보이게 해 주세요.')
        t=t.replace('<main>', '<main><section class="panel"><h2>앉은 웹캠 촬영 시험</h2><p>카메라 화면 전체를 분석합니다. 사용자 영상 아래에 별도 가로 경계를 적용하지 않습니다. 준비·마무리의 손내림은 화면 밖이어도 됩니다. 주요 동작에서 보이지 않는 부분은 근거 부족으로 표시합니다.</p><p>공개 참고의 학습 범위와 기존 모델은 유지했습니다. 촬영은 학습·튜닝에 사용하지 않습니다.</p></section>',1)
    else:
        t=t.replace('손과 상체가 화면 안에 들어오도록 촬영해 주세요.', '앉아서 촬영 가능 · 준비할 때 손은 화면 밖이어도 됩니다. 주요 동작의 손·팔과 어깨가 보이게 해 주세요.')
    with (U/name).open('x',encoding='utf8') as f:f.write(t)
pins={str(B/n):sha(B/n) for n in ['adapter.py','engine.py','live_engine.py','server.py','evidence_guard.py','REFERENCE_BANK.json','NORMAL_RANGES.json','index.html','app.js']}
(U/'CODE_MANIFEST.json').write_text(json.dumps(dict(base_pins=pins,changed_adapter_functions=changed,
    models_changed=False,reference_bank_changed=False,thresholds_changed=False,frames_removed=False),indent=2),encoding='utf8')
print('Only analyze_video boundary selection/empty-input check changed; copied UI.')
