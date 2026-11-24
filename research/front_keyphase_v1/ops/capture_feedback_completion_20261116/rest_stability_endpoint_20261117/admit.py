from pathlib import Path
import json, hashlib, ast, datetime, shutil
D = Path(__file__).resolve().parent
P = D.parent / 'seated_stop_short_cycle_20261117'
read = lambda p: json.loads(Path(p).read_text('utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    checks = {}
    for rel in ['checks/STOP_CHECK.json', 'web/CONSUMER_CHECK.json', 'web/CARD_CHECK.json', 'web/CAPTURE_CHECK.json']:
        p = D / rel
        assert read(p)['passed'], rel
        checks[str(p)] = sha(p)
    stop = read(D / 'checks/STOP_CHECK.json')
    assert any(r['before'] is None and r['after'] is not None for r in stop['rows'])
    pins = read(P / 'web/ADMISSION.json')['pins'].copy()
    assert all(sha(p) == h for p, h in pins.items()), 'Frozen baseline changed'
    old = len(pins)
    # Endpoint candidate is intentionally NOT imported or admitted.
    active = ['base.py', 'stable_rest.py', 'stable_runtime.py', 'serve.py']
    for name in active:
        p = D / name
        ast.parse(p.read_text('utf-8'))
        pins[str(p)] = sha(p)
    for name in ['CONTRACT.json', 'ENGINEERING_NOTE.json']:
        pins[str(D / name)] = sha(D / name)
    for row in read(D / 'web/CAPTURE_CHECK.json')['rows']:
        folder = Path(row['directory'])
        out = D / 'public_capture_artifacts' / f"{row['word']}_{row['kind']}"
        out.mkdir(parents=True, exist_ok=True)
        for name in ['recorded.mp4', 'rtmw/observations.npz', 'auto_end_diagnostic.json']:
            source = folder / name
            assert source.is_file(), source
            if source.resolve() != (out / source.name).resolve():
                shutil.copy2(source, out / source.name)
    q = dict(admitted=True, at=datetime.datetime.now().astimezone().isoformat(), pins=pins, checks=checks,
             prior_pins_preserved=old, training_runs=0, private_data=False, physical_camera_verified=False,
             stop_change_admitted=True, endpoint_candidate_admitted=False,
             scope='Public frontal automatic-stop stability only; existing feedback weights, endpoint logic and C/T3-F preserved.')
    (D / 'web/ADMISSION.json').write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ADMITTED', len(pins), 'prior', old)

if __name__ == '__main__':
    main()
