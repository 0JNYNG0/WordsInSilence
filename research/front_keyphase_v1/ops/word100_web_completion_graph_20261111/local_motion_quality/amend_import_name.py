"""One-off technical import-name amendment; never prepares or trains."""
from pathlib import Path
import hashlib
import json
import datetime

W = Path(__file__).resolve().parent
EXPECTED = Path('C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/local_motion_quality').resolve()
assert W == EXPECTED

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    assert not (W/'INPUTS_COMPLETE.json').exists()
    assert not (W/'inputs').exists() or not any((W/'inputs').iterdir())
    assert not (W/'FROZEN_IMPORT_AMENDMENT.json').exists()
    frozen_path=W/'FROZEN.json'
    initial=frozen_path.read_bytes()
    old=json.loads(initial)
    for path,digest in old['pins'].items():
        assert sha(Path(path).read_bytes()) == digest, path
    initial_path=W/'FROZEN_INITIAL_IMPORT_FAILURE.json'
    assert not initial_path.exists()
    initial_path.write_bytes(initial)
    snapshots=W/'initial_import_failure_code'
    snapshots.mkdir(exist_ok=False)
    originals={n:(W/n).read_bytes() for n in ('model.py','prepare.py','train.py','test_model.py')}
    for name,data in originals.items():
        (snapshots/name).write_bytes(data)
    assert not (W/'quality_model.py').exists()
    (W/'model.py').rename(W/'quality_model.py')
    changes=[]
    for name in ('prepare.py','train.py','test_model.py'):
        data=originals[name]
        updated=data.replace(b'from model import ',b'from quality_model import ')
        if name == 'prepare.py':
            updated=updated.replace(b"W/'model.py'",b"W/'quality_model.py'")
        assert updated != data, name
        (W/name).write_bytes(updated)
        changes.append(dict(path=str(W/name),before_sha256=sha(data),after_sha256=sha(updated),snapshot=str(snapshots/name)))
    changes.append(dict(before_path=str(W/'model.py'),after_path=str(W/'quality_model.py'),before_sha256=sha(originals['model.py']),after_sha256=sha((W/'quality_model.py').read_bytes()),snapshot=str(snapshots/'model.py'),content_changed=False))
    new=json.loads(initial)
    new_pins={}
    for path,digest in old['pins'].items():
        target=W/'quality_model.py' if Path(path)==W/'model.py' else Path(path)
        new_pins[str(target)]=sha(target.read_bytes())
    new['pins']=new_pins
    assert {k:v for k,v in old.items() if k!='pins'} == {k:v for k,v in new.items() if k!='pins'}
    refreshed=(json.dumps(new,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    frozen_path.write_bytes(refreshed)
    report=dict(at=datetime.datetime.now().isoformat(),reason='Local model.py shadowed legacy EvidenceHeads model during role-core import before input preparation. Rename only and refresh affected code pins.',
        initial_frozen_path=str(initial_path),initial_frozen_sha256=sha(initial),amended_frozen_sha256=sha(refreshed),
        changes=changes,non_pin_frozen_fields_identical=True,data_split_labels_training_config_changed=False,
        preserved_failure_files=[str(p) for p in W.glob('FAILURE_*.json')],input_preparation_run=False,training_run=False)
    (W/'FROZEN_IMPORT_AMENDMENT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Import rename complete; initial freeze/code preserved; data/config unchanged.')

if __name__=='__main__':
    main()
