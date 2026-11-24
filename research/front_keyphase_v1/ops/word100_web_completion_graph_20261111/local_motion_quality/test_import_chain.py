"""Import-only smoke test: checkpoints, input arrays and model calls forbidden."""
from pathlib import Path
import sys,json,datetime
sys.dont_write_bytecode=True
import numpy as np
import torch

W=Path(__file__).resolve().parent

def forbidden(*args,**kwargs):
    raise AssertionError('Import-only test attempted array/checkpoint loading or model execution')

def main():
    original=(np.load,torch.load,torch.nn.Module._call_impl)
    try:
        np.load=torch.load=torch.nn.Module._call_impl=forbidden
        import quality_model
        import prepare
        role=prepare.load('_local_quality_import_only',prepare.R/'role_core.py')
        trainer=prepare.load('_local_quality_train_import_only',W/'train.py')
        legacy=sys.modules.get('model')
        assert legacy is not None and hasattr(legacy,'EvidenceHeads')
        assert Path(legacy.__file__).resolve() != W/'quality_model.py'
        assert trainer.LocalMotionQuality is quality_model.LocalMotionQuality
        assert callable(role.frozen_model)
        result=dict(at=datetime.datetime.now().isoformat(),passed=True,
            quality_module=quality_model.__file__,legacy_model_module=legacy.__file__,
            numpy_array_loads=0,checkpoint_loads=0,model_forward_calls=0,
            preparation_run=False,training_run=False)
        (W/'IMPORT_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False))
    finally:
        np.load,torch.load,torch.nn.Module._call_impl=original

if __name__=='__main__':
    main()
