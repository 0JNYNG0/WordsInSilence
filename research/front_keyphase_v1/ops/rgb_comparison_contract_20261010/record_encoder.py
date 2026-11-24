"""Read installed source metadata. Does not import torch or construct a model."""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import re
from pathlib import Path

unit = Path(__file__).resolve().parent
root = Path(importlib.util.find_spec('torchvision').origin).parent
paths = [root/'models/mobilenetv3.py', root/'transforms/_presets.py', root/'version.py']
model = paths[0].read_text(encoding='utf-8')
small = model.split('class MobileNet_V3_Small_Weights')[1].split('@register_model')[0]
url = re.search(r'url="([^"]+)"', small).group(1)
tree = ast.parse(paths[1].read_text(encoding='utf-8'))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ImageClassification')
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
kw = dict(zip([arg.arg for arg in init.args.kwonlyargs], init.args.kw_defaults))
values = {k: ast.literal_eval(kw[k]) for k in ('resize_size','mean','std','antialias')}
assert values == {'resize_size':256, 'mean':(0.485,0.456,0.406), 'std':(0.229,0.224,0.225), 'antialias':True}
assert ast.unparse(kw['interpolation']) == 'InterpolationMode.BILINEAR'
assert 'crop_size=224' in small
assert url == 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'
meta = {
    'status':'SOURCE_RECIPE_VERIFIED_LOCAL_WEIGHTS_NOT_ACQUIRED',
    'encoder_candidate':'torchvision.models.mobilenet_v3_small',
    'weight_enum':'MobileNet_V3_Small_Weights.IMAGENET1K_V1', 'weight_url':url,
    'filename_hash_prefix':'047dcff4', 'local_weight_path':None, 'local_full_sha256':None,
    'weight_download_performed':False,
    'installed_torch':importlib.metadata.version('torch'),
    'installed_torchvision':importlib.metadata.version('torchvision'),
    'num_parameters_official_full_classifier':2542856, 'published_weight_size_mb':9.829,
    'official_preprocessing':{'resize_short_edge':256,'center_crop':224,'interpolation':'BILINEAR',
                              'antialias':True,'rescale':'uint8 RGB / 255',
                              'mean':values['mean'],'std':values['std']},
    'proposed_domain_preprocessing_not_executed':{
        'source':'original full square hand region with padding', 'resize':[224,224],
        'center_crop':False,'normalization':'same as official',
        'reason':'avoid trimming ROI edges; different from official evaluation recipe',
        'existing128_upscale_is_not_original224':True},
    'source_pins':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'license_status':{'library':'BSD-3-Clause',
                      'weights':'TorchVision notes pretrained models may have training-data-derived terms; library license alone does not establish weight/data rights',
                      'planned_use':'local academic research; deployment/redistribution permission not asserted'},
    'references':['https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html',
                  'https://github.com/pytorch/vision#pre-trained-model-license',
                  'https://github.com/pytorch/vision/blob/main/LICENSE'],
    'training_or_inference':False}
with (unit/'ENCODER_CONTRACT.json').open('x',encoding='utf-8') as f:
    json.dump(meta,f,indent=2,ensure_ascii=False)
print(json.dumps({'status':'PASS','weight_url':url,'source_only':True}))
