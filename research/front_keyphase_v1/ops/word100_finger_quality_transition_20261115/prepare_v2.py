from quality_common import *
assert not(U/'TRAIN_FROZEN_V2.json').exists()
src=(U/'train.py').read_text('utf-8').replace('from quality_model import','from quality_model_v2 import').replace('FIT_LABELS.json','FIT_LABELS_V2.json').replace('TRAIN_FROZEN.json','TRAIN_FROZEN_V2.json').replace('TRAIN_INPUTS.npz','TRAIN_INPUTS_V2.npz').replace('TRAINED.json','TRAINED_V2.json').replace("U/'models'","U/'models_v2'").replace("r['person']in[3,6,9]","r['person']in[1,3,4,6,8,9]").replace("r['role']=='fit'","r['role'].startswith('fit')").replace('quality_model.py','quality_model_v2.py')
(U/'train_v2.py').write_text(src,'utf-8')
src=(U/'quality_runtime.py').read_text('utf-8').replace('from quality_model import','from quality_model_v2 import').replace("U/'models'","U/'models_v2'")
(U/'quality_runtime_v2.py').write_text(src,'utf-8')
src=(U/'evaluate.py').read_text('utf-8').replace('from quality_runtime import','from quality_runtime_v2 import').replace("f'{role}_RESULTS.json'","f'{role}_RESULTS_V2.json'")
(U/'evaluate_v2.py').write_text(src,'utf-8')
write(U/'V2_RATIONALE.json',dict(reason='V1 missed all7 wrong chains and marked12 stable finger moments as changing in diagnostic14. Add FITpeople1/4/8 for quality and FITfront-facing rotations3/6/9 as stable-shape controls. Separate encoders: full geometry+neighbors for quality, direction-independent bending/length geometry for shape change. No reserve15 predictions.',limitations='2D geometry cannot always distinguish projection from flexion or reconstruct invisible fingers. No RGB at inference, no coordinate imputation.',thresholds='same all3seeds .85/.15; same600steps; no threshold search',web='unchanged until gates',fit_expansion='PLAN initial3,6,9 expanded within declared FIT01..09 before training'))
