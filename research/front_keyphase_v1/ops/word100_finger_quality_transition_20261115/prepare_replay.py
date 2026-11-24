from quality_common import *
src=(H/'replay_shadow.py').read_text('utf-8').replace('from holdout_common import *','from quality_common import *').replace('from guard_runtime import','from quality_runtime_v2 import').replace("GuardedObserver('require_agreement')","GuardedObserver('quality_only')").replace('rejected guard','frozen quality_only candidate')
(U/'replay_shadow.py').write_text(src,'utf-8')
