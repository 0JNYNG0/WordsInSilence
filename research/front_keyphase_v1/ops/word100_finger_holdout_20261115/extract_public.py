from holdout_common import *
import os
m=module('holdout_fixed_public_extraction',O/'word100_public_g4_20261114/prepare.py');m.U=U;m.O=O
write(U/'CURRENT_EXECUTION.json',dict(status='RTMW_PUBLIC_EXTRACTION',pid=os.getpid(),source_plan=str(U/'FROZEN.json')))
m.main()
write(U/'CURRENT_EXECUTION.json',dict(status='RGB_LABEL_REVIEW',pid=None))
