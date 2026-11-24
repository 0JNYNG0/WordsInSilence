from quality_common import *
import os
m=module('quality_fixed_extraction',O/'word100_public_g4_20261114/prepare.py');m.U=U;m.O=O
write(U/'CURRENT_EXECUTION.json',dict(status='PUBLIC_EXTRACTION',pid=os.getpid()));m.main();write(U/'CURRENT_EXECUTION.json',dict(status='RGB_REVIEW',pid=None))
