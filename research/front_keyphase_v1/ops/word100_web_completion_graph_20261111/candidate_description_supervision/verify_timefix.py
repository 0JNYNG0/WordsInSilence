"""The same saved-array verifier applied to the separately corrected input contract."""
from transfer_local_timefix_core import *
source=(W/'verify_local_sequence.py').read_text(encoding='utf-8')
source=source.replace('from local_sequence_core import *','from transfer_local_timefix_core import *\nQ=TQ')
source=source.replace("['local_sequence']","['local_pretrained']")
assert source!= (W/'verify_local_sequence.py').read_text(encoding='utf-8')
exec(compile(source,str(Path(__file__)), 'exec'))
