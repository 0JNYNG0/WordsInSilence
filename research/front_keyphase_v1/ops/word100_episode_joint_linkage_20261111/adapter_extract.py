import importlib.util
from pathlib import Path
u=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_adapter_extract',u/'extract.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.U=u/'proposal_adapter';m.main()
