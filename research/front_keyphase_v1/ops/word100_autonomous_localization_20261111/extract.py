from pathlib import Path
import importlib.util
U=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_autoloc_extract',U.parent/'word100_episode_joint_linkage_20261111/extract.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.U=U;m.main()
