from core import *
E=W/'boundary_direction_repair/evaluation'
m=load('_boundary_shadow',W/'shadow_components.py');m.W=E;m.S=E/'shadow_components';m.main()
