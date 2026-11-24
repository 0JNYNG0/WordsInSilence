from core import *
N=W/'dense_window_supervision'
s=load('_dense_supervision_shadow',W/'shadow_components.py');s.W=N/'evaluation';s.S=s.W/'shadow_components';s.main()
