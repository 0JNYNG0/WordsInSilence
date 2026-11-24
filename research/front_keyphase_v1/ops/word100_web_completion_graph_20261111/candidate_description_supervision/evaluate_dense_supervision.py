from core import *
N=W/'dense_window_supervision'
m=load('_dense_supervision_eval',W/'dense_proposal_probe.py');m.W=N;m.D=N/'evaluation';m.main()
