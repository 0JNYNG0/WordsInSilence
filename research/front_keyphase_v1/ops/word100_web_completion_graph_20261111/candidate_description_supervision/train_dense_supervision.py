from core import *
N=W/'dense_window_supervision'
assert read(N/'PRETRAIN_VERIFICATION.json')['pass_']
m=load('_dense_fit_train',W/'train.py');m.W=N;m.main()
