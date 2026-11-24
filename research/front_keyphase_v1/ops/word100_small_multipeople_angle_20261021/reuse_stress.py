"""New public cohort, same evidence-only counterfactual controls."""
import sys
import study
sys.modules["run"]=study
s=study.module("new_cohort_stress",study.M/"stress.py")
s.main()
