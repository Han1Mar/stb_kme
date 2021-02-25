"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
"""

This experiment is based on randomly rotated, 2D Gaussian distributed toy data.
Each of the 50 bags has a different amount of samples (between 10 to 300); this
corresponds to setting (c) Imbalanced Bags.

In order to run the experiments on the IMBALANCED data set please run
> python RUN_ALL.py 
or
proceed as follows:
- Run MODEL_naive.py and FINAL_naive.py *once* which generate the toy data used for
  the model optimization (see ‘../Results/imbalanced_kme/NaiveData/') and the final 
  estimation of the generalization error (‘../Results/imbalanced_kme/FinalData/'). 
  Each other method will be performed on those generated data sets.
- Run MODEL_<method>.py to find the optimal values for the model parameters, e.g.
  run MODEL_STB_weight.py to find optimal values for zeta and gamma. The results will
  be save in ‘../Results/imbalanced_kme/NaiveData/opt_param_<method>.mat'.
- Only after the optimal model parameters were found, run FINAL_<method>.py to estimate
  the generalized KME estimation error of the method; saved in     ‘../Results/imbalanced_kme/FinalData/KME_error_<method>.mat'.
- Once every method was run, you can plot a figure similar to the one seen in the paper
  which summarizes the final results (PLOT_Result.py)