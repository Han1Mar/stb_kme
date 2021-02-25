"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
"""

This experiment is based on the MISR1 data set which is a collection of 800 bags with each 100 samples of 16 features. More information on the data set can be found in

>>> Wang, Z., Lan, L., and Vucetic, S. (2011). Mixture model for multiple instance
    regression and   applications in remote sensing. IEEE Transactions on Geoscience 
    and Remote Sensing, 50(6):2226–2237. <<<

You received a copy of the MISR1.mat data set for your convenience.

In order to run the experiments on the MISR1 data set please run
> python RUN_ALL.py
or
proceed as follows:
- Make sure that the folder ‘../Results/aod_kme/Data/' holds the MISR1.mat data set
- Run the script DATA_preparation.py to preprocess the data, which standardizes and
  retrieves only relevant features
- Run the script KERNEL_preparation.py to precompute all relevant kernels. 
- Make sure to run DATA_preparation.py and KERNEL_preparation.py only *once* so that 
  every approach operates on the same (subsampled) data.
- Run the experiments on the MISR data set for a specific method by running the 
  corresponding script, e.g. run FINAL_STB_weight.py to estimate the KME estimation 
  error of STB weight. It finds optimal values for the model parameters (saved in 
  opt_param_<method>.mat), saves the cross-validation error of the optimal parameter 
  combination (CV_error_<method>.csv) and the final KME estimation error computed on 
  the test data (KME_error_<method>.csv) in folder '../Results/aod_kme/TrialData/'.
- Once every method was run, you can print a summary of the results with 
  PLOT_Results.py.
