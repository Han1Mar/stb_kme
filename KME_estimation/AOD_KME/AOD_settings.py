"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
AOD_settings.py:
    Sets the settings of the experiment.
"""

unbiased    = True       # unbiased estimation of the MMD^2
replace     = True       # replace negative values of MMD^2 with zero 
save_neighs = True       # save info about found neighbors
kw1_A           = 1.     # kernel width (used for Gaussian kernel)
subsample_size  = 20     # number of observations used to estimate the KMEs
num_trials      = 100    # number of trials (repetitions of experiments)
T_full  = 800            # number of bags (all)
T_train = 400            # number of bags used for training
T_test  = 400            # number of bags used for testing

# where the data is stored
FN_Datapath = '../Results/aod_kme/Data/'
FN_trial    = '../Results/aod_kme/TrialData/'