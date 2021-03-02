"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
WINE_settings.py:
    Sets the settings of the experiment.
"""

### update as desired ########################################################
unbiased    = True       # unbiased estimation of the MMD^2
replace     = True       # replace negative values of MMD^2 with zero 
save_neighs = True       # save the number of neighbors
use_mdnf    = True       # use MDNF (or DNF) kernel
subsample_size  = 50     # number of observations used to estimate the KMEs
num_trials      = 100    # number of trials (repetitions of experiments)
T_full  = 15             # number of bags (all)
T_train = T_full - 1     # number of bags used for training
T_test  = 1              # number of bags used for testing

# where to store the results
str_samples = str(subsample_size)+'_samples/'
if use_mdnf:
    str_kernel  = 'mDNF_Kernel/'
else:
    str_kernel  = 'DNF_Kernel/'
FN = '../Results/wine_kme/'+str_samples+str_kernel+'TrialData/'       
FN_Datapath  = '../Results/wine_kme/Data/'  # where the raw data set is saved
FN_X     = FN_Datapath+'X.mat'              # where the used data is saved
##############################################################################