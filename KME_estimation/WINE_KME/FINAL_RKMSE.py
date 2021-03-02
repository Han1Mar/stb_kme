"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
FINAL_RKMSE.py:
    - computes the KME estimation error on the wine data set and saves it
"""
import sys
sys.path.append('../')
import utiels as u
import WINE_settings as s
import numpy as np

unbiased    = s.unbiased            # unbiased estimation of the MMD^2
replace     = s.replace             # replace negative values of MMD^2 with zero 
save_neighs = s.save_neighs         # save the number of neighbors
subsample_size  = s.subsample_size  # number of observations used to estimate the KMEs
num_trials      = s.num_trials      # number of trials (repetitions of experiments)
T_full  = s.T_full                  # number of bags (all)
T_train = s.T_train                 # number of bags used for training
T_test  = s.T_test                  # number of bags used for testing
FN = s.FN                           # where to store the results

#### FINAL EXPERIMENTS #######################################################
print('Starting Final Experiments: ')
KME_error = np.zeros([num_trials, T_full])

for trial in range(num_trials):
    if trial == 0 or np.mod(trial,10) == 0:
        print('... trial = '+str(trial+1)+' / '+str(num_trials))
    # data folder: FN + setting number + trial number
    curr_FN = FN+str(trial)+'/'
    sum_K_          = np.genfromtxt(curr_FN+'sum_K.csv', delimiter=',')
    sum_no_diag_    = np.genfromtxt(curr_FN+'sum_no_diag.csv', delimiter=',')
    num_obs_        = np.genfromtxt(curr_FN+'num_obs.csv', delimiter=',')
    sum_no_diag_Z_  = np.genfromtxt(curr_FN+'sum_no_diag_Z.csv', delimiter=',')
    sum_XZ_         = np.genfromtxt(curr_FN+'sum_XZ.csv', delimiter=',')
    num_obs_Z_      = np.genfromtxt(curr_FN+'num_obs_Z.csv', delimiter=',')
    
    # compute weighting matrix
    W = u.compute_RKMSE_weight(sum_K_, sum_no_diag_, num_obs_)

    KME_error[trial,:] = u.compute_KME_error(sum_K_, num_obs_, sum_XZ_, sum_no_diag_Z_, num_obs_Z_, W, T_full, unbiased, replace)
    
    # save the results after each setting
    np.savetxt(FN+'KME_error_RKMSE.csv', KME_error, delimiter=',')
print(np.mean(KME_error))
print('Done')