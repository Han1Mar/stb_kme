"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
FINAL_RKMSE.py:
    - computes the KME estimation error on the MISR1 data set and saves it
"""
import sys
sys.path.append('../')
import utiels as u
import numpy as np

### update as desired ########################################################
unbiased    = True       # unbiased estimation of the MMD^2
replace     = True       # replace negative values of MMD^2 with zero 
kw1_A           = 1.     # kernel width
subsample_size  = 20     # number of observations used to estimate the KMEs
num_trials      = 100    # number of trials (repetitions of experiments)
T_full  = 800            # number of bags (all)
T_train = 400            # number of bags used for training
T_test  = 400            # number of bags used for testing

FN = '../Results/aod_kme/TrialData/'    # where to store the results
##############################################################################

#### FINAL EXPERIMENTS #######################################################
print('Starting Final Experiments: ')
KME_error = np.zeros([num_trials, T_test])

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
    test_idx        = np.sort(np.genfromtxt(curr_FN+'test_idx.csv', delimiter=',', dtype=int))
    
    assert(T_test == len(test_idx))
    
    # get only the bags for training (finding opt. parameters)
    test_mask             = np.zeros(T_full, dtype=bool)
    test_mask[test_idx]   = True
    test_mask             = np.outer(test_mask, test_mask)
    sum_K           = np.reshape(sum_K_[test_mask], (T_test, T_test))
    sum_no_diag     = sum_no_diag_[test_idx]
    num_obs         = num_obs_[test_idx]
    sum_no_diag_Z   = sum_no_diag_Z_[test_idx]
    sum_XZ          = np.reshape(sum_XZ_[test_mask], (T_test, T_test))
    num_obs_Z       = num_obs_Z_[test_idx]

    # compute weighting matrix
    W = u.compute_RKMSE_weight(sum_K, sum_no_diag, num_obs)

    KME_error[trial,:] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T_test, unbiased, replace)
    
    # save the results after each setting
    np.savetxt(FN+'KME_error_RKMSE.csv', KME_error, delimiter=',')
print('Done')