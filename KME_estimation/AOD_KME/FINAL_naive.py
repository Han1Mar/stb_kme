"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
FINAL_naive.py:
    - computes the KME estimation error on the MISR1 data set and saves it
"""
import sys
sys.path.append('../')
import utiels as u
import AOD_settings as s
import numpy as np

unbiased    = s.unbiased            # unbiased estimation of the MMD^2
replace     = s.replace             # replace negative values of MMD^2 with zero 
subsample_size  = s.subsample_size  # number of observations used to estimate the KMEs
num_trials      = s.num_trials      # number of trials (repetitions of experiments)
T_full  = s.T_full                  # number of bags (all)
T_train = s.T_train                 # number of bags used for training
T_test  = s.T_test                  # number of bags used for testing
FN = s.FN_trial                     # where to store the results

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

    KME_error[trial,:] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, np.eye(T_test), T_test, unbiased, replace)
    
    # save the results after each setting
    np.savetxt(FN+'KME_error_naive.csv', KME_error, delimiter=',')

print('Done')