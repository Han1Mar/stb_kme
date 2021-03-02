"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
KERNEL_preparation.py:
    - precomputes the kernels such that every method can be applied on the
      same data and the results will be comparable.
"""
import sys
sys.path.append('../')
import utiels as u
import WINE_settings as s
import numpy as np
import os
from scipy.io import loadmat

unbiased    = s.unbiased            # unbiased estimation of the MMD^2
replace     = s.replace             # replace negative values of MMD^2 with zero 
save_neighs = s.save_neighs         # save the number of neighbors
subsample_size  = s.subsample_size  # number of observations used to estimate the KMEs
num_trials      = s.num_trials      # number of trials (repetitions of experiments)
T_full  = s.T_full                  # number of bags (all)
T_train = s.T_train                 # number of bags used for training
T_test  = s.T_test                  # number of bags used for testing
FN = s.FN                           # where to store the results
FN_X = s.FN_X                       # where the data is located

### update as desired ########################################################
np.random.seed(1337)

if s.use_mdnf:
    kernel      = u.MDNFkernel
    normalize   = False
else:
    kernel      = u.DNFkernel
    normalize   = True       # whether to use a normalized dnf kernel
##############################################################################

# load the data
Z_mat = loadmat(FN_X)
Z = Z_mat['X']
T,n,D = np.shape(Z)
num_obs_full = [n]*T
num_obs_sample = [subsample_size]*T

for trial in range(num_trials):
    if trial == 0 or np.mod(trial,10) == 0:
        print('... trial = '+str(trial+1)+' / '+str(num_trials))
    #train_idx, test_idx = np.split(t,2)
    subsample_idx = np.random.choice(range(n), subsample_size, replace=False)
    X = Z[:,subsample_idx,:]
    
    sum_no_diag_Z, sum_XZ, num_obs_Z = u.compute_test_kernel_sums(X, Z, T, unbiased, kernel, normalize, D)
    sum_K, sum_no_diag, task_var, num_obs = u.compute_kernel_sums(X, 1., T, kernel, normalize, D)

    # compute average inter task MMD^2 (gives also the estimated kernel matrix of the naive approach)
    G_naive = u.compute_MMD2_matrix_naive(sum_K, sum_no_diag, num_obs, T, unbiased, replace)

    # write G_naive and other needed information
    curr_FN = FN + str(trial) + '/'
    # create result folder
    if not os.path.exists(curr_FN):
        os.makedirs(curr_FN)
    np.savetxt(curr_FN+'G_naive.csv', G_naive, delimiter=',')
    np.savetxt(curr_FN+'sum_K.csv', sum_K, delimiter=',')
    np.savetxt(curr_FN+'sum_no_diag.csv', sum_no_diag, delimiter=',')
    np.savetxt(curr_FN+'task_var.csv', task_var, delimiter=',')
    np.savetxt(curr_FN+'num_obs.csv', num_obs_sample, delimiter=',')
    np.savetxt(curr_FN+'sum_no_diag_Z.csv', sum_no_diag_Z, delimiter=',')
    np.savetxt(curr_FN+'sum_XZ.csv', sum_XZ, delimiter=',')
    np.savetxt(curr_FN+'num_obs_Z.csv', num_obs_full, delimiter=',')
    np.savetxt(curr_FN+'subsample_idx.csv', subsample_idx, delimiter=',')
