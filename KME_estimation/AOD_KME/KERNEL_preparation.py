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
import AOD_settings as s
import numpy as np
import os
from scipy.io import loadmat

unbiased    = s.unbiased      # unbiased estimation of the MMD^2
replace     = s.replace       # replace negative values of MMD^2 with zero 
kw1_A           = s.kw1_A     # kernel width (used for Gaussian kernel)
subsample_size  = s.subsample_size      # number of observations used to estimate the KMEs
num_trials      = s.num_trials          # number of trials (repetitions of experiments)
FN_XA       = s.FN_Datapath+'X_A.mat'   # where the data set is stored
FN_trial    = s.FN_trial                # where the results will be saved

### update as desired ########################################################
kernel = u.Gaussiankernel                       # kernel
# kernel = u.Linearkernel
##############################################################################

# load the data
X_Amat = loadmat(FN_XA)
X_A = X_Amat['X_A']
T,n,D = np.shape(X_A)
num_obs_full = [n]*T
num_obs_sample = [subsample_size]*T
t = np.linspace(0,T-1,T, dtype=int)

for trial in range(num_trials):
    if trial == 0 or np.mod(trial,10) == 0:
        print('... trial = '+str(trial+1)+' / '+str(num_trials))
    np.random.shuffle(t)
    train_idx, test_idx = np.split(t,2)
    subsample_idx = np.random.choice(range(n), subsample_size, replace=False)
    X = X_A[:,subsample_idx,:]
    
    sum_no_diag_Z, sum_XZ, num_obs_Z = u.compute_test_kernel_sums(X, X_A, T, unbiased, kernel, kw1_A)
    sum_K, sum_no_diag, task_var, num_obs = u.compute_kernel_sums(X, 1., T, kernel, kw1_A)

    # compute average inter task MMD^2 (gives also the estimated kernel matrix of the naive approach)
    G_naive = u.compute_MMD2_matrix_naive(sum_K, sum_no_diag, num_obs, T, unbiased, replace)

    # write G_naive and other needed information
    curr_FN = FN_trial + str(trial) + '/'
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
    np.savetxt(curr_FN+'train_idx.csv', train_idx, delimiter=',', fmt='%i')
    np.savetxt(curr_FN+'test_idx.csv', test_idx, delimiter=',', fmt='%i')
