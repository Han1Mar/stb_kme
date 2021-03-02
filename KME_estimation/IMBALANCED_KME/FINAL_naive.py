"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
FINAL_naive.py:
    - creates the data for the model optimization so that every method can
      be performed on the same data
"""
import sys
sys.path.append('../')
import utiels as u
import IMBALANCED_settings as s
import numpy as np
import os
import scipy.io as io

unbiased    = s.unbiased          # unbiased estimation of the MMD^2
replace     = s.replace           # replace negative values of MMD^2 with zero 
kernel      = u.Gaussiankernel    # kernel 
kw1         = s.kw1               # kernel width
num_trials  = s.num_trials_final  # number of trials

FN_save = s.FN_final  # where to save the data, results

T  = s.T   # number of tasks (bags)
N  = s.N   # number of samples (train data) per task
NZ = s.NZ  # number of sampler (test data) per task

mu     = s.mu  # center of each task
S      = s.S   # covariance matrix which will be randomly rotated

print('Starting Experiments: ')

one_hot   = np.eye(T, dtype=bool)

kme_error = np.zeros([num_trials, T])
KME_error  = {'Error': np.zeros(T), 'std': np.zeros(T)}

for trial in range(num_trials):
    if trial == 0 or np.mod(trial,10) == 0:
        print('... trial = '+str(trial+1)+' / '+str(num_trials))
    # data folder: FN + setting number + trial number
    curr_FN = FN_save+str(trial)+'/'
    X, Z, Angles = u.gen_indep_data(mu, S, N, NZ)
    Angles = np.array(Angles)

    # compute the kernel matrices
    sum_no_diag_Z, sum_XZ, num_obs_Z = u.compute_test_kernel_sums(X, Z, T, unbiased, kernel, kw1)
    sum_K, sum_no_diag, task_var, num_obs = u.compute_kernel_sums(X, 1., T, kernel, kw1)

    # compute average inter task MMD^2 (gives also the estimated kernel matrix of the naive approach)
    G_naive = u.compute_MMD2_matrix_naive(sum_K, sum_no_diag, num_obs, T, unbiased, replace)

    # write G_naive and other needed information
    # create result folder
    if not os.path.exists(curr_FN):
        os.makedirs(curr_FN)
    np.savetxt(curr_FN+'G_naive.csv', G_naive, delimiter=',')
    np.savetxt(curr_FN+'sum_K.csv', sum_K, delimiter=',')
    np.savetxt(curr_FN+'sum_no_diag.csv', sum_no_diag, delimiter=',')
    np.savetxt(curr_FN+'task_var.csv', task_var, delimiter=',')
    np.savetxt(curr_FN+'num_obs.csv', num_obs, delimiter=',')
    np.savetxt(curr_FN+'Angles.csv', Angles, delimiter=',')
    np.savetxt(curr_FN+'sum_no_diag_Z.csv', sum_no_diag_Z, delimiter=',')
    np.savetxt(curr_FN+'sum_XZ.csv', sum_XZ, delimiter=',')
    np.savetxt(curr_FN+'num_obs_Z.csv', num_obs_Z, delimiter=',')

    kme_error[trial,:] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, np.eye(T), T, unbiased, replace)

KME_error['Error'] = np.mean(kme_error, axis=0)
KME_error['std']   = np.std(kme_error, axis=0)
# save the results after each setting
io.savemat(FN_save+'KME_error_naive.mat', KME_error)
print('Done')