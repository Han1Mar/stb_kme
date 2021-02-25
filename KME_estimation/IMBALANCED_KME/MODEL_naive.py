"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
MODEL_naive.py:
    - creates the data for the model optimization so that every method can
      be performed on the same data
"""
import sys
sys.path.append('../')
import utiels as u
import numpy as np
import os

### update as desired ########################################################
unbiased    = True          # unbiased estimation of the MMD^2
replace     = True          # replace negative values of MMD^2 with zero 
kernel = u.Gaussiankernel   # kernel 
kw1    = 2.25               # kernel width
num_trials = 100            # number of trials

FN = '../Results/imbalanced_kme/NaiveData/' # where to save the data, results

T  = 50             # number of tasks (bags)
n  = [10,10,20,20,30,30,40,40,50,50,60,60,70,70,80,80,90,90,100,100,125,150,\
      200,250,300]
n  = np.sort(n*2)   
N  = np.array(n)    # number of samples (train data) per task
NZ = int(1000)      # number of sampler (test data) per task

mu     = [0,0]                      # center of each task
S      = np.array([[1,0],[0,10]])   # covariance matrix which will be ...
                                    # ... randomly rotated
##############################################################################

# create result folder
if not os.path.exists(FN):
    os.makedirs(FN)

print('Starting Experiments: ')

for trial in range(num_trials):
    if trial == 0 or np.mod(trial,10) == 0:
        print('... trial = '+str(trial+1)+' / '+str(num_trials))
    X, Z, Angles = u.gen_indep_data(mu, S, N, NZ)
    Angles = np.array(Angles)

    # compute the kernel matrices
    sum_no_diag_Z, sum_XZ, num_obs_Z = u.compute_test_kernel_sums(X, Z, T, unbiased, kernel, kw1)
    sum_K, sum_no_diag, task_var, num_obs = u.compute_kernel_sums(X, 1., T, kernel, kw1)

    # compute average inter task MMD^2 (gives also the estimated kernel matrix of the naive approach)
    G_naive = u.compute_MMD2_matrix_naive(sum_K, sum_no_diag, num_obs, T, unbiased, replace)

    # write G_naive and other needed information
    # data folder: FN + setting number + trial number
    curr_FN = FN+str(trial)+'/'
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

print('Done')