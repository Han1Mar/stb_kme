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

assert(len(sys.argv) == 2)
differentNrBags = (sys.argv[1] == 'differentNrBags')

### update as desired ########################################################
unbiased    = True          # unbiased estimation of the MMD^2
replace     = True          # replace negative values of MMD^2 with zero 
kernel = u.Gaussiankernel   # kernel 
kw1    = 2.25               # kernel width
num_trials = 200            # number of trials

# values that shall be tested in the setting
setting_range = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300])
num_settings = len(setting_range)

NZ      = int(1000)                 # number of sampler (test data) per task
mu     = [0,0]                      # center of each task
S      = np.array([[1,0],[0,10]])   # covariance matrix which will be ...
                                    # ... randomly rotated
##############################################################################
if differentNrBags:
    n = 50
    FN = '../Results/differentNrBags_kme/NaiveData/'
else:
    T = 50
    FN = '../Results/differentBagsizes_kme/NaiveData/'

# create result folder
if not os.path.exists(FN):
    os.makedirs(FN)

print('Starting Experiments: ')

for s_idx in range(num_settings):
    if differentNrBags:
        T = setting_range[s_idx]
        N = np.array([n]*T)
    else:
        n = setting_range[s_idx]
        N = np.array([n]*T)
    
    print('... n = '+str(n)+' , T = '+str(T)+' ('+str(s_idx+1)+' of '+str(num_settings)+')')
    
    for trial in range(num_trials):
        X, Z, Angles = u.gen_indep_data(mu, S, N, NZ)
        Angles = np.array(Angles)

        # compute the kernel matrices
        sum_no_diag_Z, sum_XZ, num_obs_Z = u.compute_test_kernel_sums(X, Z, T, unbiased, kernel, kw1)
        sum_K, sum_no_diag, task_var, num_obs = u.compute_kernel_sums(X, 1., T, kernel, kw1)

        # compute average inter task MMD^2 (gives also the estimated kernel matrix of the naive approach)
        G_naive = u.compute_MMD2_matrix_naive(sum_K, sum_no_diag, num_obs, T, unbiased, replace)

        # write G_naive and other needed information
        # data folder: FN + setting number + trial number
        curr_FN = FN+str(s_idx)+'/'+str(trial)+'/'
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