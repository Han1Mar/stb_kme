"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
CLUSTERED_FINAL_naive.py:
    - creates the data for the final estimation of the generalized KME
      estimation error
"""
import sys
sys.path.append('../')
import utiels as u
import TOY_settings as s
import numpy as np
import os
import scipy.io as io

### update as desired ########################################################
unbiased    = s.unbiased         # unbiased estimation of the MMD^2
replace     = s.replace          # replace negative values of MMD^2 with zero 
kernel      = u.Gaussiankernel   # kernel 
kw1         = s.kw1              # kernel width
num_trials = s.num_trials_final  # number of trials

FN_save = s.FN_final_Clustered      # where to save the data, results
# tested radius of circle on which the tasks lie equally spaced on
setting_range = s.setting_range_Clustered 
num_settings  = len(setting_range)  # number of settings
num_centroids = s.num_centroids_Clustered                  # number of centroids
T       = 50                        # number of tasks (bags)
N       = [50]*T                    # number of samples (train data) per task
NZ      = s.NZ                      # number of sampler (test data) per task
TperC   = int(T/num_centroids)      # number of tasks per centroid

S      = s.S    # covariance matrix which will be randomly rotated
##############################################################################

KME_error  = {'Error': np.zeros(num_settings), 'std': np.zeros(num_settings)}

print('Starting Experiments: ')

for s_idx in range(num_settings):
    radius = setting_range[s_idx]
    centroids = u.gen_centroids(num_centroids, radius)
    Mu = centroids * TperC
    
    print('... radius = '+str(radius)+' ('+str(s_idx+1)+' of '+str(num_settings)+')')
    
    T_train = T-1
    T_test = 1
    one_hot = np.eye(T, dtype=bool)
    W       = np.eye(T, dtype=bool)
    kme_e   = np.zeros([num_trials, T])
    
    for trial in range(num_trials):
        # generate data
        X, Z, Angles = u.gen_indep_data(Mu, S, N, NZ)
        Angles = np.array(Angles)
        
        # compute the kernel matrices
        sum_no_diag_Z, sum_XZ, num_obs_Z = u.compute_test_kernel_sums(X, Z, T, unbiased, kernel, kw1)
        sum_K, sum_no_diag, task_var, num_obs = u.compute_kernel_sums(X, 1., T, kernel, kw1)

        # compute average inter task MMD^2 (gives also the estimated kernel matrix of the naive approach)
        G  = u.compute_MMD2_matrix_naive(sum_K, sum_no_diag, num_obs, T, unbiased, replace)
        
        # data folder: FN + setting number + trial number
        curr_FN = FN_save+str(s_idx)+'/'+str(trial)+'/'
        # create result folder
        if not os.path.exists(curr_FN):
            os.makedirs(curr_FN)
        np.savetxt(curr_FN+'G_naive.csv', G, delimiter=',')
        np.savetxt(curr_FN+'sum_K.csv', sum_K, delimiter=',')
        np.savetxt(curr_FN+'sum_no_diag.csv', sum_no_diag, delimiter=',')
        np.savetxt(curr_FN+'task_var.csv', task_var, delimiter=',')
        np.savetxt(curr_FN+'num_obs.csv', num_obs, delimiter=',')
        np.savetxt(curr_FN+'Angles.csv', Angles, delimiter=',')
        np.savetxt(curr_FN+'sum_no_diag_Z.csv', sum_no_diag_Z, delimiter=',')
        np.savetxt(curr_FN+'sum_XZ.csv', sum_XZ, delimiter=',')
        np.savetxt(curr_FN+'num_obs_Z.csv', num_obs_Z, delimiter=',')
        
        kme_e[trial,:] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, np.eye(T), T, unbiased, replace)
            
    KME_error['Error'][s_idx] = np.mean(kme_e)
    KME_error['std'][s_idx]   = np.std(kme_e)
    # save the results after each setting
    io.savemat(FN_save+'KME_error_naive.mat', KME_error)
print('Done')