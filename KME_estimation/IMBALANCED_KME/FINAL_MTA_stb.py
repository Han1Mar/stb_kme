"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
FINAL_MTA_stb.py:
    - estimate generalization error using optimal parameter values 
"""
import sys
sys.path.append('../')
import utiels as u
import numpy as np
import scipy.io as io

### update as desired ########################################################
unbiased    = True          # unbiased estimation of the MMD^2
replace     = True          # replace negative values of MMD^2 with zero 
save_neighs = True          # save the number of neighbors
kernel = u.Gaussiankernel   # kernel 
kw1    = 2.25               # kernel width
num_trials = 200            # number of trials

FN_load = '../Results/imbalanced_kme/NaiveData/' # where to find opt. param.
FN_save = '../Results/imbalanced_kme/FinalData/' # where to save the results

T  = 50             # number of tasks (bags)
##############################################################################

opt_param = io.loadmat(FN_load+'opt_param_MTA_stb.mat')
zeta       = opt_param['Zeta'][0,0]
gamma     = opt_param['Gamma'][0,0]

print('Starting Experiments: ')

one_hot   = np.eye(T, dtype=bool)

kme_error = np.zeros([num_trials, T])
KME_error  = {'Error': np.zeros(T), 'std': np.zeros(T)}
if save_neighs:
    nei = np.zeros([num_trials, T])
    NEI = np.zeros(T)

for trial in range(num_trials):
    if trial == 0 or np.mod(trial,10) == 0:
        print('... trial = '+str(trial+1)+' / '+str(num_trials))
    # data folder: FN + setting number + trial number
    curr_FN = FN_save+str(trial)+'/'
    G_naive     = np.genfromtxt(curr_FN+'G_naive.csv', delimiter=',')
    sum_K       = np.genfromtxt(curr_FN+'sum_K.csv', delimiter=',')
    sum_no_diag = np.genfromtxt(curr_FN+'sum_no_diag.csv', delimiter=',')
    task_var    = np.genfromtxt(curr_FN+'task_var.csv', delimiter=',')
    num_obs     = np.genfromtxt(curr_FN+'num_obs.csv', delimiter=',')
    sum_no_diag_Z = np.genfromtxt(curr_FN+'sum_no_diag_Z.csv', delimiter=',')
    sum_XZ        = np.genfromtxt(curr_FN+'sum_XZ.csv', delimiter=',')
    num_obs_Z     = np.genfromtxt(curr_FN+'num_obs_Z.csv', delimiter=',')

    # compute weighting matrix
    A = u.compute_MTA_similarity(G_naive, T, 'stb', zeta, task_var)
    W = u.compute_MTA_weight(A, task_var, gamma, T)
    kme_error[trial,:] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T, unbiased, replace)
    if save_neighs:
        nei[trial, :] = np.sum(A>0, axis=1)

KME_error['Error'] = np.mean(kme_error, axis=0)
KME_error['std']   = np.std(kme_error, axis=0)
# save the results after each setting
io.savemat(FN_save+'KME_error_MTA_stb.mat', KME_error)
if save_neighs:
        NEI = np.mean(nei, axis=0)
        np.savetxt(FN_save+'Neighbors_MTA_stb.csv', NEI, delimiter=',')
print('Done')