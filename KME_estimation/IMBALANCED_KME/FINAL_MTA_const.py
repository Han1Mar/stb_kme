"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
FINAL_MTA_const.py:
    - estimate generalization error using optimal parameter values 
"""
import sys
sys.path.append('../')
import utiels as u
import IMBALANCED_settings as s
import numpy as np
import scipy.io as io

unbiased    = s.unbiased          # unbiased estimation of the MMD^2
replace     = s.replace           # replace negative values of MMD^2 with zero 
num_trials  = s.num_trials_final  # number of trials
T           = s.T                 # number of tasks (bags)

FN_load = s.FN_model # where to find opt. param.
FN_save = s.FN_final # where to save the results

gamma = io.loadmat(FN_load+'opt_param_MTA_const.mat')['Gamma'][0,0]

print('Starting Experiments: ')

CV_error  = np.zeros(T)
one_hot   = np.eye(T, dtype=bool)

kme_error = np.zeros([num_trials, T])
KME_error  = {'Error': np.zeros(T), 'std': np.zeros(T)}

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
    A = u.compute_MTA_similarity(G_naive, T, 'const')
    W  = u.compute_MTA_weight(A, task_var, gamma, T)    
    kme_error[trial,:] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T, unbiased, replace)

KME_error['Error'] = np.mean(kme_error, axis=0)
KME_error['std']   = np.std(kme_error, axis=0)
# save the results after each setting
io.savemat(FN_save+'KME_error_MTA_const.mat', KME_error)
print('Done')