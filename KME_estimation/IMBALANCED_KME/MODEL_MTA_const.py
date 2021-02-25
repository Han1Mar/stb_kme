"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
MODEL_MTA_const.py:
    - model optimization on the training data
"""
import sys
sys.path.append('../')
import utiels as u
import numpy as np
import scipy.io as io

### update as desired ########################################################
unbiased    = True          # unbiased estimation of the MMD^2
replace     = True          # replace negative values of MMD^2 with zero 
kernel = u.Gaussiankernel   # kernel 
kw1    = 2.25               # kernel width
num_trials = 100            # number of trials

FN = '../Results/imbalanced_kme/NaiveData/' # where to save the data, results

T  = 50             # number of tasks (bags)

num_modelparams = {'Gamma': 41}
Gamma           = np.linspace(0., 5., num_modelparams['Gamma'])
##############################################################################

print('Starting Experiments: ')

CV_error  = np.zeros(T)
opt_param = {'Gamma': 0}
one_hot   = np.eye(T, dtype=bool)

kme_error = np.zeros([num_trials, T, num_modelparams['Gamma']])

for trial in range(num_trials):
    if trial == 0 or np.mod(trial,10) == 0:
        print('... trial = '+str(trial+1)+' / '+str(num_trials))
    # data folder: FN + setting number + trial number
    curr_FN = FN+str(trial)+'/'
    G_naive     = np.genfromtxt(curr_FN+'G_naive.csv', delimiter=',')
    sum_K       = np.genfromtxt(curr_FN+'sum_K.csv', delimiter=',')
    sum_no_diag = np.genfromtxt(curr_FN+'sum_no_diag.csv', delimiter=',')
    task_var    = np.genfromtxt(curr_FN+'task_var.csv', delimiter=',')
    num_obs     = np.genfromtxt(curr_FN+'num_obs.csv', delimiter=',')
    sum_no_diag_Z = np.genfromtxt(curr_FN+'sum_no_diag_Z.csv', delimiter=',')
    sum_XZ        = np.genfromtxt(curr_FN+'sum_XZ.csv', delimiter=',')
    num_obs_Z     = np.genfromtxt(curr_FN+'num_obs_Z.csv', delimiter=',')

    A = u.compute_MTA_similarity(G_naive, T, 'const')

    for j_g, gamma_idx in enumerate(range(num_modelparams['Gamma'])):
        gamma = Gamma[gamma_idx]
        W  = u.compute_MTA_weight(A, task_var, gamma, T)    
        kme_error[trial,:,j_g] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T, unbiased, replace)

# get optimal value and save its CV error
mean_e        = np.mean(kme_error, axis=(0,1))
opt_idx_gamma = np.where(mean_e == np.min(mean_e))[0]
CV_error      = np.mean(kme_error[:,:,opt_idx_gamma], axis=0)
opt_param['Gamma']  = Gamma[opt_idx_gamma]        
# save the results after each setting
np.savetxt(FN+'CV_error_MTA_const.csv',  CV_error,  delimiter=',')
io.savemat(FN+'opt_param_MTA_const.mat', opt_param)
print('Optimal Parameter: Gamma = '+str(opt_param['Gamma']))
print('Done')