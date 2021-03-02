"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
MODEL_STB.py:
    - model optimization on the training data
"""
import sys
sys.path.append('../')
import utiels as u
import IMBALANCED_settings as s
import numpy as np
import scipy.io as io

unbiased   = s.unbiased          # unbiased estimation of the MMD^2
replace    = s.replace           # replace negative values of MMD^2 with zero 
num_trials = s.num_trials_model  # number of trials
T          = s.T                 # number of tasks (bags)
FN         = s.FN_model          # where to save the data, results

### update as desired ########################################################
num_modelparams = {'Zeta': 31}
Zeta   = np.linspace(0.,3.0,num_modelparams['Zeta'])
##############################################################################

print('Starting Experiments: ')

CV_error  = np.zeros(T)
opt_param = {'Zeta': 0}
one_hot   = np.eye(T, dtype=bool)

kme_error = np.zeros([num_trials, T, num_modelparams['Zeta']])

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

    for j_z, zeta_idx in enumerate(range(num_modelparams['Zeta'])):
        zeta = Zeta[zeta_idx]
        # compute weighting matrix
        W  = u.compute_STB_neighbors(G_naive, task_var, zeta)
        W  = u.compute_STB_weight(W, 'stb')
        kme_error[trial,:,j_z] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T, unbiased, replace)

# get optimal value and save its CV error
mean_e      = np.mean(kme_error, axis=(0,1))
opt_idx_zeta = np.where(mean_e == np.min(mean_e))[0]
CV_error    = np.mean(kme_error[:,:,opt_idx_zeta], axis=0)
opt_param['Zeta']    = Zeta[opt_idx_zeta]     
# save the results after each setting
np.savetxt(FN+'CV_error_STB.csv',  CV_error,  delimiter=',')
io.savemat(FN+'opt_param_STB.mat', opt_param)
print('Optimal Parameter: Zeta = '+str(opt_param['Zeta']))
print('Done')