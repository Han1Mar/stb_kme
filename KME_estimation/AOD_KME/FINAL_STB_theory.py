"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
FINAL_STB_theory.py:
    - performs model optimization on part of the data to find optimal values
    - computes the KME estimation error on the MISR1 data set and saves it
    - here: gamma (not as notated in paper) refers to the mult. constant c
"""
import sys
sys.path.append('../')
import utiels as u
import AOD_settings as s
import numpy as np
import scipy.io as io

unbiased    = s.unbiased            # unbiased estimation of the MMD^2
replace     = s.replace             # replace negative values of MMD^2 with zero 
save_neighs = s.save_neighs         # save info about found neighbors
subsample_size  = s.subsample_size  # number of observations used to estimate the KMEs
num_trials      = s.num_trials      # number of trials (repetitions of experiments)
T_full  = s.T_full                  # number of bags (all)
T_train = s.T_train                 # number of bags used for training
T_test  = s.T_test                  # number of bags used for testing
FN = s.FN_trial                     # where to store the results

### update as desired ########################################################
# value range that shall be tested for the model parameters
# for the linear kernel, the number of tested model parameters can be reduced
# to 21, i.e. num_modelparams = {'Zeta': 21, 'Gamma': 21}
num_modelparams = {'Zeta': 41, 'Gamma': 41}
Zeta     = np.linspace(0., 10., num_modelparams['Zeta'])
Gamma    = np.linspace(0., 10., num_modelparams['Gamma'])
Gamma[0] = 0.1
##############################################################################

### MODEL OPTIMIZATION #######################################################
print('Starting Model Optimization: ')

CV_error  = np.zeros(num_trials)
KME_error = np.zeros([num_trials, T_test])
opt_param = {'Zeta': np.zeros(num_trials), 'Gamma': np.zeros(num_trials)}
if save_neighs:
    NEI = np.zeros(num_trials)

for trial in range(num_trials):
    if trial == 0 or np.mod(trial,10) == 0:
        print('...... trial = '+str(trial+1)+' / '+str(num_trials))
    cv_error = np.zeros([T_train, num_modelparams['Zeta'], num_modelparams['Gamma']])
    # data folder: FN + setting number + trial number
    curr_FN = FN+str(trial)+'/'
    G_naive_        = np.genfromtxt(curr_FN+'G_naive.csv', delimiter=',')
    sum_K_          = np.genfromtxt(curr_FN+'sum_K.csv', delimiter=',')
    sum_no_diag_    = np.genfromtxt(curr_FN+'sum_no_diag.csv', delimiter=',')
    task_var_       = np.genfromtxt(curr_FN+'task_var.csv', delimiter=',')
    num_obs_        = np.genfromtxt(curr_FN+'num_obs.csv', delimiter=',')
    sum_no_diag_Z_  = np.genfromtxt(curr_FN+'sum_no_diag_Z.csv', delimiter=',')
    sum_XZ_         = np.genfromtxt(curr_FN+'sum_XZ.csv', delimiter=',')
    num_obs_Z_      = np.genfromtxt(curr_FN+'num_obs_Z.csv', delimiter=',')
    train_idx       = np.sort(np.genfromtxt(curr_FN+'train_idx.csv', delimiter=',', dtype=int))
    test_idx        = np.sort(np.genfromtxt(curr_FN+'test_idx.csv', delimiter=',', dtype=int))
    
    assert(T_train == len(train_idx))
    assert(T_test == len(test_idx))
    
    # get only the bags for training (finding opt. parameters)
    train_mask              = np.zeros(T_full, dtype=bool)
    train_mask[train_idx]   = True
    train_mask              = np.outer(train_mask, train_mask)
    G_naive         = np.reshape(G_naive_[train_mask], (T_train, T_train))
    sum_K           = np.reshape(sum_K_[train_mask], (T_train, T_train))
    sum_no_diag     = sum_no_diag_[train_idx]
    task_var        = task_var_[train_idx]
    num_obs         = num_obs_[train_idx]
    sum_no_diag_Z   = sum_no_diag_Z_[train_idx]
    sum_XZ          = np.reshape(sum_XZ_[train_mask], (T_train, T_train))
    num_obs_Z       = num_obs_Z_[train_idx]
    

    for j_z, zeta_idx in enumerate(range(num_modelparams['Zeta'])):
        zeta = Zeta[zeta_idx]
        A  = u.compute_STB_neighbors(G_naive, task_var, zeta)
        
        for j_g, gamma_idx in enumerate(range(num_modelparams['Gamma'])):
            gamma = Gamma[gamma_idx]
            W  = u.compute_STB_weight(A, 'theory', zeta, gamma)
            cv_error[:,j_z,j_g] =  u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T_train, unbiased, replace)

    # get optimal value and save its CV error
    mean_e                      = np.mean(cv_error, axis=0)
    opt_idx_zeta, opt_idx_gamma  = np.where(mean_e == np.min(mean_e))
    CV_error[trial]             = np.mean(cv_error[:,opt_idx_zeta, opt_idx_gamma])
    opt_param['Zeta'][trial]     = Zeta[opt_idx_zeta]   
    opt_param['Gamma'][trial]   = Gamma[opt_idx_gamma]      
    # save the results after each setting
    np.savetxt(FN+'CV_error_STB_theory.csv',  CV_error,  delimiter=',')
    io.savemat(FN+'opt_param_STB_theory.mat', opt_param)
    
    #### FINAL EXPERIMENTS #######################################################
    zeta = opt_param['Zeta'][trial]
    gamma = opt_param['Gamma'][trial]
    
    # get only the bags for training (finding opt. parameters)
    test_mask             = np.zeros(T_full, dtype=bool)
    test_mask[test_idx]   = True
    test_mask             = np.outer(test_mask, test_mask)
    G_naive         = np.reshape(G_naive_[test_mask], (T_test, T_test))
    sum_K           = np.reshape(sum_K_[test_mask], (T_test, T_test))
    sum_no_diag     = sum_no_diag_[test_idx]
    task_var        = task_var_[test_idx]
    num_obs         = num_obs_[test_idx]
    sum_no_diag_Z   = sum_no_diag_Z_[test_idx]
    sum_XZ          = np.reshape(sum_XZ_[test_mask], (T_test, T_test))
    num_obs_Z       = num_obs_Z_[test_idx]

    # compute weighting matrix
    A  = u.compute_STB_neighbors(G_naive, task_var, zeta)
    W  = u.compute_STB_weight(A, 'theory', zeta, gamma)
    KME_error[trial,:] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T_test, unbiased, replace)
    
    np.savetxt(FN+'KME_error_STB_theory.csv', KME_error, delimiter=',')
    if save_neighs:
        NEI[trial] = np.mean(np.sum(A>0, axis=1))
        np.savetxt(FN+'Neighbors_STB_theory.csv', NEI, delimiter=',')

print('Done')