"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
FINAL_STB.py:
    - performs model optimization on part of the data to find optimal values
    - computes the KME estimation error on the wine data set and saves it
"""
import sys
sys.path.append('../')
import utiels as u
import WINE_settings as s
import numpy as np
import scipy.io as io

unbiased    = s.unbiased            # unbiased estimation of the MMD^2
replace     = s.replace             # replace negative values of MMD^2 with zero 
save_neighs = s.save_neighs         # save the number of neighbors
subsample_size  = s.subsample_size  # number of observations used to estimate the KMEs
num_trials      = s.num_trials      # number of trials (repetitions of experiments)
T_full  = s.T_full                  # number of bags (all)
T_train = s.T_train                 # number of bags used for training
T_test  = s.T_test                  # number of bags used for testing
FN = s.FN                           # where to store the results

### update as desired ########################################################
# value range that shall be tested for the model parameters
Zeta = np.array([-1.,0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5, \
                 7.,7.5,8.,8.5,9.,9.5,10.])
num_modelparams = {'Zeta': len(Zeta)}
##############################################################################

### MODEL OPTIMIZATION #######################################################
print('Starting Model Optimization: ')

CV_error  = np.zeros(num_trials)
KME_error = np.zeros([num_trials, T_full])
opt_param = {'Zeta': np.zeros(num_trials)}
if save_neighs:
    NEI = np.zeros(num_trials)

for trial in range(num_trials):
    if trial == 0 or np.mod(trial,10) == 0:
        print('...... trial = '+str(trial+1)+' / '+str(num_trials))
    cv_error = np.zeros([T_train, num_modelparams['Zeta']])
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
    
    # leave-one-out CV
    for t in range(T_full):
        # creating indices for training and testing
        train_idx = list(np.arange(T_full, dtype=int))
        test_idx  = [train_idx.pop(t)]
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
            # compute weighting matrix
            W  = u.compute_STB_neighbors(G_naive, task_var, zeta)
            W  = u.compute_STB_weight(W, 'stb')
            cv_error[:,j_z] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T_train, unbiased, replace)
    
        # get optimal value and save its CV error
        mean_e                      = np.mean(cv_error, axis=0)
        opt_idx_zeta                = np.where(mean_e == np.min(mean_e))[0]
        if len(opt_idx_zeta)>1:
            print('WARNING: more than one ZETA for which cv_error is minimal! Check ZETA range! Choosing first one.')
            opt_idx_zeta = opt_idx_zeta[0]
        CV_error[trial]             = np.mean(cv_error[:,opt_idx_zeta])
        opt_param['Zeta'][trial]    = Zeta[opt_idx_zeta]     
        # save the results after each setting
        np.savetxt(FN+'CV_error_STB.csv',  CV_error,  delimiter=',')
        io.savemat(FN+'opt_param_STB.mat', opt_param)
    
        #### FINAL EXPERIMENTS #######################################################
        zeta = opt_param['Zeta'][trial]
    
        # use all bags for testing (but get only the error for bag t)
        # compute weighting matrix
        W  = u.compute_STB_neighbors(G_naive_, task_var_, zeta)
        W  = u.compute_STB_weight(W, 'stb')
        KME_error[trial,t] = u.compute_KME_error(sum_K_, num_obs_, sum_XZ_, sum_no_diag_Z_, num_obs_Z_, W, T_full, unbiased, replace)[t]
        
        np.savetxt(FN+'KME_error_STB.csv', KME_error, delimiter=',')
        if save_neighs:
            NEI[trial] = np.mean(np.sum(W>0, axis=1))
            np.savetxt(FN+'Neighbors_STB.csv', NEI, delimiter=',')

print(np.mean(KME_error))
print('Done')