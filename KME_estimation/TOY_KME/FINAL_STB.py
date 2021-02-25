"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
FINAL_STB.py:
    - performs model optimization 
    - pass 'differentNrBags', 'differentBagsizes' or 'clustered' (without '') 
      as argument when called to indicate which experiment shall be performed
"""
import sys
sys.path.append('../')
import utiels as u
import numpy as np
import scipy.io as io

differentNrBags = 'differentNrBags'
differentBagsizes = 'differentBagsizes'
clustered = 'clustered'

assert(len(sys.argv) == 2)
experiment = sys.argv[1]
assert(experiment==differentNrBags or experiment==differentBagsizes or experiment==clustered)

### update as desired ########################################################
unbiased    = True          # unbiased estimation of the MMD^2
replace     = True          # replace negative values of MMD^2 with zero 
save_neighs = True          # save info about neighbors
kernel = u.Gaussiankernel   # kernel 
kw1    = 2.25               # kernel width
num_trials = 200            # number of trials
##############################################################################

if experiment == differentNrBags:
    n = 50
    FN_save = '../Results/differentNrBags_kme/FinalData/'
    FN_load = '../Results/differentNrBags_kme/NaiveData/'
    setting_range = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300])
elif experiment == differentBagsizes:
    T = 50
    FN_save = '../Results/differentBagsizes_kme/FinalData/'
    FN_load = '../Results/differentBagsizes_kme/NaiveData/'
    setting_range = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300])
elif experiment == clustered:
    T = 50
    N = [50]*T
    FN_save = '../Results/clustered_kme/FinalData/'
    FN_load = '../Results/clustered_kme/NaiveData/'
    setting_range = np.linspace(0,5,21)
      
num_settings = len(setting_range)
KME_error  = {'Error': np.zeros(num_settings), 'std': np.zeros(num_settings)}
if save_neighs:
    NEI = np.zeros(num_settings)

opt_param = io.loadmat(FN_load+'opt_param_STB.mat') # 'Zeta'

print('Starting Experiments: ')

for s_idx in range(num_settings):
    if experiment == differentNrBags:
        T = setting_range[s_idx]
        N = np.array([n]*T)
        print('... n = '+str(n)+' , T = '+str(T)+' ('+str(s_idx+1)+' of '+str(num_settings)+')')
    elif experiment == differentBagsizes:
        n = setting_range[s_idx]
        N = np.array([n]*T)
        print('... n = '+str(n)+' , T = '+str(T)+' ('+str(s_idx+1)+' of '+str(num_settings)+')')
    else:
        radius = setting_range[s_idx]
        print('... radius = '+str(radius)+' ('+str(s_idx+1)+' of '+str(num_settings)+')')
    
    one_hot = np.eye(T, dtype=bool)
    kme_e   = np.zeros([num_trials, T])
    if save_neighs:
        nei     = np.zeros(num_trials)
    
    # load the parameters
    zeta   = opt_param['Zeta'][0,s_idx]
    
    for trial in range(num_trials):
        # data folder: FN + setting number + trial number
        curr_FN = FN_save+str(s_idx)+'/'+str(trial)+'/'
        G_naive     = np.genfromtxt(curr_FN+'G_naive.csv', delimiter=',')
        sum_K       = np.genfromtxt(curr_FN+'sum_K.csv', delimiter=',')
        sum_no_diag = np.genfromtxt(curr_FN+'sum_no_diag.csv', delimiter=',')
        task_var    = np.genfromtxt(curr_FN+'task_var.csv', delimiter=',')
        num_obs     = np.genfromtxt(curr_FN+'num_obs.csv', delimiter=',')
        sum_no_diag_Z = np.genfromtxt(curr_FN+'sum_no_diag_Z.csv', delimiter=',')
        sum_XZ        = np.genfromtxt(curr_FN+'sum_XZ.csv', delimiter=',')
        num_obs_Z     = np.genfromtxt(curr_FN+'num_obs_Z.csv', delimiter=',')

        # compute weighting matrix
        W  = u.compute_STB_neighbors(G_naive, task_var, zeta)
        W  = u.compute_STB_weight(W, 'stb')
        
        if save_neighs:
            nei[trial] = np.mean(np.sum(W>0, axis=1))
        
        kme_e[trial,:] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T, unbiased, replace)
            
    KME_error['Error'][s_idx] = np.mean(kme_e)
    KME_error['std'][s_idx]   = np.std(kme_e)
    
    # save the results after each setting
    io.savemat(FN_save+'KME_error_STB.mat', KME_error)
    if save_neighs:
        NEI[s_idx] = np.mean(nei)
        np.savetxt(FN_save+'Neighbors_STB.csv', NEI, delimiter=',')
print('Done')