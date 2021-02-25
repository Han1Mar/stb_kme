"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
MODEL_STB.py:
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
kernel = u.Gaussiankernel   # kernel 
kw1    = 2.25               # kernel width
num_trials = 200            # number of trials
                                    
# check every possible value for the specified model parameter in first setting
exhaustive_search = {'Zeta': False}

assert(experiment==differentNrBags or experiment==differentBagsizes or experiment==clustered)

# In each setting not the whole parameter range is tested, but only a ...
# ... subset of values. This subset is centered around the optimal value of ...
# ... the last tested setting, i.e. test_range_(s_idx+1) = ...
# ... = [lastbest_idx(s_idx)-test_range, lastbest_idx(s_idx)+test_range]
testrange         = 10
num_modelparams = {'Zeta': 31}
Zeta   = np.linspace(0.,3.0,num_modelparams['Zeta'])
###############################################################################

if experiment == differentNrBags:
    n = 50
    lastbest_idx = {'Zeta': 11}  # Zeta:1.1
    FN = '../Results/differentNrBags_kme/NaiveData/'
    setting_range = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300])
elif experiment == differentBagsizes:
    T = 50
    lastbest_idx = {'Zeta': 1} # Zeta:0.1
    FN = '../Results/differentBagsizes_kme/NaiveData/'
    setting_range = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300])
elif experiment == clustered:
    T = 50
    N = [50]*T
    lastbest_idx = {'Zeta': 11} # Zeta:1.1
    FN = '../Results/clustered_kme/NaiveData/'
    setting_range = np.linspace(0,5,21)
    
num_settings = len(setting_range)

print('Starting Experiments: ')

CV_error  = np.zeros(num_settings)
opt_param = {'Zeta': np.zeros(num_settings)}

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

    min_idx = {'Zeta': max(0, lastbest_idx['Zeta']-testrange)}
    max_idx = {'Zeta': min(num_modelparams['Zeta'], lastbest_idx['Zeta']+testrange)}

    if s_idx == 0:
        if exhaustive_search['Zeta']:
            min_idx['Zeta']   = 0
            max_idx['Zeta']   = num_modelparams['Zeta']
    
    kme_error = np.zeros([num_trials, T, max_idx['Zeta']-min_idx['Zeta']])
    
    for trial in range(num_trials):
        # data folder: FN + setting number + trial number
        curr_FN = FN+str(s_idx)+'/'+str(trial)+'/'
        G_naive     = np.genfromtxt(curr_FN+'G_naive.csv', delimiter=',')
        sum_K       = np.genfromtxt(curr_FN+'sum_K.csv', delimiter=',')
        sum_no_diag = np.genfromtxt(curr_FN+'sum_no_diag.csv', delimiter=',')
        task_var    = np.genfromtxt(curr_FN+'task_var.csv', delimiter=',')
        num_obs     = np.genfromtxt(curr_FN+'num_obs.csv', delimiter=',')
        sum_no_diag_Z = np.genfromtxt(curr_FN+'sum_no_diag_Z.csv', delimiter=',')
        sum_XZ        = np.genfromtxt(curr_FN+'sum_XZ.csv', delimiter=',')
        num_obs_Z     = np.genfromtxt(curr_FN+'num_obs_Z.csv', delimiter=',')

        for j_z, zeta_idx in enumerate(range(min_idx['Zeta'], max_idx['Zeta'])):
            zeta = Zeta[zeta_idx]
            # compute weighting matrix
            W  = u.compute_STB_neighbors(G_naive, task_var, zeta)
            W  = u.compute_STB_weight(W, 'stb')
            
            kme_error[trial,:,j_z] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T, unbiased, replace)

    # get optimal value and save its CV error
    _,_, end_t = np.shape(kme_error)
    mean_e = np.mean(kme_error[:,:, 0:end_t], axis=(0,1))
    opt_idx_zeta = np.where(mean_e == np.min(mean_e))[0]
    CV_error[s_idx] = np.min(mean_e)
    lastbest_idx['Zeta']       = min_idx['Zeta'] + opt_idx_zeta[0]
    opt_param['Zeta'][s_idx]   = Zeta[lastbest_idx['Zeta']]         
    # save the results after each setting
    np.savetxt(FN+'CV_error_STB.csv', CV_error, delimiter=',')
    io.savemat(FN+'opt_param_STB.mat', opt_param)
print('Done')