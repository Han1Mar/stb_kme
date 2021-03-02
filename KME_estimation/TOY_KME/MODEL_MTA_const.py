"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
MODEL_MTA_const.py:
    - performs model optimization 
    - pass 'differentNrBags', 'differentBagsizes' or 'clustered' (without '') 
      as argument when called to indicate which experiment shall be performed
"""
import sys
sys.path.append('../')
import utiels as u
import TOY_settings as s
import numpy as np
import scipy.io as io

differentNrBags = s.differentNrBags
differentBagsizes = s.differentBagsizes
clustered = s.clustered

assert(len(sys.argv) == 2)
experiment = sys.argv[1]
assert(experiment==differentNrBags or experiment==differentBagsizes or experiment==clustered)

unbiased    = s.unbiased         # unbiased estimation of the MMD^2
replace     = s.replace          # replace negative values of MMD^2 with zero 
num_trials  = s.num_trials_model # number of trials

### update as desired ########################################################
# check every possible value for the specified model parameter in first setting
exhaustive_search = {'Gamma': False} 

# In each setting not the whole parameter range is tested, but only a ...
# ... subset of values. This subset is centered around the optimal value of ...
# ... the last tested setting, i.e. test_range_(s_idx+1) = ...
# ... = [lastbest_idx(s_idx)-test_range, lastbest_idx(s_idx)+test_range]
testrange         = 10
num_modelparams   = {'Gamma': 41}
Gamma = np.linspace(0., 5., num_modelparams['Gamma'])
##############################################################################
if experiment == differentNrBags:
    n = 50
    lastbest_idx = {'Gamma': 3}  # Gamma: 0.075
    FN = s.FN_model_NrBags
    setting_range = s.setting_range
elif experiment == differentBagsizes:
    T = 50
    lastbest_idx = {'Gamma': 2}  # Gamma:0.05
    FN = s.FN_model_Bagsizes
    setting_range = s.setting_range
elif experiment == clustered:
    T = 50
    N = [50]*T
    lastbest_idx = {'Gamma': 3}  # Gamma: 0.075
    FN = s.FN_model_Clustered
    setting_range = s.setting_range_Clustered

num_settings = len(setting_range)

print('Starting Experiments: ')

CV_error  = np.zeros(num_settings)
opt_param = {'Gamma': np.zeros(num_settings)}

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

    min_idx = {'Gamma': max(0, lastbest_idx['Gamma']-testrange)}
    max_idx = {'Gamma': min(num_modelparams['Gamma'], lastbest_idx['Gamma']+testrange)}
    
    if s_idx == 0:
        if exhaustive_search['Gamma']:
            min_idx['Gamma']   = 0
            max_idx['Gamma']   = num_modelparams['Gamma']
             
    kme_error = np.zeros([num_trials, T, max_idx['Gamma']-min_idx['Gamma']])
    
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

        A = u.compute_MTA_similarity(G_naive, T, 'const')
            
        for j_g, gamma_idx in enumerate(range(min_idx['Gamma'], max_idx['Gamma'])):
            gamma = Gamma[gamma_idx]
            W  = u.compute_MTA_weight(A, task_var, gamma, T)
            kme_error[trial,:, j_g] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T, unbiased, replace)

    # get optimal value and save its CV error
    _,_,end_g = np.shape(kme_error)
    mean_e = np.mean(kme_error[:,:, 0:end_g], axis=(0,1))
    opt_idx_gamma = np.where(mean_e == np.min(mean_e))[0]
    CV_error[s_idx] = np.min(mean_e)
    
    lastbest_idx['Gamma']     = min_idx['Gamma'] + opt_idx_gamma[0]
    opt_param['Gamma'][s_idx] = Gamma[lastbest_idx['Gamma']]

    # save the results after each setting
    np.savetxt(FN+'CV_error_MTA_const.csv', CV_error, delimiter=',')
    io.savemat(FN+'opt_param_MTA_const.mat', opt_param)
print('Done')