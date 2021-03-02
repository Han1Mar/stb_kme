"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
MODEL_MTA_stb.py:
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
# load optimal parameter values from past model optimization
load_param        = {'Zeta': False, 'Gamma': False}
# check every possible value for the specified model parameter in first setting
exhaustive_search = {'Zeta': False, 'Gamma': False}

assert(experiment==differentNrBags or experiment==differentBagsizes or experiment==clustered)
assert(not(load_param['Zeta']   and load_param['Gamma']))
assert(not(load_param['Zeta']   and exhaustive_search['Zeta']))
assert(not(load_param['Gamma'] and exhaustive_search['Gamma']))

# In each setting not the whole parameter range is tested, but only a ...
# ... subset of values. This subset is centered around the optimal value of ...
# ... the last tested setting, i.e. test_range_(s_idx+1) = ...
# ... = [lastbest_idx(s_idx)-test_range, lastbest_idx(s_idx)+test_range]
testrange         = 10
num_modelparams   = {'Zeta': 41, 'Gamma': 41}
Zeta  = np.linspace(0.,  10.,num_modelparams['Zeta'])
Gamma = np.linspace(0., 10000., num_modelparams['Gamma'])
###############################################################################

if experiment == differentNrBags:
    n = 50
    lastbest_idx = {'Zeta': 0, 'Gamma': 2}  # Zeta:1.0,  Gamma: 5.
    FN = s.FN_model_NrBags
    setting_range = s.setting_range
elif experiment == differentBagsizes:
    T = 50
    lastbest_idx = {'Zeta': 5, 'Gamma': 2} # Zeta:0.48 , Gamma: 5.
    FN = s.FN_model_Bagsizes
    setting_range = s.setting_range
elif experiment == clustered:
    T = 50
    N = [50]*T
    lastbest_idx = {'Zeta': 13, 'Gamma': 4} # Zeta:1.3 , Gamma: 10.
    FN = s.FN_model_Clustered
    setting_range = s.setting_range_Clustered
    
num_settings = len(setting_range)

print('Starting Experiments: ')

CV_error  = np.zeros(num_settings)
opt_param = {'Zeta': np.zeros(num_settings), 'Gamma': np.zeros(num_settings)}

if load_param['Zeta']:
    oldResults = io.loadmat(FN+'opt_param_MTA_stb.mat')
    opt_param['Zeta'] = oldResults['Zeta']
if load_param['Gamma']:
    oldResults = io.loadmat(FN+'opt_param_MTA_stb.mat')
    opt_param['Gamma'] = oldResults['Gamma']

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

    min_idx = {'Zeta': max(0, lastbest_idx['Zeta']-testrange), \
               'Gamma': max(0, lastbest_idx['Gamma']-testrange)}
    max_idx = {'Zeta'  : min(num_modelparams['Zeta']  , lastbest_idx['Zeta']+testrange), \
               'Gamma': min(num_modelparams['Gamma'], lastbest_idx['Gamma']+testrange)}
        
    if s_idx == 0:
        if exhaustive_search['Zeta']:
            min_idx['Zeta']   = 0
            max_idx['Zeta']   = num_modelparams['Zeta']
        if exhaustive_search['Gamma']:
            min_idx['Gamma']   = 0
            max_idx['Gamma']   = num_modelparams['Gamma']
            
    kme_error = np.zeros([num_trials, T, max_idx['Zeta']-min_idx['Zeta'], max_idx['Gamma']-min_idx['Gamma']])
    
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
            if load_param['Zeta']:
                zeta = opt_param['Zeta'][0,s_idx]
            else:
                zeta = Zeta[zeta_idx]
            # compute weighting matrix
            A = u.compute_MTA_similarity(G_naive, T, 'stb', zeta, task_var)
            
            for j_g, gamma_idx in enumerate(range(min_idx['Gamma'], max_idx['Gamma'])):
                if load_param['Gamma']:
                    gamma = opt_param['Gamma'][0,s_idx]
                else:
                    gamma = Gamma[gamma_idx]
                W = u.compute_MTA_weight(A, task_var, gamma, T)
                
                kme_error[trial,:,j_z,j_g] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T, unbiased, replace)
                
                if load_param['Gamma']:
                    break
            if load_param['Zeta']:
                break

    # get optimal value and save its CV error
    _,_, end_t, end_g = np.shape(kme_error)
    if load_param['Zeta']:
        end_t = 1
    if load_param['Gamma']:
        end_g = 1
    mean_e = np.mean(kme_error[:,:, 0:end_t, 0:end_g], axis=(0,1))
    opt_idx_zeta, opt_idx_gamma = np.where(mean_e == np.min(mean_e))
    CV_error[s_idx] = np.min(mean_e)
    if not(load_param['Zeta']):
        lastbest_idx['Zeta']       = min_idx['Zeta'] + opt_idx_zeta[0]
        opt_param['Zeta'][s_idx]   = Zeta[lastbest_idx['Zeta']]
    if not(load_param['Gamma']):
        lastbest_idx['Gamma']     = min_idx['Gamma'] + opt_idx_gamma[0]
        opt_param['Gamma'][s_idx] = Gamma[lastbest_idx['Gamma']]
           
    # save the results after each setting
    np.savetxt(FN+'CV_error_MTA_stb.csv', CV_error, delimiter=',')
    io.savemat(FN+'opt_param_MTA_stb.mat', opt_param)
print('Done')