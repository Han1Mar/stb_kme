"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
FINAL_RKMSE.py:
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
num_trials  = s.num_trials_final # number of trials

if experiment == differentNrBags:
    n = 50
    FN_save = s.FN_final_NrBags
    setting_range = s.setting_range
elif experiment == differentBagsizes:
    T = 50
    FN_save = s.FN_final_Bagsizes
    setting_range = s.setting_range
elif experiment == clustered:
    T = 50
    N = [50]*T
    FN_save = s.FN_final_Clustered
    setting_range = s.setting_range_Clustered
    
num_settings = len(setting_range)

KME_error  = {'Error': np.zeros(num_settings), 'std': np.zeros(num_settings)}

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
    
    for trial in range(num_trials):
        # data folder: FN + setting number + trial number
        curr_FN = FN_save+str(s_idx)+'/'+str(trial)+'/'
        sum_K       = np.genfromtxt(curr_FN+'sum_K.csv', delimiter=',')
        sum_no_diag = np.genfromtxt(curr_FN+'sum_no_diag.csv', delimiter=',')
        num_obs     = np.genfromtxt(curr_FN+'num_obs.csv', delimiter=',')
        sum_no_diag_Z = np.genfromtxt(curr_FN+'sum_no_diag_Z.csv', delimiter=',')
        sum_XZ        = np.genfromtxt(curr_FN+'sum_XZ.csv', delimiter=',')
        num_obs_Z     = np.genfromtxt(curr_FN+'num_obs_Z.csv', delimiter=',')

        # compute weighting matrix
        W  = u.compute_RKMSE_weight(sum_K, sum_no_diag, num_obs)
        
        kme_e[trial,:] = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T, unbiased, replace)
            
    KME_error['Error'][s_idx] = np.mean(kme_e)
    KME_error['std'][s_idx]   = np.std(kme_e)
    
    # save the results after each setting
    io.savemat(FN_save+'KME_error_RKMSE.mat', KME_error)
print('Done')