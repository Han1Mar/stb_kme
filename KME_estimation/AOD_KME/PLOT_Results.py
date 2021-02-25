"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
PLOT_Results.py:
    - prints a summary of the results to the console (not very pretty, though) 
"""
import numpy as np
from scipy.io import loadmat

FN = '../Results/aod_kme/TrialData/'

fn_err = 'KME_error_'
fn_nei = 'Neighbors_'

methods = ['Naive', 'STB', 'STB_weight', 'STB_theory','MTA_const', 'MTA_stb', 'RKMSE']

avg_n = np.mean(np.genfromtxt(FN+fn_err+'naive.csv', delimiter=','))


print('AVERAGE KME ERROR: ######')
for m in methods:
    err = np.genfromtxt(FN+fn_err+m+'.csv', delimiter=',') # shape: trial, T
    print(str(np.mean(err)) + '\t: \t'+m)
   
print('STANDARD DEVIATION: #####')     
for m in methods:
    err = np.genfromtxt(FN+fn_err+m+'.csv', delimiter=',') # shape: trial, T
    print(str(np.std(err)) + '\t: \t'+m)
    
print('AVERAGE NR. NEIGHBORS: ##')    
for m in methods:
    if m == 'STB' or m == 'STB_weight' or m == 'STB_theory' or m == 'MTA_stb':
        nei = np.genfromtxt(FN+fn_nei+m+'.csv', delimiter=',') # shape: T
        print(str(np.mean(nei)) + '\t: \t'+m)
        
print('AVERAGE OPTIMAL PARAMETERS:')
for m in methods:
    if m == 'STB' or m == 'STB_weight' or m == 'STB_theory' or m == 'MTA_const' or m == 'MTA_stb':
        op = loadmat(FN+'opt_param_'+m+'.mat')
        if m == 'STB':
            print(m+'\t\t\t Zeta: '+str(np.mean(op['Zeta'][0,:])))
        if m == 'STB_weight':
            print(m+'\t Zeta: '+str(np.mean(op['Zeta'][0,:]))+'\t Gamma: '+str(np.mean(op['Gamma'][0,:])))
        if m == 'STB_theory':
            print(m+'\t Zeta: '+str(np.mean(op['Zeta'][0,:]))+'\t Gamma: '+str(np.mean(op['Gamma'][0,:])))
        if m == 'MTA_const':
            print(m+'\t\t\t\t Gamma: '+str(np.mean(op['Gamma'][0,:])))
        if m == 'MTA_stb':
            print(m+'\t\t Zeta: '+str(np.mean(op['Zeta'][0,:]))+'\t Gamma: '+str(np.mean(op['Gamma'][0,:])))
            
print('DECREASE IN KME ERROR OVER NAIVE IN PERCENT: ######')
for m in methods:
    err = np.genfromtxt(FN+fn_err+m+'.csv', delimiter=',') # shape: trial, T
    p = (avg_n - np.mean(err))/avg_n
    print(str(p*100) + '\t: \t'+m)
    
    