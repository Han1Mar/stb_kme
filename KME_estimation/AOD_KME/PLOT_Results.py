"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
PLOT_Results.py:
    - prints a summary of the results to the stdout
"""
import numpy as np
from scipy.io import loadmat

FN = '../Results/aod_kme/TrialData/'

fn_err = 'KME_error_'
fn_nei = 'Neighbors_'
fn_opt = 'opt_param_'

methods = ['naive', 'RKMSE', 'STB', 'MTA_const', 'STB_theory', 'MTA_stb', 'STB_weight']

# get experiment information
KME_errors      = np.zeros(len(methods))
KME_decrease    = np.zeros(len(methods))
STD             = np.zeros(len(methods))
NEI             = np.zeros(len(methods))
OPT             = np.zeros([len(methods), 2]) # 0:zeta, 1:gamma

for midx, method in enumerate(methods):
    err = np.genfromtxt(FN+fn_err+method+'.csv', delimiter=',')
    KME_errors[midx] = np.mean(err)
    # get decrease in error compared to naive
    KME_decrease[midx] = ((KME_errors[0] - KME_errors[midx])/KME_errors[0])*100.
    # standard deviation of error
    STD[midx] = np.std(err)
    # get neighbors for appropriate methods
    if ('STB' in method) or ('stb' in method):
        NEI[midx] = np.mean(np.genfromtxt(FN+fn_nei+method+'.csv', delimiter=','))
    else:
        NEI[midx] = float('NaN')
    # get optimal parameter values
    if method != 'naive' and method != 'RKMSE':
        op = loadmat(FN+fn_opt+method+'.mat')
        if 'Zeta' in op.keys():
            OPT[midx, 0] = np.mean(op['Zeta'][0,:])
        else:
            OPT[midx, 0] = float('NaN')
        if 'Gamma' in op.keys():
            OPT[midx, 1] = np.mean(op['Gamma'][0,:])
        else:
            OPT[midx, 1] = float('NaN') 
    else:
        OPT[midx, :] = float('NaN')

# plot the model information to the console
white = '             '
print(white.replace(' '*len('Method'), 'Method', 1)+'| KME Error \t | Std. \t\t | Decrease  | Neighbors | Zeta \t | Gamma')
print('-------------------------------------------------------------------------------------------')
for midx, method in enumerate(methods): 
    white = '             '
    print(white.replace(' '*len(method), method, 1)+'| %.7f' % KME_errors[midx]+'\t | %.7f' % STD[midx]+\
          '\t |  %.3f' % KME_decrease[midx] +'\t |  %4.2f' % NEI[midx]+\
          '\t |  %5.3f' % OPT[midx,0]+'\t |  %.3f' % OPT[midx,1])


