"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
PLOT_Results.py:
    - prints a summary of the results to the console (not very pretty, though) 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.io import loadmat
import Plot_Globals as g

dnf_kernel = True
print_legend = False
saving = True
plot_decrease = True
plot_raw      = True
print_modelInfo = True

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True

FN1 = '../Results/wine_kme/'
FN_sizes = ['20_samples', '50_samples', '100_samples', '150_samples', '200_samples']
x        = [20, 50, 100, 150, 200]
if dnf_kernel:
    FN_kernel = '/DNF_kernel/'
    filename  = 'DNF_kernel_KME_error'
else:
    FN_kernel = '/MDNF_kernel/'
    filename  = 'MDNF_kernel_KME_error'
FN2 = 'TrialData/'

fn_err = 'KME_error_'
fn_nei = 'Neighbors_'
fn_opt = 'opt_param_'

methods = ['naive', 'RKMSE', 'STB', 'MTA_const', 'STB_theory', 'MTA_stb', 'STB_weight']

# get experiment information
KME_errors      = np.zeros([len(FN_sizes), len(methods)])
KME_decrease    = np.zeros([len(FN_sizes), len(methods)])
STD             = np.zeros([len(FN_sizes), len(methods)])
NEI             = np.zeros([len(FN_sizes), len(methods)])
OPT             = np.zeros([len(FN_sizes), len(methods), 2]) # 0:zeta, 1:gamma

for sidx, size in enumerate(FN_sizes):
    FN = FN1+size+FN_kernel+FN2
    for midx, method in enumerate(methods):
        err = np.genfromtxt(FN+fn_err+method+'.csv', delimiter=',')
        KME_errors[sidx, midx] = np.mean(err)
        # get decrease in error compared to naive
        KME_decrease[sidx, midx] = ((KME_errors[sidx, 0] - KME_errors[sidx, midx])/KME_errors[sidx, 0])*100.
        # standard deviation of error
        STD[sidx, midx] = np.std(err)
        # get neighbors for appropriate methods
        if ('STB' in method) or ('stb' in method):
            NEI[sidx, midx] = np.mean(np.genfromtxt(FN+fn_nei+method+'.csv', delimiter=','))
        else:
            NEI[sidx, midx] = float('NaN')
        # get optimal parameter values
        if method != 'naive' and method != 'RKMSE':
            op = loadmat(FN+fn_opt+method+'.mat')
            if 'Zeta' in op.keys():
                OPT[sidx, midx, 0] = np.mean(op['Zeta'][0,:])
            else:
                OPT[sidx, midx, 0] = float('NaN')
            if 'Gamma' in op.keys():
                OPT[sidx, midx, 1] = np.mean(op['Gamma'][0,:])
            else:
                OPT[sidx, midx, 1] = float('NaN') 
        else:
            OPT[sidx, midx, :] = float('NaN')

# plot the model information to the console
if print_modelInfo:
    if dnf_kernel:
        print('DNF KERNEL ')
    else:
        print('MDNF KERNEL ')
    for midx, method in enumerate(methods): 
        print('----------------------------------------------------------------------------------------------')
        print(method)
        print('Bag Size | KME Error \t | Std. \t\t | Decrease  | Neighbors | Zeta \t | Gamma')
        for sidx, size in enumerate(FN_sizes):
            print(str(x[sidx])+'\t\t | %.7f' % KME_errors[sidx, midx]+'\t | %.7f' % STD[sidx, midx]+\
                  '\t |  %.3f' % KME_decrease[sidx, midx] +'\t |  %4.2f' % NEI[sidx, midx]+\
                  '\t |  %5.3f' % OPT[sidx, midx,0]+'\t |  %.3f' % OPT[sidx, midx,1])

# plotting figures
if plot_decrease:        
    # plot decrease in percent
    plt.figure(figsize=g.fig11_wide)
    for midx, method in enumerate(methods):
        plt.plot(x, KME_decrease[:, midx], lw=g.liwi, label=g.Labels[method], c=g.Colors[method], ls=g.Lstyle[method], marker=g.Marker[method])
    if print_legend:
        lgd = plt.legend(fontsize=g.fsl, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.37))
    plt.xlim([18.5,201.5])
    plt.xticks(x)
    plt.xlabel('Bag Size', fontsize=g.fsl)
    plt.ylabel('Decrease in KME Error', fontsize=g.fsl) 
    if saving and print_legend:
        plt.savefig(FN1+filename+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig(FN1+filename+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight', format='pdf')
    elif saving:
        plt.savefig(FN1+filename+'.png', bbox_inches='tight')
        plt.savefig(FN1+filename+'.pdf', bbox_inches='tight', format='pdf') 

if plot_raw:
    plt.rcParams['text.latex.preamble'] = r'\newcommand{\mathdefault}[1][]{}'
    # plot raw errors with log scale
    plt.figure(figsize=g.fig11_wide)
    for midx, method in enumerate(methods):
        plt.plot(x, KME_errors[:, midx], lw=g.liwi, label=g.Labels[method], c=g.Colors[method], ls=g.Lstyle[method], marker=g.Marker[method])
    if print_legend:
        lgd = plt.legend(fontsize=g.fsl, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.37))
    plt.xlim([18.5,201.5])
    plt.xticks(x)
    plt.xlabel('Bag Size', fontsize=g.fsl)
    plt.ylabel('KME Error (log scale)', fontsize=g.fsl)
    plt.yscale('log')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    if dnf_kernel:
        locs = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]
        formatter.set_powerlimits((-1,1))
    else:
        locs = [0.0015, 0.01, 0.025, 0.05, 0.1, 0.2]
        formatter.set_powerlimits((-1,1))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(ticker.FixedLocator(locs))
    plt.grid(axis='y', linestyle=':')
    if saving and print_legend:
        plt.savefig(FN1+filename+'_logscale.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig(FN1+filename+'_logscale.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight', format='pdf')
    elif saving:
        plt.savefig(FN1+filename+'_logscale.png', bbox_inches='tight')
        plt.savefig(FN1+filename+'_logscale.pdf', bbox_inches='tight', format='pdf')    