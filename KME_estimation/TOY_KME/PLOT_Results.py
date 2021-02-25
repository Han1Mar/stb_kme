"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
PLOT_Results.py:
    - plots the optimal parameter values, the cross-validation error, the 
      number of neighbors and also the final performance of each method
"""
import numpy as np
import matplotlib.pyplot as plt
import Plot_Globals as g
from scipy.io import loadmat

differentNrBags = 'differentNrBags'
differentBagsizes = 'differentBagsizes'
clustered = 'clustered'

### CHANGE IF NEEDED #########################################################
experiment = differentNrBags        # change setting
saving = True                       # save the plots
final  = False                      # plot for final or model opt.
neighbors = True                    # plot info about neighbors
print_legend = True                 # plot the legend (only visible in png)
##############################################################################

if experiment == differentNrBags:
    FN_mo  = '../Results/DifferentNrBags_KME/NaiveData/'
    FN_err = '../Results/DifferentNrBags_KME/FinalData/'
    xl = 'Nr. of Bags'
    setting_range = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300])
elif experiment == differentBagsizes:
    FN_mo  = '../Results/DifferentBagsizes_KME/NaiveData/'
    FN_err = '../Results/DifferentBagsizes_KME/FinalData/'
    xl = 'Bag Size'
    setting_range = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300])
elif experiment == clustered:
    FN_mo  = '../Results/Clustered_KME/NaiveData/'
    FN_err = '../Results/Clustered_KME/FinalData/'
    xl = 'Radius'
    setting_range = np.linspace(0,5,21)

err_fn = 'CV_error_'
mo_fn  = 'opt_param_'
kme_fn = 'KME_error_'
reg_fn = 'REG_error_'
nei_fn = 'Neighbors_'
e      = 'Error'
s      = 'std'

if not(final):
    fn = FN_mo + err_fn
    #err_naive = np.genfromtxt(fn+'naive.csv', delimiter=',')
    err_stb = np.genfromtxt(fn+'STB.csv', delimiter=',')
    err_stb_weight = np.genfromtxt(fn+'STB_weight.csv', delimiter=',')
    err_stb_theory = np.genfromtxt(fn+'STB_theory.csv', delimiter=',')
    err_mta_const  = np.genfromtxt(fn+'MTA_const.csv', delimiter=',')
    err_mta_stb    = np.genfromtxt(fn+'MTA_stb.csv', delimiter=',')
    
    fn = FN_mo+mo_fn
    
    op_stb = loadmat(fn+'STB.mat')
    op_stb_weight = loadmat(fn+'STB_weight.mat')
    op_stb_theory = loadmat(fn+'STB_theory.mat')
    op_mta_const  = loadmat(fn+'MTA_const.mat')
    op_mta_stb    = loadmat(fn+'MTA_stb.mat')
    
    # PLOT ERRORS
    plt.figure(figsize=g.fig11_wide)
    #plt.plot(setting_range, err_naive, lw=g.liwi, label=g.Labels['Naive'], c=g.Colors['Naive'])
    plt.plot(setting_range, err_stb, lw=g.liwi, label=g.Labels['STB'], c=g.Colors['STB'])
    plt.plot(setting_range, err_stb_weight, lw=g.liwi, label=g.Labels['STB_weight'], c=g.Colors['STB_weight'])
    plt.plot(setting_range, err_stb_theory, lw=g.liwi, label=g.Labels['STB_theory'], c=g.Colors['STB_theory'])
    plt.plot(setting_range, err_mta_const, lw=g.liwi, label=g.Labels['MTA_const'], c=g.Colors['MTA_const'])
    plt.plot(setting_range, err_mta_stb, lw=g.liwi, label=g.Labels['MTA_stb'], c=g.Colors['MTA_stb'])
    plt.legend(fontsize=g.fsl)
    plt.title('CV KME Estimation Error', fontsize=g.fst)
    plt.xlabel(xl, fontsize=g.fsl)
    plt.ylabel('KME Error', fontsize=g.fsl)
    if saving:
        plt.savefig(FN_mo+'CV_error.png')
    
    
    # PLOT OP STB
    plt.figure(figsize=g.fig11_wide)
    plt.plot(setting_range, op_stb['Zeta'].T, label='Zeta', lw=g.liwi, ls=g.Lstyle['Zeta'], c=g.Colors['STB'])
    plt.legend(fontsize=g.fsl)
    plt.xlabel(xl, fontsize=g.fsl)
    plt.ylabel('Value', fontsize=g.fsl)
    plt.title('Model Parameters: '+ g.Labels['STB'])
    if saving:
        plt.savefig(FN_mo+'opt_param_STB.png')
        
    # PLOT OP STB WEIGHT
    plt.figure(figsize=g.fig11_wide)
    plt.plot(setting_range, op_stb_weight['Gamma'].T, label='Gamma', lw=g.liwi, ls=g.Lstyle['Gamma'], c=g.Colors['STB_weight'])
    plt.plot(setting_range, op_stb_weight['Zeta'].T, label='Zeta', lw=g.liwi, ls=g.Lstyle['Zeta'], c=g.Colors['STB_weight'])
    plt.legend(fontsize=g.fsl)
    plt.xlabel(xl, fontsize=g.fsl)
    plt.ylabel('Value', fontsize=g.fsl)
    plt.title('Model Parameters: '+ g.Labels['STB_weight'])
    if saving:
        plt.savefig(FN_mo+'opt_param_STB_weight.png')
        
    # PLOT OP STB THEORY
    plt.figure(figsize=g.fig11_wide)
    plt.plot(setting_range, op_stb_theory['Gamma'].T, label='Gamma', lw=g.liwi, ls=g.Lstyle['Gamma'], c=g.Colors['STB_theory'])
    plt.plot(setting_range, op_stb_theory['Zeta'].T, label='Zeta', lw=g.liwi, ls=g.Lstyle['Zeta'], c=g.Colors['STB_theory'])
    plt.legend(fontsize=g.fsl)
    plt.xlabel(xl, fontsize=g.fsl)
    plt.ylabel('Value', fontsize=g.fsl)
    plt.title('Model Parameters: '+ g.Labels['STB_theory'])
    if saving:
        plt.savefig(FN_mo+'opt_param_STB_theory.png')
    
    # PLOT OP MTA CONST
    plt.figure(figsize=g.fig11_wide)
    plt.plot(setting_range, op_mta_const['Gamma'].T, label='Gamma', lw=g.liwi, ls=g.Lstyle['Gamma'], c=g.Colors['MTA_const'])
    plt.legend(fontsize=g.fsl)
    plt.xlabel(xl, fontsize=g.fsl)
    plt.ylabel('Value', fontsize=g.fsl)
    plt.title('Model Parameters: '+ g.Labels['MTA_const'])
    if saving:
        plt.savefig(FN_mo+'opt_param_MTA_const.png')
        
    # PLOT OP MTA STB
    plt.figure(figsize=g.fig11_wide)
    plt.plot(setting_range, op_mta_stb['Gamma'].T, label='Gamma', lw=g.liwi, ls=g.Lstyle['Gamma'], c=g.Colors['MTA_stb'])
    plt.plot(setting_range, op_mta_stb['Zeta'].T, label='Zeta', lw=g.liwi, ls=g.Lstyle['Zeta'], c=g.Colors['MTA_stb'])
    plt.legend(fontsize=g.fsl)
    plt.xlabel(xl, fontsize=g.fsl)
    plt.ylabel('Value', fontsize=g.fsl)
    plt.title('Model Parameters: '+ g.Labels['MTA_stb'])
    if saving:
        plt.savefig(FN_mo+'opt_param_MTA_stb.png')
             
    if neighbors:
        fn = FN_err+nei_fn
        n_s  = np.genfromtxt(fn+'STB.csv', delimiter=',')
        n_sw  = np.genfromtxt(fn+'STB_weight.csv', delimiter=',')
        n_st  = np.genfromtxt(fn+'STB_theory.csv', delimiter=',')
        n_ms = np.genfromtxt(fn+'MTA_stb.csv', delimiter=',')
        plt.figure(figsize=g.fig11_wide)
        plt.plot(setting_range, n_s, lw=g.liwi, label=g.Labels['STB'], c=g.Colors['STB'])
        plt.plot(setting_range, n_st, lw=g.liwi, label=g.Labels['STB_theory'], c=g.Colors['STB_theory'])
        plt.plot(setting_range, n_sw, lw=g.liwi, label=g.Labels['STB_weight'], c=g.Colors['STB_weight'])
        plt.plot(setting_range, n_ms, lw=g.liwi, label=g.Labels['MTA_stb'], c=g.Colors['MTA_stb'])
        plt.legend(fontsize=g.fsl)
        plt.title('Average Nr. of Neighbors', fontsize=g.fst)
        plt.xlabel(xl, fontsize=g.fsl)
        plt.ylabel('Nr. of Neighbors', fontsize=g.fsl)
        if saving:
            plt.savefig(FN_err+'Neighbors.png')

else:
    fn = FN_err+kme_fn
    e_n   = loadmat(fn+'naive.mat')
    e_s   = loadmat(fn+'STB.mat')
    e_sw  = loadmat(fn+'STB_weight.mat')
    e_st  = loadmat(fn+'STB_theory.mat')
    e_mc  = loadmat(fn+'MTA_const.mat')
    e_ms  = loadmat(fn+'MTA_stb.mat')
    e_ss  = loadmat(fn+'RKMSE.mat')
    
    p_n  = (e_n[e].T - e_n[e].T)/e_n[e].T
    p_s  = (e_n[e].T - e_s[e].T)/e_n[e].T
    p_sw = (e_n[e].T - e_sw[e].T)/e_n[e].T
    p_st = (e_n[e].T - e_st[e].T)/e_n[e].T
    p_mc = (e_n[e].T - e_mc[e].T)/e_n[e].T
    p_ms = (e_n[e].T - e_ms[e].T)/e_n[e].T
    p_ss = (e_n[e].T - e_ss[e].T)/e_n[e].T
    # KME ERROR
    plt.figure(figsize=g.fig11_wide)
    plt.plot(setting_range, p_n, lw=g.liwi, label=g.Labels['Naive'], c=g.Colors['Naive'], ls=g.Lstyle['Naive'], marker=g.Marker['Naive'])
    plt.plot(setting_range, p_ss, lw=g.liwi, label=g.Labels['RKMSE'], c=g.Colors['RKMSE'], ls=g.Lstyle['RKMSE'], marker=g.Marker['RKMSE'])
    plt.plot(setting_range, p_s, lw=g.liwi, label=g.Labels['STB'], c=g.Colors['STB'], ls=g.Lstyle['STB'], marker=g.Marker['STB'])
    plt.plot(setting_range, p_mc, lw=g.liwi, label=g.Labels['MTA_const'], c=g.Colors['MTA_const'], ls=g.Lstyle['MTA_const'], marker=g.Marker['MTA_const'])
    plt.plot(setting_range, p_st, lw=g.liwi, label=g.Labels['STB_theory'], c=g.Colors['STB_theory'], ls=g.Lstyle['STB_theory'], marker=g.Marker['STB_theory'])
    plt.plot(setting_range, p_ms, lw=g.liwi, label=g.Labels['MTA_stb'], c=g.Colors['MTA_stb'], ls=g.Lstyle['MTA_stb'], marker=g.Marker['MTA_stb'])
    plt.plot(setting_range, p_sw, lw=g.liwi, label=g.Labels['STB_weight'], c=g.Colors['STB_weight'], ls=g.Lstyle['STB_weight'], marker=g.Marker['STB_weight'])
    if print_legend:
        lgd = plt.legend(fontsize=g.fsl, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.37))
    plt.xlabel(xl, fontsize=g.fsl)
    plt.ylabel('Decrease in KME Error', fontsize=g.fsl)
    ylims = plt.ylim()
    plt.ylim([-0.005, ylims[1]])
    if experiment == clustered:
        plt.xlim([np.min(setting_range)-0.05, np.max(setting_range)+0.05])
    else:
        plt.xlim([np.min(setting_range)-2.5, np.max(setting_range)+2.5])
    if saving and print_legend:
        plt.savefig(FN_err+'KME_error_percent_paper.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    elif saving:
        plt.savefig(FN_err+'KME_error_percent_paper.png', bbox_inches='tight')