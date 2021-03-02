"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
PLOT_Results.py:
    - plots the optimal parameter values, the cross-validation error, the 
      number of neighbors and also the final performance of each method
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import Plot_Globals as g
from scipy.io import loadmat

differentNrBags = 'differentNrBags'
differentBagsizes = 'differentBagsizes'
clustered = 'clustered'

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True

### CHANGE IF NEEDED #########################################################
experiment    = clustered     # change setting
saving        = True               # save the plots
final         = True                # plot for final or model opt.
plot_decrease = True               # plot decrease over naive in percent or raw errors
plot_logScale = True                # plot raw errors but on log scale
print_legend  = False               # plot the legend (only visible in png)
##############################################################################

if experiment == differentNrBags:
    FN_mo  = '../Results/differentNrBags_kme/NaiveData/'
    FN_err = '../Results/differentNrBags_kme/FinalData/'
    xl = 'Nr. of Bags'
    filename = 'nrbags_KME_error'
    setting_range = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300])
elif experiment == differentBagsizes:
    FN_mo  = '../Results/differentBagsizes_kme/NaiveData/'
    FN_err = '../Results/differentBagsizes_kme/FinalData/'
    filename = 'bagsizes_KME_error'
    xl = 'Bag Size'
    setting_range = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300])
elif experiment == clustered:
    FN_mo  = '../Results/clustered_kme/NaiveData/'
    FN_err = '../Results/clustered_kme/FinalData/'
    xl = 'Radius'
    filename = 'clustered_KME_error'
    setting_range = np.linspace(0,5,21)

err_fn = 'CV_error_'
mo_fn  = 'opt_param_'
kme_fn = 'KME_error_'
reg_fn = 'REG_error_'
nei_fn = 'Neighbors_'
e      = 'Error'
s      = 'std'

fn = FN_err+kme_fn
e_n   = loadmat(fn+'naive.mat')
e_s   = loadmat(fn+'STB.mat')
e_sw  = loadmat(fn+'STB_weight.mat')
e_st  = loadmat(fn+'STB_theory.mat')
e_mc  = loadmat(fn+'MTA_const.mat')
e_ms  = loadmat(fn+'MTA_stb.mat')
e_ss  = loadmat(fn+'RKMSE.mat')

if plot_decrease:
    p_n  = (e_n[e].T - e_n[e].T)/e_n[e].T * 100.
    p_s  = (e_n[e].T - e_s[e].T)/e_n[e].T * 100.
    p_sw = (e_n[e].T - e_sw[e].T)/e_n[e].T * 100.
    p_st = (e_n[e].T - e_st[e].T)/e_n[e].T * 100.
    p_mc = (e_n[e].T - e_mc[e].T)/e_n[e].T * 100.
    p_ms = (e_n[e].T - e_ms[e].T)/e_n[e].T * 100.
    p_ss = (e_n[e].T - e_ss[e].T)/e_n[e].T * 100.
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
        plt.savefig(FN_err+filename+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig(FN_err+filename+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight', format='pdf')
    elif saving:
        plt.savefig(FN_err+filename+'.png', bbox_inches='tight')
        plt.savefig(FN_err+filename+'.pdf', bbox_inches='tight', format='pdf')
if plot_logScale:
    plt.rcParams['text.latex.preamble'] = r'\newcommand{\mathdefault}[1][]{}'
    filename = filename + '_logscale'
    # KME ERROR
    plt.figure(figsize=g.fig11_wide)
    plt.plot(setting_range, e_n[e].T, lw=g.liwi, label=g.Labels['Naive'], c=g.Colors['Naive'], ls=g.Lstyle['Naive'], marker=g.Marker['Naive'])
    plt.plot(setting_range, e_ss[e].T, lw=g.liwi, label=g.Labels['RKMSE'], c=g.Colors['RKMSE'], ls=g.Lstyle['RKMSE'], marker=g.Marker['RKMSE'])
    plt.plot(setting_range, e_s[e].T, lw=g.liwi, label=g.Labels['STB'], c=g.Colors['STB'], ls=g.Lstyle['STB'], marker=g.Marker['STB'])
    plt.plot(setting_range, e_mc[e].T, lw=g.liwi, label=g.Labels['MTA_const'], c=g.Colors['MTA_const'], ls=g.Lstyle['MTA_const'], marker=g.Marker['MTA_const'])
    plt.plot(setting_range, e_st[e].T, lw=g.liwi, label=g.Labels['STB_theory'], c=g.Colors['STB_theory'], ls=g.Lstyle['STB_theory'], marker=g.Marker['STB_theory'])
    plt.plot(setting_range, e_ms[e].T, lw=g.liwi, label=g.Labels['MTA_stb'], c=g.Colors['MTA_stb'], ls=g.Lstyle['MTA_stb'], marker=g.Marker['MTA_stb'])
    plt.plot(setting_range, e_sw[e].T, lw=g.liwi, label=g.Labels['STB_weight'], c=g.Colors['STB_weight'], ls=g.Lstyle['STB_weight'], marker=g.Marker['STB_weight'])
    if print_legend:
        lgd = plt.legend(fontsize=g.fsl, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.37))
        fig = lgd.figure
        fig.canvas.draw()
        bbox  = lgd.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array([-5,-5,5,5])))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(FN_err+'legend.pdf', dpi="figure", format='pdf', bbox_inches=bbox) 
    plt.xlabel(xl, fontsize=g.fsl)
    plt.ylabel('KME Error (log scale)', fontsize=g.fsl)
    plt.yscale('log')
    if experiment==differentBagsizes:
        locs = [0.001, 0.002, 0.004, 0.006, 0.008 ,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    else:
        locs = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012]
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(ticker.FixedLocator(locs))
    plt.grid(axis='y', linestyle=':')
    if experiment == clustered:
        plt.xlim([np.min(setting_range)-0.05, np.max(setting_range)+0.05])
    else:
        plt.xlim([np.min(setting_range)-2.5, np.max(setting_range)+2.5])
    if saving:
        plt.savefig(FN_err+filename+'.png', bbox_inches='tight')
        plt.savefig(FN_err+filename+'.pdf', bbox_inches='tight', format='pdf')