"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import Plot_Globals as g
from scipy.io import loadmat


### CHANGE IF NEEDED ###
saving = True
#######################

FN_err = '../Results/imbalanced_kme/FinalData/'

xl = 'Bag'
n = [10,10,20,20,30,30,40,40,50,50,60,60,70,70,80,80,90,90,100,100,125,150,200,250,300]
n = np.sort(n*2)
N = np.array(n)
B = range(len(N))

kme_fn = 'KME_error_'
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

num_N = len(N)
avg_n   = np.zeros(num_N)
avg_s   = np.zeros(num_N)
avg_sw  = np.zeros(num_N)
avg_st  = np.zeros(num_N)
avg_mc  = np.zeros(num_N)
avg_ms  = np.zeros(num_N)
avg_ss  = np.zeros(num_N)

for i in range(num_N):
    idx = np.where(N == N[i])[0]
    avg_n[i]  = np.mean(e_n[e][0,idx])
    avg_s[i]  = np.mean(e_s[e][0,idx])
    avg_sw[i] = np.mean(e_sw[e][0,idx])
    avg_st[i] = np.mean(e_st[e][0,idx])
    avg_mc[i] = np.mean(e_mc[e][0,idx])
    avg_ms[i] = np.mean(e_ms[e][0,idx])
    avg_ss[i] = np.mean(e_ss[e][0,idx])
    
# KME ERROR
p_n  = (avg_n - avg_n)/avg_n
p_s  = (avg_n - avg_s)/avg_n
p_sw = (avg_n - avg_sw)/avg_n
p_st = (avg_n - avg_st)/avg_n
p_mc = (avg_n - avg_mc)/avg_n
p_ms = (avg_n - avg_ms)/avg_n
p_ss = (avg_n - avg_ss)/avg_n
B = np.linspace(1,50,50)
fig,ax = plt.subplots(figsize=g.fig11_wide)
ax.step(B, p_n, lw=g.liwi, label=g.Labels['Naive'], c=g.Colors['Naive'], where='mid', ls=g.Lstyle['Naive'], marker=g.Marker['Naive'])
ax.step(B, p_ss, lw=g.liwi, label=g.Labels['RKMSE'], c=g.Colors['RKMSE'], where='mid', ls=g.Lstyle['RKMSE'], marker=g.Marker['RKMSE'])
ax.step(B, p_s, lw=g.liwi, label=g.Labels['STB'], c=g.Colors['STB'], where='mid', ls=g.Lstyle['STB'], marker=g.Marker['STB'])
ax.step(B, p_st, lw=g.liwi, label=g.Labels['STB_theory'], c=g.Colors['STB_theory'], where='mid', ls=g.Lstyle['STB_theory'], marker=g.Marker['STB_theory'])
ax.step(B, p_sw, lw=g.liwi, label=g.Labels['STB_weight'], c=g.Colors['STB_weight'], where='mid', ls=g.Lstyle['STB_weight'], marker=g.Marker['STB_weight'])
ax.step(B, p_mc, lw=g.liwi, label=g.Labels['MTA_const'], c=g.Colors['MTA_const'], where='mid', ls=g.Lstyle['MTA_const'], marker=g.Marker['MTA_const'])
ax.step(B, p_ms, lw=g.liwi, label=g.Labels['MTA_stb'], c=g.Colors['MTA_stb'], where='mid', ls=g.Lstyle['MTA_stb'], marker=g.Marker['MTA_stb'])
ax.step(B, p_n, lw=g.liwi, c=g.Colors['Naive'], where='mid', ls=g.Lstyle['Naive'], marker=g.Marker['Naive'])
ax.step(B, p_ss, lw=g.liwi, c=g.Colors['RKMSE'], where='mid', ls=g.Lstyle['RKMSE'], marker=g.Marker['RKMSE'])
ax.set_xlabel(xl, fontsize=g.fsl)
ax.set_ylabel('Decrease in KME Error', fontsize=g.fsl)
ylims = ax.get_ylim()
ax.set_ylim([-0.005, ylims[1]])
ax.set_xlim([0.5,50.5])
ax2 = ax.twinx()
ax2.bar(B, N, fill=True, color='lightgrey')
ax2.set_ylabel('Bagsize', color='grey',fontsize=g.fsl)
ax2.tick_params(axis='y', colors='grey')
ax2.yaxis.label.set_color('grey')
ylims = ax2.get_ylim()
ax2.set_ylim([0,ylims[1]])
ax.set_zorder(ax2.get_zorder()+1)
ax.patch.set_visible(False)
if saving:
    fig.savefig(FN_err+'KME_error_averaged_paper.png', bbox_inches='tight')