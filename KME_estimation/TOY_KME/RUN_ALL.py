"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
RUN_ALL.py:
    - runs every method (model optimization and final estimation) on
      synthetic data sets for different settings: 
      {clustered, differentBagsizes, differentNrBags}
    - please run PLOT_results.py afterwards to visualize the results
"""
import sys
import subprocess

differentNrBags = 'differentNrBags'
differentBagsizes = 'differentBagsizes'
clustered = 'clustered'

#assert(len(sys.argv) == 2)
#experiment = sys.argv[1]
#assert(experiment==differentNrBags or experiment==differentBagsizes or experiment==clustered)
experiment = clustered

## MODEL OPTIMIZATION 
Methods = ['naive', 'STB', 'STB_weight', 'STB_theory', 'MTA_const', 'MTA_stb']
for m in Methods:
    print('MODEL: '+m)
    if experiment==clustered and m=='naive':
        p = subprocess.Popen('python CLUSTERED_MODEL_'+m+'.py '+experiment, shell=True)
    else:
        p = subprocess.Popen('python MODEL_'+m+'.py '+experiment, shell=True)
    p.wait()
        
### FINAL ESTIMATION
Methods = ['naive', 'RKMSE', 'STB', 'STB_weight', 'STB_theory', 'MTA_const', 'MTA_stb']
for m in Methods:
    print('FINAL: '+m)
    if experiment==clustered and m=='naive':
        p = subprocess.Popen('python CLUSTERED_FINAL_'+m+'.py '+experiment, shell=True)
    else:
        p = subprocess.Popen('python FINAL_'+m+'.py '+experiment, shell=True)
    p.wait()