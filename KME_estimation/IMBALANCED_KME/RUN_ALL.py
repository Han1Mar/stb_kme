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
      synthetic, imbalanced data sets
    - please run PLOT_results.py afterwards to visualize the results
"""
import subprocess


Methods = ['naive', 'STB', 'STB_weight', 'STB_theory', 'MTA_const', 'MTA_stb']
## MODEL OPTIMIZATION 
for m in Methods:
    print('MODEL: '+m)
    p = subprocess.Popen('python MODEL_'+m+'.py', shell=True)
    p.wait()
 
Methods = ['naive', 'RKMSE', 'STB', 'STB_weight', 'STB_theory', 'MTA_const', 'MTA_stb']       
### FINAL ESTIMATION
for m in Methods:
    print('FINAL: '+m)
    p = subprocess.Popen('python FINAL_'+m+'.py', shell=True)
    p.wait()