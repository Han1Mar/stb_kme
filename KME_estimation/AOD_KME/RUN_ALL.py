"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
RUN_ALL.py:
    - runs every method (model optimization and final estimation) on the AOD
      MISR1 data set.
    - please run PLOT_results.py afterwards to visualize the results
"""
import subprocess


Methods = ['naive', 'RKMSE', 'STB', 'STB_weight', 'STB_theory', 'MTA_const', 'MTA_stb']

p = subprocess.Popen('python DATA_preparation.py', shell=True)
p.wait()
p = subprocess.Popen('python KERNEL_preparation.py', shell=True)
p.wait()

### FINAL ESTIMATION
for m in Methods:
    print('FINAL: '+m)
    p = subprocess.Popen('python FINAL_'+m+'.py', shell=True)
    p.wait()