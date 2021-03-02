"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
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
    p = subprocess.Popen('python FINAL_'+m+'.py', shell=False, stdout=subprocess.PIPE)
    p.wait()
    out = p.communicate()[0]
    print(out.decode('utf-8'))