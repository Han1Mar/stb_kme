"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
IMBALANCED_settings.py:
    Sets the settings of the experiment.
"""

import numpy as np

unbiased    = True          # unbiased estimation of the MMD^2
replace     = True          # replace negative values of MMD^2 with zero 
save_neighs = True          # save info about found neighbors
kw1    = 2.25               # kernel width
num_trials_final = 200      # number of trials for testing
num_trials_model = 100      # number of trials for training

T  = 50             # number of tasks (bags)
n  = [10,10,20,20,30,30,40,40,50,50,60,60,70,70,80,80,90,90,100,100,125,150,\
      200,250,300]
n  = np.sort(n*2)   
N  = np.array(n)    # number of samples (train data) per task
NZ = int(1000)      # number of sampler (test data) per task

mu     = [0,0]                      # center of each task
S      = np.array([[1,0],[0,10]])   # covariance matrix which will be ...
                                    # ... randomly rotated

# where to save the data
FN_model = '../Results/imbalanced_kme/ModelData/'
FN_final = '../Results/imbalanced_kme/FinalData/'