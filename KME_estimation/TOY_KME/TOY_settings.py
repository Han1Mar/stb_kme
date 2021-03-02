"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
TOY_settings.py:
    Sets the settings of the experiment.
"""

import numpy as np

# experiment names
differentNrBags = 'differentNrBags'
differentBagsizes = 'differentBagsizes'
clustered = 'clustered'

unbiased    = True          # unbiased estimation of the MMD^2
replace     = True          # replace negative values of MMD^2 with zero 
save_neighs = True          # save info about neighbors
kw1    = 2.25               # kernel width
num_trials_model = 100      # number of trials for training
num_trials_final = 200      # number of trials for testing

# values that shall be tested in the setting
# nr of bags or bag sizes:
setting_range = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300])
# tested radius of circle on which the tasks lie equally spaced on
setting_range_Clustered = np.linspace(0,5,21) 
num_centroids_Clustered = 5

NZ      = int(1000)                 # number of sampler (test data) per task
mu     = [0,0]                      # center of each task
S      = np.array([[1,0],[0,10]])   # covariance matrix which will be ...
                                    # ... randomly rotated

# where to save and load the data
FN_final_NrBags      = '../Results/differentNrBags_kme/FinalData/'
FN_model_NrBags      = '../Results/differentNrBags_kme/ModelData/'
FN_final_Bagsizes    = '../Results/differentBagsizes_kme/FinalData/'
FN_model_Bagsizes    = '../Results/differentBagsizes_kme/ModelData/'
FN_final_Clustered   = '../Results/clustered_kme/FinalData/'
FN_model_Clustered   = '../Results/clustered_kme/ModelData/'