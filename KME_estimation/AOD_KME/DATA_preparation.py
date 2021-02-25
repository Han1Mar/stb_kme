"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
DATA_perparation.py:
    Prepares the original MISR1.mat data set
    - load the data
    - standardizes each feature such that it has zero mean and unit standard
      deviation
    - splits the features (X_A: features 0 to 11; X_B: features 12 to 15) 
      because some of the features (X_B) are constant per bag and are not 
      needed for improving the KME estimation
    - X_A: (800,100,12) array, that holds the relevant, standardized MISR1 
           data
"""
import sys
sys.path.append('../')
import AOD_utiels as au
from scipy.io import savemat

### update as desired ###
FN_Datapath = '../Results/aod_kme/Data/' 
FN_MISR = FN_Datapath+'MISR1.mat'


# prepare the data
X, y, num_obs = au.load_MISR(FN_MISR)
X_A, X_B = au.standardize_split(X)
savemat(FN_Datapath+'X_A.mat', mdict={'X_A':X_A, 'y':y, 'num_obs':num_obs})
savemat(FN_Datapath+'X_B.mat', mdict={'X_B':X_B, 'y':y, 'num_obs':num_obs})