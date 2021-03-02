"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
"""
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing

def load_MISR(FN, data_name='MISR1'):
    """Loads MISR1.mat data set.

    Args:
      FN: String, path to MISR1.mat data set (including file-ending).
      data_name: String, key used for the MISR1 data set in the mat-dictionary

    Returns:
      X: (800,) list, of (100,16) arrays, holding the MISR1 data set
      y: (800,) list, of the AOD-labels of the corresponding bags
      num_obs: (800,) list, number of observations (here:100) of each bag
    """
    MISR = loadmat(FN)[data_name]
    ids = np.array(MISR[:,0], dtype=int)
    
    max_id = np.max(ids)
    min_id = np.min(ids)
    
    y = []
    X = []
    num_obs = []
    for i in range(min_id,max_id+1):
        n = np.sum(ids == i)
        if n != 0:
            num_obs.append(n)
            xs = MISR[np.where(ids == i)]
            X.append(xs[:,1:17])
            y.append(xs[0,-1])
            if n != np.sum(xs[:,-1] == y[-1]):
                print('Different labels in one bag!')
                break
    return X,y,num_obs

def standardize_split(X):
    """Standardizes the features (over all observations and bags) such that 
    they have zero mean and unit standard deviation. Also splits the available
    data into to parts X_A and X_B where X_A holds the relevant data and X_B
    the features that are constant per bag.

    Args:
      X: (800,) list, of (100,16) arrays, holding the MISR1 data set

    Returns:
      X_A: (800,100,12) array, subset of features of X that are relevant
      X_B: (800,100,4) array, subset of features of X that are constant per
           bag
    """
    # Only works for MISR1!
    X_scaled = np.array(X)
    T,N,D = np.shape(X_scaled)
    for d in range(D):
        x = X_scaled[:,:,d].flatten()
        x = preprocessing.scale(x)
        X_scaled[:,:,d] = np.reshape(x,(T,N))
    X_A = X_scaled[:,:,0:12]
    X_B = X_scaled[:,:,12::]
    return X_A, X_B

