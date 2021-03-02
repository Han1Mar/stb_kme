"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
DATA_perparation.py:
    Prepares the original WINE data set
    - loads the train          (84,641x42), 
                test           (24,184x42) and 
                validation set (12,091x42) of all 21 bags
      and creates a single data set
    - each feature is either 0 or 1 (but not boolean)
    - the features 'points' and 'price' (idx: [0,1]), which are part of the original data,
      are not used here
    - features [2:41] contain the actual features, i.e. descriptions of the wine
    - features [41:62] contain the country (used as bag information)
    - selects only the bags with more than 460 samples (resulting in 15 bags)
    - for bags with more than 460 samples, a random subsample is selected
    - X: (15,461,39) array, that holds the wine data (TxnxD)
"""
import numpy as np
import WINE_settings as s
from scipy.io import savemat

### update as desired ###
FN_Datapath = s.FN_Datapath
T = 21
D = 39
X_train = []
X_test  = []
X_val   = []
countries = np.genfromtxt(FN_Datapath+'validation.csv', delimiter=',', max_rows = 1, dtype='str')[41:62]
num_obs_all = np.zeros(T, dtype=int)

# load each bag individual
train = np.genfromtxt(FN_Datapath+'train.csv', delimiter=',', skip_header=1, dtype=int)
for country in range(T):
    obs_idx = np.where(train[:,41+country])[0]
    X_train.append(train[obs_idx, 2:41])
    num_obs_all[country] += len(X_train[country])
    
test = np.genfromtxt(FN_Datapath+'test.csv', delimiter=',', skip_header=1, dtype=int)
for country in range(T):
    obs_idx = np.where(test[:,41+country])[0]
    X_test.append(test[obs_idx, 2:41])
    num_obs_all[country] += len(X_test[country])

val = np.genfromtxt(FN_Datapath+'validation.csv', delimiter=',', skip_header=1, dtype=int)
for country in range(T):
    obs_idx = np.where(val[:,41+country])[0]
    X_val.append(val[obs_idx, 2:41])
    num_obs_all[country] += len(X_val[country])

X_all = []
for country in range(T):
    A = np.zeros([num_obs_all[country], D], dtype=int)
    len_train = len(X_train[country])
    len_test  = len(X_test[country])
    len_val   = len(X_val[country])
    A[0:len_train,:] = X_train[country]
    A[len_train:len_train+len_test] = X_test[country]
    A[len_train+len_test::] = X_val[country]
    X_all.append(A)

# Use only the countries with more than 400 observations   
# Select from larger bags only a subset of samples such that every ...
# ... bag has the same number of observations 
idx = np.where(num_obs_all >= 400)
min_num_obs = np.min(num_obs_all[idx])
num_obs = np.array([min_num_obs]*len(idx[0]))
X = []
for i in idx[0]:
    subsample_idx = np.random.choice(range(num_obs_all[i]), min_num_obs, replace=False)
    subsample = X_all[i][subsample_idx[0:min_num_obs],:]
    X.append(subsample)
    
# save as mat
savemat(FN_Datapath+'X.mat', mdict={'X':X, 'y':countries[idx], 'num_obs':num_obs})