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
from scipy.spatial import distance_matrix

### KERNEL FUNCTIONS #########################################################
def Gaussiankernel(X, Y, sigma):
    """Compute the Gaussian kernel for the given data.

    Args:
      X: (N,D) array, N observations of D dimensions
      Y: (M,D) array, M observations of D dimensions
      sigma: scalar, kernel width

    Returns:
      K: (N,M) array, kernel matrix where (K)_n,m = exp(||x_n - y_m||^2 / (-2*sigma^2))
    """
    K = np.exp(distance_matrix(X,Y)**2 / ((-2.)*sigma**2))
    return K

def Linearkernel(X,Y,_):
    """Compute the linear kernel for the given data.

    Args:
      X: (N,D) array, N observations of D dimensions
      Y: (M,D) array, M observations of D dimensions

    Returns:
      K: (N,M) array, kernel matrix where (K)_n,m = <x_n, y_m>
    """
    K = np.dot(X,Y.T)
    return K

def DNFkernel(X,Y,normalize,D):
    """Compute the disjunctive normal form kernel for the given data. Computes
    the number of bits (True, False) that are the same.

    Args:
      X: (N,D) array, N observations of D dimensions ([0,1] but not bool)
      Y: (M,D) array, M observations of D dimensions ([0,1] but not bool)
      normalize: bool, normalize kernel so that it is between 0 and 1
      D: int, number of dimensions; used to normalize the kernel 

    Returns:
      K: (N,M) array, kernel matrix where (K)_n,m = 
         -1 + 2**((<x_n, y_m> + <not(x_n),not(y_m)>)/D)
         if normalize=False, the exponent is not divided by D
    """
    not_X = np.ones(D, dtype=int) - X
    not_Y = np.ones(D, dtype=int) - Y
    K = np.dot(X,Y.T) + np.dot(not_X, not_Y.T)
    if normalize:
        K = K / float(D)
    K = -1. + 2**(K)
    return K

def MDNFkernel(X,Y,normalize,D):
    """Compute the monotone disjunctive normal form kernel for the given data. 
    Computes the number of Trues that are the same.

    Args:
      X: (N,D) array, N observations of D dimensions ([0,1] but not bool)
      Y: (M,D) array, M observations of D dimensions ([0,1] but not bool)
      normalize: bool, normalize kernel so that it is between 0 and 1, note
                 that the kernel is only 1 iff all bits are 1 and equal
      D: int, number of dimensions; used to normalize the kernel 

    Returns:
      K: (N,M) array, kernel matrix where (K)_n,m = 
         -1 + 2**(<x_n, y_m>/D)
         if normalize=False, the exponent is not divided by D
    """
    K = np.dot(X,Y.T)
    if normalize:
        K = K / float(D)
    K = -1. + 2**(K)
    return K

### MAIN FUNCTIONS ###########################################################
def estimate_threshold(K, zeta):
    """Estimates the threshold that regulates how large the distance between 
    KME t to all other KMEs can be s.t. they are still considered as neighbors.
    threshold_t = zeta * 1/(2*N^2*(N-1)) *
                 sum_(n != n')^N { k(x_n,x_n) - 2*k(x_n, x_n') + k(x_n',x_n')
    This threshold (without zeta) is the estimated MSE of task t.

    Args:
      K: (N,M) array, kernel matrix of task t
      zeta: scalar, regulates how large the distance between KMEs can be s.t.
           they are still considered as neighbors

    Returns:
      threshold: scalar, threshold
    """
    N = len(K)
    mask = np.ones((N,N))*(-2.) + np.eye(N)*((N-1)*2+2)
    threshold = (np.sum(np.multiply(K,mask)) / (2.*N*N*(N-1))) * zeta
    return threshold


def compute_kernel_sums(X, zeta, T, kernel, *args):
    """Computes the kernel matrices and the thresholds of each task in X.
    Note that the kernel matrices are not stored explicitly but only the sum
    of their entries.

    Args:
      X: (T,) list with (N,D) arrays, or (T,N,D) array, N observations of 
         D dimensions
      zeta: scalar, regulates how large the distance between KMEs can be s.t.
           they are still considered as neighbors
      T: int, number of tasks (bags)
      kernel: function, kernel function 
      *args: parameters of the kernel function, e.g. sigma for Gaussian kernel

    Returns:
      sum_K: (T,T) array, (sum_K)_t,t' = sum(kernel(X[t], X[t']))
      sum_no_diag: (T,) array, (sum_no_diag)_t = sum(kernel(X[t], X[t])) - 
                   diag(kernel(X[t], X[t]))
      thresholds: (T,), threshold for each task
      num_obs: (T,) array, number of observations for each task (N)
    """
    sum_K = np.zeros((T,T))
    sum_no_diag = np.zeros(T)
    thresholds = np.zeros(T)
    num_obs = np.zeros(T)
    for t in range(T):
        for tt in range(t, T):
            K = kernel(X[t], X[tt], *args)
            sum_K[t,tt] = np.sum(K)
            sum_K[tt,t] = sum_K[t,tt]
            if t == tt:
                num_obs[t] = len(X[t])
                thresholds[t] = estimate_threshold(K, zeta)
                sum_no_diag[t] = sum_K[t,tt] - np.sum(np.diag(K))
    return sum_K, sum_no_diag, thresholds, num_obs


def compute_test_kernel_sums(X, Z, T, unbiased, kernel, *args):
    """Computes the kernel matrices between the training and test data for
    each task. Note that the kernel matrices are not stored explicitly but 
    only the sum of their entries.

    Args:
      X: (T,) list with (N,D) arrays, or (T,N,D) array, N observations of 
         D dimensions. Training data.
      Z: (T,) list with (M,D) arrays, or (T,M,D) array, M observations of 
         D dimensions with M >> N. Test data.
      T: int, number of tasks (bags)
      unbiased: bool, use unbiased MMD estimation for calculation of the kme 
                error (see compute_KME_error(...) for more detail) 
      kernel: function, kernel function 
      *args: parameters of the kernel function, e.g. sigma for Gaussian kernel

    Returns:
      sum_no_diag_Z: (T,) array, (sum_no_diag_Z)_t = sum(kernel(Z[t], Z[t])) - 
                     diag(kernel(Z[t], Z[t]))
      sum_XZ: (T,T) array, (sum_XZ)_t,t' = sum(kernel(X[t], Z[t']))
      num_obs_Z: (T,) array, number of test samples of each task 
    """
    sum_no_diag_Z = np.zeros(T)
    sum_XZ = np.zeros((T,T))
    num_obs_Z = np.zeros(T)
    for t in range(T):
        num_obs_Z[t] = len(Z[t])
        K_Z = kernel(Z[t], Z[t], *args) 
        if unbiased:
            np.fill_diagonal(K_Z, 0)
        sum_no_diag_Z[t] = np.sum(K_Z)
        for tt in range(T):
            K = kernel(X[t], Z[tt], *args)
            sum_XZ[t,tt] = np.sum(K)
    return sum_no_diag_Z, sum_XZ, num_obs_Z


def compute_MMD2_matrix_naive(sum_K, sum_no_diag, num_obs, T, unbiased, replace):
    """Computes the matrix that holds the pairwise MMD^2 between each pair of 
    KMEs. Here, each KME is estimated using the naive approach (NE). See also:
    compute_MMD2_matrix_weighted(...).

    Args:
      sum_K: (T,T) array, (sum_K)_t,t' = sum(kernel(X[t], X[t']))
      sum_no_diag: (T,) array, (sum_no_diag)_t = sum(kernel(X[t], X[t])) - 
                   diag(kernel(X[t], X[t]))
      num_obs: (T,) array, number of observations for each task (N)
      T: int, number of tasks (bags)
      unbiased: bool, use unbiased (or biased) estimate of the squared MMD
      replace: bool, replace (or not replace) the estimated MMD^2 with 0 if
               it is negative. 

    Returns:
      G: (T,T) array, estimated MMD^2, (G)_t,t' = MMD^2(H, X[t], X[t'])
    """
    G = np.zeros((T,T))
    D = np.zeros(T)
    for t in range(T):
        N_t = num_obs[t]
        if D[t] == 0:
            if unbiased:
                D[t] = np.sum(sum_no_diag[t]) / (N_t*(N_t-1.))
            else:
                D[t] = sum_K[t,t] / (N_t*N_t)
        #for tt in range(t+1,T):
        for tt in range(t,T):
            N_tt = num_obs[tt]
            if D[tt] == 0:
                if unbiased:
                    D[tt] = np.sum(sum_no_diag[tt]) / (N_tt*(N_tt-1.))
                else:
                    D[tt] = sum_K[tt,tt] / (N_tt*N_tt)
            D_t_tt = sum_K[t,tt] / (N_t*N_tt)
            G[t,tt] = D[t] + D[tt] - 2.*D_t_tt
            if replace:
                G[t,tt] = np.max([G[t,tt], 0])
            G[tt,t] = G[t,tt]
    return G


def compute_MMD2_matrix_weighted(sum_K, sum_no_diag, num_obs, W, T, unbiased, replace):
    """Computes the matrix that holds the pairwise MMD^2 between each pair of 
    KMEs. Here, each KME is estimated as the weighted sum of the naive 
    estimations. See also: compute_MMD2_matrix_naive(...).

    Args:
      sum_K: (T,T) array, (sum_K)_t,t' = sum(kernel(X[t], X[t']))
      sum_no_diag: (T,) array, (sum_no_diag)_t = sum(kernel(X[t], X[t])) - 
                   diag(kernel(X[t], X[t]))
      num_obs: (T,) array, number of observations for each task (N)
      W: (T,T) array, weights to weight the naive estimations with, i.e.
         weighted KME_t is the sum of (W)_t,t'*naive KME_t' over all t'.
      T: int, number of tasks (bags)
      unbiased: bool, use unbiased (or biased) estimate of the squared MMD
      replace: bool, replace (or not replace) the estimated MMD^2 with 0 if
               it is negative. 

    Returns:
      G: (T,T) array, estimated MMD^2, (G)_t,t' = MMD^2(H, X[t], X[t'])
    """
    ### updated
    G = np.zeros((T,T))
    D = np.zeros(T)
    N_TT = np.outer(num_obs, num_obs)
    sum_K = sum_K / N_TT
    if unbiased:
        sum_no_diag = sum_no_diag / (num_obs*(num_obs-1.))
    else:
        sum_no_diag = sum_no_diag / (np.diag(N_TT))
    for t in range(T):
        if D[t] == 0:
            mask = np.outer(W[t,:], W[t,:])
            if unbiased:
                diag_mask = np.diag(mask).copy()
                np.fill_diagonal(mask,0)
                D[t] = np.sum(mask*sum_K) + np.sum(diag_mask*sum_no_diag)
            else:
                D[t] = np.sum(mask*sum_K)
        for tt in range(t,T):
            if D[tt] == 0:
                mask = np.outer(W[tt,:], W[tt,:])
                if unbiased: 
                    diag_mask = np.diag(mask).copy()
                    np.fill_diagonal(mask,0)
                    D[tt] = np.sum(mask*sum_K) + np.sum(diag_mask*sum_no_diag)
                else:
                    D[tt] = np.sum(mask*sum_K)
            mask = np.outer(W[t,:], W[tt,:])
            D_t_tt = np.sum(mask*sum_K)
            G[t,tt] = D[t] + D[tt] - 2.*D_t_tt
            if replace:
                G[t,tt] = np.max([G[t,tt], 0])
            G[tt,t] = G[t,tt]
    return G


def compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T, unbiased, replace):
    """Computes the KME estimation error which is the MMD^2 between the 
    estimated KME (based on training data X) and the true or proxy KME (based
    on much larger test data Z).

    Args:
      sum_K: (T,T) array, (sum_K)_t,t' = sum(kernel(X[t], X[t']))
      num_obs: (T,) array, number of training observations for each task (N)
      sum_no_diag: (T,) array, (sum_no_diag)_t = sum(kernel(X[t], X[t])) - 
                   diag(kernel(X[t], X[t]))
      sum_XZ: (T,T) array, (sum_XZ)_t,t' = sum(kernel(X[t], Z[t']))
      sum_no_diag_Z: (T,) array, (sum_no_diag_Z)_t = sum(kernel(Z[t], Z[t])) - 
                     diag(kernel(Z[t], Z[t]))
      num_obs_Z: (T,) array, number of test samples of each task 
      W: (T,T) array, (W)_t,t' holds the weight which is used to weight the
         naive KME of task t' for the calculation of the KME of task t.
      T: int, number of tasks (bags)
      unbiased: bool, use unbiased (or biased) estimate of the squared MMD
      replace: bool, replace (or not replace) the estimated MMD^2 with 0 if
               it is negative. 

    Returns:
      error: (T,) array, estimation error for each task
    """
    error = np.zeros(T)
    one_hot = np.eye(T)
    N_XX = np.outer(num_obs, num_obs)
    N_XZ = np.outer(num_obs, num_obs_Z)
    if unbiased:
        N_ZZ = num_obs_Z*(num_obs_Z-1.)
    else:
        N_ZZ = num_obs_Z*num_obs_Z
        
    sum_K   = sum_K / N_XX
    sum_XZ  = sum_XZ/ N_XZ
    sum_no_diag_Z   = sum_no_diag_Z/ N_ZZ
    
    for t in range(T):
        # X_term (diagonal terms included)
        mask = np.outer(W[t,:], W[t,:])
        D_X  = np.sum(mask*sum_K)
        # Z_term (diagonal excluded)
        D_Z = sum_no_diag_Z[t]
        # X_Z_term
        mask  = np.outer(W[t,:], one_hot[t,:])
        D_X_Z = np.sum(mask*sum_XZ)
        error[t] = D_X + D_Z - (2.*D_X_Z)
        if replace:
            error[t] = np.max([error[t], 0])
    return error


### KME ESTIMATION APPROACHES ################################################
def compute_STB_neighbors(G_naive, est_MSE, zeta):
    """Determines the neighboring tasks for each task.

    Args:
      G_naive: (T,T) array, estimated MMD^2 between the naive estimations of 
               the KMEs (see compute_MMD_matrix_naive)
      est_MSE: (T,) distance between KMEs can be s.t.
                they are still considered as neighbors without zeta (see 
                estimate_threshold(...))
      zeta: scalar, regulates how large the distance between KMEs can be s.t.
           they are still considered as neighbors
    Returns:
      W: (T,T) array, (W)_t,t' = True, if task t has task t' as neighbor. Note
         that W is not necessarily symmetric.
    """
    W = (G_naive <= zeta*est_MSE[:,None])
    np.fill_diagonal(W,True)
    return W


def compute_STB_weight(W, method, *args):
    """Computes the weighting of each KME based on the neighboring tasks using
    the specified STB method. For the formulas of each method, see Appendix 7.
    For all methods, it holds that (W)_t,t' = 0 if t' is not a neighbor of t.

    Args:
      W: (T,T) array, (W)_t,t' = True, if task t has task t' as neighbor. Note
         that W is not necessarily symmetric.
      method: String, must be one of the following: {'weighted', 'stb', 
              'theory'}. Let V_t denote the number of neighbors of task t.
              'stb': uses uniform weighting for every neighbor.
                     (W)_t,t' = 1/V_t
              'weighted': weighting for task t is different than for its
                          neighbors. 
                          (W)_t,t  = gamma + (1-gamma)/V_t
                          (W)_t,t' = (1-gamma)/V_t
              'theory': as in 'weighted' with 
                        gamma_t = c*zeta*(V_t-1) / ((1+c*zeta)*(V_t-1)+ 1)
      args: model parameters needed for method:
            'stb': []
            'weighted': [gamma]
            'theory': [zeta, c]
    Returns:
      W: (T,T) array, (W)_t,t' holds the weight which is used to weight the
         naive KME of task t' for the calculation of the KME of task t.
    """
    if method == 'weighted':
        alpha = args[0]
        w = np.sum(W,axis=1)[:,np.newaxis]
        W = W*((1.-alpha)/w)
        np.fill_diagonal(W, np.diag(W)+alpha)
    elif method == 'stb':
        W = (W.T/np.sum(W,axis=1)).T
    elif method=='theory':
        zeta    = args[0]
        alpha  = args[1]
        w      = np.sum(W,axis=1)
        const  = alpha*zeta
        gammas = (const*(w-1.))/((1.+const)*(w-1.) + 1.)
        off_diag = ((1.-gammas)/w)[:,np.newaxis]
        W = W*off_diag
        np.fill_diagonal(W, np.diag(W)+gammas)
    else:
        print('Unknown method in compute_STB_weight!')
    return W


def compute_MTA_similarity(G_naive, T, method, *args):
    """Computes the task relatedness for the specified MTA method. The MTA 
    approach was introduced in
    >> Feldman, S., Gupta, M. R., and Frigyik, B. A. (2014). Revisiting Stein’s 
       paradox: multi-task averaging. Journal of Machine Learning Research, 
       15(106):3621–3662. <<
    that we modified such that it can be used for the estimation of KMEs.
    For the formulas of each method, see Appendix 7.

    Args:
      G_naive: (T,T) array, estimated MMD^2 between the naive estimations of 
               the KMEs (see compute_MMD_matrix_naive)
      T: int, number of tasks (bags)
      method: String, must be one of the following: {stb', 'const'}.
              'const': assumes every task is related to ever other task with a
                       constant amount of similarity (average distance between
                       the naive estimations of the KMEs, see G_naive).
              'stb': uses the proposed similarity test (see 
                     compute_STB_neighbors(...)) as task relatedness.
              'gauss': uses a Gaussian kernel to estimate the task similarity
              'gaussvar': uses a Gaussian kernel to estimate the task
                          similarity where the kernel width depends on the
                          task_var
      args: model parameters needed for method:
            'const': []
            'stb': [zeta, est_MSE] where est_MSE is defined as the maximum 
                   distance between KMEs can be s.t. they are still considered 
                   as neighbors without zeta (see estimate_threshold(...))
    Returns:
      A: (T,T) array, (A)_t,t' holds the similarity between task t and t'.
    """
    if method == 'const':
        a = 2./(1./(T*(T-1.))*np.sum(G_naive))
        A = np.ones((T,T))*a
    elif method == 'stb':
        # args[0]: zeta, args[1]: task_var
        A = (G_naive <= args[0]*args[1][:,None])
    elif method == 'gauss':
        # args[0]: zeta
        A = np.exp(G_naive / (-2.*args[0]))
    elif method == 'gaussvar':
        # args[0]: zeta, args[1]: task_var
        temp = (G_naive.T/args[1]).T
        A = np.exp(temp/ (-2.*args[0]))
    else:
        print('Unknown method in compute_MTA_similarity!')
        A = np.eye(T)
    return A


def compute_laplacian(A):
    """Computes the graph laplacian

    Args:
      A: (T,T) array, (A)_t,t' holds the similarity between task t and t'.
      
    Returns:
      L: (T,T) array, graph Laplacian of A
    """
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return L


def compute_MTA_weight(A, est_MSE, gamma, T):
    """Computes the weighting of each KME based on similar tasks using the 
    specified MTA method. For the formulas of each method, see Appendix 7.

    Args:
      A: (T,T) array, (A)_t,t' holds the similarity between task t and t'.
      est_MSE: (T,) estimated 'variance' of task t (see: estimate_threshold)
      gamma: scalar, regularization parameter
      T: int, number of tasks (bags)
      
    Returns:
      W: (T,T) array, (W)_t,t' holds the weight which is used to weight the
         naive KME of task t' for the calculation of the KME of task t.
    """
    L = compute_laplacian(A)
    W = np.eye(T) + (gamma/T*np.dot(np.diag(est_MSE), L))
    W = np.linalg.inv(W)
    return W


def compute_RKMSE_weight(sum_K, sum_no_diag, num_obs):
    """Computes the weighting of each KME. This approach was proposed in
    >>> Muandet, K., Sriperumbudur, B., Fukumizu, K., Gretton, A., and 
        Sch¨olkopf, B. (2016). Kernel mean shrinkage estimators. Journal of 
        Machine Learning Research, 17(48):1–41. <<< 
    It estimates each KME individually (i.e. (W)_t,t' = 0 for all t!=t') and
    shrinks it towards 0. For the formulas this method, see Appendix 7.

    Args:
      sum_K: (T,T) array, (sum_K)_t,t' = sum(kernel(X[t], X[t']))
      sum_no_diag: (T,) array, (sum_no_diag)_t = sum(kernel(X[t], X[t])) - 
                   diag(kernel(X[t], X[t]))
      num_obs: (T,) array, number of training observations for each task (N)
      
    Returns:
      W: (T,T) array, (W)_t,t' holds the weight which is used to weight the
         naive KME of task t' for the calculation of the KME of task t.
    """
    rho     = np.diag(sum_K)/(num_obs*num_obs)
    varrho  = (np.diag(sum_K) - sum_no_diag) / num_obs
    w       = (varrho - rho) / ((num_obs - 1.)*rho + (1./num_obs - 1.)*varrho)
    w       = 1. - (w / (1. + w))
    W       = np.diag(w)
    return W
    

### FUNCTIONS FOR GENERATING THE TOY DATA ####################################
def gen_data(mu, S, N, T, angle_min=((-1.)*(np.pi/4.)), angle_max=(np.pi/4.)):
    """Generates synthetic data sets. The generated data will be Gaussian 
    distributed with mean(s) mu. Each task has a different covariance matrix, 
    given by S rotated by a random angle (sampled from an uniform distribution 
    with [angle_min, angle_max]).

    Args:
      mu: (D,) or (T,D) array, mean or means of each task of dimension D
      S: (D,D) array, matrix that is randomly rotated to form the covariance 
         matrices
      N: (T,) array, number of observations for each task 
      T: int, number of tasks (bags)
      angle_min: smallest possible angle
      angle_max: largest possible angle
      
    Returns:
      Data: (T,) list with (N,D) arrays, list of random data with N observations 
            of D dimensions
      angles: (T,) list, random angles that were used to rotate the covariance 
              matrix of task t
    """
    centroids = (len(mu) == T)
    Data = []
    angles = []
    for t in range(T):
        # random rotation matrix
        theta_t = np.random.uniform(angle_min, angle_max)
        angles.append(theta_t)
        c = np.cos(theta_t)
        s = np.sin(theta_t)
        R_t = np.array([[c, (-1.)*s],[s, c]])
        S_t = np.dot(R_t,np.dot(S, R_t.T))
        # generate random data
        if centroids:
            Data.append(np.random.multivariate_normal(mu[t], S_t, N[t]))  
        else:
            Data.append(np.random.multivariate_normal(mu, S_t, N[t]))  
    return Data, angles


def gen_indep_data(mu, S, N_X, N_Z, angle_min=((-1.)*(np.pi/4.)), angle_max=(np.pi/4.)):
    """Generates synthetic data sets for training and testing. 
    The generated data will be Gaussian distributed with mean(s) mu. Each task 
    has a different covariance matrix, given by S rotated by a random angle 
    (sampled from an uniform distribution with [angle_min, angle_max]).

    Args:
      mu: (D,) or (T,D) array, mean or means of each task of dimension D
      S: (D,D) array, matrix that is randomly rotated to form the covariance 
         matrices
      N_X: (T,) array, number of training samples for each task 
      N_Z: int, number of test samples (const. for each task) 
      angle_min: smallest possible angle
      angle_max: largest possible angle
      
    Returns:
      X: (T,) list with (N_X,D) arrays, list of random training data with N_X 
         observations of D dimensions
      Z: (T,) list with (N_X,D) arrays, list of random test data with N_Z 
         observations of D dimensions
      angles: (T,) list, random angles that were used to rotate the covariance 
              matrix of task t
    """
    X = []
    Z = []
    angles = []
    T = len(N_X)
    centroids = (len(mu) == T)
    for t in range(T):
        # random rotation matrix
        theta_t = np.random.uniform(angle_min, angle_max)
        angles.append(theta_t)
        c = np.cos(theta_t)
        s = np.sin(theta_t)
        R_t = np.array([[c, (-1.)*s],[s, c]])
        S_t = np.dot(R_t,np.dot(S, R_t.T))
        # generate random data
        if centroids:
            X.append(np.random.multivariate_normal(mu[t], S_t, N_X[t]))
            Z.append(np.random.multivariate_normal(mu[t], S_t, N_Z))
        else:
            X.append(np.random.multivariate_normal(mu, S_t, N_X[t]))
            Z.append(np.random.multivariate_normal(mu, S_t, N_Z))
    return X, Z, angles


def gen_centroids(num_centroids, radius):
    """Generates data points that are equally spaced on a circle of given 
    radius.

    Args:
      num_centroids: int, number of data points that shall be generated
      radius: scalar, radius or distance between each generated data point and
              the origin.
      
    Returns:
      centroids: (num_centroids,) list, two dimensional data points that lie 
                 equally spaced on a circle.
    """
    Theta  = np.linspace(0, np.pi, num_centroids, endpoint=False)
    vector = np.array([radius, 0]) 
    centroids = []
    for idx in range(num_centroids):
        c = np.cos(Theta[idx])
        s = np.sin(Theta[idx])
        R_t = np.array([[c, (-1.)*s],[s, c]])
        rotated_vec = np.dot(R_t,np.dot(vector, R_t.T))
        centroids.append(rotated_vec)
    return centroids
    

### AUXILARIES ###############################################################
def splitData(Data, train_test_fraction):
    """Splits the given data into training and test set.

    Args:
      Data: (T,) list with (N,D) arrays, list of random data with N observations 
            of D dimensions
      train_test_fraction: [0,1], percent of the data used for training
      
    Returns:
      Train: (T,) list with (N*train_test_fraction,D) arrays, list of training
             data with N*train_test_fraction observations of D dimensions
      Test: (T,) list with (N,D) arrays, equal to Data
    """
    T = len(Data)
    Train = []
    Test = []
    for t in range(T):
        N = len(Data[t])
        frac = int(train_test_fraction*N)
        train_idx = np.random.choice(np.linspace(0,N-1,N, dtype=int), frac)
        Train.append(Data[t][train_idx, :])
        Test.append(Data[t])
    return Train, Test


def shuffle_matrix(A, shuffle_indices):
    """Sorts matrix A (rows and columns) accordings to shuffle_indices such 
    that (A)_i,j = (B)_shuffle_indices[i], shuffle_indices[j]

    Args:
      A: (D,D) arrays, symmetric matrix that needs to be shuffled
      shuffle_indices: (D,) array, indices used for sorting/shuffling
      
    Returns:
      B: (D,D) array, holds same data as A but was sorted according to the
         given indices
    """
    B = np.zeros_like(A)
    B[:,:] = A[shuffle_indices,:]
    B[:,:] = B[:,shuffle_indices]
    return B


### FUNCTIONS FOR TESTING ####################################################  
def test_G(sum_K, sum_no_diag, num_obs, W, unbiased, replace):
    """Should return same result as compute_MMD_matrix_weighted(...).

    Args:
      sum_K: (T,T) array, (sum_K)_t,t' = sum(kernel(X[t], X[t']))
      sum_no_diag: (T,) array, (sum_no_diag)_t = sum(kernel(X[t], X[t])) - 
                   diag(kernel(X[t], X[t]))
      num_obs: (T,) array, number of observations for each task (N)
      W: (T,T) array, weights to weight the naive estimations with, i.e.
         weighted KME_t is the sum of (W)_t,t'*naive KME_t' over all t'.
      unbiased: bool, use unbiased (or biased) estimate of the squared MMD
      replace: bool, replace (or not replace) the estimated MMD^2 with 0 if
               it is negative. 

    Returns:
      G: (T,T) array, estimated MMD^2, (G)_t,t' = MMD^2(H, X[t], X[t'])
    """
    T = len(sum_K)
    DTT = np.zeros((T,T))
    G = np.zeros((T,T))
    if unbiased:
        # compute D matrix
        DT = np.zeros(T)
        for t in range(T):
            for b in range(T):
                for bb in range(T):
                    if b==bb:
                        DT[t] += W[t,b]*W[t,b]*(1./num_obs[b])*(1./(num_obs[b]-1.))*sum_no_diag[b]
                    else:
                        DT[t] += W[t,b]*W[t,bb]*(1./num_obs[b])*(1./num_obs[bb])*sum_K[b,bb]
        for t in range(T):
            for tt in range(T):
                for b in range(T):
                    for bb in range(T):
                        DTT[t,tt] += W[t,b]*W[tt,bb]*(1./num_obs[b])*(1./num_obs[bb])*sum_K[b,bb]
        # compute G matrix
        for t in range(T):
            for tt in range(T):
                G[t,tt] = DT[t] + DT[tt] - 2*DTT[t,tt]
                if replace:
                    G[t,tt] = np.max([G[t,tt], 0])
    else:
        for t in range(T):
            for tt in range(T):
                for b in range(T):
                    for bb in range(T):
                        DTT[t,tt] += W[t,b]*W[tt,bb]*(1./num_obs[b])*(1./num_obs[bb])*sum_K[b,bb]
        # compute G matrix
        for t in range(T):
            for tt in range(T):
                G[t,tt] = DTT[t,t] + DTT[tt,tt] - 2*DTT[t,tt]
                if replace:
                    G[t,tt] = np.max([G[t,tt], 0])
    return G


def test_KME_error(sum_X, num_obs_X, sum_XZ, sum_no_diag_Z, num_obs_Z, W, replace):
    """Should return same results as compute_KME_error(...).

    Args:
      sum_X: (T,T) array, (sum_K)_t,t' = sum(kernel(X[t], X[t']))
      num_obs_X: (T,) array, number of training observations for each task (N)
      sum_XZ: (T,T) array, (sum_XZ)_t,t' = sum(kernel(X[t], Z[t']))
      sum_no_diag_Z: (T,) array, (sum_no_diag_Z)_t = sum(kernel(Z[t], Z[t])) - 
                     diag(kernel(Z[t], Z[t]))
      num_obs_Z: (T,) array, number of test samples of each task 
      W: (T,T) array, weighting for each KME
      unbiased: bool, use unbiased (or biased) estimate of the squared MMD
      replace: bool, replace (or not replace) the estimated MMD^2 with 0 if
               it is negative. 

    Returns:
      error: (T,) array, estimation error for each task
    """
    T = len(sum_X)
    error = np.zeros(T)
    for t in range(T):
        DX = 0.
        DXZ = 0.
        DZ  = sum_no_diag_Z[t] / (num_obs_Z[t]*(num_obs_Z[t]-1.))
        for b in range(T):
            DXZ += W[t,b]*sum_XZ[b,t]/ (num_obs_X[b]*num_obs_Z[t])
            for bb in range(T):
                DX  += W[t,b]*W[t,bb]*sum_X[b,bb]/(num_obs_X[b]*num_obs_X[bb])
        error[t] = DX + DZ - 2.*DXZ
        if replace:
            error[t] = np.max([error[t], 0.])
    return error