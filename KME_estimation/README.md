# Estimation of Kernel Mean Embeddings

This code can be used to reproduce the experiments of
> Marienwald, Hannah, Fermanian, Jean-Baptiste and Blanchard, Gilles. (2021). High-Dimensional Multi-Task Averaging and Application to Kernel Mean Embedding. In *International Conference on Artificial Intelligence and Statistics*, PMLR.

More specifically, see ...
* ... `TOY_KME` for the experiments on the synthetic data reported in Section 4.1, Figure 1(a), (b) and (d)
* ... `IMBALANCED_KME` for the experiments on the synthetic data reported in Section 4.1., Figure 1(c)
* ... `AOD_KME` for the experiments on the MISR1 data set reported in Section 4.2, Table 1 or
* ... `WINE_KME` for the experiments on the wine data set reported in Section 4.2, Figure 2.

More information can be found in the READMEs of the subdirectories.

`utiels.py` contains the main functions for applying the similarity-test based methods that were proposed in the paper. A detailed description of the tested methods can be found in Section 7 of the Supplemental of the paper.

Below you can find an example on how to compute the error of the KME estimation for bags $i \in {1, \ldots T}$. $X = {\lbrace X^{(i)} \rbrace}_{i = 1}^{T}$ contains the bag data, which is used to estimate the KME $\hat{\mu}_i$. The estimation error is computed as the unbiased, squared Maximum Mean Discrepancy (MMD) between the estimation $\hat{\mu}_i$ and the true KME $\mu_i$. Since $\mu_i$ is unknown, it must be approximated by another (naive) estimation based on independent data $Z = {\lbrace Z^{(i)} \rbrace}_{i = 1}^{T}$ of the same distribution as $X$ but large bag sizes. A Gaussian kernel might be used as `kernel` where `kargs` are its hyperparameters.      
### Example
```python
import utiels as u
import numpy as np

"""
Example: Computes the KME estimation error which is the MMD^2 between the estimated KME (based on training data X) and the true or proxy KME (based on much larger test data Z).

    Args:
      X: (T,) list with (N,D) arrays, or (T,N,D) array, N observations of D dimensions
      Z: (T,) list with (M,D) arrays, or (T,M,D) array, M observations of D dimensions with M >> N
      T: int, number of tasks (bags)
      unbiased: bool, use unbiased (or biased) estimate of the squared MMD
      replace: bool, replace (or not replace) the estimated MMD^2 with 0 if it is negative. 
      zeta: scalar, regulates how large the distance between KMEs can be s.t. they are still considered as neighbors
      gamma: scalar, in [0,1], see Supplemental Section 7 for a detailed description
      kernel: function, kernel function 
      *kargs: parameters of the kernel function, e.g. sigma for Gaussian kernel

    Output:
      error_naive: (T,) array, estimation error for each task for naive method
      error_stb_weight: (T,) array, estimation error for each task for STB weight
"""

# NAIVE ESTIMATION
# compute all relevant kernel matrices
sum_no_diag_Z, sum_XZ, num_obs_Z = u.compute_test_kernel_sums(X, Z, T, unbiased, kernel, kargs)
sum_K, sum_no_diag, task_var, num_obs = u.compute_kernel_sums(X, 1., T, kernel, kargs)
# compute average inter task MMD^2 (gives also the estimated gram matrix of the naive approach)
G_naive = u.compute_MMD2_matrix_naive(sum_K, sum_no_diag, num_obs, T, unbiased, replace)
# compute the error of the naive estimation
error_naive = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, np.eye(T), T, unbiased, replace)

# STB WEIGHT
# compute weighting matrix
W  = u.compute_STB_neighbors(G_naive, task_var, zeta)
W  = u.compute_STB_weight(W, 'weighted', gamma)
# compute the error of the estimation performed by STB weight
error_stb_weight = u.compute_KME_error(sum_K, num_obs, sum_XZ, sum_no_diag_Z, num_obs_Z, W, T, unbiased, replace)
```