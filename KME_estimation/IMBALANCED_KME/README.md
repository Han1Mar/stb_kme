# IMBALANCED Experiments

This code can be used to reproduce the experiments reported in **Section 4.1, Figure 1(c)** of   
> Marienwald, Hannah, Fermanian, Jean-Baptiste and Blanchard, Gilles. (2021). High-Dimensional Multi-Task Averaging and Application to Kernel Mean Embedding. In *International Conference on Artificial Intelligence and Statistics*, PMLR.
---

This experiment is based on randomly rotated, 2D Gaussian distributed toy data.
Each of the 50 bags has a different number of samples (between 10 to 300); this
corresponds to setting (c) Imbalanced Bags.

You can change the settings of the experiments in `IMBALANCED_settings.py.`

In order to run the experiments on the MISR1 data set please run     
`> python RUN_ALL.py`       
or
proceed as follows:
- Run `MODEL_naive.py` and `FINAL_naive.py` *once* which generate the toy data used for
  the model optimization (see `‘../Results/imbalanced_kme/ModelData/'`) and the final 
  estimation of the generalization error (`‘../Results/imbalanced_kme/FinalData/'`). 
  Each other method will be performed on those generated data sets.
- Run `MODEL_<method>.py` to find the optimal values for the model parameters, e.g.
  run `MODEL_STB_weight.py` to find optimal values for $\zeta$ and $\gamma$. The results will
  be save in `‘../Results/imbalanced_kme/ModelData/opt_param_<method>.mat'`.
- Only after the optimal model parameters were found, run `FINAL_<method>.py` to estimate
  the generalized KME estimation error of the method; saved in     
  `‘../Results/imbalanced_kme/FinalData/KME_error_<method>.mat'`.

Once every method was run, you can plot a Figure similar to the one seen in the paper
which summarizes the final results (`PLOT_Result.py`)