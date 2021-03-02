# TOY Experiments

This code can be used to reproduce the experiments reported in **Section 4.1, Figure 1(a), (b) and (d)** of   
> Marienwald, Hannah, Fermanian, Jean-Baptiste and Blanchard, Gilles. (2021). High-Dimensional Multi-Task Averaging and Application to Kernel Mean Embedding. In *International Conference on Artificial Intelligence and Statistics*, PMLR.
---

This experiment is based on randomly rotated, 2D Gaussian distributed toy data.
These scripts can be used to reproduce the experiments for the settings:        
(a) Different Bagsizes (use `differentBagsizes` as `<setting>`)        
(b) Different Number of Bags (use `differentNrBags` as `<setting>`)           
(d) Clustered Bags (use `clustered` as `<setting>`)           

You can change the settings of the experiments in `TOY_settings.py.`

Most scripts in this folder expect `<setting>` as argument, e.g. run 
`> python MODEL_STB_weight.py differentBagsizes`
to run the model optimization for STB weight on setting (a).

In order to run the experiments on the data set please run
`> python RUN_ALL.py differentBagsizes`
or
`> python RUN_ALL.py differentNrBags`
or 
`> python RUN_ALL.py clustered`   
or
proceed as follows:
- Run `MODEL_naive.py` and `FINAL_naive.py` *once* which generate the toy data used for
  the model optimization (see `‘../Results/<setting>_kme/ModelData/'`) and the final 
  estimation of the generalization error (`‘../Results/<setting>_kme/FinalData/'`). 
  Each other method will be performed on those generated data sets.
  For the clustered setting run `CLUSTERED_MODEL_naive.py` and `CLUSTERED_FINAL_naive.py`.
- Run `MODEL_<method>.py` to find the optimal values for the model parameters, e.g.
  run `MODEL_STB_weight.py` to find optimal values for $\zeta$ and $\gamma$. The results will
  be save in `‘../Results/<setting>_kme/ModelData/opt_param_<method>.mat'`.
- Only after the optimal model parameters were found, run `FINAL_<method>.py` to estimate
  the generalized KME estimation error of the method; saved in 
  `‘../Results/<setting>_kme/FinalData/KME_error_<method>.mat'`.


Once every method was run, you can plot a Figure similar to the one seen in the paper
which summarizes the final results (`PLOT_Result.py`)