# WINE Experiments

This code can be used to reproduce the experiments reported in **Section 4.2, Figure 2** of   
> Marienwald, Hannah, Fermanian, Jean-Baptiste and Blanchard, Gilles. (2021). High-Dimensional Multi-Task Averaging and Application to Kernel Mean Embedding. In *International Conference on Artificial Intelligence and Statistics*, PMLR.
---

This experiment is based on the wine data set which can be found on kaggle:
[https://www.kaggle.com/dbahri/wine-ratings]
and was first used in:

>Gupta, Maya R., Bahri, Dara, Cotter, Andrew and Canini, Kevin. (2018). Diminishing Returns Shape Constraints for Interpretability and Regularization. In *Proceedings of the 32nd International Conference on Neural Information Processing Systems (NeurIPS '18).

**Please download the data set (train, test, validation.csv) before running the experiments and copy it to `‘../Results/wine_kme/Data/``.**

The data set consists of wine characteristics represented as binary features.
Wines are grouped by their origin such that the wines of one country form a bag.
The original data set is highly imbalanced. However, for the experiments we
selected a random subset of samples to form the final data set. The final 
data set consists of 15 bags with each 461 samples of 39-dimensional 
${0,1}$-samples.
All 461 samples are used for the estimation of the (proxy) true KME. A randomly
selected subset of samples of size 20,50,100,150 or 200 is used for the 
estimation of the KME. 

You can change the settings of the experiments in `WINE_settings.py`.

In order to run the experiments on the wine data set please run
`> python RUN_ALL.py`
or
proceed as follows:
- Make sure that the folder `‘../Results/wine_kme/Data/`' holds the feature 
  representations of each bag (train, test, validation.csv).
- Run the script `DATA_preparation.py` to preprocess the data, which reads in 
  the data of each bag, selects subsets and saves them in a single list.
- Run the script `KERNEL_preparation.py` to precompute all relevant kernels. 
- Make sure to run `DATA_preparation.py` and `KERNEL_preparation.py` only *once* so that 
  every approach operates on the same (subsampled) data.
- Run the experiments on the wine data set for a specific method by running the 
  corresponding script, e.g. run `FINAL_STB_weight.py` to estimate the KME estimation 
  error of STB weight. It finds optimal values for the model parameters (saved in 
  `opt_param_<method>.mat`), saves the cross-validation error of the optimal parameter 
  combination (`CV_error_<method>.csv`) and the final KME estimation error computed on 
  the test data (`KME_error_<method>.csv`) in folder `'../Results/wine_kme/TrialData/'`.

Once every method was run, you can print a summary of the results with `PLOT_Results.py`.