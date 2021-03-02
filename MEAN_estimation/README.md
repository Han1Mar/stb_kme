# Estimation of High-Dimensional Means

This code can be used to reproduce the experiments reported in **Section 8 of the Supplemental** of    
> Marienwald, Hannah, Fermanian, Jean-Baptiste and Blanchard, Gilles. (2021). High-Dimensional Multi-Task Averaging and Application to Kernel Mean Embedding. In *International Conference on Artificial Intelligence and Statistics*, PMLR.
---

These scripts are meant for GNU Octave -- they might be compatible with matlab but we did not check.

Note: there is only one data point per distribution here (if there are $N$ points per distribution, these can be replaced by the empirical average for each distribution since the methods only depend on this information; this equivalent to changing the variance by a factor $\frac{1}{N}$)

* `stb_shrink_means.m` is the main function for the methods proposed in the paper.
* `mtt_cst.m` is the MTA-const method from
> Feldman, Sergey, Maya R. Gupta and Bela A. Frigyik. (2014). Revisiting Stein's paradox: multi-task averaging. *Journal for Machine Learning Research (JMLR)*, 15(1), 3441-3482.
* `mtt_stb.m` applies the MTA method of Feldman et al 2014 using the affinity matrix (a graph) found by the similarity tests.
* `generate_means.m` generates the distribution means according to different structured models.
* `comparison_exp.m` is the main script to generate the results reported (run comparison_exp(i) for i=1,...,4)
