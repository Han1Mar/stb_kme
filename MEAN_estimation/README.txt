"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> High-Dimensional Multi-Task Averaging and 
    Application to Kernel Mean Embedding <<<
---------------------------------------------------
"""

Scripts & functions for the experiments with Gaussian data for the
paper "High-Dimensional Multiple Task Averaging and Application to
Kernel Mean Embedding"

These scripts are meant for GNU Octave -- they might be compatible
with matlab but we did not check.

Note: there is only one data point per distribution here (if there are
n points per distribution, these can be replaced by the empirical
average for each distribution since the methods only depend on this
information; this equivalent to changing the variance by a factor 1/n)

stb_shrink_means.m is the main function for the STB methods proposed in
the paper.

mtt_cst.m is the "MTA-const" method (from Feldman et al 2014)

mtt_stb.m applies the MTA method of Feldman et al 2014 using the
affinity matrix (a graph) found by the similarity tests

basic_stein.m is the Positive-Part James-Stein estimator

generate_means.m generates the distribution means according to
different structured models

comparison_exp.m is the main script to generate the results reported

script_finalpaper.m runs all experiments reported in the paper.
