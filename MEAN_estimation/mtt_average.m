function estim = mtt_average(X, c=0.1)
  %% Convex combination of each line and average
  %% Basic version of "multi-task averaging"
  %% X is a N x d data matrix

  N = size(X)(1);

  estim = (1-c)*X + c* ones(N,N)/N * X;

endfunction

