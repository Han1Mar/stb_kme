function estim = mtt_cst(X, c=0.1)
  %% Shrink naive est. towards overall average
  %% Basic version of "multi-task averaging" when no info present
  %% X is a N x d data matrix
  %% c is shrinking factor

  N = size(X)(1);

  estim = (1-c)*X + c* ones(N,N)/N * X;

endfunction

