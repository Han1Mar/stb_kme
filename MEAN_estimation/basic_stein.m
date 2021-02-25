function estim = basic_stein(X, m=0, sigma=1)
  %% Positive-part JS estimator
  %% X is a N x d data matrix
  %% Shrinkage towards m
  %% sigma = noise variance

  [n,d] = size(X);

  if (m==0)
    m = zeros(1,d);
  endif;

  M = repmat(m, [n 1]);

  distsq = sum( (M-X).^2, 2);

  shrink = (1 - (d-3)*(d>3)*sigma^2./distsq);

  shrink = shrink.*(shrink>0);

  
  estim = repmat(shrink,[1 d]).*(X-M) + M;

endfunction

