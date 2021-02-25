function estim = mtt_LapGaus(X, c = 0.1, distsq = 0, band = 0.5)
  %% Version of "multi-task averaging"
  %% Using Gaussian kernel as affinity matrix
  %% X is a N x d data matrix

  %% Compute distances (if not given)

  [N,d] = size(X);
  
  if (distsq == 0)

    xnorm = sum(X.^2,2);    
    distsq = repmat(xnorm,1,N) + repmat(xnorm',N,1) -2*X*X';
    
  endif;

  %% Gaussian affinity matrix
  
  Aff = exp(-distsq/(2*d*band^2));

  %% Laplacian matrix

  Lap = diag(Aff*ones(N,1)) - Aff;

  %% Transformation matrix

  Hat = inv(eye(N) + c*Lap);

  estim = Hat*X;

endfunction

