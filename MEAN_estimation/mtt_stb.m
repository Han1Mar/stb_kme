function estim = mtt_stb(X, t=0.1, sigma=1, c = 0.1, distsq = 0)
  %% Version of "multi-task averaging"
  %% Using test result matrix as affinity
  %% X is a N x d data matrix

  %% the data are supposed to be N(mu_i, I )
  %% X is a N x d data matrix


  [N,d] = size(X);
  

  %% Compute distances (if not given)

  if (distsq == 0)

    xnorm = sum(X.^2,2);    
    distsq = repmat(xnorm,1,N) + repmat(xnorm',N,1) - 2*X*X';
    
  endif;
  
  %% Test
  
  tests = (distsq <= (2+t)*sigma^2*d);

  %% use it as affinity matrix
  
  Aff = tests;

  %% Laplacian matrix

  Lap = diag(Aff*ones(N,1)) - Aff;

  %% Transformation matrix

  Hat = inv(eye(N) + c*Lap);

  estim = Hat*X;

endfunction

