function estim = aggregate_means(X,t=0.1,sigma=1, distsq = 0)

  %% the data are supposed to be N(mu_i, I )
  %% X is a N x d data matrix


  d = size(X)(2);

  %% Compute distances (if not given)

  if (distsq == 0)

    xnorm = sum(X.^2,2);    
    distsq = repmat(xnorm,1,N) + repmat(xnorm',N,1) - 2*X*X';
    
  endif;
  
  %% Test
  
  tests = (distsq <= (2+t)*sigma^2*d);
  
  tests = diag(1./sum(tests,2)) * tests; %normalize

  %% Estimate

  estim = tests*X ;

endfunction




