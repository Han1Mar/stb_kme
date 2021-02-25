function estim = shrink_means(X,t=0.1,sigma=1, gamma=0, distsq = 0,
			      est_type = "theory" )

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

  %% Number of neighbors of each task, including itself

  v = sum(tests,2) ;
  
  %% Weights for estimation

  
  switch est_type

    case "theory"

      gammai = gamma*(v-1)./((1+gamma)*(v-1)+1);

      weights = diag(gammai)  + diag((1-gammai)./v) * tests ;

    case "shrink"

      weights = gamma * eye(N) + (1-gamma) * diag(1./v) * tests ;
  
  endswitch
  
  %% Estimate

  

  estim = weights*X ;

endfunction




