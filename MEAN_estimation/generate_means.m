function mu = generate_means(varargin)			     
  %% This function generates random means drawn
  %% according to a specified model.
  %%
  %% Models:
  %%    1: gaussian means in lower dimension (mudim,muspread)
  %%    2: sparse gaussian means (mudim, muspread)
  %%    3: uniform means in lower dimension (mudim, muspread)
  %%    4: sparse uniform means (mudim, muspread)
  %%    5: mixture of k gaussian means in lower dimension (nclusters, mudim)
  %%    6: 2 opposite half gaussian means in lower dimension
  %%    7: means uniform on a sphere of lower dimension (mudim, muspread [radius])

  %% sigma: variance
  %% N: number of means
  %% d: ambient dimensionality

  [foo, model, N, d, mudim, muspread, nclusters, clusterspread, shift] = ...
  parseparams(varargin,
	     "MODEL", 1,
	     "N", 500,
	     "D", 1000,
	     "MUDIM", 10,
	     "MUSPREAD", 1,
	     "NCLUSTERS", 5,
	     "CLUSTERSPREAD", 10,
	     "SHIFT", 0);
  
  switch  model
    case 1
    %% model 1: gaussian means in lower dimension
    mu = [ muspread*randn(N,mudim) zeros(N,d-mudim)];

  case 2
    %% model 2: sparse gaussian means
    mu = zeros(N,d);

    for (i=1:N)
      mu(i,randperm(d,mudim)) = muspread*randn(1,mudim);
    endfor

  case 3
    %% model 3: uniform means in lower dimension
    mu = [ muspread*2*(rand(N,mudim)-0.5*(1-shift)) zeros(N,d-mudim)];

  case 4
    %% model 4: sparse uniform means (with shift to set off simple shrinkage)
	
    mu = zeros(N,d);
    
    for (i=1:N)
      mu(i,randperm(d,mudim)) = muspread*2*(rand(1,mudim)-0.5*(1-shift));
    endfor

  case 5
    %% model 5: nclusters gaussian clusters in lower dimension

    mu = [ muspread*randn(N,mudim) zeros(N,d-mudim)];

    centers = clusterspread*randn(nclusters,d) ;

    mu = mu + centers(randi(nclusters,N,1),:) ;

    % try to find unfavorable case for bias ?
    
    % mu(1:(N/2), d) = 1*sigma*sqrt(d);

  case 6
    %% model 6: 2 opposite half gaussian means in lower dimension
    %% (trying to find unfavorable case for bias)

    mu = zeros(N,d);

    mu(1:N, 1) = musigma*randn(N,1);
    mu(1:(N/2), 1) = abs(mu(1:(N/2), 1));
    mu(((N/2)+1):end, 1) = -0.5*musigma - abs(mu(((N/2)+1):end, 1));

  case 7
    %% model 7: means uniform on a sphere of lower dimension (radius muspread)

    mu = randn(N,mudim+1);

    %% normalize

    mu = muspread*diag(1./sqrt(sum(mu.^2,2)))*mu;
    
    mu = [ mu zeros(N,d-mudim-1)];

    
    
  endswitch


endfunction
