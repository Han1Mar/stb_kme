function [naive, stein] =  basicexp ( model = 1, sigma = 1, N = 500,
				      t = 0, d = 1000)

%% Dimensionality

%d = 1000;

%% Noise

%sigma = 1;

%% Number of means

%N = 500;

%% Testing threshold

%t = 0;

%% Draw means

%model = 5

switch  model
  case 1
    %% model 1: gaussian means in lower dimension

    mudim = 10;
    musigma = 1;

    mu = [ musigma*randn(N,mudim) zeros(N,d-mudim)];

  case 2
    %% model 2: sparse gaussian means

    mudim = 10;
    musigma = 1;
    
    mu = zeros(N,d);

    for (i=1:N)
      mu(i,randperm(d,mudim)) = musigma*randn(1,mudim);
    endfor

  case 3
    %% model 3: uniform means in lower dimension
    mudim = 10;
    muspread = 1;

    mu = [ muspread*2*(rand(N,mudim)-0.5) zeros(N,d-mudim)];

  case 4
    %% model 4: sparse uniform means
	
    mudim = 10;
    muspread = 1;
    
    mu = zeros(N,d);
    
    for (i=1:N)
      mu(i,randperm(d,mudim)) = muspread*2*(rand(1,mudim)-0.5);
    endfor

  case 5
    %% model 5: 2 gaussian means in lower dimension

    mudim = d-1;
    musigma = 0.25;

    mu = [ musigma*randn(N,mudim) zeros(N,d-mudim)];

    % try to find unfavorable case for bias ?
    
    mu(1:(N/2), d) = 1*sigma*sqrt(d);

  case 6
    %% model 6: 2 opposite half gaussian means in lower dimension

    musigma = 10000;

    mu = zeros(N,d);

    mu(1:N, 1) = musigma*randn(N,1);
    mu(1:(N/2), 1) = abs(mu(1:(N/2), 1));
    mu(((N/2)+1):end, 1) = -0.5*musigma - abs(mu(((N/2)+1):end, 1));
    
    
    
endswitch

%% Draw data

X = mu + sigma*randn(N,d);

%% Compute distances

xnorm = sum(X.^2,2);

distq = repmat(xnorm,1,N) + repmat(xnorm',N,1) -2*X*X';

%% Test

tests = (distq <= (2+t)*sigma^2*d);

tests = diag(1./sum(tests,2)) * tests; %normalize

%% Estimate

estim = tests*X ;

%% Compare results

naive = sum(sum((X-mu).^2))/(N*d) ;

stein = sum(sum((estim-mu).^2))/(N*d) ;


endfunction
