function comparison_exp(experiment, ntrials=20, generate_table_line = true)
  %%
  %% Main script for experiments. Currently 4 experimental settings
  %% available, switch with first argument.
  %%
  %% 1: UNIF setting
  %% 2: CLUSTER setting
  %% 3: SPHERE setting
  %% 4: SPARSE setting
  %%
  %% generate_table_line = true to generate corresponding line
  %% reported in AISTATS paper
  %%
  %%
  
  sigma = 1;
  N = 2000;
  d = 1000;
  gammamax = 0.3;

  %% Range of tuning parameter (will be sometimes appropriately scaled
  %% for appropriate range)
  
  c = 0:0.05:1;

  
  cumul_results_mtt_cst = zeros(length(c),1);
  cumul_results_stb_zero = zeros(length(c),1);
  cumul_results_mtt_stb= zeros(length(c),1);
  cumul_results_stb_shrink = zeros(length(c),1);
  cumul_results_stb_theory = zeros(length(c),1);

  results_stein = 0;

  for (trial = 1:ntrials)
    
  switch  experiment

    case 1

      N = 2000;
      mu = generate_means("MODEL",3,"MUSPREAD",20,"N",N,"SHIFT",0,"MUDIM",10) ;
      model = "UNIF";
      

    case 2

      N = 2000;
      mu = generate_means("MODEL",5,"NCLUSTERS",20,"MUSPREAD",0.1,
			  "CLUSTERSPREAD",1,"MUDIM",d,"N",N);
      model = "CLUSTER"; 

    case 3

      N = 2000;
      mu = generate_means("MODEL",7,"MUDIM",5,"MUSPREAD",50,"N",N);
      model = "SPHERE";

    case 4
      
      N = 2000;
      d = 50;
      mu =   generate_means("MODEL",4,"MUDIM",2,"MUSPREAD",10,"SHIFT",1,"D",50,"N",N);
      model = "SPARSE";
      
  endswitch

  %% generate data
  
  X = mu + sigma*randn(N,d);

  %% compute sq distances once and for all

  xnorm = sum(X.^2,2);    
  distsq = repmat(xnorm,1,N) + repmat(xnorm',N,1) - 2*X*X';




  results_mtt_cst = zeros(length(c),1);

  for (i = 1:length(c))

    estim = mtt_cst(X,c(i));
    results_mtt_cst(i) = sum(sum((estim-mu).^2))/(N*d);
    
  endfor
  
  %% results_stb_aggregate = zeros(length(c),1);

  %% for (i = 1:length(c))
    
  %%   estim = aggregate_means(X,2*c(i),sigma,distsq);
  %%   results_stb_aggregate(i) = sum(sum((estim-mu).^2))/(N*d);
    
  %% endfor

  results_stb_zero = zeros(length(c),1);

  for (i = 1:length(c))
    
    estim = stb_shrink_means(X,2*c(i),sigma,0,distsq);
    results_stb_zero(i) = sum(sum((estim-mu).^2))/(N*d);
    
  endfor


  
%%% now select the t that optimizes stb-0 above for remaining methods

  [mini, imini] = min(results_stb_zero);

  tselect = 2*c(imini);   %% the value used for the threshold factor is 2+t
  
%%%
  
  results_mtt_stb= zeros(length(c),1);

  for (i = 1:length(c))

    estim = mtt_stb(X,tselect,sigma,c(i)*0.2,distsq);
    results_mtt_stb(i) = sum(sum((estim-mu).^2))/(N*d);
    
  endfor

  results_stb_theory = zeros(length(c),1);
  
  for (i = 1:length(c))
    
    estim = stb_shrink_means(X,tselect,sigma,c(i)*gammamax,distsq,"theory");
    results_stb_theory(i) = sum(sum((estim-mu).^2))/(N*d);
    
  endfor

  
  results_stb_shrink = zeros(length(c),1);

  for (i = 1:length(c))

    estim = stb_shrink_means(X,tselect,sigma,c(i)*gammamax,distsq,"shrink");
    results_stb_shrink(i) = sum(sum((estim-mu).^2))/(N*d);
    
  endfor

  estim = basic_stein(X,0,1);
  results_stein = results_stein + sum(sum((estim-mu).^2))/(N*d*ntrials);
    

  cumul_results_mtt_cst = cumul_results_mtt_cst + results_mtt_cst/ntrials;
  cumul_results_stb_zero = cumul_results_stb_zero + results_stb_zero/ntrials;
  cumul_results_mtt_stb = cumul_results_mtt_stb + results_mtt_stb/ntrials;
  cumul_results_stb_shrink = cumul_results_stb_shrink + results_stb_shrink/ntrials;
  cumul_results_stb_theory = cumul_results_stb_theory + results_stb_theory/ntrials;

  endfor


  if (generate_table_line)

    printf("Table line for AISTATS paper (revision incl. PPJS):\n\n")
    
    printf("{\\bf %s} & %2.3f & %2.3f & %2.3f & %2.3f & %2.3f & %2.3f\n\n",model,...
	   1-results_stein,...
	   1-min(cumul_results_mtt_cst), 1-min(cumul_results_mtt_stb),...
	   1-min(cumul_results_stb_zero), 1-min(cumul_results_stb_theory),...
	   1-min(cumul_results_stb_shrink));
	
  endif

  printf("[Model %s]\n",model)
    printf("PP-James-Stein:")
  printf("%2.2f \n", results_stein);
  printf("MTT-constant:")
  printf("%2.2f ", cumul_results_mtt_cst);
  printf("\n")
  printf("MTT-stb:")
  printf("%2.2f ", cumul_results_mtt_stb);
  printf("\n")
  printf("STB-0:");
  printf("%2.2f ", cumul_results_stb_zero);
  printf("\n")
  printf("STB-theory:")
  printf("%2.2f ", cumul_results_stb_theory);
  printf("\n")
  printf("\STB-weight:")
  printf("%2.2f ", cumul_results_stb_shrink);
  printf("\n")

  
endfunction
  
