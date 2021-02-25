function comparison_exp(experiment, ntrials=20, generate_table_line = true)
  %%
  %% Main script for experiments. Currently 4 experimental settings
  %% available, switch with first argument.
  %%
  %% "UNIF.1","UNIF.2","UNIF.3": UNIF setting with d=
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
  gammamax = 0.2;

  %% Range of search for test factor treshold will be 2+trange

  trange = 0:0.1:4;
  
  %% Range of tuning [shrinking] parameter (will be sometimes appropriately scaled
  %% for appropriate range)
  
  c = 0:0.05:2;

  switch  experiment  % adjustment to search range in some cases to
		      % avoid optimum at upper bound
	  
    case "SPHERE.0"
      trange = [ 0:0.1:2, 2:0.2:8] ;
      c = [ 0:0.025:1 , 1.05:0.05:2, 2.1:0.1:4];

    case "SPHERE.a"
      trange = [ 0:0.1:2, 2:0.2:8] ;
      c = [ 0:0.025:1 , 1.05:0.05:2, 2.1:0.1:4 4.5:0.5:10];

    case "UNIF.0"
      c = [ 0:0.025:1 , 1.05:0.05:2, 2.1:0.2:10];


  end;
  
  
  
  cumul_results_mtt_cst = zeros(length(c),1);
  cumul_results_stb_zero = zeros(length(trange),1);
  cumul_results_mtt_stb= zeros(length(c),1);
  cumul_results_stb_shrink = zeros(length(c),1);
  cumul_results_stb_theory = zeros(length(c),1);

  results_stein = 0;

  for (trial = 1:ntrials)
    
    switch  experiment

      case "UNIF.0"
	
	N = 2000;
	d = 100;
	mu = generate_means("MODEL",3,"MUSPREAD",20,"N",N,"MUDIM",10,"D",d,"SHIFT",1) ;
	model = "UNIF";
      

    case "UNIF.1"
           
      N = 2000;
      d = 250;
      mu = generate_means("MODEL",3,"MUSPREAD",20,"N",N,"MUDIM",10,"D",d,"SHIFT",1) ;
      model = "UNIF";

    case "UNIF.2"

      N = 2000;
      d = 500;
      mu = generate_means("MODEL",3,"MUSPREAD",20,"N",N,"MUDIM",10,"D",d,"SHIFT",1) ;
      model = "UNIF";
      
    case "UNIF.3"

      N = 2000;
      d = 1000;
      mu = generate_means("MODEL",3,"MUSPREAD",20,"N",N,"MUDIM",10,"D",d,"SHIFT",1) ;
      model = "UNIF";
      

    case "CLUSTER.1"

      N = 2000;
      d = 25;
      mu = generate_means("MODEL",5,"NCLUSTERS",20,"MUSPREAD",0.1,
			  "CLUSTERSPREAD",1,"MUDIM",d,"D",d,"N",N);
      model = "CLUSTER"; 

      
    case "CLUSTER.2"

      N = 2000;
      d = 50;
      mu = generate_means("MODEL",5,"NCLUSTERS",20,"MUSPREAD",0.1,
			  "CLUSTERSPREAD",1,"MUDIM",d,"D",d,"N",N);
      model = "CLUSTER"; 

      
    case "CLUSTER.3"

      N = 2000;
      d = 100;
      mu = generate_means("MODEL",5,"NCLUSTERS",20,"MUSPREAD",0.1,
			  "CLUSTERSPREAD",1,"MUDIM",d,"D",d,"N",N);
      model = "CLUSTER";

    case "CLUSTER.4"

      N = 2000;
      d = 250;
      mu = generate_means("MODEL",5,"NCLUSTERS",20,"MUSPREAD",0.1,
			  "CLUSTERSPREAD",1,"MUDIM",d,"D",d,"N",N);
      model = "CLUSTER"; 
      
    case "SPHERE.a"
      
      N = 2000;
      d = 50;
      mu = generate_means("MODEL",7,"MUDIM",5,"MUSPREAD",50,"N",N,"D",d);
      model = "SPHERE";
      
      
    case "SPHERE.0"

      N = 2000;
      d = 100;
      mu = generate_means("MODEL",7,"MUDIM",5,"MUSPREAD",50,"N",N,"D",d);
      model = "SPHERE";

      
    case "SPHERE.1"

      N = 2000;
      d = 250;
      mu = generate_means("MODEL",7,"MUDIM",5,"MUSPREAD",50,"N",N,"D",d);
      model = "SPHERE";

    case "SPHERE.2"
	    
      N = 2000;
      d = 500;
      mu = generate_means("MODEL",7,"MUDIM",5,"MUSPREAD",50,"N",N,"D",d);
      model = "SPHERE";

    case "SPHERE.3"

      N = 2000;
      d = 1000;
      mu = generate_means("MODEL",7,"MUDIM",5,"MUSPREAD",50,"N",N,"D",d);
      model = "SPHERE";

    case "SPARSE.1"
      
      N = 2000;
      d = 25;
      mu = generate_means("MODEL",4,"MUDIM",2,"MUSPREAD",10,"SHIFT",1,"D",d,"N",N) ;
      model = "SPARSE";

    case "SPARSE.2"
      
      N = 2000;
      d = 50;
      mu = generate_means("MODEL",4,"MUDIM",2,"MUSPREAD",10,"SHIFT",1,"D",d,"N",N) ;
      model = "SPARSE";
      
    case "SPARSE.3"
      
      N = 2000;
      d = 100;
      mu = generate_means("MODEL",4,"MUDIM",2,"MUSPREAD",10,"SHIFT",1,"D",d,"N",N) ;
      model = "SPARSE";

    case "SPARSE.4"
      
      N = 2000;
      d = 250;
      mu = generate_means("MODEL",4,"MUDIM",2,"MUSPREAD",10,"SHIFT",1,"D",d,"N",N) ;
      model = "SPARSE";


    case "SPARSE.5"
      
      N = 2000;
      d = 500;
      mu = generate_means("MODEL",4,"MUDIM",2,"MUSPREAD",10,"SHIFT",1,"D",d,"N",N) ;
      model = "SPARSE";

    case "SPARSE.6"
      
      N = 2000;
      d = 1000;
      mu = generate_means("MODEL",4,"MUDIM",2,"MUSPREAD",10,"SHIFT",1,"D",d,"N",N) ;
      model = "SPARSE";

    case "SPARSE.Z"
      
      N = 2000;
      d = 200;
      mu = generate_means("MODEL",4,"MUDIM",2,"MUSPREAD",20,"SHIFT",1,"D",d,"N",N) ;
      model = "SPARSE";
      

    case "MIX_CLUSTER_SPARSE"
      
      N = 2001;
      d = 100;

      mu1 = generate_means("MODEL",4,"MUDIM",2,"MUSPREAD",10,"SHIFT",1,"D",d,"N",1000) ;

      mu2 = generate_means("MODEL",5,"NCLUSTERS",20,"MUSPREAD",0.1,
			  "CLUSTERSPREAD",1,"MUDIM",d,"D",d,"N",1001);

      mu = [ mu1-2 ; mu2+2];
      
      model = "MIX_CLUSTER_SPARSE";

    case "MIX_SPHERE_UNIF"
      
      N = 2001;
      d = 250;

      mu1 = generate_means("MODEL",3,"MUSPREAD",20,"N",1000,"MUDIM",10,"D",d) ;
      mu2 = generate_means("MODEL",7,"MUDIM",5,"MUSPREAD",50,"N",1001,"D",d);

      mu = [ mu1-2 ; mu2+2];
      
      model = "MIX_SPHERE_UNIF";

      c=2*c;

    case "MIX_CLUSTERS"

      N = 2001;
      d = 50;
      mu1 = generate_means("MODEL",5,"NCLUSTERS",20,"MUSPREAD",0.1,
			   "CLUSTERSPREAD",1,"MUDIM",d,"D",d,"N",1000);
      mu2 = generate_means("MODEL",5,"NCLUSTERS",200,"MUSPREAD",0.1,
			  "CLUSTERSPREAD",2,"MUDIM",d,"D",d,"N",1001);

      mu = [mu1-1;mu2+1];
      
      model = "MIX_CLUSTERS";
      
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

  %%% special treatment for stb_zero: its minimum error will determine
  %% test threshold for all other methods

  
  results_stb_zero = zeros(length(trange),1);

  for (i = 1:length(trange))
    
    estim = stb_shrink_means(X,trange(i),sigma,0,distsq);
    results_stb_zero(i) = sum(sum((estim-mu).^2))/(N*d);
    
  endfor


  
%%% now select the t that optimizes stb-0 above for remaining methods

  [mini, imini] = min(results_stb_zero);

  tselect = trange(imini);   % the value used for the threshold factor
			% will be 2+tselect
  
%%%
  
  results_mtt_stb= zeros(length(c),1);

  for (i = 1:length(c))

    estim = mtt_stb(X,tselect,sigma,c(i),distsq);
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
  results_stein = results_stein + sum(sum((estim-mu).^2))/(N*d);
    

  cumul_results_mtt_cst = cumul_results_mtt_cst + results_mtt_cst/ntrials;
  cumul_results_stb_zero = cumul_results_stb_zero + results_stb_zero/ntrials;
  cumul_results_mtt_stb = cumul_results_mtt_stb + results_mtt_stb/ntrials;
  cumul_results_stb_shrink = cumul_results_stb_shrink + results_stb_shrink/ntrials;
  cumul_results_stb_theory = cumul_results_stb_theory + results_stb_theory/ntrials;

  endfor

  

  if (generate_table_line)

    %printf("Table line for AISTATS paper:\n\n")
    
    printf(" %u & {\\bf %s} & %2.3f & %2.3f & %2.3f & %2.3f & %2.3f & %2.3f \\\\ \n\n",...
	   d, model,...
	   results_stein/ntrials,...
	   min(cumul_results_mtt_cst), min(cumul_results_mtt_stb),...
	   min(cumul_results_stb_zero), min(cumul_results_stb_theory),...
	   min(cumul_results_stb_shrink));
	
  endif

  if ( (min(cumul_results_mtt_cst) == cumul_results_mtt_cst(1)))
    printf("Warning: Minimum attained at lower search boundary for MTT-cst\n");
  endif;

  if ( (min(cumul_results_mtt_cst) == cumul_results_mtt_cst(end)))
    printf("Warning: Minimum attained at upper search boundary for MTT-cst\n");
  endif;


  if ( (min(cumul_results_mtt_stb) == cumul_results_mtt_stb(1)))
    printf("Warning: Minimum attained at lower search boundary for MTT-stb\n");
  endif;

  if ( (min(cumul_results_mtt_stb) == cumul_results_mtt_stb(end)))
    printf("Warning: Minimum attained at upper search boundary for MTT-stb\n");
  endif;
       

  if ( (min(cumul_results_stb_zero) == cumul_results_stb_zero(1)))
    printf("Warning: Minimum attained at lower search boundary for\
 STB-0 [threshold factor used is exactly 2]\n");
  endif;

  if ( (min(cumul_results_stb_zero) == cumul_results_stb_zero(end)))
    printf("Warning: Minimum attained at upper search boundary for\
 STB-0 [determines test threshold factor]\n");
  endif;

  
  if ( (min(cumul_results_stb_theory) == cumul_results_stb_theory(1)))
    printf("Warning: Minimum attained at lower search boundary for\
 STB-theory [-> identical to STB-0]\n");
  endif;

  if ( (min(cumul_results_stb_theory) == cumul_results_stb_theory(end)))
    printf("Warning: Minimum attained at upper search boundary for STB-theory\n");
  endif;


  if ( (min(cumul_results_stb_shrink) == cumul_results_stb_shrink(1)))
    printf("Warning: Minimum attained at lower search boundary for\
 STB-weight [->identical to STB-0]\n");
  endif;

  if ( (min(cumul_results_stb_shrink) == cumul_results_stb_shrink(end)))
    printf("Warning: Minimum attained at upper search boundary for STB-weight\n");
  endif;



  ## printf("[Model %s]\n",model)
  ##   printf("PP-James-Stein:")
  ## printf("%2.2f \n", results_stein);
  ## printf("MTT-constant:")
  ## printf("%2.2f ", cumul_results_mtt_cst);
  ## printf("\n")
  ## if 
  ## printf("MTT-stb:")
  ## printf("%2.2f ", cumul_results_mtt_stb);
  ## printf("\n")
  ## printf("STB-0:");
  ## printf("%2.2f ", cumul_results_stb_zero);
  ## printf("\n")
  ## printf("STB-theory:")
  ## printf("%2.2f ", cumul_results_stb_theory);
  ## printf("\n")
  ## printf("\STB-weight:")
  ## printf("%2.2f ", cumul_results_stb_shrink);
  ## printf("\n")

  
endfunction
  
