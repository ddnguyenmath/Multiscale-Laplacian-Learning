%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict the dataset
%	using pre-defined parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function predict(dataset,kernel_num)

	%% reading optimal paramters from data file
	opt_param_dir='opt_parameters';
	for i=0:kernel_num-1
		HermiteOrder = i;

		opt_file = strcat(dataset,'.t',int2str(i),'.params.opt');

		fid = fopen(strcat(opt_param_dir,'/',opt_file));
		data = textscan(fid,'%s %f','delimiter',' ');
		fclose(fid);
		h_NN 		=	data{2}(1);
		h_SIGMA 	= 	data{2}(2);
		h_gamma_A 	=	data{2}(3);
		h_gamma_I	=	data{2}(4);
		h_DEGREE 	=	data{2}(5);	 
		h_coeff 	= 	data{2}(6);
		options_h   =   generate_options(h_NN,h_SIGMA,h_gamma_A,h_gamma_I,h_DEGREE,HermiteOrder,h_coeff);
		opt(i+1)	= 	options_h;
		% opth = opt(i+1).HermiteOrder
	end

	results = mlap(dataset,opt,1);
	avg_error_lapSVM = mean(vertcat(results.err_svm));
	avg_error_lapRLS = mean(vertcat(results.err_rlsc));
					
	
	fprintf('LapSVM = %.3f\n',avg_error_lapSVM)
	fprintf('LapRLS = %.3f\n',avg_error_lapRLS)



function options=generate_options(NN,SIGMA,gamma_A,gamma_I,DEGREE,HermiteOrder,coeff)
	options=make_options;
	options.NN=NN;
	options.Kernel='rbf';
	options.KernelParam=SIGMA;
	% options.KernelParam=1;
	options.gamma_A=gamma_A; 
	options.gamma_I=gamma_I; 
	options.GraphWeights='heat';
	%options.GraphWeightParam='default';
	options.GraphWeightParam=SIGMA;
	options.LaplacianDegree=DEGREE;
	options.HermiteOrder = HermiteOrder;
	options.coeff = coeff;
	options.Verbose=0;
	% options.UseBias=1;
	% options.UseHinge=1;
	% options.GraphNormalize=0;


