clear;
% clc;

% ——————————————————————————————————————————————————————————————————————— %
% Determination of the experimental setting
% ——————————————————————————————————————————————————————————————————————— %
% Add the path of auxiliary functions
addpath('./funcs');

% Set the random number generator
reset(RandStream.getGlobalStream);

% Set hyper-parameters for model
ps      = 6;                                                                % patch size
nlsp    = 10;                                                               % number of nonlocal patch
cls_num = 200;                                                              % number of class

% Set hyper-parameters for training set generation 
path_imgs     = '../data/FoE400/';                                          % path of source images
im_num        = 400;                                                        % number of training images  
flag_gen      = 0;                                                          % data generation flag: if true, generate data; otherwise, use the data generated in advance
Par.patsize   = ps;
Par.patnum    = nlsp;
Par.step      = 3;
Par.SearchWin = 15;     

% Set hyper-parameters for data divition            
batch_num = 10;                                                             % number of batch (this depends on your CPU memory)

% Set hyper-parameters for EM-GSMM (i.e., the parameter learning algorithm for GSMM)
iter_EM_MAX     = 50;                                                       % maximal iteration number of the outer loop (EM)
iter_Inner_max  = 1e2;                                                      % maximal iteration number of the inner loop (OptManiMulitBallGBB)
inner_criterion = 1e-5;                                                     % stoppint criterion for the inner loop

% Set hyper-parameters for OptManiMulitBallGBB
opts.record = 0;                                                            % record = 0, no print out
opts.mxitr  = 1e3;                                                          % max number of iterations
opts.gtol   = 1e-5;                                                         % stop control for the projected gradient
opts.xtol   = 1e-5;                                                         % stop control for ||X_k - X_{k-1}||
opts.tau    = 1e-3;     	

% Create a direction to store the trained model
dir_results = './result/';
if exist(dir_results, 'dir') == 0
    mkdir(dir_results);
end

% ——————————————————————————————————————————————————————————————————————— %
% Data Prepraration
% ——————————————————————————————————————————————————————————————————————— %
% Generate training set (Extract patch groups from the source images)
tic;
Path_TrainingSet = ['./TrainingSet_' num2str(im_num) '_ps' num2str(ps) '.mat'];
if (~flag_gen) && exist(Path_TrainingSet, 'file')
    load(Path_TrainingSet);
else
    for ii = 1:im_num
        Path_O_Img = strcat(path_imgs, sprintf('test_%03d.png', ii));
        im = double(imread(Path_O_Img));
        im = im/255; 
        if ii == 1
            [Neighbor_arr, Num_arr, Self_arr] =	NeighborIndex(im, Par);
            NL_mat_Total = zeros(Par.patnum, length(Num_arr), im_num);
            O_Img_Total  = zeros([size(im), im_num]);
            subs = Im2Patch(reshape(1:length(im(:)), size(im)), Par.patsize);
        end
        CurPat               = reshape(im(subs(:)),size(subs));
        O_Img_Total(:,:,ii)  = im;
        NL_mat_Total(:,:,ii) = Block_matching(CurPat, Par, Neighbor_arr, Num_arr, Self_arr);  
    end
    save(Path_TrainingSet, 'im_num', 'subs', 'O_Img_Total', 'NL_mat_Total');
end
toc;

% Set auxiliary variables used for extracting training batches     
im_batch   = im_num/batch_num;
im_len     = size(O_Img_Total,1) * size(O_Img_Total,2);
index_len  = size(NL_mat_Total,2) * im_batch;
subs_Batch = repmat(subs(:), [1, im_batch]);
subs_Batch = subs_Batch + (0: im_len: im_len*(im_batch-1));
subs_Batch = reshape(subs_Batch, [size(subs), im_batch]);

% ——————————————————————————————————————————————————————————————————————— %
% Model Training (Alg. 1: EM-GSMM)
% ——————————————————————————————————————————————————————————————————————— %
% ----------------------------------------------------------------------- %
% (Line 1 in Alg. 1): Initialize model parameters at random
% ----------------------------------------------------------------------- %
W_Total     = ones( cls_num, 1 );
M_Total     = zeros( nlsp*ps^2, cls_num );
U_Total     = rand( ps^2, ps^2, cls_num );
V_Total     = rand( nlsp, nlsp, cls_num );
Sigma_Total = ones( nlsp*ps^2, cls_num );

for kk = 1:cls_num
    U_Total(:,:,kk) = orth(U_Total(:,:,kk));
    V_Total(:,:,kk) = orth(V_Total(:,:,kk));
end

% ----------------------------------------------------------------------- %
% (Line 2 in Alg. 1): Start the outer loop
% ----------------------------------------------------------------------- %
f_obs       = zeros(1,iter_EM_MAX);
Nk_Record   = zeros( cls_num, iter_EM_MAX);
for iter_EM = 1:iter_EM_MAX
    iter_EM
    
    % ------------------------------------------------------------------- %
    % (Line 3 in Alg. 1) E-Step: Update responsibilities
    % ------------------------------------------------------------------- %
    tic;
    if iter_EM == 1  
        % At 1st interation, initialize responsibilities at random 
        flag = 0;
        while flag == 0
            idx       = zeros(cls_num,2);
            Mean      = zeros(ps^2 * nlsp,cls_num);
            Temp      = randsample(size(NL_mat_Total,2)*im_num, cls_num);
            idx(:, 1) = Temp - (ceil(Temp/size(NL_mat_Total,2))-1)*size(NL_mat_Total,2);
            idx(:, 2) = ceil(Temp/size(NL_mat_Total,2));
            for ii = 1:cls_num
                Temp_Img   = O_Img_Total(:,:,idx(ii,2));
                subs_Mean  = subs(:,NL_mat_Total(:, idx(ii,1), idx(ii,2)));
                Temp_PG    = Temp_Img(subs_Mean(:));
                Mean(:,ii) = Temp_PG - mean(Temp_PG,1); 
            end
        
            label_Total = zeros(size(NL_mat_Total,2)*im_num/batch_num, batch_num);
            for ii = 1:batch_num
                imgs = im_batch*(ii-1)+1 : im_batch*ii;
                PG   = extract_pg(subs_Batch, O_Img_Total(:,:,imgs), NL_mat_Total(:,:,imgs));
                PG   = PG - mean(PG,1);   % normalization
        
                [~,label]  = max(bsxfun(@minus,Mean'*PG,dot(Mean,Mean,1)'/2),[],1);
                label_Total(:, ii) = label';
            end
            label_Total       = label_Total(:)';
            [u,~,label_Total] = unique(label_Total);
            if cls_num == length(u)
                flag = 1;
            end
        end
        n = size(NL_mat_Total,2)*im_num;
        R = sparse(1:n,label_Total,1,n,cls_num,n);
        R = full(R');
    else
        % At other interations, calculate responsibilities as Eq. (22)
        R = zeros(cls_num, size(NL_mat_Total,2)*im_num);
        for ii = 1:batch_num
            pdf_Total = zeros(cls_num, size(NL_mat_Total,2)*im_batch);
            
            index = index_len*(ii-1)+1 : index_len*ii;
            imgs  = im_batch*(ii-1)+1 : im_batch*ii;
            PG    = extract_pg(subs_Batch, O_Img_Total(:,:,imgs), NL_mat_Total(:,:,imgs));
            PG    = PG - mean(PG,1);   % normalization
          
            PG_num = size(PG, 2);
            for kk = 1:cls_num
                % Opt: X - M
                means = M_Total(:,kk);
                Temp1 = PG - means(:);                              
                % Opt: U'(X - M)V
                Temp1 = reshape(Temp1, ps^2, nlsp*PG_num);
                Temp2 = U_Total(:,:,kk)' * Temp1;
                Temp2 = reshape(Temp2', nlsp, ps^2*PG_num);
                Temp1 = Temp2' * V_Total(:,:,kk);
                Temp1 = reshape(Temp1, PG_num, nlsp*ps^2);
                Temp1 = Temp1';
                
                Temp1 = (Temp1.^2) ./ (Sigma_Total(:,kk)+eps);
                Temp1 = sum(Temp1, 1);
                pdf_Total(kk,:) = -Temp1/2;
                pdf_Total(kk,:) = pdf_Total(kk,:) ...          
                                - sum(log(Sigma_Total(:,kk)+eps)/2); 
            end
            Temp           = max(pdf_Total);
            pdf_Total      = pdf_Total - Temp;
            pdf_Total      = exp(pdf_Total);
            R_part         = pdf_Total .* W_Total;
            f_obs(iter_EM) = f_obs(iter_EM) ...
                           + sum(log(sum(R_part,1)+eps) + Temp);
            R_part         = R_part./(sum(R_part,1)+eps);
            R(:, index)    = R_part;
        end
        f_obs(iter_EM)  = f_obs(iter_EM)/size(R,2);
    end
    toc;
    
    % ------------------------------------------------------------------- %
    % (Line 4 in Alg. 1) M-Step: Update mixing coefficients 
    % ------------------------------------------------------------------- %
    Nk      = sum(R, 2);
    W_Total = Nk/sum(Nk);
    Nk_Record(:,iter_EM) = Nk;
    
    % ------------------------------------------------------------------- %
    % (Optinal) M-Step: Update mean vectors ("zero-mean" is assmued here!)
    % ------------------------------------------------------------------- %
    M_Total     = zeros( nlsp*ps^2, cls_num );                              
    
    % ------------------------------------------------------------------- %
    % (Auxiliary): Prepare auxiliary variables for other steps 
    % ------------------------------------------------------------------- %
    tic;
    Mat_A_Total = zeros( nlsp*ps^2, nlsp*ps^2, cls_num );
    for ii = 1:batch_num
        index = index_len*(ii-1)+1 : index_len*ii;
        imgs  = im_batch*(ii-1)+1 : im_batch*ii;
        PG    = extract_pg(subs_Batch, O_Img_Total(:,:,imgs), NL_mat_Total(:,:,imgs));
        PG    = PG - mean(PG,1);  
        for kk = 1:cls_num
            means = M_Total(:,kk);
            Temp  = PG - means(:);                                          % Opt: X - M
            Temp  = Temp .* sqrt(R(kk,index));
            Temp  = Temp * Temp';
            Mat_A_Total(:,:,kk) = Mat_A_Total(:,:,kk) + Temp;
        end
    end
    for kk = 1:cls_num
        Mat_A_Total(:,:,kk) = Mat_A_Total(:,:,kk)/Nk(kk) ...            
                            + eye(size(Mat_A_Total,1))*(1e-6);              % add a prior for numerical stability  
    end
    toc;
    
    % ------------------------------------------------------------------- %
    % (Line 5 in Alg. 1): Start the inner loop
    % ------------------------------------------------------------------- %
    tic;
    parfor kk = 1:cls_num                                                   % use 'parfor' for acceleration 
        U         = U_Total(:,:,kk);
        V         = V_Total(:,:,kk);
        Sigma     = Sigma_Total(:,kk);
        Mat_A     = Mat_A_Total(:,:,kk);
        for iter_Inner = 1:iter_Inner_max
            % ----------------------------------------------------------- %
            % (Auxiliary): Record history result to check the stopping criterion
            % ----------------------------------------------------------- %
            Sigma_Old = Sigma;                          
            U_Old     = U;                              
            V_Old     = V;                              
            
            % ----------------------------------------------------------- %
            % (Line 6 in Alg. 1) M-Step: Update Sigma
            % ----------------------------------------------------------- %
            Temp  = kron( V, U );     
            Sigma = diag( Temp'*Mat_A*Temp );    
            Sigma = reshape(Sigma, ps^2, nlsp);   
            Sigma = Sigma + 1e-6;   %

            % ----------------------------------------------------------- %
            % (Line 7 in Alg. 1) M-Step: Update right-multiplying matrix U
            % ----------------------------------------------------------- %
            % Prepare auxiliary variables: Mat_V
            Mat_V = zeros(ps^2, ps^2, nlsp);
            for jj = 1:nlsp
                Temp  = kron( V(:,jj), eye(ps^2) );
                Mat_V(:,:,jj) = Temp'*Mat_A*Temp;
            end
            % Prepare auxiliary variables: A_U
            A_U = zeros(ps^2, ps^2, ps^2);
            for ii = 1:ps^2
                Temp        = reshape(Mat_V(:), ps^2*ps^2, nlsp)...
                            ./repmat(Sigma(ii,:)+eps, ps^2*ps^2, 1);
                A_U(:,:,ii) = reshape(sum(Temp,2), ps^2, ps^2);
            end
            % Optimize U using the optimizer 'OptStiefelGBB'
            [U, out_U] = OptStiefelGBB(U, @func_U, opts, A_U);

            % ----------------------------------------------------------- %
            % (Line 8 in Alg. 1) M-Step: Update left-multiplying matrix V
            % ----------------------------------------------------------- %
            % Prepare auxiliary variables: Mat_U
            Mat_U = zeros(nlsp, nlsp, ps^2);
            for ii = 1:ps^2
                Temp = kron( eye(nlsp), U(:,ii) );
                Mat_U(:,:,ii) = Temp'*Mat_A*Temp;
            end
            % Prepare auxiliary variables: A_V
            A_V = zeros(nlsp, nlsp, nlsp);
            for jj = 1:nlsp
                Temp        = reshape(Mat_U(:), nlsp*nlsp, ps^2)...
                            ./repmat(Sigma(:,jj)'+eps, nlsp*nlsp, 1);
                A_V(:,:,jj) = reshape(sum(Temp,2), nlsp, nlsp);
            end
            % Optimize V using the optimizer 'OptStiefelGBB'
            [V, out_V] = OptStiefelGBB(V, @func_V, opts, A_V);
            
            % ----------------------------------------------------------- %
            % (Auxiliary): Check the stopping criterion 
            % ----------------------------------------------------------- %
            err_Sigma = norm(Sigma_Old(:)-Sigma(:))/(eps+norm(Sigma_Old(:)));       
            err_U = norm(U_Old(:)-U(:))/(eps+norm(U_Old(:)));           
            err_V = norm(V_Old(:)-V(:))/(eps+norm(V_Old(:)));           
            if err_Sigma < inner_criterion ... 
            && err_U < inner_criterion ...
            && err_V < inner_criterion 
                break;
            end
        end
        % Update Sigma_Total, U_Total, V_Total
        Sigma_Total(:,kk) = Sigma(:);
        U_Total(:,:,kk)   = U;
        V_Total(:,:,kk)   = V;
    end 
    toc;

    % ----------------------------------------------------------- %
    % (Auxiliary): Save the intermediate results
    % ----------------------------------------------------------- %
    % Save results at each iteration
    path = [dir_results num2str(iter_EM, '%02d') '.mat'];
    save(path, 'W_Total', 'M_Total', 'Sigma_Total', 'U_Total', 'V_Total');
    % Save record
    save([dir_results 'record.mat'], 'Nk_Record', 'f_obs');
end

% Remove the path of auxiliary functions
rmpath('./funcs');

        