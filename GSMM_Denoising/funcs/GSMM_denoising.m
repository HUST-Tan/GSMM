function [E_Img, psnrs] = GSMM_denoising(N_Img, Par)
%This fucntion denoises the input noisy image with the prior learned by GSMM.
    
    [Height, Width] = size(N_Img);

    % Load model parameters of GSMM
    M_Total       = Par.model.M_Total;
    U_Total       = Par.model.U_Total;
    V_Total       = Par.model.V_Total;
    Sigma_Total   = Par.model.Sigma_Total;
    
    % Determine neighbor patches to be searched 
    [Neighbor_arr, Num_arr, Self_arr] =	NeighborIndex(N_Img, Par);

    % Iterations for Half Quadratic Split (HQS) method (Alg. 2)
    E_Img  = N_Img;
    psnrs  = zeros(Par.iter, 2);
    k_cand = ones(Par.clsnum, length(Num_arr));
    for iter_HQS = 1:Par.iter
        % ——————————————————————————————————————————————————————————————— %
        % Step 1 (Line 3 in Alg. 2): Update beta
        % ——————————————————————————————————————————————————————————————— %
        beta = Par.betas(iter_HQS);
        nSig = sqrt(1/beta);

        % ——————————————————————————————————————————————————————————————— %
        % Step 2 (Line 4 in Alg. 2): Extract patch groups
        % ——————————————————————————————————————————————————————————————— %
        % 2-1: Convert the noisy image to patches
        CurPat = Im2Patch( E_Img, Par.ps );

        % 2-2: Find Non-Local similar patches for each patch
        if mod(iter_HQS-1, 1) == 0
        NL_mat = Block_matching(CurPat, Par.nl_acc, Neighbor_arr, Num_arr, Self_arr);
        end
        if iter_HQS == 1
            Neighbor_arr = NL_mat;
            Num_arr      = Par.nl_acc * ones(1, length(Num_arr));
        end

        % ——————————————————————————————————————————————————————————————— %
        % Step 3 (Line 5 in Alg. 2): Denoise patch groups
        % ——————————————————————————————————————————————————————————————— %
        % 3-1: Remove DC component
        NL_mat = NL_mat(1:Par.patnum,:);
        PG     = CurPat(:, NL_mat(:)); 
        PG     = reshape(PG, size(PG,1)*Par.patnum, size(PG,2)/Par.patnum);
        DC     = mean(PG,1);
        PG     = PG - DC;

        % 3-2: Do classification every 3 iterations
        if mod(iter_HQS-1, 3) == 0
            Lambda_Total = 1./(Sigma_Total+nSig^2);
            pdf_Total    = zeros(Par.clsnum, length(Num_arr));
            for kk = 1:Par.clsnum
                index  = find(k_cand(kk,:) == 1);
                PG_num = length(index);
                if isempty(index)
                    continue;
                end
                % Opt: X - M
                means = M_Total(:,kk);
                Temp1 = PG(:, index) - means(:);                   
                % Opt: U'(X - M)V
                Temp1 = reshape(Temp1, Par.ps^2, Par.patnum*PG_num);
                Temp2 = U_Total(:,:,kk)' * Temp1;
                Temp2 = reshape(Temp2', Par.patnum, Par.ps^2*PG_num);
                Temp1 = Temp2' * V_Total(:,:,kk);
                Temp1 = reshape(Temp1, PG_num, Par.patnum*Par.ps^2);
                % Opt: (U'(X - M)V).^2
                Temp1 = Temp1.^2;
                
                Temp1 = Temp1 * Lambda_Total(:,kk);
                Temp1 = Temp1';
                pdf_Total(kk,index) = - Temp1/2 ...          
                                      - sum(log(Sigma_Total(:,kk)+nSig^2)/2);
            end
            [~, k_opt]  = sort(pdf_Total, 1, 'descend');
            sub         = k_opt(1:Par.k_acc,:) + (0:1:length(Num_arr)-1)*Par.clsnum;
            k_cand      = zeros(Par.clsnum, length(Num_arr)); 
            k_cand(sub) = 1;
            k_opt       = k_opt(1,:);   
        end

        % 3-3: Denoise patch groups
        subs_orign = Im2Patch( reshape(1:Height*Width, Height, Width), Par.ps );
        for kk = 1:Par.clsnum
            index = find(k_opt == kk);
            if isempty(index)
                continue;
            end
            % Do transformation
            means = M_Total(:,kk);                  
            Temp  = kron(V_Total(:,:,kk), U_Total(:,:,kk)); % Opt: V @ U
            alpha = Temp' * (PG(:, index) - means);         % Opt: (V @ U)'*vec(X-M) = U'(X - M)V                       
            % Shrink coefficients
            switch Par.norm
                case 'L1'
                    a_old  = alpha;
                    weight = sqrt(Sigma_Total(:,kk)) + eps;
                    weight = Par.c*(nSig^2)./weight;
                    alpha  = sign(alpha).*max((abs(alpha) - weight), 0);
                    alpha  = alpha .* (1-Par.mask) + a_old .* Par.mask;
                case 'L2'
                    weight = Sigma_Total(:,kk)./(nSig^2 + Sigma_Total(:,kk));
                    alpha  = alpha .* weight;
                otherwise
                    fprintf('Please set the norm type as L1 or L2!');
            end
            % Do inverse transformation
            PG(:, index) = Temp * alpha + means;
        end

        % 3-4: Add DC component back
        PG = PG + DC;

        % ——————————————————————————————————————————————————————————————— %
        % Step 4 (Line 6 in Alg. 2): Aggregate denoised patch groups
        % ——————————————————————————————————————————————————————————————— %
        subs  = subs_orign(:, NL_mat(:));
        W_Img = accumarray(subs(:), 1);
        W_Img = reshape(W_Img, Height, Width);
        I_Img = accumarray(subs(:), PG(:));
        I_Img = reshape(I_Img, Height, Width);
        I_Img = I_Img ./ W_Img;

        % ——————————————————————————————————————————————————————————————— %
        % Step 5 (Line 7 in Alg. 2): Add filted residule back (Iterative Regularization)
        % ——————————————————————————————————————————————————————————————— %
        E_Img = (beta*I_Img + Par.lambda*N_Img)./(beta+Par.lambda+eps);
        
        % ——————————————————————————————————————————————————————————————— %
        % Record the intermediate results (optinal)
        % ——————————————————————————————————————————————————————————————— %
        if isfield(Par, 'O_Img')
            % Calculate and Show PNSR result
            psnrs(iter_HQS,1) = csnr( I_Img*255, Par.O_Img*255, 0, 0 );
            psnrs(iter_HQS,2) = csnr( E_Img*255, Par.O_Img*255, 0, 0 );
            fprintf('Iter %d: I_Img PSNR:%2.2f, E_Img PSNR is:%2.2f \n',...
                     iter_HQS, psnrs(iter_HQS,1), psnrs(iter_HQS,2));
        end
    end
end

