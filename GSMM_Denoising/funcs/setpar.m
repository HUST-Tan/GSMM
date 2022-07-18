function Par = setpar(nSig, norm)
%This fucntion selects the GSMM model and determines the parameters for
%the GSMM denoising algorithm.

% Set the noise level and the norm type according the input variable
Par.nSig = nSig;
Par.norm = norm;

% Set the parameters used for accelaration
Par.k_acc  = 20;
Par.nl_acc = 200;

% Set the patch size and the parameter 'c' in Eq. (52)
if nSig < 20/255
    Par.c  = 1.2;
    Par.ps = 6;
elseif nSig < 40/255
    Par.c  = 1.1;
    Par.ps = 7;
else
    Par.c  = 1;
    Par.ps = 9;
end

% Load the trained model and extract its hyper-parameters
path_model = sprintf('./models/win15_step3_%dx%d_nlsp10_cls200_TrainedOn_FoE400.mat', Par.ps, Par.ps);
Par.model  = load(path_model);
Par.patnum = size(Par.model.V_Total, 1);
Par.clsnum = size(Par.model.U_Total,3);

% Set the parameters for searching non-local similar patches
Par.SearchWin = 15;
Par.step      = 3;

% Set parameters for the Half-Quadratic-Split(HQS) Method (Alg. 2)
Par.iter   = 6;
Par.lambda = 1 * (1/nSig^2);
Par.betas  = [1 4 8 16 32 64 128] * (1/nSig^2);

% Set mask for L1 norm (This is used to keep coefficients in the first column unchanged.)
mask = zeros(Par.ps^2, Par.patnum-1);
mask = padarray(mask, [0 1], 1, 'pre');
Par.mask = mask(:);

end

