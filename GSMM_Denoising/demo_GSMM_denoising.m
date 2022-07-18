clear;
close all

% Add the path of auxiliary functions
addpath('./funcs');

% Set the random number generator    
reset(RandStream.getGlobalStream);

% Determine the noise level for this experiment
nSig = 25/255;   % noise level

% Load the test image and Genenrate the corresponding noisy image
path_Img = '../data/Set10/Monarch_full.png';
O_Img    = imread(path_Img);
O_Img    = double(O_Img)/255;
N_Img    = O_Img + nSig * randn(size(O_Img)); 

% Print the psnr of the noisy image to be processed
PSNR  = csnr( O_Img*255, N_Img*255, 0, 0 );
fprintf( 'Noisy Image: nSig = %2.3f, PSNR = %2.2f \n', nSig*255, PSNR );  

% Set parameters for the GSMM denoising algorithm
norm = 'L2';        % Select the norm type: 'L1' (GSMM_l1) or 'L2' (GSMM_LF)
Par  = setpar(nSig, norm);

% Denoise the image with the prior learned by GSMM
Par.O_Img = O_Img;  % Uncomment to calculate PSNRs for intermediate results
E_Img = GSMM_denoising(N_Img, Par);

% Print information of processed result
E_Img(E_Img<0) = 0;
E_Img(E_Img>1) = 1;
PSNR  = csnr( O_Img*255, E_Img*255, 0, 0 );
fprintf( 'Estimated Image: PSNR = %2.2f \n\n', PSNR ); 

% Store the denoised result in PNG format
imwrite(uint8(E_Img*255), 'img_denoised.png');

% Remove the path of auxiliary functions
rmpath('./funcs');