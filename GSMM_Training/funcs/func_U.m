function [f, g] = func_U(U, A_Total)
%FUNC_U �˴���ʾ�йش˺����ժҪ
%   �˴���ʾ��ϸ˵��
    ps = sqrt(size(U,1));
    f  = 0;
    g  = zeros(size(U));
%     f  = gpuArray(0);
%     g  = gpuArray.zeros(size(U));
    for ii = 1:ps^2
        A       = A_Total(:,:,ii);
        f       = f + 0.5*U(:,ii)'*A*U(:,ii);
        g(:,ii) = A*U(:,ii);
    end
end

