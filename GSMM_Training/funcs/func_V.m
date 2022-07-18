function [f, g] = func_V(V, A_Total)
%FUNC_U �˴���ʾ�йش˺����ժҪ
%   �˴���ʾ��ϸ˵��
    nlsp = size(V,1);
    f    = 0;
    g    = zeros(size(V));
%     f  = gpuArray(0);
%     g  = gpuArray.zeros(size(V));
    for jj = 1:nlsp
        A       = A_Total(:,:,jj);
        f       = f + 0.5*V(:,jj)'*A*V(:,jj);
        g(:,jj) = A*V(:,jj);
    end
end

