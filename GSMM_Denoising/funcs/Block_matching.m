function  [Init_Index]  =  Block_matching(X, nlsp, Neighbor_arr, Num_arr, SelfIndex_arr)
L           =   length(Num_arr);
Init_Index  =   zeros(nlsp,L);

for  i  =  1 : L
    Patch           = X(:,SelfIndex_arr(i));
    Neighbors       = X(:,Neighbor_arr(1:Num_arr(i),i));
    Dist            = sum((repmat(Patch,1,size(Neighbors,2))-Neighbors).^2);
    [~, index]      = sort(Dist);
    Init_Index(:,i) = Neighbor_arr(index(1:nlsp),i);
end
