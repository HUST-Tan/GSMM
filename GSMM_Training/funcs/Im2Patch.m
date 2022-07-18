function  [Y]  =  Im2Patch( Img, patsize )
TotalPatNum = (size(Img,1)-patsize+1)*(size(Img,2)-patsize+1);      %Total Patch Number in the image
Y           =   zeros(patsize*patsize, TotalPatNum);      %Current Patches
% Y           =   zeros(patsize*patsize, TotalPatNum, 'single');      %Current Patches
k           =   0;

for i  = 1:patsize
    for j  = 1:patsize
        k       =  k+1;
        E_patch =  Img(i:end-patsize+i,j:end-patsize+j);       
        Y(k,:)  =  E_patch(:)';
    end
end