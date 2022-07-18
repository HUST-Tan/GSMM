function   [Px] =  Get_PG_Non_PG0( im, win, ps ,nlsp, step, type)
[h, w]  =  size(im);
S         =  win;
maxr         =  h-ps+1;
maxc         =  w-ps+1;
r         =  [1:step:maxr];
r         =  [r r(end)+1:maxr];
c         =  [1:step:maxc];
c         =  [c c(end)+1:maxc];
X = zeros(ps^2,maxr*maxc,'single');
Px = [];
if nlsp ==1
    k    =  0;
    for i  = 1:ps
        for j  = 1:ps
            k    =  k+1;
            blk     =  im(r-1+i,c-1+j);
            Px(k,:) =  blk(:)';
        end
    end
else
    k    =  0;
    for i  = 1:ps
        for j  = 1:ps
            k    =  k+1;
            blk  =  im(i:end-ps+i,j:end-ps+j);
            X(k,:) =  blk(:)';
        end
    end
    % Index image
    Index     =   (1:maxr*maxc);
    Index    =   reshape(Index, maxr, maxc);
    N1    =   length(r);
    M1    =   length(c);
    blk_arr   =  zeros(nlsp, N1*M1 );
    for  i  =  1 :N1
        for  j  =  1 : M1
            row     =   r(i);
            col     =   c(j);
            off     =  (col-1)*maxr + row;
            off1    =  (j-1)*N1 + i;
            
            rmin    =   max( row-S, 1 );
            rmax    =   min( row+S, maxr );
            cmin    =   max( col-S, 1 );
            cmax    =   min( col+S, maxc );
            
            idx     =   Index(rmin:rmax, cmin:cmax);
            idx     =   idx(:);
            neighbor       =   X(:,idx);
            seed       =   X(:,off);
            
            dis     =   (neighbor(1,:) - seed(1)).^2;
            for k = 2:ps^2
                dis   =  dis + (neighbor(k,:) - seed(k)).^2;
            end
            dis = dis./ps^2;
            [~,ind]   =  sort(dis);
            indc        =  idx( ind( 1 : nlsp ) );
            blk_arr(:,off1)  =  indc;
            X_nl = X(:,indc); % or X_nl = neighbor(:,ind( 1 : nlsp ));
            % Removes DC component from image patch group
            if type == 1
                % - Way_1 - 
                DC = mean(X_nl,2);
                X_nl = bsxfun(@minus, X_nl, DC);
            else
                % - Way_2 -
                DC = mean(X_nl(:));
                X_nl = X_nl - DC;
            end
            Px = [Px X_nl];
%             % Select the smooth patches
%             sv=var(X_nl);
%             if max(sv) <= delta
%                 Px0 = [Px0 X_nl];
%             else
%                 Px = [Px X_nl];
%             end
        end
    end
end