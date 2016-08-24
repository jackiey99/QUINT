function P = BLin_W2P(W,flg)


%Setup Transition Matrix for random walk
%The diagnal of W is already set zero

%size(W)
W = (triu(W,1) + tril(W,-1));
if flg==0%row normalization and then transport
    D0 = (sparse(diag(sum(W,2))));
    [m,n] = size(W);
    for i=1:m
        if D0(i,i)>0
            D0(i,i) = 1/D0(i,i);
        end
    end
    %row normalization
    P = D0 * W;
    P = P';
elseif flg==-1 %row normalization and NO transport
    D0 = (sparse(diag(sum(W,2))));
    [m,n] = size(W);
    for i=1:m
        if D0(i,i)>0
            D0(i,i) = 1/D0(i,i);
        end
    end
    %row normalization
    P = D0 * W;
else


D0 = sum(W);
D0 = max(D0,0.00000000001);
W = sparse(W);

if min(D0)>0
    if flg==1%Graph Laplacian
        D = sparse(diag(D0.^(-0.5)));
        P = D * W * D;
    elseif flg==2%normalize row 1/2
        
        D1 = sparse(diag(sum(W).^0.5));%colum sum
        D2 = sparse(diag(sum(W,2).^(0.5)));%row sum
        W2 = D2 *  W * D1;
        
        D0 = inv(sparse(diag(sum(W2))));
        P = W2 * D0;
        
    elseif flg==3%normalize row -1/2      
        D1 = sparse(diag(sum(W).^(-0.5)));%colum sum
        D2 = sparse(diag(sum(W,2).^(-0.5)));%row sum
        P = D2 *  W * D1;
        
        %D0 = inv(sparse(diag(sum(W2))));
        %P = W2 * D0;
        
        %D2 = sparse(diag(sum(W,2).^(-1)));%row sum
        %W2 = D2 * W ;
        
        %D0 = inv(sparse(diag(sum(W2))));
        %P = W2 * D0;
    elseif flg==4    
        %Dc = sparse(diag(sum(W).^(-1)));%sum of column
        D0 = inv(sparse(diag(sum(W))));
        P = W * D0;
    elseif flg>=5&flg<=6% row normalized by d^
        coff = flg-5;
        D1 = sparse(diag(sum(W).^(-coff)));%colum sum
        D2 = sparse(diag(sum(W,2).^(-coff)));%row sum
        W2 = D2 *  W * D1;
        
        D0 = inv(sparse(diag(sum(W2))));
        P = W2 * D0;
    else%log(D+1)
        D2 = sum(W,2);%row sum
        [m,n] = size(W);
        for i=1:m
            for j=1:n
                W2(i,j) = W(i,j) /(log(D2(i)+1));
            end
        end
        D0 = inv(sparse(diag(sum(W2))));
        P = W2 * D0;
    end       
    
else%row by row
end
end
    