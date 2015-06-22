function [ Z eigenvalue Partial_retained_variance retained_variance ] = Extract_PCA( XY, K )


[m n] = size(XY);
X = XY(:,1:n-1);
Y = XY(:,n);
sigma =  (1/m)* X'*X;
[U S V] = svd(sigma); 
[l c] = size(S);
col = S(:,1:K);
eigenvalue =sum(S(:,K));
Partial_retained_variance = sum(S(:,K))/sum(S(:))*100;
retained_variance = sum(col(:))/sum(S(:))*100;
U_red = U(:,1:K);
Z = X * U_red;
Z = [Z Y];
end

