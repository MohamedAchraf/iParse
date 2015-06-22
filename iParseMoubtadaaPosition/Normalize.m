function [ XY_normal ] = Normalize( XY , b )
% Normalization of a given matrix.
%
%   Syntaxe : A = Normalize(B, n)
%
%   If 'n' is not equal to zero then a
%   sample of n rows will be displayed.
%
% Example:
%     A =
%          1     9     2
%          3     1     9
%          6    10     9
% 
%     Normalize(A, 2);
% 
%        -0.1325   -1.1488    9.0000
%         1.0596    0.6757    9.0000
    
[m n] = size(XY);
X = XY(:,1:n-1);
Y = XY(:,n);

mu = mean(X);
sigma = std(X);

for i=1:size(X,2)
    X(:,i) = ( X(:,i) - mu(i)) ./ sigma(i);
end

XY_normal = [X Y];

if( b ~= 0  )
    index = randsample(1:length(XY_normal), b);
    XY_normal(index,:);
end
end

