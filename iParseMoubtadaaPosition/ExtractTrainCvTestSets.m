function [ x_train y_train x_cv y_cv x_test y_test ] = ExtractTrainCvTestSets( XY ,...
                                                   trainPourcentage,...
                                                   cvPourcentage,...
                                                   testPourcentage,...
                                                   withShuffle,...
                                                   DisplaySample)
% ExtractTrainCvTestSets Divide given data into train
% cross validation and test sets.
%
%   Proportion of each set is given in the argument.
%   If 'withShuffle' is eqaul to 1 then data wil be
%   shuffled. if DisplaySample is not equal to zero
%   the 'DisplaySample' rows of each data sets will 
%   be displayed.
%   
% Example:
%	u =
%	     6     8     8     4     1
%	     3     3     6     9     1
%	     8     8     8     6     1
%	     4    10     3     9     1
%	    10    10     1     5     1
%	     3     9     2     1     2
%	     8     6     7     5     2
%	     9     1     1     4     2
%	     5     1    10    10     2
%	     7     1    10     9     2
%	     
%	[ut vt ucv vcv utest vtest] = ExtractTrainCvTestSets(u, 60, 20, 20, 1, 2);
%	Train sample :
%	    10    10     1     5
%	     8     8     8     6
%	
%	     1
%	     1
%	
%	Cross Validation sample :
%	     9     1     1     4
%	     4    10     3     9
%	
%	     2
%	     1
%	
%	Test sample :
%	     8     6     7     5
%	     7     1    10     9
%	
%	     2
%	     2
%

[m n] = size(XY);

if(withShuffle == 1)
    index = randperm(m);
    XY = XY(index,:);
end

X = XY(:,1:n-1);
Y = XY(:,n);

limit_train_set = round(m * trainPourcentage / 100);
limit_cv_set = round(m * (trainPourcentage + cvPourcentage) / 100);

x_train = X(1:limit_train_set, :);
y_train = Y(1:limit_train_set);

x_cv = X(limit_train_set+1: limit_cv_set, :);
y_cv = Y(limit_train_set+1: limit_cv_set);

x_test = X(limit_cv_set+1: end, :);
y_test = Y(limit_cv_set+1: end);


if(DisplaySample ~= 0)    
    fprintf('Train sample :')
    x_train(1:DisplaySample,:)
    y_train(1:DisplaySample,:)

    fprintf('Cross Validation sample :')
    x_cv(1:DisplaySample,:)
    y_cv(1:DisplaySample,:)
    
    fprintf('Test sample :')
    x_test(1:DisplaySample,:)
    y_test(1:DisplaySample,:)
end




end