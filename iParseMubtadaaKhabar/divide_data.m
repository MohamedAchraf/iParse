function [x_dev y_dev x_cv y_cv x_test y_test ] = divide_data(x, y)

[m,n] = size(x);

cv_index = round(m*.6)+1;
test_index = round(m*.8);

id = randperm(m);
xy = [x y];

xy_shuffle = xy(id,:);

x_dev = xy_shuffle(1:cv_index-1,1:n);
y_dev = xy_shuffle(1:cv_index-1,end);

x_cv = xy_shuffle(cv_index:test_index,1:n);
y_cv = xy_shuffle(cv_index:test_index,end);

x_test = xy_shuffle(test_index+1:end,1:n);
y_test = xy_shuffle(test_index+1:end,end);




end

