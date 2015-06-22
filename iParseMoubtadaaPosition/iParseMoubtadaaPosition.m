%% =================== 1- INITIALIZATION ===================
fprintf('[INFO]\tINITIALIZATION...\n');
close all; clear all; clc;
set(0, 'defaultfigurewindowstyle', 'docked');
warning('off','all');
%% ============ 2- DATA LOAD AND NORMALIZATION =============
fprintf('[INFO]\tDATA LOAD AND NORMALIZATION...\n');
XY = load('iParseMoubtadaaPosition_total.dat');
XY_Normal = Normalize(XY, 0);
[m n] = size(XY_Normal);
% -------------------Running PCA --------------------------
fprintf('[INFO]\tRUNNING PCA...\n');
K = n;
retained_variance = 0.0;
p_retained_variance = 0.0;
i=1;
while (i<n) && (retained_variance < 99.0)
    [Z eigenvalue p_retained_variance retained_variance ] = Extract_PCA( XY, i );
    K=i;
    %fprintf('\t\tPC N°: %d |Eigenvalue: %10.2f |Variance: %2.2f%%|Total Variance: %3.2f%%\n', K,eigenvalue,p_retained_variance,retained_variance );
    i=i+1;
end
fprintf('\t\tOptimal PC: Z-Dimension = %d | Retained Variance = %0.2f%%\n', K,retained_variance );
[Z retained_variance ] = Extract_PCA( XY, K );
%XY_Normal = Z;
[m n] = size(XY_Normal);
% -------------------End Running PCA --------------------------
XY_Normal = [ones(m,1) XY_Normal];
%% ============== 3- SETUP TRAIN CV TEST SETS ===============
fprintf('[INFO]\tSETUP TRAIN, CV AND TEST SETS...\n');
[x_train y_train x_cv y_cv x_test y_test] =...
    ExtractTrainCvTestSets(...
    XY_Normal,70, 15, 15,...
    1, 0);
%% =============== 4- INITIALIZE NN PARAMS =================
fprintf('[INFO]\tINITIALIZE NN PARAMS. ...\n');
input_layer_size = n;
hidden_layer_size = 2;
num_labels = length(unique(y_train));
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
iterations =  [20:20:1000]; 
lambdas = [0.00001 0.00003  0.0003 0.001 0.003 0.01 0.03 0.1 0.3];
opt_iteration = 0;
opt_alpha = 0;
opt_accuracy = 0;
accuracyList = [];
totalAccuarcyList = zeros(1,length(iterations));
%% ====================== 5- TRAINING NN ======================
fprintf('[INFO]\tTRAINING NN...\n');
for lambda = lambdas
    for iteration = iterations
        % ----------------- Compute Thetas --------------
        options = optimset('MaxIter', iteration);
        costFunction = @(p) iParseCostFunction(p, input_layer_size, hidden_layer_size, num_labels, x_train, y_train, lambda);
        [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
        
        % ----------------- Reshape Thetas --------------
        Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
        Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
        
        % ----------------- Predict on Cross Validation Set --------------
        pred = predict(Theta1, Theta2, x_cv);
        accuracy = mean(double(pred == y_cv)) * 100;
        accuracyList = [accuracyList accuracy ];
        fprintf('\t\tIter.=%4.0d | Lambda=%0.5f | Accuracy=%0.2f%%', iteration, lambda, accuracy);
        if(accuracy > opt_accuracy)
            opt_accuracy = accuracy;
            opt_lambda = lambda;
            opt_iteration = iteration;
            fprintf('\t[Optimal Params.]');
        end        
        fprintf('\n');
    end
    totalAccuarcyList = vertcat(totalAccuarcyList, accuracyList);
    accuracyList = [];
end
fprintf('[INFO]\tPLOT NN TRAINING PARAMETERS...\n');
figure;
surf(iterations, lambdas, totalAccuarcyList(2:length(lambdas)+1,:),'FaceColor','interp','FaceLighting','phong');
ylabel('\lambda');
xlabel('Iterations')
zlabel('Accuarcy');
%% =========== 6- TEST NN WITH OPTIMAL PARAMS. =============
fprintf('[INFO]\tTEST NN WITH OPTIMAL PARAMS. ...\n');
% ----------------- Compte Thetas --------------
options = optimset('MaxIter', opt_iteration);
costFunction = @(p) iParseCostFunction(p, input_layer_size, hidden_layer_size, num_labels, x_train, y_train, opt_lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% ----------------- Reshape Thetas --------------
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% ----------------- Predict on Test Set --------------
pred = predict(Theta1, Theta2, x_test);
accuracy = mean(double(pred == y_test)) * 100;
fprintf('\t\tNN Accuracy with Optimal Params. = %0.2f%%\n', accuracy);
%% =================== 7- Plotting Learning Curves I ======================
fprintf('[INFO]\tLearning Curves I : [Train_Set, CV_Set] = Error(Train_Size)...\n');
train_sizes = 20:20:size(x_train,1);
total_cost_cv = [];
total_cost_dev = [];
options = optimset('MaxIter', opt_iteration);
fprintf('\t\tProgression : 10%%|');
ranges = [10 20 30 40 50 60 70 80 90 100]; range = 1;
for train_size = train_sizes
    
    p = round((train_size/max(train_sizes))*100);
    if( p > ranges(range))        
        fprintf('%d%%|', ranges(range+1));
        range = range + 1 ;
    end;    
    
    xx_train = x_train(1:train_size,:);
    yy_train = y_train(1:train_size);
    costFunction = @(p) iParseCostFunction(p, ...
        input_layer_size, ...
        hidden_layer_size, ...
        num_labels, xx_train, yy_train, opt_lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_labels, (hidden_layer_size + 1));
    %----------------------------------
    [JJ GG JL] = iParseCostFunction(nn_params,input_layer_size, ...
        hidden_layer_size, ...
        num_labels, xx_train, yy_train, opt_lambda);
    total_cost_dev = [total_cost_dev JL];    
    %-----------------------------------
     [JJ GG JL] = iParseCostFunction(nn_params,input_layer_size, ...
        hidden_layer_size, ...
        num_labels, x_cv, y_cv, opt_lambda);
    total_cost_cv = [total_cost_cv JL];
    %-----------------------------------
end
fprintf('\n\t\tPlot Error on Training and CV Set...\n');
figure;
plot(total_cost_dev, '-*b', 'MarkerSize',3, 'MarkerFaceColor','b');
hold on;

plot(total_cost_cv, '-sm', 'MarkerSize',3, 'MarkerFaceColor','m');
legend('J_{train}(\theta)','J_{cv}(\theta)');
xlabel('Training Set Size');
ylabel('Error');
title('Learning Curves I : Error(Train Set, CV Set) = f(Train Size)');
hold off;

%% =================== 8- Plotting Learning Curves II =====================
fprintf('[INFO]\tLearning Curves II : [Train_Set, CV_Set] = Error(Iterations)...\n');
Iterations = 100:100:1000;
total_cost_cv = [];
total_cost_dev = [];
fprintf('\t\tProgression : 10%%|');
range = 1;
for Iteration = Iterations
    
     p = round((train_size/max(train_sizes))*100);
    if( p > ranges(range))        
        fprintf('%d%%|', ranges(range+1));
        range = range + 1 ;
    end;    
    
    
    options = optimset('MaxIter', Iteration);   
    costFunction = @(p) iParseCostFunction(p, ...
        input_layer_size, ...
        hidden_layer_size, ...
        num_labels, x_train, y_train, opt_lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_labels, (hidden_layer_size + 1));
    %----------------------------------
    [JJ GG JL] = iParseCostFunction(nn_params,input_layer_size, ...
        hidden_layer_size, ...
        num_labels, x_train, y_train, opt_lambda);
    total_cost_dev = [total_cost_dev JL];    
    %-----------------------------------
     [JJ GG JL] = iParseCostFunction(nn_params,input_layer_size, ...
        hidden_layer_size, ...
        num_labels, x_cv, y_cv, opt_lambda);
    total_cost_cv = [total_cost_cv JL];
    %-----------------------------------
end
fprintf('\n\t\tPlot Error on Training and CV Set...\n');
figure;
plot(total_cost_dev, '-*b', 'MarkerSize',3, 'MarkerFaceColor','b');
hold on;

plot(total_cost_cv, '-sm', 'MarkerSize',3, 'MarkerFaceColor','m');
legend('J_{train}(\theta)','J_{cv}(\theta)');
xlabel('iterations');
ylabel('Error');
title('Learning Curves II : Error(Train Set, CV Set) = f(iterations)');
%%
fprintf('[INFO]\tEnd.\n');