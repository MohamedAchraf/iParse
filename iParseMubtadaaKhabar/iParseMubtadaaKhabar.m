clear ; close all; clc; 
%% ===============Initialization ===================
fprintf('[INFO]\tInitialization...\n');
set(0, 'defaultFigureWindowStyle','Docked');
warning('off','all');
addpath('D:\ML\iparse');
%% ============ Setup NN parameters =================
fprintf('[INFO]\tSetup NN parameters...\n');
input_layer_size  = 12;   % 1x12 Input
hidden_layer_size = 25;   % 25 hidden units
num_labels = 2;           % 2 labels
%-------- Initial thetas
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =========== Loading, Normalization and Dividing Data =============
fprintf('[INFO]\tLoading Data...\n');
xy = load('iParse.dat');
x = xy(:,1:input_layer_size);

fprintf('[INFO]\tNormalization...\n');
mu = mean(x);
sigma = std(x);
for i=1:size(x,2)
    x(:,i) = ( x(:,i) - mu(i) )./ sigma(i);
end

fprintf('[INFO]\tSetup Train Set, Validation Set and Test Set...\n');
y = xy(:,input_layer_size+1);
m = size(x, 1);
[x_dev y_dev x_cv y_cv x_test y_test] = divide_data(x, y);

%% =================== Training NN ===================
fprintf('[INFO]\tTraining NN...\n');
iterations = 20:20:100;%[1:9 10:5:45 50:10:90 100:100:500 600:200:1000];
lambdas =  [0.001 0.003 0.01 0.03 0.1 0.3 1] ;
accuracyList = [];
totalAccuarcyList = zeros(1,length(iterations));
maxAccuracy = 0;
optLambda = 0;
optIteration = 0;
for lambda = lambdas
    for iteration = iterations
        options = optimset('MaxIter', iteration);
        costFunction = @(p) iParseCostFunction(p, ...
            input_layer_size, ...
            hidden_layer_size, ...
            num_labels, x_dev, y_dev, lambda);
        [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
        
        Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
            hidden_layer_size, (input_layer_size + 1));
        
        Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
            num_labels, (hidden_layer_size + 1));
        %% =============== Predict on Cross Validation Set ===============
        pred = predict(Theta1, Theta2, x_cv);
        accuracy = mean(double(pred == y_cv)) * 100;
        accuracyList = [accuracyList accuracy ];
        fprintf('\t\tLambda=%0.4f | Iter.=%4.0d | Accuracy=%0.2f%%', lambda, iteration, accuracy);
        if(accuracy > maxAccuracy)
            maxAccuracy = accuracy;
            optLambda = lambda;
            optIteration = iteration;
            fprintf('\t[Optimal Params.]');
        end
        fprintf('\n');
    end
    totalAccuarcyList = vertcat(totalAccuarcyList, accuracyList);
    accuracyList = [];
end
fprintf('\t\tOptimal Params. : Lambda=%0.4f | Iter.=%4.0d | Accuracy=%0.2f%%\n', optLambda, optIteration, maxAccuracy);
fprintf('[INFO]\tPlot NN Training Parameters...\n');
figure;
surf(iterations, lambdas, totalAccuarcyList(2:length(lambdas)+1,:),'FaceColor','interp','FaceLighting','phong');
ylabel('\lambda');
xlabel('Iterations')
zlabel('Accuarcy');
fprintf('\t\tPress Enter to continue..\n');
pause;
%% ==================== Train with optimal parameters ===================
fprintf('[INFO]\tTraining NN with Optimal values...\n');
options = optimset('MaxIter', optIteration);
costFunction = @(p) iParseCostFunction(p, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, x_dev, y_dev, optLambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));
%% ========================  Predict On test set =========================
fprintf('[INFO]\tAccuracy on Test Set = %0.2f%% \n', accuracy);
pred = predict(Theta1, Theta2, x_test);
accuracy = mean(double(pred == y_test)) * 100;
%% ==================== Plotting Learning Curves I =========================
fprintf('[INFO]\tLearning Curves I : [Train_Set, CV_Set] = Error(Train_Size)...\n');
train_sizes = 20:20:size(x_dev,1);
total_cost_cv = [];
total_cost_dev = [];
options = optimset('MaxIter', optIteration);
fprintf('\t\tProgression :');
ranges = [10 20 30 40 50 60 70 80 90 100];
range = 1;
for train_size = train_sizes
    
    p = round((train_size/max(train_sizes))*100);
    if( p > ranges(range))        
        fprintf('%d%%|', ranges(range+1));
        range = range + 1 ;
    end;    

    xx_dev = x_dev(1:train_size,:);
    yy_dev = y_dev(1:train_size);
    costFunction = @(p) iParseCostFunction(p, ...
        input_layer_size, ...
        hidden_layer_size, ...
        num_labels, xx_dev, yy_dev, optLambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_labels, (hidden_layer_size + 1));
    %----------------------------------
    [JJ GG JL] = iParseCostFunction(nn_params,input_layer_size, ...
        hidden_layer_size, ...
        num_labels, xx_dev, yy_dev, optLambda);
    total_cost_dev = [total_cost_dev JL];    
    %-----------------------------------
     [JJ GG JL] = iParseCostFunction(nn_params,input_layer_size, ...
        hidden_layer_size, ...
        num_labels, x_cv, y_cv, optLambda);
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
fprintf('\t\tPress Enter to continue..\n');
pause;
%% ==================== Plotting Learning Curves II =======================
fprintf('[INFO]\tLearning Curves II : [Train_Set, CV_Set] = Error(Iterations)...\n');
Iterations = 20:20:1000; %[1:9 10:5:45 50:10:90 100:100:500 600:200:1000 ];

total_cost_cv = [];
total_cost_dev = [];
fprintf('\t\tProgression :');
range = 1;
for Iteration = Iterations
    
     p = round((Iteration/max(Iterations))*100);
    if( p > ranges(range))        
        fprintf('%d%%|', ranges(range+1));
        range = range + 1 ;
    end;    
    
    options = optimset('MaxIter', Iteration);   
    costFunction = @(p) iParseCostFunction(p, ...
        input_layer_size, ...
        hidden_layer_size, ...
        num_labels, x_dev, y_dev, optLambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_labels, (hidden_layer_size + 1));
    %----------------------------------
    [JJ GG JL] = iParseCostFunction(nn_params,input_layer_size, ...
        hidden_layer_size, ...
        num_labels, x_dev, y_dev, optLambda);
    total_cost_dev = [total_cost_dev JL];    
    %-----------------------------------
     [JJ GG JL] = iParseCostFunction(nn_params,input_layer_size, ...
        hidden_layer_size, ...
        num_labels, x_cv, y_cv, optLambda);
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