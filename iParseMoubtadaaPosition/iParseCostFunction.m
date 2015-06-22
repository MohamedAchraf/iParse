function [J grad J_learning] = iParseCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   x, y, lambda)


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(x, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


	x = [ones(m,1) x];  
	a2 = sigmoid(Theta1 * x');
	a2 = [ones(m,1) a2'];
	h_theta = sigmoid(Theta2 * a2'); % h_theta equals z3
	yk = zeros(num_labels, m); 
	
    for i=1:m	
        yk(y(i),i)=1;	
	end

	J = (1/m) * sum ( sum (  (-yk) .* log(h_theta)  -  (1-yk) .* log(1-h_theta) ));
    J_learning = (1/m) * sum ( sum (  (-yk) .* log(h_theta)  -  (1-yk) .* log(1-h_theta) ));


	t1 = Theta1(:,2:size(Theta1,2));
	t2 = Theta2(:,2:size(Theta2,2));

	Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);
	J = J + Reg;

	for t=1:m,
	a1 = x(t,:); 
	z2 = Theta1 * a1';

	a2 = sigmoid(z2);
	a2 = [1 ; a2]; % add bias

	z3 = Theta2 * a2;

	a3 = sigmoid(z3); % final activation layer a3 == h(theta)

	
	
	z2=[1; z2];

	delta_3 = a3 - yk(:,t);
	delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2);

	delta_2 = delta_2(2:end); 

	Theta1_grad = Theta1_grad + delta_2 * a1; 
	Theta2_grad = Theta2_grad + delta_3 * a2';

	end;


	Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
	
	Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));
	
	
	Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
	
	Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
