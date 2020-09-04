function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m,1), X];
z2 = a1 * transpose(Theta1);
a2 = sigmoid(z2);

a2 = [ones(size(a2,1),1) a2];
z3 = a2 * transpose(Theta2);
a3 = h_net = sigmoid(z3);

y(y == 0) = 10;
y_new = zeros(size(y,1), num_labels);

for n = 1:m
  y_new(n,y(n)) = 1;
end
J_i = 0;
for k = 1:num_labels
  h = h_net(:,k);
  y_temp = y_new(:,k);  
  log_h = log(h);
  one_minus_h = ones(size(h,1),1) - h;
  log_one_minus_h = log(one_minus_h) ;
  J_i =  J_i + (-1/m * ((transpose(y_temp) * log_h)+(transpose(ones(m,1)-y_temp)* log_one_minus_h)));
end

J_reg = 0;
theta_1_sq_sum = 0;
for n = 1:hidden_layer_size
  theta_1_temp = Theta1(n,:);
  theta_1_temp(1,1) = 0.0;
  theta_1_sq_sum = theta_1_sq_sum + (theta_1_temp * transpose(theta_1_temp));

end

theta_2_sq_sum = 0;
for n = 1:num_labels
  theta_2_temp = Theta2(n,:);
  theta_2_temp(1,1) = 0.0;
  theta_2_sq_sum = theta_2_sq_sum + (theta_2_temp * transpose(theta_2_temp));
end

J_reg = (lambda / (2 * m)) * (theta_1_sq_sum + theta_2_sq_sum);

J = J_i + J_reg;
% -------------------------------------------------------------
%BACK PROPAGATION
Theta1_n = Theta1(:,2:end); 
Theta2_n = Theta2(:,2:end);
 
del_3 = a3 - y_new;
del_2 = del_3 * Theta2_n .* sigmoidGradient(z2);

Del_1 = transpose(a1) * del_2;
Del_2 = transpose(a2) * del_3;

Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = 1/m * transpose(Del_1) + (lambda/m) * Theta1;
Theta2_grad = 1/m * transpose(Del_2) + (lambda/m) * Theta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end