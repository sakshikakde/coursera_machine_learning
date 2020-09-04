function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;
J_i = 0.0;
J_reg = 0.0;

t = h - y;
J_i = (1/(2*m)) * (transpose(t) * t);

theta_t = theta;
theta_t(1,1) = 0.0;
J_reg = (lambda/(2*m)) * (transpose(theta_t) * theta_t);

J = J_i + J_reg;

% =========================================================================

grad = (1/m) * transpose(X) * t;
grad = grad + ((lambda/m) * theta_t);
grad = grad(:);

end
