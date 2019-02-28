function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
result1 = X*theta;
result2 = sigmoid(result1);
result = log(result2);
y1 = -y;
t1 = y1'*result;
y2 = 1-y;
result3 = log(1-result2);
t2 = y2'*result3;
result4 = theta'*theta-(theta(1))^2;
J = (t1-t2)/m + (lambda*result4)/(2*m);

grad1 = (lambda/m)*theta;
grad1(1) = (m/lambda)*grad(1);
grad = (X'*(result2-y))/m+grad1;



% =============================================================

end
