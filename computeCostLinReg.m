function [J,grad] = computeCostLinReg(X, y, theta, lambda)
  m = length(y);
  grad = zeros(size(theta));
  
  h = X*theta;
  e = h-y;
  theta2 = theta(2:end);
  
  J = 1/(2*m) * (e'*e + lambda*(theta2'*theta2));
  grad(1) = 1/m * X(:,1)'*e;
  grad(2:end)= 1/m * X(:,2:end)'*e + lambda/m*theta2;
  grad = grad(:);  
end