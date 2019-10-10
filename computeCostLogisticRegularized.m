function [J, grad] = computeCostLogisticRegularized(theta, X, y, lambda)

  m = length(y);                
  J = 0;                        
  grad = zeros(size(theta));    

  temp = theta' * X';
  h = (sigmoid(temp));
  theta_temp = theta(2:end);
  J = (1/m) * (-y' * log(h') - (1-y)' * log(1-h)') + (lambda/(2*m)) * theta_temp' * theta_temp;
  grad = (1/m) * (X' * (h' - y));
  fix = ones(size(grad));
  fix(1,1) = 0;
  reg = fix  * lambda/m .* theta;
  grad = grad + reg;
end