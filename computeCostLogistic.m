function [J, grad] = computeCostLogistic(theta, X, y)
  m = length(y);                   
  J = 0;                           
  grad = zeros(size(theta));       

  temp = theta' * X' ;
  h = (sigmoid(temp));
  grad = (1/m) * (X' * (h' - y));  
  J = (1/m) * (-y' * log(h') - (1-y)' * log(1-h)');
end
