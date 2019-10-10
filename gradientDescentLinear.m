function [theta, J_history] = gradientDescentLinear(X, y, theta, alpha, iter)
  m = length(y);                  
  J_history = zeros(iter, 1);
  
  for i = 1:iter
    delta = (1/m) * (X' * (X*theta-y)); 
    theta = theta - alpha * delta;
    J_history(i) = computeCostLinear(X, y, theta);  
  end
end