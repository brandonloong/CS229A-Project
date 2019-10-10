function p = predictLogistic(theta, X)
  m = size(X, 1);               
  p = zeros(m, 1);
  t = 0.5;

  p = (sigmoid(X * theta) >= t);
end