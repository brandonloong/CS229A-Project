function g = sigmoidGradient(z)

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================

sig = sigmoid(z);
g = sig .* (1 - sig);

% =============================================================

end